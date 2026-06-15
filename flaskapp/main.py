from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from backend.routers import test # test router

import json as _json
import importlib
import logging
import os
import time
import urllib.request as _urlrequest
import urllib.error as _urlerror
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Load environment variables from .env before importing modules that depend on them
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
load_dotenv()

auth = importlib.import_module("backend.routers.auth")
memories = importlib.import_module("backend.routers.memories")
dashboard = importlib.import_module("backend.routers.dashboard")
spotify = importlib.import_module("backend.routers.spotify")
ai_features = importlib.import_module("backend.routers.ai_features")

_nlp_processor = importlib.import_module("backend.nlp_processor")
process_unprocessed_memories = _nlp_processor.process_unprocessed_memories
_hf_inference_endpoints = _nlp_processor._hf_inference_endpoints
_get_hf_emotion_model_id = _nlp_processor._get_hf_emotion_model_id
_get_hf_timeout_seconds = _nlp_processor._get_hf_timeout_seconds

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DMJ Backend")

HTTP_REQUESTS_TOTAL = Counter(
    "dmj_http_requests_total",
    "Total HTTP requests handled by the backend",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "dmj_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

NLP_JOB_TOTAL = Counter(
    "dmj_nlp_jobs_total",
    "Total NLP background jobs processed",
    ["status"],
)

NLP_JOB_DURATION_SECONDS = Histogram(
    "dmj_nlp_job_duration_seconds",
    "NLP background job duration in seconds",
)

NLP_JOB_RUNNING = Gauge(
    "dmj_nlp_job_running",
    "Whether the NLP background job is currently running",
)

# Allow local dev frontend to call the API on any local port.
# You can override with CORS_ALLOW_ORIGINS="http://localhost:3000,http://127.0.0.1:3001"
configured_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS","https://v0-djjv2.vercel.app,https://dmemoryjar.vercel.app").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    route = request.scope.get("route")
    path = getattr(route, "path_format", request.url.path)

    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        path=path,
        status=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        path=path,
    ).observe(duration)

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ai/model-health")
def model_health():
    """Lightweight model availability check.

    - If `HF_API_TOKEN` is not set the endpoint reports `degraded` (fallbacks used).
    - If set, the endpoint will attempt a short inference call to the configured emotion model
      via the available inference endpoints and return `ok` if any respond successfully.
    """
    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        return {"status": "degraded", "reason": "HF_API_TOKEN not set; using local fallbacks"}

    model_id = _get_hf_emotion_model_id()
    endpoints = _hf_inference_endpoints(model_id)
    timeout = min(_get_hf_timeout_seconds(), 5)

    body = _json.dumps({"inputs": "test", "options": {"wait_for_model": True}}).encode("utf-8")

    last_error = None
    for endpoint in endpoints:
        req = _urlrequest.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
            },
        )

        try:
            with _urlrequest.urlopen(req, timeout=timeout) as res:  # nosec B310
                payload = _json.loads(res.read().decode("utf-8"))

            if isinstance(payload, dict) and payload.get("error"):
                last_error = payload.get("error")
                continue

            return {"status": "ok", "endpoint": endpoint}

        except _urlerror.HTTPError as e:
            last_error = f"HTTP {e.code} via {endpoint}"
        except _urlerror.URLError as e:
            last_error = f"Network error via {endpoint}: {getattr(e, 'reason', str(e))}"
        except Exception as e:
            last_error = str(e)

    return {"status": "unavailable", "reason": "all endpoints failed", "detail": last_error}


@app.get("/")
def root():
    return {
        "service": "DMJ Backend",
        "status": "ok",
        "health": "/healthz",
        "docs": "/docs",
        "test": "/test",
    }


app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(memories.router, prefix="/memories", tags=["memories"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(spotify.router, prefix="/spotify", tags=["spotify"])
app.include_router(ai_features.router, prefix="/ai", tags=["ai"])
app.include_router(test.router)


scheduler = BackgroundScheduler()


def _scheduler_enabled() -> bool:
    return os.getenv("ENABLE_NLP_SCHEDULER", "false").strip().lower() in {"1", "true", "yes", "on"}


def _scheduler_interval_seconds() -> int:
    value = os.getenv("NLP_SCHEDULER_INTERVAL_SECONDS", "30")
    try:
        seconds = int(value)
        return max(seconds, 10)
    except ValueError:
        return 30


def process_memories_job():
    """Job that processes unprocessed memories in the background."""
    NLP_JOB_RUNNING.set(1)
    start_time = time.perf_counter()
    try:
        process_unprocessed_memories()
        NLP_JOB_TOTAL.labels(status="success").inc()
    except Exception as e:
        NLP_JOB_TOTAL.labels(status="error").inc()
        logger.error(f"Error in memory processing job: {e}")
    finally:
        NLP_JOB_DURATION_SECONDS.observe(time.perf_counter() - start_time)
        NLP_JOB_RUNNING.set(0)


@app.on_event("startup")
def startup_event():
    if _scheduler_enabled():
        scheduler.add_job(
            process_memories_job,
            "interval",
            seconds=_scheduler_interval_seconds(),
            id="process_memories",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Application startup - NLP scheduler running")
    else:
        logger.info("Application startup - NLP scheduler disabled")


@app.on_event("shutdown")
def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Application shutdown - NLP scheduler stopped")
    else:
        logger.info("Application shutdown")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)