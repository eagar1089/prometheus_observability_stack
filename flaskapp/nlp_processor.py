"""
AI NLP Processing Pipeline Template
1. Text Preprocessing & Cleaning (text_preprocessor.py)
2. Emotion Analysis
3. Keyword & Topic Extraction
4. Entity Recognition
5. Embedding Generation
6. Store in MongoDB + FAISS
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging
import json
import os
import math
from urllib import error, request

from backend.connection import get_collection
from backend.crud import update_memory_with_nlp
from backend.text_preprocessor import TextPreprocessor, preprocess_unprocessed_memories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper functions for Hugging Face API calls
# ----------------------------------------------------------------------
def _get_hf_timeout_seconds() -> int:
    value = os.getenv("HF_INFERENCE_TIMEOUT_SECONDS", "20")
    try:
        return int(value)
    except ValueError:
        return 20


def _get_hf_emotion_model_id() -> str:
    return os.getenv("HF_EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base").strip()


def _get_hf_topic_model_id() -> str:
    return os.getenv("HF_TOPIC_MODEL", "facebook/bart-large-mnli").strip()


def _hf_inference_endpoints(model_id: str) -> List[str]:
    explicit_base = os.getenv("HF_INFERENCE_BASE_URL", "").strip().rstrip("/")
    endpoints: List[str] = []
    if explicit_base:
        endpoints.append(f"{explicit_base}/{model_id}")

    endpoints.extend([
        f"https://router.huggingface.co/hf-inference/models/{model_id}",
        f"https://api-inference.huggingface.co/models/{model_id}",
    ])

    deduped: List[str] = []
    seen = set()
    for endpoint in endpoints:
        if endpoint not in seen:
            deduped.append(endpoint)
            seen.add(endpoint)
    return deduped


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------
def _get_keybert_top_n() -> int:
    value = os.getenv("KEYBERT_TOP_N", "8")
    try:
        parsed = int(value)
        return max(1, min(parsed, 20))
    except ValueError:
        return 8


def _get_topic_candidate_labels() -> List[str]:
    return [
        "Work & Productivity",
        "Health & Wellness",
        "Emotions & Mental Health",
        "Relationships & Family",
        "Learning & Growth",
        "Finance",
        "Travel & Leisure",
        "Daily Life",
    ]
def _get_topic_score_threshold() -> float:
    value = os.getenv("TOPIC_MIN_SCORE", "0.2")
    try:
        parsed = float(value)
        return max(0.0, min(parsed, 1.0))
    except ValueError:
        return 0.2


def _get_topic_max_labels() -> int:
    value = os.getenv("TOPIC_MAX_LABELS", "2")
    try:
        parsed = int(value)
        return max(1, min(parsed, 5))
    except ValueError:
        return 2


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------
def _flatten_hf_labels(payload: object) -> List[Dict[str, float]]:
    if not isinstance(payload, list):
        return []

    if payload and isinstance(payload[0], list):
        candidates = payload[0]
    else:
        candidates = payload

    parsed: List[Dict[str, float]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        score = item.get("score", 0.0)
        try:
            parsed.append({"label": label, "score": float(score)})
        except (TypeError, ValueError):
            continue
    return parsed


def _dedupe_text_items(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for item in items:
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


# ----------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess(text)
    return result["cleaned"]


# ----------------------------------------------------------------------
# Emotion scoring (Hugging Face API)
# ----------------------------------------------------------------------
EMOTION_BUCKET_LABELS = {
    "joy": {"joy", "amusement", "excitement", "optimism", "contentment", "happy", "excited", "content"},
    "sadness": {"sadness", "disappointment", "grief", "remorse", "hurt", "lonely", "disappointed"},
    "anger": {"anger", "annoyance", "rage", "frustration", "frustrated", "annoyed", "furious"},
    "fear": {"fear", "nervousness", "anxiety", "worry", "anxious", "nervous", "worried"},
    "surprise": {"surprise", "realization", "amazed", "amaze", "shocked"},
    "disgust": {"disgust", "disapproval", "embarrassment", "dislike", "uncomfortable"},
}
def _neutral_emotion_scores() -> Dict[str, float]:
    return {
        "joy": 0.0,
        "sadness": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "surprise": 0.0,
        "disgust": 0.0,
    }


FALLBACK_EMOTION_KEYWORDS = {
    "joy": [
        "happy", "joy", "grateful", "excited", "peaceful", "calm", "good", "great", "love", "relaxed",
        "productive", "proud", "hopeful", "content",
    ],
    "sadness": [
        "sad", "down", "lonely", "empty", "tired", "cry", "hopeless", "upset", "depressed", "hurt",
        "heavy", "off", "invisible", "burden", "burdened", "drained", "numb", "stuck", "foggy",
        "weighed", "overwhelmed", "creeping", "isolated", "low",
    ],
    "anger": [
        "angry", "mad", "furious", "annoyed", "frustrated", "irritated", "rage", "hate",
    ],
    "fear": [
        "afraid", "fear", "anxious", "worried", "panic", "nervous", "scared", "stress", "stressed",
    ],
    "surprise": [
        "surprised", "unexpected", "suddenly", "shocked", "amazed", "wow",
    ],
    "disgust": [
        "disgust", "gross", "nasty", "awful", "sick", "uncomfortable",
    ],
}


def _fallback_emotion_scores(text: str) -> Dict[str, float]:
    scores = _neutral_emotion_scores()
    normalized = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return scores

    token_set = set(tokens)
    for emotion, keywords in FALLBACK_EMOTION_KEYWORDS.items():
        count = 0
        for token in tokens:
            if token in keywords:
                count += 1

        # Capture a few multi-word feelings that show up often in journal entries.
        if emotion == "sadness":
            if "felt" in token_set and "heavy" in token_set:
                count += 1
            if "felt" in token_set and "off" in token_set:
                count += 1
            if "creeping" in token_set and "back" in token_set:
                count += 1

        if count > 0:
            scores[emotion] = float(count)

    total = sum(scores.values())
    if total <= 0:
        return scores

    return {label: round(value / total, 4) for label, value in scores.items()}


def dominant_mood_from_scores(emotion_scores: Optional[Dict[str, float]]) -> str:
    if not emotion_scores:
        return "neutral"

    normalized: Dict[str, float] = {}
    for label, value in emotion_scores.items():
        try:
            normalized[str(label).lower()] = float(value)
        except (TypeError, ValueError):
            continue

    if not normalized:
        return "neutral"

    max_score = max(normalized.values())
    if max_score <= 0:
        return "neutral"

    top_labels = [label for label, score in normalized.items() if score == max_score]
    if len(top_labels) != 1:
        return "neutral"

    top_label = top_labels[0]
    if top_label in {"happy", "contentment", "optimism", "amusement", "excitement"}:
        return "joy"
    return top_label


def _bucketize_emotions(label_scores: List[Dict[str, float]]) -> Dict[str, float]:
    bucket_scores = _neutral_emotion_scores()
    for item in label_scores:
        label = item["label"]
        score = float(item["score"])
        for bucket, aliases in EMOTION_BUCKET_LABELS.items():
            if label in aliases:
                bucket_scores[bucket] += score
                break

    total = sum(bucket_scores.values())
    if total > 0:
        return {k: round(v / total, 4) for k, v in bucket_scores.items()}
    return bucket_scores


def extract_emotion_scores(text: str) -> Dict[str, float]:
    if not text or not text.strip():
        return _neutral_emotion_scores()

    hf_api_token = os.getenv("HF_API_TOKEN")
    if not hf_api_token:
        logger.warning("HF_API_TOKEN missing. Using fallback lexical emotion scoring.")
        return _fallback_emotion_scores(text)

    hf_timeout_seconds = _get_hf_timeout_seconds()
    model_id = _get_hf_emotion_model_id()
    body = json.dumps({"inputs": text, "options": {"wait_for_model": True}}).encode("utf-8")

    for endpoint in _hf_inference_endpoints(model_id):
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {hf_api_token}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=hf_timeout_seconds) as res:  # nosec B310
                payload = json.loads(res.read().decode("utf-8"))

            if isinstance(payload, dict) and payload.get("error"):
                logger.warning("Hugging Face API error from %s: %s", endpoint, payload.get("error"))
                continue

            label_scores = _flatten_hf_labels(payload)
            if label_scores:
                return _bucketize_emotions(label_scores)

        except error.HTTPError as e:
            logger.warning("HF HTTP error (%s) via %s", e.code, endpoint)
        except error.URLError as e:
            logger.warning("HF network error via %s: %s", endpoint, e.reason)
        except Exception as e:
            logger.warning("HF emotion scoring error via %s: %s", endpoint, str(e))

    logger.warning("Hugging Face emotion inference unavailable. Using fallback lexical emotion scoring.")
    return _fallback_emotion_scores(text)


# ----------------------------------------------------------------------
# Keyword extraction (lightweight)
# ----------------------------------------------------------------------
def extract_keywords(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    top_n = _get_keybert_top_n()
    try:
        preprocessor = TextPreprocessor()
        return preprocessor.extract_keywords(text, top_n=top_n)
    except Exception as e:
        logger.error("Keyword extraction failed: %s", str(e))
        return []


# ----------------------------------------------------------------------
# Topic categorization
# ----------------------------------------------------------------------
def _fallback_topic_classification(text: str) -> List[str]:
    topics = []
    work_keywords = ["work", "email", "project", "deliverable", "deadline"]
    health_keywords = ["walk", "exercise", "sleep", "health", "tired"]
    mood_keywords = ["grateful", "happy", "sad", "anxious", "stressed"]

    text_lower = text.lower()
    if any(k in text_lower for k in work_keywords):
        topics.append("Work & Productivity")
    if any(k in text_lower for k in health_keywords):
        topics.append("Health & Wellness")
    if any(k in text_lower for k in mood_keywords):
        topics.append("Emotions & Mental Health")

    return topics or ["Daily Life"]


def categorize_topics(text: str, keywords: List[str]) -> List[str]:
    if not text or not text.strip():
        return ["Daily Life"]

    candidate_labels = _get_topic_candidate_labels()
    min_score = _get_topic_score_threshold()
    max_labels = _get_topic_max_labels()
    text_for_classification = text
    if keywords:
        text_for_classification = f"{text}\nKeywords: {', '.join(keywords[:10])}"

    hf_api_token = os.getenv("HF_API_TOKEN")
    if not hf_api_token:
        logger.warning("HF_API_TOKEN missing. Using fallback topic classification.")
        return _fallback_topic_classification(text)

    hf_timeout = _get_hf_timeout_seconds()
    model_id = _get_hf_topic_model_id()

    try:
        body = json.dumps(
            {
                "inputs": text_for_classification,
                "parameters": {"candidate_labels": candidate_labels, "multi_label": True},
                "options": {"wait_for_model": True},
            }
        ).encode("utf-8")

        for endpoint in _hf_inference_endpoints(model_id):
            req = request.Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {hf_api_token}",
                    "Content-Type": "application/json",
                },
            )

            try:
                with request.urlopen(req, timeout=hf_timeout) as res:  # nosec B310
                    result = json.loads(res.read().decode("utf-8"))

                if isinstance(result, dict) and result.get("error"):
                    logger.warning("HF topic API error: %s", result.get("error"))
                    continue

                labels = result.get("labels", []) if isinstance(result, dict) else []
                scores = result.get("scores", []) if isinstance(result, dict) else []

                ranked_topics: List[str] = []
                for label, score in zip(labels, scores):
                    if float(score) >= min_score:
                        ranked_topics.append(str(label))
                    if len(ranked_topics) >= max_labels:
                        break

                if ranked_topics:
                    return ranked_topics
                if labels:
                    return [str(labels[0])]

            except Exception as e:
                logger.warning("HF topic request failed via %s: %s", endpoint, str(e))
                continue

    except Exception as e:
        logger.error("Topic classification failed: %s", str(e))

    return _fallback_topic_classification(text)


# ----------------------------------------------------------------------
# Named Entity Recognition (disabled for memory)
# ----------------------------------------------------------------------
def extract_entities(text: str) -> List[str]:
    logger.info("NER disabled for memory optimization")
    return []


# ----------------------------------------------------------------------
# Embedding generation (lightweight deterministic)
# ----------------------------------------------------------------------
def generate_embedding(text: str) -> Dict:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimensions = 64
    vector = [0.0] * dimensions

    tokens = [token for token in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split() if token]
    if not tokens:
        return {"vector": vector, "model": model_name}

    for token in tokens:
        index = hash(token) % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        vector = [round(value / norm, 6) for value in vector]

    return {"vector": vector, "model": model_name}


# ----------------------------------------------------------------------
# Embedding storage (Mongo-backed fallback)
# ----------------------------------------------------------------------
def store_embedding_in_faiss(vector: List[float], memory_id: str, faiss_index) -> int:
    embedding_collection = get_collection("memory_embeddings")
    now = datetime.utcnow()

    embedding_collection.update_one(
        {"memory_id": memory_id},
        {
            "$set": {
                "memory_id": memory_id,
                "vector": vector,
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
    )

    return abs(hash(memory_id)) % 1_000_000_000


# ----------------------------------------------------------------------
# Main processing loop (called by scheduler)
# ----------------------------------------------------------------------
def process_unprocessed_memories(batch_size: int = 50) -> Dict:
    col = get_collection("memories")

    preprocessing_result = preprocess_unprocessed_memories(batch_size)

    unprocessed = list(
        col.find(
            {
                "preprocessing": {"$exists": True},
                "nlp_insights": {"$exists": False},
            }
        ).limit(batch_size)
    )

    processed_count = 0
    failed_count = 0
    errors = []

    for memory in unprocessed:
        try:
            memory_id = str(memory["_id"])
            preprocessed = memory.get("preprocessing", {})
            cleaned_text = preprocessed.get("cleaned", "")
            preprocessing_keywords = preprocessed.get("keywords", [])

            if not cleaned_text:
                continue

            logger.info("Processing memory %s...", memory_id)

            emotion_scores = extract_emotion_scores(cleaned_text)
            keywords = extract_keywords(cleaned_text) or preprocessing_keywords
            topics = categorize_topics(cleaned_text, keywords)
            entities = extract_entities(cleaned_text)
            embedding_data = generate_embedding(cleaned_text)
            embedding_id = store_embedding_in_faiss(
                embedding_data["vector"],
                memory_id,
                faiss_index=None,
            )

            mood = dominant_mood_from_scores(emotion_scores)

            nlp_data = {
                "content_clean": cleaned_text,
                "mood": mood,
                "embedding_id": embedding_id,
                "nlp_insights": {
                    "emotion_scores": emotion_scores,
                    "keywords": keywords,
                    "topics": topics,
                    "entities": entities,
                },
            }

            if update_memory_with_nlp(memory_id, nlp_data):
                processed_count += 1
                logger.info("✓ Processed %s", memory_id)
            else:
                failed_count += 1
                errors.append(f"Failed to update {memory_id}")

        except Exception as e:
            failed_count += 1
            error_msg = f"Error processing {memory.get('_id')}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

    return {
        "preprocessing": preprocessing_result,
        "nlp_processing": {
            "total": len(unprocessed),
            "processed": processed_count,
            "failed": failed_count,
            "errors": errors,
        },
    }
