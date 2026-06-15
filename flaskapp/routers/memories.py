from typing import List
import logging

from fastapi import APIRouter, Depends, HTTPException

from backend import crud, schemas
from backend.auth_deps import verify_firebase_token
from backend.nlp_processor import extract_emotion_scores, extract_keywords, categorize_topics, dominant_mood_from_scores


router = APIRouter()
logger = logging.getLogger(__name__)


def _stage_log(stage: str) -> None:
	message = f"[NLP][analyze] {stage}"
	print(message)
	logger.info(message)


def _save_stage_log(stage: str) -> None:
	message = f"[MEMORY][save] {stage}"
	print(message)
	logger.info(message)


def _generate_simple_summary(text: str, max_words: int = 28) -> str:
	"""Create a lightweight summary from raw memory text."""
	cleaned = " ".join(text.split())
	if not cleaned:
		return ""

	words = cleaned.split(" ")
	if len(words) <= max_words:
		return cleaned

	return " ".join(words[:max_words]).rstrip(".,;: ") + "..."


def _top_tags(keywords: List[str], topics: List[str], max_tags: int = 5) -> List[str]:
	"""Combine and normalize keywords/topics into compact tag list."""
	tags: List[str] = []
	seen = set()

	for item in (keywords or []) + (topics or []):
		value = str(item).strip().lower()
		if not value:
			continue
		value = value.replace("&", "and")
		if value in seen:
			continue
		seen.add(value)
		tags.append(value)
		if len(tags) >= max_tags:
			break

	return tags


@router.get("/", response_model=List[schemas.MemoryDB])
async def list_memories(user: dict = Depends(verify_firebase_token)):
	docs = crud.list_memories(uid=user.get("uid"))
	return docs


@router.post("/", status_code=201)
async def create_memory(
	payload: schemas.MemoryCreate,
	user: dict = Depends(verify_firebase_token),
):
	_save_stage_log("1/3 received request")
	data = payload.dict(exclude_none=True)
	data["uid"] = user.get("uid")
	_save_stage_log(f"2/3 persisting for uid={data['uid']}")
	inserted_id = crud.create_memory(data)
	_save_stage_log(f"3/3 completed id={inserted_id}")
	return {"id": inserted_id, "status": "created", "message": "Memory stored. Will be processed by AI."}


@router.post("/analyze", response_model=schemas.MemoryAnalyzeResponse)
async def analyze_memory(payload: schemas.MemoryAnalyzeRequest, user: dict = Depends(verify_firebase_token)):
	_stage_log("1/6 received request")
	text = payload.content.strip()
	if not text:
		_stage_log("validation failed: empty content")
		raise HTTPException(status_code=400, detail="Memory content is required")

	_stage_log("2/6 emotion scoring")
	emotion_scores = extract_emotion_scores(text)
	mood = dominant_mood_from_scores(emotion_scores)
	_stage_log(f"emotion scoring done, mood={mood}")

	_stage_log("3/6 keyword extraction")
	keywords = extract_keywords(text)
	_stage_log(f"keyword extraction done, count={len(keywords)}")

	_stage_log("4/6 topic categorization")
	topics = categorize_topics(text, keywords)
	_stage_log(f"topic categorization done, count={len(topics)}")

	_stage_log("5/6 summary + tags")
	ai_summary = _generate_simple_summary(text)
	tags = _top_tags(keywords, topics)
	_stage_log(f"summary + tags done, tags={len(tags)}")

	_stage_log("6/6 response ready")

	return {
		"ai_summary": ai_summary,
		"mood": mood,
		"tags": tags,
		"nlp_insights": {
			"emotion_scores": emotion_scores,
			"keywords": keywords,
			"topics": topics,
			"entities": [],
		},
	}


@router.get("/{memory_id}", response_model=schemas.MemoryDB)
async def get_memory(memory_id: str, user: dict = Depends(verify_firebase_token)):
	"""Get a specific memory by ID."""
	memory = crud.get_memory_by_id(memory_id, uid=user.get("uid"))
	if not memory:
		raise HTTPException(status_code=404, detail="Memory not found")
	return memory


@router.put("/{memory_id}", response_model=schemas.MemoryDB)
async def update_memory(memory_id: str, payload: schemas.MemoryUpdate, user: dict = Depends(verify_firebase_token)):
	updates = payload.dict(exclude_none=True)
	if not updates:
		raise HTTPException(status_code=400, detail="No updates provided")

	existing = crud.get_memory_by_id(memory_id, uid=user.get("uid"))
	if not existing:
		raise HTTPException(status_code=404, detail="Memory not found")

	updated = crud.update_memory_by_id(memory_id, updates, uid=user.get("uid"))
	if not updated:
		raise HTTPException(status_code=500, detail="Failed to update memory")

	memory = crud.get_memory_by_id(memory_id, uid=user.get("uid"))
	if not memory:
		raise HTTPException(status_code=404, detail="Memory not found")

	return memory


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, user: dict = Depends(verify_firebase_token)):
	"""Delete a specific memory by ID for the authenticated user."""
	existing = crud.get_memory_by_id(memory_id, uid=user.get("uid"))
	if not existing:
		raise HTTPException(status_code=404, detail="Memory not found")

	deleted = crud.delete_memory_by_id(memory_id, uid=user.get("uid"))
	if not deleted:
		raise HTTPException(status_code=500, detail="Failed to delete memory")

	return {"status": "deleted", "id": memory_id}
