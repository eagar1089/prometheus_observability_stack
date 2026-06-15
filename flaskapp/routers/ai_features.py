from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable

from fastapi import APIRouter, Depends

from backend import crud, schemas
from backend.auth_deps import verify_firebase_token

router = APIRouter()


MOOD_SCORE = {
    "happy": 2,
    "joy": 2,
    "calm": 1,
    "peaceful": 1,
    "reflective": 0,
    "neutral": 0,
    "surprise": 0,
    "sadness": -1,
    "disgust": -1,
    "fear": -2,
    "anger": -2,
}

NEGATIVE_MOODS = {"sadness", "anger", "fear", "disgust"}
STOP_WORDS = {
    "what", "when", "where", "how", "why", "did", "the", "and", "for", "this", "that", "with",
    "from", "have", "was", "were", "you", "about", "my", "your", "into", "over", "under", "there",
}


def _parse_date(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _user_memories(uid: str, limit: int = 400) -> list[dict]:
    return crud.list_memories(limit=limit, uid=uid)


def _recent_topics(memory: dict) -> Iterable[str]:
    topics = []
    for tag in memory.get("tags") or []:
        if isinstance(tag, str) and tag.strip():
            topics.append(tag.strip().lower())

    nlp = memory.get("nlp_insights") or {}
    for topic in nlp.get("topics") or []:
        if isinstance(topic, str) and topic.strip():
            topics.append(topic.strip().lower())
    return topics


@router.get("/weekly-reflection", response_model=schemas.WeeklyReflectionResponse)
async def weekly_reflection(user: dict = Depends(verify_firebase_token)):
    memories = _user_memories(user.get("uid", ""), limit=500)

    now = datetime.utcnow()
    monday_delta = now.weekday()
    week_start = (now - timedelta(days=monday_delta)).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = week_start + timedelta(days=7)

    weekly = []
    for memory in memories:
        created_at = _parse_date(memory.get("created_at"))
        if not created_at:
            continue
        if week_start <= created_at < week_end:
            weekly.append(memory)

    if not weekly:
        return schemas.WeeklyReflectionResponse(
            period_start=week_start.date().isoformat(),
            period_end=(week_end - timedelta(days=1)).date().isoformat(),
            total_entries=0,
            dominant_mood="neutral",
            top_topics=[],
            summary="No entries this week yet. Add a memory and I will generate your weekly reflection.",
        )

    mood_counts = Counter((memory.get("mood") or "neutral").lower() for memory in weekly)
    dominant_mood = mood_counts.most_common(1)[0][0]

    topic_counts = Counter()
    for memory in weekly:
        topic_counts.update(_recent_topics(memory))
    top_topics = [name for name, _ in topic_counts.most_common(3)]

    score_values = [MOOD_SCORE.get((memory.get("mood") or "neutral").lower(), 0) for memory in weekly]
    avg_score = sum(score_values) / max(len(score_values), 1)

    if avg_score <= -0.8:
        tone = "You went through a heavy emotional stretch"
    elif avg_score >= 0.8:
        tone = "You showed a strong positive emotional pattern"
    else:
        tone = "Your week looked emotionally balanced overall"

    if top_topics:
        summary = (
            f"{tone}. You logged {len(weekly)} entries, with {dominant_mood} as the dominant mood, "
            f"and recurring themes around {', '.join(top_topics)}."
        )
    else:
        summary = f"{tone}. You logged {len(weekly)} entries, with {dominant_mood} as the dominant mood."

    return schemas.WeeklyReflectionResponse(
        period_start=week_start.date().isoformat(),
        period_end=(week_end - timedelta(days=1)).date().isoformat(),
        total_entries=len(weekly),
        dominant_mood=dominant_mood,
        top_topics=top_topics,
        summary=summary,
    )


@router.get("/mood-anomaly", response_model=schemas.MoodAnomalyResponse)
async def mood_anomaly(user: dict = Depends(verify_firebase_token)):
    memories = _user_memories(user.get("uid", ""), limit=500)
    now = datetime.utcnow()

    recent_start = now - timedelta(days=3)
    baseline_start = now - timedelta(days=17)
    baseline_end = now - timedelta(days=3)

    recent = []
    baseline = []
    for memory in memories:
        created_at = _parse_date(memory.get("created_at"))
        if not created_at:
            continue
        if created_at >= recent_start:
            recent.append(memory)
        elif baseline_start <= created_at < baseline_end:
            baseline.append(memory)

    def negative_ratio(items: list[dict]) -> float:
        if not items:
            return 0.0
        negatives = sum(1 for item in items if (item.get("mood") or "neutral").lower() in NEGATIVE_MOODS)
        return negatives / len(items)

    recent_ratio = negative_ratio(recent)
    baseline_ratio = negative_ratio(baseline)

    if len(recent) < 2:
        return schemas.MoodAnomalyResponse(
            level="insufficient",
            summary="Not enough recent memories to detect an anomaly yet.",
            recent_negative_ratio=round(recent_ratio, 3),
            baseline_negative_ratio=round(baseline_ratio, 3),
            recent_entries=len(recent),
        )

    delta = recent_ratio - baseline_ratio
    if delta >= 0.25:
        level = "alert"
        summary = "Emotional dip detected in recent days. Try a short reset routine and log one positive action."
    elif delta <= -0.2:
        level = "uptrend"
        summary = "Positive shift detected versus your baseline pattern. Keep your current habits going."
    else:
        level = "stable"
        summary = "Mood pattern looks stable compared to your baseline period."

    return schemas.MoodAnomalyResponse(
        level=level,
        summary=summary,
        recent_negative_ratio=round(recent_ratio, 3),
        baseline_negative_ratio=round(baseline_ratio, 3),
        recent_entries=len(recent),
    )


@router.post("/companion-chat", response_model=schemas.CompanionChatResponse)
async def companion_chat(payload: schemas.CompanionChatRequest, user: dict = Depends(verify_firebase_token)):
    memories = _user_memories(user.get("uid", ""), limit=300)
    prompt = payload.question.lower().strip()
    tokens = [
        token
        for token in "".join(c if c.isalnum() else " " for c in prompt).split()
        if len(token) > 2 and token not in STOP_WORDS
    ]

    if not tokens:
        return schemas.CompanionChatResponse(
            answer="Ask with a bit more detail, for example: what stressed me this week?",
            references=[],
        )

    ranked = []
    for memory in memories:
        joined = " ".join(
            [
                str(memory.get("content") or ""),
                str(memory.get("ai_summary") or ""),
                " ".join(memory.get("tags") or []),
                str(memory.get("mood") or ""),
            ]
        ).lower()

        score = sum(1 for token in tokens if token in joined)
        if score > 0:
            ranked.append((memory, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    top = ranked[:3]

    if not top:
        return schemas.CompanionChatResponse(
            answer="I could not find a close match in your recent memories. Try mentioning a mood, topic, or event keyword.",
            references=[],
        )

    references = []
    snippets = []
    for memory, _ in top:
        memory_id = memory.get("id", "")
        created_at = str(memory.get("created_at") or "")
        text = (memory.get("ai_summary") or memory.get("content") or "").strip()
        snippet = text[:140] + ("..." if len(text) > 140 else "")

        references.append(
            schemas.CompanionReference(
                memory_id=memory_id,
                created_at=created_at,
                snippet=snippet,
            )
        )
        snippets.append(f"{created_at[:10]}: {snippet}")

    answer = "Here are the closest memories I found: " + " | ".join(snippets)
    return schemas.CompanionChatResponse(answer=answer, references=references)
