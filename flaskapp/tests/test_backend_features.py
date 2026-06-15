import asyncio
import math
from datetime import datetime, timedelta

import pytest

from backend import nlp_processor
from backend.routers import ai_features, spotify
from backend.schemas import SpotifySuggestRequest


def test_bucketize_emotions_normalizes_alias_labels():
    result = nlp_processor._bucketize_emotions(
        [
            {"label": "amusement", "score": 0.25},
            {"label": "excitement", "score": 0.75},
        ]
    )

    assert result["joy"] == pytest.approx(1.0)
    assert result["sadness"] == pytest.approx(0.0)
    assert sum(result.values()) == pytest.approx(1.0)


def test_dominant_mood_from_scores_handles_ties_and_aliases():
    assert nlp_processor.dominant_mood_from_scores({"joy": 0.6, "sadness": 0.4}) == "joy"
    assert nlp_processor.dominant_mood_from_scores({"happy": 0.9}) == "joy"
    assert nlp_processor.dominant_mood_from_scores({"joy": 0.5, "sadness": 0.5}) == "neutral"


def test_fallback_emotion_scores_detects_subtle_sadness_language():
    text = (
        "I don’t even know why today felt so heavy. Nothing major happened, but everything felt off. "
        "It’s like carrying something invisible that no one else can see. I tried distracting myself, "
        "but the feeling kept creeping back. Maybe I just needed to sit with it instead of running."
    )

    scores = nlp_processor._fallback_emotion_scores(text)

    assert scores["sadness"] > 0
    assert nlp_processor.dominant_mood_from_scores(scores) == "sadness"


def test_generate_embedding_returns_normalized_vector():
    embedding = nlp_processor.generate_embedding("alpha beta gamma")

    assert embedding["model"] == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert len(embedding["vector"]) == 64
    assert math.isclose(sum(value * value for value in embedding["vector"]), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_hf_inference_endpoints_order_and_deduplication(monkeypatch):
    monkeypatch.setenv("HF_INFERENCE_BASE_URL", "https://example.com/custom")

    endpoints = nlp_processor._hf_inference_endpoints("model-id")

    assert endpoints == [
        "https://example.com/custom/model-id",
        "https://router.huggingface.co/hf-inference/models/model-id",
        "https://api-inference.huggingface.co/models/model-id",
    ]


def test_build_query_combines_mood_keywords_and_topics():
    payload = SpotifySuggestRequest(
        mood=" calm ",
        keywords=["focus", "growth", "reflection", "ignored"],
        topics=["work", "health", "extra"],
    )

    assert spotify._build_query(payload) == "calm focus growth reflection work health"


def test_to_track_parses_spotify_search_item():
    track = spotify._to_track(
        {
            "name": "Demo Song",
            "external_urls": {"spotify": "https://open.spotify.com/track/demo"},
            "artists": [{"name": "Demo Artist"}],
            "album": {"images": [{"url": "https://example.com/cover.jpg"}]},
            "preview_url": "https://example.com/preview.mp3",
        }
    )

    assert track is not None
    assert track.title == "Demo Song"
    assert track.artist == "Demo Artist"
    assert track.album_image == "https://example.com/cover.jpg"
    assert track.preview_url == "https://example.com/preview.mp3"


def test_weekly_reflection_summarizes_entries(monkeypatch):
    now = datetime.utcnow()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    memories = [
        {
            "created_at": (week_start + timedelta(days=1)).isoformat(),
            "mood": "happy",
            "tags": ["work"],
            "nlp_insights": {"topics": ["Learning & Growth"]},
        },
        {
            "created_at": (week_start + timedelta(days=2)).isoformat(),
            "mood": "happy",
            "tags": ["focus"],
            "nlp_insights": {"topics": ["Health & Wellness"]},
        },
    ]

    monkeypatch.setattr(ai_features, "_user_memories", lambda uid, limit=400: memories)

    response = asyncio.run(ai_features.weekly_reflection(user={"uid": "user-1"}))

    assert response.total_entries == 2
    assert response.dominant_mood == "happy"
    assert "logged 2 entries" in response.summary
    assert "happy" in response.summary


def test_mood_anomaly_detects_alert(monkeypatch):
    now = datetime.utcnow()
    memories = [
        {
            "created_at": (now - timedelta(days=1)).isoformat(),
            "mood": "sadness",
        },
        {
            "created_at": (now - timedelta(days=2)).isoformat(),
            "mood": "anger",
        },
        {
            "created_at": (now - timedelta(days=7)).isoformat(),
            "mood": "happy",
        },
        {
            "created_at": (now - timedelta(days=8)).isoformat(),
            "mood": "calm",
        },
    ]

    monkeypatch.setattr(ai_features, "_user_memories", lambda uid, limit=400: memories)

    response = asyncio.run(ai_features.mood_anomaly(user={"uid": "user-1"}))

    assert response.level == "alert"
    assert response.recent_entries == 2
    assert response.recent_negative_ratio == pytest.approx(1.0)
    assert response.baseline_negative_ratio == pytest.approx(0.0)
