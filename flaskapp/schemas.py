from typing import Optional, List, Dict
from datetime import datetime

from pydantic import BaseModel, Field


class EmotionScores(BaseModel):
    """Emotion analysis from NLP."""
    joy: Optional[float] = Field(None, description="Joy score 0-1")
    sadness: Optional[float] = Field(None, description="Sadness score 0-1")
    anger: Optional[float] = Field(None, description="Anger score 0-1")
    fear: Optional[float] = Field(None, description="Fear score 0-1")
    surprise: Optional[float] = Field(None, description="Surprise score 0-1")
    disgust: Optional[float] = Field(None, description="Disgust score 0-1")



class NLPInsights(BaseModel):
    emotion_scores: Optional[EmotionScores] = Field(None, description="Emotion sentiment analysis")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Extracted keywords/phrases")
    topics: Optional[List[str]] = Field(default_factory=list, description="Identified topics (e.g., Work, Health, Relationships)")
    entities: Optional[List[str]] = Field(default_factory=list, description="Named entities (people, places)")


class MemoryCreate(BaseModel):
    content: str = Field(..., description="Raw text content of the memory")
    content_clean: Optional[str] = Field(None, description="Cleaned/normalized version of content")
    mood: Optional[str] = Field(None, description="Detected mood (e.g., happy, sad, reflective)")
    ai_summary: Optional[str] = Field(None, description="AI-generated summary of the memory")
    tags: Optional[List[str]] = Field(default_factory=list, description="Associated tags")
    recorded_by: Optional[str] = Field(None, description="Input method: text, voice, etc.")
    nlp_insights: Optional[NLPInsights] = Field(None, description="NLP extraction results")
    embedding_id: Optional[int] = Field(None, description="Reference to FAISS index ID for vector search")


class MemoryUpdate(BaseModel):
    content: Optional[str] = Field(None, description="Raw text content of the memory")
    mood: Optional[str] = Field(None, description="Updated mood")
    ai_summary: Optional[str] = Field(None, description="Updated AI summary")
    tags: Optional[List[str]] = Field(None, description="Updated tags")


class MemoryAnalyzeRequest(BaseModel):
    content: str = Field(..., description="Raw text content to analyze")


class MemoryAnalyzeResponse(BaseModel):
    ai_summary: str = Field(..., description="Generated concise summary")
    mood: str = Field(..., description="Detected mood label")
    tags: List[str] = Field(default_factory=list, description="Detected keyword/topic tags")
    nlp_insights: Optional[NLPInsights] = Field(None, description="Detailed NLP extraction results")


class MemoryDB(MemoryCreate):
    id: str = Field(..., description="MongoDB ObjectId as string")
    uid: Optional[str] = Field(None, description="Firebase user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    is_processed: bool = Field(False, description="Whether NLP extraction/embedding has been completed")


class StatsResponse(BaseModel):
    total_memories: int
    avg_mood_score: Optional[float] = None
    most_common_mood: Optional[str] = None
    top_emotions: Optional[Dict[str, float]] = None
    top_topics: Optional[List[str]] = None


class SpotifySuggestRequest(BaseModel):
    mood: Optional[str] = Field("neutral", description="Detected mood")
    keywords: List[str] = Field(default_factory=list, description="Top keywords from latest memory")
    topics: List[str] = Field(default_factory=list, description="Top topics from latest memory")


class SpotifyTrack(BaseModel):
    title: str
    artist: str
    url: str
    album_image: Optional[str] = None
    preview_url: Optional[str] = None


class SpotifySuggestResponse(BaseModel):
    mood: str
    query: str
    primary: SpotifyTrack
    alternatives: List[SpotifyTrack] = Field(default_factory=list)


class WeeklyReflectionResponse(BaseModel):
    period_start: str
    period_end: str
    total_entries: int
    dominant_mood: str
    top_topics: List[str] = Field(default_factory=list)
    summary: str


class MoodAnomalyResponse(BaseModel):
    level: str
    summary: str
    recent_negative_ratio: float
    baseline_negative_ratio: float
    recent_entries: int


class CompanionChatRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question about memory history")


class CompanionReference(BaseModel):
    memory_id: str
    created_at: str
    snippet: str


class CompanionChatResponse(BaseModel):
    answer: str
    references: List[CompanionReference] = Field(default_factory=list)
