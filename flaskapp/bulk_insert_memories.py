#!/usr/bin/env python3
"""
Fast bulk insert of test memories for the past 5 days into MongoDB Atlas.

Usage:
    python bulk_insert_memories.py --uid <firebase_user_id> [--count-per-day <n>]

Example:
    python bulk_insert_memories.py --uid user-123 --count-per-day 3
"""

import sys
import argparse
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import os
import time

SAMPLE_MEMORIES = [
    "Today felt surprisingly productive and calm at the same time. I managed to complete most of the tasks I had been putting off, and that gave me a sense of control. There were moments where I almost got distracted, but I pulled myself back. Ending the day with a clear mind feels rewarding.",

    "Woke up feeling a bit low and couldn’t really figure out why. Nothing specific went wrong, but everything just felt heavier than usual. I tried to stay busy to distract myself, but the feeling kept coming back. Maybe I just need to sit with it instead of avoiding it.",

    "Spent a good amount of time alone today and it didn’t feel lonely at all. I actually enjoyed the quiet and the space to think. It helped me organize my thoughts and slow things down. Days like this remind me how important it is to pause.",

    "There was this constant sense of worry in the background today. Even while doing normal things, my mind kept jumping to future problems. It made it hard to stay present. I really need to work on managing this better.",

    "Something exciting is coming up and I can feel the energy building inside me. I kept imagining different possibilities and outcomes. It’s a mix of excitement and nervousness, but mostly positive. I’m actually looking forward to what’s next.",

    "Today was pretty ordinary, nothing really stood out. I followed my usual routine and got through the day without much effort. It wasn’t bad, just very neutral. Sometimes that kind of stability is actually nice.",

    "I found myself thinking deeply about my goals today. Some things that once seemed important don’t feel the same anymore. It’s interesting how perspectives change over time. I think I’m slowly understanding what I truly want.",

    "There were a few small moments today that genuinely made me smile. A random conversation, a good song, a quiet break. Nothing major, but it added up. It made me realize happiness doesn’t always have to be big.",

    "The day carried a quiet heaviness that I couldn’t explain. I tried to shake it off by staying busy, but it stayed in the background. It wasn’t overwhelming, just persistent. Maybe I need to understand it instead of ignoring it.",

    "Had a really satisfying day where things actually went as planned. Finished my work on time and even had extra time to relax. That doesn’t happen often. It felt good to be in control for once.",

    "I decided to slow things down today and not rush into anything. It made a noticeable difference in how I felt. My thoughts were clearer and I wasn’t as stressed. Maybe this is something I should practice regularly.",

    "Deadlines are getting closer and I can feel the pressure increasing. I keep thinking about everything that could go wrong. It’s exhausting to stay in that mindset. I need to take a step back and plan things calmly.",

    "Tried doing something different today just for the sake of it. I didn’t overthink or hesitate, just went for it. It turned out to be a good experience. I should probably take more risks like this.",

    "It was just another routine day, nothing particularly exciting happened. Still, I managed to stay consistent with my work. That consistency might not feel exciting now, but it matters in the long run.",

    "I realized today how much I’ve grown over time. Things that once stressed me out don’t affect me the same way anymore. It’s not a huge change, but it’s noticeable. That gives me some confidence.",

    "Spent some time doing something I genuinely enjoy and lost track of time. It felt refreshing to be fully present in that moment. I didn’t think about anything else. I need more of that in my life.",

    "Something small triggered a deeper emotional response today. It stayed on my mind longer than expected. I think there are still things I haven’t fully processed. Maybe I should give it more attention.",

    "The day felt balanced overall, not too stressful and not too exciting. Everything was manageable and steady. It’s not the kind of day you remember, but it’s still valuable in its own way.",

    "Connected with someone after a long time and it felt really good. The conversation was easy and natural. It reminded me how important these connections are. It definitely improved my mood.",

    "Spent some time reflecting on my priorities and what really matters. I realized that some things I stress about aren’t that important. It’s a bit confusing but also clarifying. I think I’m slowly figuring things out."
]
MOODS = ["happy", "sad", "calm", "anxious", "excited", "neutral", "reflective", "sadness", "joy"]


def bulk_insert_memories(
    uid,
    mongo_uri,
    db_name,
    collection_name,
    count_per_day=1,
    connect_timeout_ms=20000,
    retries=3,
):
    """Bulk insert memories for the past 5 days."""

    # Connect to MongoDB with retry for transient Atlas replica set elections.
    client = None
    for attempt in range(1, retries + 1):
        try:
            client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=connect_timeout_ms,
                connectTimeoutMS=connect_timeout_ms,
                socketTimeoutMS=connect_timeout_ms,
                retryWrites=True,
            )
            client.admin.command("ping")
            print(f"✓ Connected to MongoDB (attempt {attempt}/{retries})")
            break
        except Exception as e:
            if attempt == retries:
                print(f"✗ Failed to connect to MongoDB after {retries} attempts: {e}")
                return False
            wait_seconds = min(2 ** attempt, 8)
            print(f"⚠ Connection attempt {attempt}/{retries} failed: {e}")
            print(f"  Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
    
    # Use explicit DB/collection so inserts match backend reads.
    db = client[db_name]
    collection = db[collection_name]
    
    # Generate memories for the past 5 days
    memories = []
    now = datetime.now(timezone.utc)
    
    for day_offset in range(5, 0, -1):  # 5 days ago to yesterday
        day_start = now - timedelta(days=day_offset)
        day_start = day_start.replace(hour=9, minute=0, second=0, microsecond=0)
        
        for i in range(count_per_day):
            timestamp = day_start + timedelta(hours=i * (12 // count_per_day))
            content = SAMPLE_MEMORIES[(day_offset * count_per_day + i) % len(SAMPLE_MEMORIES)]
            mood = MOODS[(day_offset + i) % len(MOODS)]
            
            memory = {
                "uid": uid,
                "content": content,
                "content_clean": content.lower(),
                "mood": mood,
                "ai_summary": content[:50] + "..." if len(content) > 50 else content,
                "tags": ["test", mood],
                "recorded_by": "script",
                "created_at": timestamp,
                "updated_at": timestamp,
                "is_processed": True,
                "nlp_insights": {
                    "emotion_scores": {
                        "joy": 0.2,
                        "sadness": 0.1,
                        "anger": 0.0,
                        "fear": 0.0,
                        "surprise": 0.0,
                        "disgust": 0.0,
                    },
                    "keywords": ["test", mood],
                    "topics": ["general"],
                    "entities": [],
                },
            }
            memories.append(memory)
    
    # Bulk insert
    try:
        if memories:
            result = collection.insert_many(memories, ordered=False)
            print(f"✓ Inserted {len(result.inserted_ids)} memories")
            print(f"  UIDs: {result.inserted_ids[:3]}..." if len(result.inserted_ids) > 3 else f"  UIDs: {result.inserted_ids}")
            return True
    except BulkWriteError as e:
        print(f"⚠ Partial insert: {e.details['nInserted']} inserted, {len(e.details.get('writeErrors', []))} errors")
        return True
    except Exception as e:
        print(f"✗ Bulk insert failed: {e}")
        return False
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk insert test memories into MongoDB")
    parser.add_argument("--uid", required=True, help="Firebase user ID")
    parser.add_argument("--count-per-day", type=int, default=1, help="Number of memories per day (default: 1)")
    parser.add_argument("--db", default=os.getenv("MONGO_DB", "dmj"), help="MongoDB database name (default: dmj)")
    parser.add_argument("--collection", default="memories", help="MongoDB collection name (default: memories)")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb+srv://dbadmin:dbadmin@cluster0.gfkc1ci.mongodb.net"), help="MongoDB URI")
    parser.add_argument("--connect-timeout-ms", type=int, default=20000, help="MongoDB connect timeout in ms (default: 20000)")
    parser.add_argument("--retries", type=int, default=3, help="MongoDB connection retries (default: 3)")
    
    args = parser.parse_args()
    
    success = bulk_insert_memories(
        args.uid,
        args.mongo_uri,
        args.db,
        args.collection,
        args.count_per_day,
        args.connect_timeout_ms,
        args.retries,
    )
    sys.exit(0 if success else 1)
