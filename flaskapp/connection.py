import os

from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

_client = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client


def get_db(db_name: str = COLLECTION_NAME):
    client = get_client()
    return client[db_name]


def get_collection(name: str, db_name: str = "dmj"):
    db = get_db(db_name)
    return db[name]


if __name__ == "__main__":
    try:
        col = get_collection("memories")
        print("Connected to collection:", col.name)
        print("Documents count:", col.count_documents({}))
    except Exception as e:
        print("Connection test failed:", e)
