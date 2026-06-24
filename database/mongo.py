"""
database/mongo.py
=================
MongoDB async data-access layer using Motor (async pymongo wrapper).

All functions are async so they integrate cleanly with FastAPI's async
request handlers without blocking the event loop.

Functions exposed
-----------------
- get_database()           → Motor AsyncIOMotorDatabase (connection singleton)
- insert_result(doc)       → inserted document id
- get_latest_result()      → most recent OCR document or None
- get_history(limit)       → list of recent OCR documents
- delete_all_history()     → number of deleted documents
- find_by_filename(name)   → list of documents for a given filename

Connection strategy
-------------------
Motor lazily creates the connection pool on first use, so ``get_database()``
is cheap to call repeatedly.  The pool is shared across all requests.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from bson import ObjectId
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# ---------------------------------------------------------------------------
# Singleton client — created once at import time
# ---------------------------------------------------------------------------
_client: Optional[AsyncIOMotorClient] = None


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        logger.info(f"Creating MongoDB client → {settings.MONGO_URI}")
        _client = AsyncIOMotorClient(settings.MONGO_URI)
    return _client


def get_database() -> AsyncIOMotorDatabase:
    """Return the Motor database object."""
    return _get_client()[settings.MONGO_DB_NAME]


def _collection():
    return get_database()[settings.MONGO_COLLECTION]


def _serialize(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serialisable dict."""
    if doc is None:
        return {}
    doc["_id"] = str(doc["_id"])   # ObjectId → str
    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()
    return doc


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

async def insert_result(data: dict) -> str:
    """
    Insert one OCR result document.

    Parameters
    ----------
    data : dict
        Must contain: filename, text, length, confidence, status.
        timestamp is added automatically if missing.

    Returns
    -------
    str
        String representation of the inserted document's ObjectId.
    """
    if "timestamp" not in data:
        data["timestamp"] = datetime.now(timezone.utc)

    try:
        result = await _collection().insert_one(data)
        doc_id = str(result.inserted_id)
        logger.info(f"MongoDB: inserted document _id={doc_id}")
        return doc_id
    except Exception as exc:
        logger.error(f"MongoDB insert_result failed: {exc}")
        raise


async def get_latest_result() -> Optional[dict]:
    """
    Retrieve the most recently inserted OCR result.

    Returns None if the collection is empty.
    """
    try:
        doc = await _collection().find_one(
            sort=[("timestamp", -1)]   # newest first
        )
        return _serialize(doc) if doc else None
    except Exception as exc:
        logger.error(f"MongoDB get_latest_result failed: {exc}")
        raise


async def get_history(limit: int = 50) -> List[dict]:
    """
    Retrieve the *limit* most recent OCR results, newest first.

    Parameters
    ----------
    limit : int
        Maximum number of documents to return (capped at 200 to prevent
        accidentally fetching millions of records).
    """
    limit = min(limit, 200)
    try:
        cursor = _collection().find().sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [_serialize(d) for d in docs]
    except Exception as exc:
        logger.error(f"MongoDB get_history failed: {exc}")
        raise


async def delete_all_history() -> int:
    """
    Delete ALL documents in the OCR results collection.

    Returns
    -------
    int
        Number of deleted documents.
    """
    try:
        result = await _collection().delete_many({})
        count = result.deleted_count
        logger.info(f"MongoDB: deleted {count} document(s).")
        return count
    except Exception as exc:
        logger.error(f"MongoDB delete_all_history failed: {exc}")
        raise


async def find_by_filename(filename: str) -> List[dict]:
    """
    Find all OCR results for a given filename (case-insensitive prefix match).

    Parameters
    ----------
    filename : str
        Filename to search for (exact match, no extension required).
    """
    try:
        # Case-insensitive regex match on the filename field
        import re
        pattern = re.compile(re.escape(filename), re.IGNORECASE)
        cursor = _collection().find({"filename": {"$regex": pattern}}).sort(
            "timestamp", -1
        )
        docs = await cursor.to_list(length=100)
        return [_serialize(d) for d in docs]
    except Exception as exc:
        logger.error(f"MongoDB find_by_filename failed: {exc}")
        raise


async def ping_database() -> bool:
    """Return True if the MongoDB server is reachable."""
    try:
        await _get_client().admin.command("ping")
        return True
    except Exception:
        return False
