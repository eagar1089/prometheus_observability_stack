import base64
import json
import os
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Depends, HTTPException

from backend import schemas
from backend.auth_deps import verify_firebase_token_optional

router = APIRouter()


SPOTIFY_USER_AGENT = "DigitalMemoryJar/1.0 (+https://localhost)"


def _spotify_token() -> str:
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=503,
            detail="Spotify is not configured. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
        )

    encoded = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    data = urlencode({"grant_type": "client_credentials"}).encode("utf-8")

    req = Request(
        "https://accounts.spotify.com/api/token",
        data=data,
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": SPOTIFY_USER_AGENT,
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=10) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
            token = payload.get("access_token")
            if not token:
                raise HTTPException(status_code=502, detail="Failed to get Spotify access token")
            return token
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = str(exc)
        raise HTTPException(status_code=502, detail=f"Spotify token request failed: {exc.code} {body}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Spotify token request failed: {exc}") from exc


def _build_query(payload: schemas.SpotifySuggestRequest) -> str:
    mood = (payload.mood or "neutral").strip()
    terms = [mood]
    terms.extend([(k or "").strip() for k in payload.keywords[:3]])
    terms.extend([(t or "").strip() for t in payload.topics[:2]])
    filtered = [term for term in terms if term]
    return " ".join(filtered) if filtered else "chill"


def _to_track(item: dict) -> schemas.SpotifyTrack | None:
    external = item.get("external_urls") or {}
    album = item.get("album") or {}
    artists = item.get("artists") or []
    artist_names = [artist.get("name", "") for artist in artists if artist.get("name")]
    images = album.get("images") or []

    url = external.get("spotify")
    name = item.get("name")
    if not url or not name:
        return None

    return schemas.SpotifyTrack(
        title=name,
        artist=", ".join(artist_names) if artist_names else "Unknown Artist",
        url=url,
        album_image=images[0].get("url") if images else None,
        preview_url=item.get("preview_url"),
    )


@router.post("/suggest", response_model=schemas.SpotifySuggestResponse)
async def spotify_suggest(
    payload: schemas.SpotifySuggestRequest,
    user: dict | None = Depends(verify_firebase_token_optional),
):
    token = _spotify_token()
    query = _build_query(payload)
    market = os.getenv("SPOTIFY_MARKET", "US")

    params = urlencode(
        {
            "q": query,
            "type": "track",
            "limit": 5,
            "market": market,
        }
    )

    req = Request(
        f"https://api.spotify.com/v1/search?{params}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": SPOTIFY_USER_AGENT,
        },
        method="GET",
    )

    try:
        with urlopen(req, timeout=10) as response:  # nosec B310
            payload_json = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:
            body = str(exc)
        raise HTTPException(status_code=502, detail=f"Spotify search failed: {exc.code} {body}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Spotify search failed: {exc}") from exc

    items = ((payload_json.get("tracks") or {}).get("items") or [])
    tracks = [track for track in (_to_track(item) for item in items) if track is not None]

    if not tracks:
        raise HTTPException(status_code=404, detail="No Spotify tracks found for this mood")

    return schemas.SpotifySuggestResponse(
        mood=(payload.mood or "neutral").lower(),
        query=query,
        primary=tracks[0],
        alternatives=tracks[1:3],
    )
