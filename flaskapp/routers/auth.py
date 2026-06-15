from fastapi import APIRouter, Depends

from backend.auth_deps import verify_firebase_token

router = APIRouter()


@router.get("/me")
async def me(user: dict = Depends(verify_firebase_token)):
    """Return current user info from Firebase token."""
    return {
        "uid": user.get("uid"),
        "email": user.get("email"),
        "email_verified": user.get("email_verified"),
    }


