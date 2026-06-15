from fastapi import APIRouter, Depends

from flaskapp import crud, schemas
from flaskapp.auth_deps import verify_firebase_token

router = APIRouter()


@router.get("/stats", response_model=schemas.StatsResponse)
async def stats(user: dict = Depends(verify_firebase_token)):
	return crud.get_stats(uid=user.get("uid"))
