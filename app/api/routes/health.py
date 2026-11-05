from fastapi import APIRouter, Depends

from app.dependencies import AppState, get_state

router = APIRouter()


@router.get("/health")
def health(state: AppState = Depends(get_state)) -> dict:
    return {
        "status": "ok",
        "collection": state.store.collection,
        "points": state.store.count(),
    }
