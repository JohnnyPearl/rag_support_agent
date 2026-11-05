from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from app.dependencies import AppState, get_state
from app.schemas.requests import AskRequest
from app.schemas.responses import CitationEvent, DoneEvent, ErrorEvent, TokenEvent

router = APIRouter()


@router.post("/ask")
async def ask(req: AskRequest, state: AppState = Depends(get_state)) -> EventSourceResponse:
    return EventSourceResponse(_event_stream(req, state))


async def _event_stream(req: AskRequest, state: AppState) -> AsyncIterator[dict]:
    try:
        graph_input = {"question": req.question, "top_k": req.top_k or state.top_k}
        result = await state.graph.ainvoke(graph_input)

        yield {
            "event": "citation",
            "data": CitationEvent(citations=result["citations"]).model_dump_json(),
        }

        async for token in state.chat.stream(result["system_prompt"], result["user_prompt"]):
            yield {"event": "token", "data": TokenEvent(text=token).model_dump_json()}

        yield {"event": "done", "data": DoneEvent().model_dump_json()}

    except Exception as exc:
        yield {"event": "error", "data": ErrorEvent(message=str(exc)).model_dump_json()}
