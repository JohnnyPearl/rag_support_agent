from typing import Literal

from pydantic import BaseModel

from app.schemas.chunks import Citation


class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    citations: list[Citation]


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
