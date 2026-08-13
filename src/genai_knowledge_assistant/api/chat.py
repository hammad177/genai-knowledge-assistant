"""Endpoint for asking questions grounded in ingested documents."""

from fastapi import APIRouter, Depends, HTTPException

from genai_knowledge_assistant.services.chat_service import ChatService
from genai_knowledge_assistant.models.chat import ChatRequest, ChatResponse
from genai_knowledge_assistant.dependencies import get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    body: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    try:
        return chat_service.ask(query=body.query, top_k=body.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")
