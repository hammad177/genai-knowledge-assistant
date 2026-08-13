"""FastAPI app entrypoint."""

from fastapi import FastAPI
from genai_knowledge_assistant.api import health, documents, chat, memory

app = FastAPI(title="GenAI Knowledge Assistant")

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(memory.router)


def main():
    import uvicorn

    uvicorn.run(
        "genai_knowledge_assistant.main:app", host="0.0.0.0", port=8000, reload=True
    )


if __name__ == "__main__":
    main()
