"""FastAPI app entrypoint."""

from fastapi import FastAPI
from genai_knowledge_assistant.observability.tracing_setup import setup_tracing
from genai_knowledge_assistant.api import health, documents, chat, memory, graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GenAI Knowledge Assistant")

# LangSmith tracing
setup_tracing()

# Routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(memory.router)
app.include_router(graph.router)


def main():
    import uvicorn

    uvicorn.run(
        "genai_knowledge_assistant.main:app", host="0.0.0.0", port=8000, reload=True
    )


if __name__ == "__main__":
    main()
