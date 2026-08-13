"""FastAPI app entrypoint."""

from fastapi import FastAPI
from genai_knowledge_assistant.api import health

app = FastAPI(title="GenAI Knowledge Assistant")

app.include_router(health.router)


def main():
    import uvicorn

    uvicorn.run(
        "genai_knowledge_assistant.main:app", host="0.0.0.0", port=8000, reload=True
    )


if __name__ == "__main__":
    main()
