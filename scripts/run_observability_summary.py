"""CLI entrypoint for the local observability summary.
Run with: uv run python scripts/run_observability_summary.py"""

from genai_knowledge_assistant.observability.summary import print_summary

if __name__ == "__main__":
    print_summary()
