"""CLI entrypoint for generating the trust report.
Run with: uv run python scripts/generate_trust_report.py"""

from genai_knowledge_assistant.evaluation.trust_report import save_trust_report

if __name__ == "__main__":
    path = save_trust_report()
    print(f"Trust report saved to {path}")
