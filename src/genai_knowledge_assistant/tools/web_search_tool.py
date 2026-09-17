"""Live web search tool, built directly on ddgs rather than the
deprecated langchain_community DuckDuckGoSearchRun wrapper."""

from ddgs import DDGS


def run_web_search(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return f"No search results found for '{query}'."

    formatted = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "No title")
        href = r.get("href", "#")
        snippet = r.get("body", "No snippet available.")
        formatted.append(f"{i}. {title}\n   URL: {href}\n   Snippet: {snippet}\n")

    return "\n".join(formatted)
