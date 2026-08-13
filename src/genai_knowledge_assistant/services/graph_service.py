"""Orchestrates entity extraction and graph storage during ingestion, and graph-based Q&A."""

from langchain_openai import ChatOpenAI

from genai_knowledge_assistant.core.entity_extraction import EntityExtractor
from genai_knowledge_assistant.repositories.graph_repository import GraphRepository
from genai_knowledge_assistant.config import settings


class GraphService:
    def __init__(self, graph_repo: GraphRepository):
        self.graph_repo = graph_repo
        self.extractor = EntityExtractor()
        self.llm = ChatOpenAI(
            model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0.2
        )

    def extract_and_store(self, text: str, source: str) -> int:
        """Runs entity/relationship extraction on ingested text and stores it in Neo4j.
        Returns the number of relationships stored."""
        extracted = self.extractor.extract(text)
        for rel in extracted.relationships:
            self.graph_repo.add_relationship(
                rel.subject, rel.predicate, rel.object, source=source
            )
        return len(extracted.relationships)

    def replace_source(self, text: str, source: str) -> int:
        self.graph_repo.delete_by_source(source)
        return self.extract_and_store(text, source)

    async def identify_entities_in_query(self, query: str) -> list[str]:
        """Asks the LLM which entity names in the query to look up in the graph."""
        prompt = (
            "List the key named entities (people, organizations, products, concepts, "
            "places) mentioned in this question, as a comma-separated list. "
            "If none, return an empty response.\n\n"
            f"Question: {query}"
        )
        response = await self.llm.ainvoke(prompt)
        raw = response.content.strip()
        if not raw:
            return []
        return [e.strip() for e in raw.split(",") if e.strip()]
