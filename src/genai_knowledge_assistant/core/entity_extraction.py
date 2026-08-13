"""Extracts named entities and relationships from text using structured LLM output."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from genai_knowledge_assistant.config import settings
from genai_knowledge_assistant.models.graph import ExtractedGraph

EXTRACTION_PROMPT = (
    "Extract named entities (people, organizations, products, concepts, places) "
    "and the relationships between them from the text below. Keep entity names "
    "short and consistent (e.g. always 'OpenAI', never 'OpenAI Inc.' and 'OpenAI' "
    "interchangeably). Only extract relationships that are explicitly stated or "
    "clearly implied — do not infer speculative connections. If no clear entities "
    "or relationships exist, return empty lists."
)


class EntityExtractor:
    def __init__(self):
        llm = ChatOpenAI(
            model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0
        )
        self.structured_llm = llm.with_structured_output(ExtractedGraph)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EXTRACTION_PROMPT),
                ("human", "{text}"),
            ]
        )
        self.chain = self.prompt | self.structured_llm

    def extract(self, text: str) -> ExtractedGraph:
        return self.chain.invoke({"text": text})
