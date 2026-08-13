"""Pydantic schemas for entity/relationship extraction and graph queries."""

from pydantic import BaseModel, Field


class Relationship(BaseModel):
    subject: str = Field(..., description="The source entity.")
    predicate: str = Field(
        ..., description="The relationship type, e.g. 'works_at', 'located_in'."
    )
    object: str = Field(..., description="The target entity.")


class ExtractedGraph(BaseModel):
    entities: list[str] = Field(
        ..., description="Distinct named entities found in the text."
    )
    relationships: list[Relationship] = Field(
        ..., description="Relationships between entities."
    )


class GraphFact(BaseModel):
    subject: str
    predicate: str
    object: str
    source: str
