"""Stores document metadata locally as JSON, keyed by content hash for dedup."""

import json
from pathlib import Path
from genai_knowledge_assistant.models.document import DocumentMetadata


class DocumentRepository:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self.storage_path.exists():
            with open(self.storage_path, "r") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def find_by_hash(self, content_hash: str) -> DocumentMetadata | None:
        for record in self._data.values():
            if record["content_hash"] == content_hash:
                return DocumentMetadata(**record)
        return None

    def save(self, metadata: DocumentMetadata) -> None:
        self._data[metadata.id] = metadata.model_dump(mode="json")
        self._save()

    def delete(self, document_id: str) -> None:
        self._data.pop(document_id, None)
        self._save()

    def list_all(self) -> list[DocumentMetadata]:
        return [DocumentMetadata(**record) for record in self._data.values()]
