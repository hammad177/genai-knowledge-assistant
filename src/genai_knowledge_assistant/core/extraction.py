"""Extracts raw text from PDFs and web URLs."""

from pathlib import Path
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


class PDFExtractor:
    def extract(self, file_path: Path) -> str:
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(p for p in pages if p.strip())


class URLExtractor:
    def extract(self, url: str) -> str:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
