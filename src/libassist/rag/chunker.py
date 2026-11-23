from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from libassist.rag.text import fold

# split data into chunks based on sections


@dataclass(frozen=True)
class Chunk:
    id: str  # citation id
    title: str
    text: str  # the section body
    source: str  # og. file

    @property
    def indexed_text(self) -> str:
        return f"{self.title}\n{self.text}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")


def chunk_markdown(doc_id: str, markdown: str, source: str | None = None) -> list[Chunk]:
    doc_title = doc_id
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for line in markdown.splitlines():
        if line.startswith("# ") and doc_title == doc_id and current is None:
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            current = (line[3:].strip(), [])
            sections.append(current)
        elif current is not None:
            current[1].append(line)

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for heading, body in sections:
        text = "\n".join(body).strip()
        if not text:
            continue
        chunk_id = f"{doc_id}#{slugify(heading)}"
        suffix = 2
        while chunk_id in seen:  # two sections with the same heading still get distinct ids
            chunk_id = f"{doc_id}#{slugify(heading)}-{suffix}"
            suffix += 1
        seen.add(chunk_id)
        chunks.append(Chunk(chunk_id, f"{doc_title} > {heading}", text, source or f"{doc_id}.md"))
    return chunks


def load_policy_chunks(policies_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(policies_dir).glob("*.md")):
        chunks.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8"), source=path.name))
    if not chunks:
        raise FileNotFoundError(f"No policy documents found in {policies_dir}")
    return chunks
