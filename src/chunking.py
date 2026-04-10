from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split into sentences keeping the delimiters
        # Using a lookbehind to split after (.!?) followed by space or newline
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_sentences).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Special case for character level split
        if separator == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # Split the text by the current separator
        parts = current_text.split(separator)
        final_chunks = []
        temp_buffer = ""

        for i, part in enumerate(parts):
            # Re-attach the separator except for the last part
            piece = part + (separator if i < len(parts) - 1 else "")
            
            if not piece:
                continue

            if len(piece) > self.chunk_size:
                # Piece is too large, flush buffer then recurse on the piece
                if temp_buffer:
                    final_chunks.append(temp_buffer)
                    temp_buffer = ""
                final_chunks.extend(self._split(piece, next_separators))
            elif len(temp_buffer) + len(piece) <= self.chunk_size:
                temp_buffer += piece
            else:
                # Buffer is full, flush it and start new buffer with this piece
                if temp_buffer:
                    final_chunks.append(temp_buffer)
                temp_buffer = piece

        if temp_buffer:
            final_chunks.append(temp_buffer)
            
        return final_chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=20),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        
        results = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            results[name] = {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                "chunks": chunks,
            }
        return results


class HeaderChunker:
    """
    Chunks text by looking for Markdown headers (# Header).
    Each chunk starts with a header and includes all text until the next header.
    """

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split using regex but keep the delimiter (header)
        # ^#+\s+.*$ matches lines starting with one or more '#'
        pattern = r"(^#+\s+.*$)"
        parts = re.split(pattern, text, flags=re.MULTILINE)

        chunks = []
        # First part is text before the first header (if any)
        if parts[0].strip():
            chunks.append(parts[0].strip())

        # parts now looks like: [pre-text, header1, body1, header2, body2, ...]
        for i in range(1, len(parts), 2):
            header = parts[i]
            body = parts[i + 1] if (i + 1) < len(parts) else ""
            combined = (header + body).strip()
            if combined:
                chunks.append(combined)

        return chunks


class HybridChunker:
    """
    A hybrid chunking strategy that first splits by Markdown headers,
    and then further splits any sections that exceed a maximum size
    using a recursive strategy.
    """

    def __init__(self, max_chunk_size: int = 1500) -> None:
        self.max_chunk_size = max_chunk_size
        self.header_chunker = HeaderChunker()
        self.recursive_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        # Step 1: Split by header
        sections = self.header_chunker.chunk(text)

        final_chunks = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                final_chunks.append(section)
            else:
                # Step 2: Sub-chunk oversized sections recursively
                sub_chunks = self.recursive_chunker.chunk(section)
                final_chunks.extend(sub_chunks)

        return final_chunks
