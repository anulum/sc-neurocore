# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GOTM content indexer

"""Content indexer for the God of the Math (GOTM) collection.

Walks repository trees, extracts text from source code and documentation,
produces semantically structured chunks suitable for driving neural
input currents in the GOTM Brain module.

The indexer is designed to be deterministic and offline — it reads files
from the local filesystem only and produces numerical vectors via
lightweight statistical methods (TF-IDF-inspired), not via external
embedding APIs.

Supported file types:
    - Python (.py) — extracts docstrings and comments
    - Markdown (.md) — full text
    - Rust (.rs), Julia (.jl), Mojo (.mojo), Go (.go) — doc comments
    - TOML, YAML, JSON — metadata fields
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# File extensions to index, with priority weights
_EXTENSION_WEIGHTS: dict[str, float] = {
    ".py": 1.0,
    ".md": 1.2,  # Documentation is high-value
    ".rs": 0.8,
    ".jl": 0.8,
    ".mojo": 0.8,
    ".go": 0.7,
    ".toml": 0.5,
    ".yaml": 0.5,
    ".yml": 0.5,
    ".json": 0.3,
    ".sv": 0.6,  # SystemVerilog
    ".v": 0.6,  # Verilog
    ".lean": 0.9,  # Lean 4 proofs
}

# Directories to skip
_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        ".pixi",
        "build",
        "dist",
        ".eggs",
        "*.egg-info",
        ".tox",
        ".pytest_cache",
    }
)

# Maximum file size to index (256 KB)
_MAX_FILE_BYTES = 256 * 1024

# Chunk target size in characters
_CHUNK_TARGET_SIZE = 2000


@dataclass
class ContentChunk:
    """A single indexed content chunk from a GOTM repository.

    Attributes
    ----------
    repo_name : str
        Repository name (e.g. ``"SC-NEUROCORE"``).
    file_path : str
        Relative path within the repository.
    chunk_index : int
        Sequential index within the file.
    text : str
        Raw text content of the chunk.
    content_type : str
        One of ``"docstring"``, ``"comment"``, ``"markdown"``, ``"code"``,
        ``"metadata"``.
    weight : float
        Priority weight based on file type.
    sha256 : str
        SHA-256 hash of the chunk text (provenance).
    """

    repo_name: str
    file_path: str
    chunk_index: int
    text: str
    content_type: str
    weight: float
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @property
    def summary(self) -> str:
        """First 200 characters of the chunk text."""
        return self.text[:200].replace("\n", " ").strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-compatible dict."""
        return {
            "repo": self.repo_name,
            "path": self.file_path,
            "chunk": self.chunk_index,
            "type": self.content_type,
            "weight": self.weight,
            "sha256": self.sha256,
            "summary": self.summary,
            "length": len(self.text),
        }


def _should_skip_dir(name: str) -> bool:
    """Check if a directory should be skipped during indexing."""
    return name in _SKIP_DIRS or name.startswith(".") or name.endswith(".egg-info")


def _extract_python_docstrings(text: str) -> list[str]:
    """Extract docstrings and significant comments from Python source."""
    chunks: list[str] = []
    # Triple-quoted strings (docstrings)
    for match in re.finditer(r'"""(.*?)"""', text, re.DOTALL):
        doc = match.group(1).strip()
        if len(doc) > 20:
            chunks.append(doc)
    for match in re.finditer(r"'''(.*?)'''", text, re.DOTALL):
        doc = match.group(1).strip()
        if len(doc) > 20:
            chunks.append(doc)
    # Comment blocks (3+ consecutive comment lines)
    comment_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment_lines.append(stripped.lstrip("# "))
        else:
            if len(comment_lines) >= 3:
                chunks.append("\n".join(comment_lines))
            comment_lines = []
    if len(comment_lines) >= 3:
        chunks.append("\n".join(comment_lines))
    return chunks


def _extract_rust_doc_comments(text: str) -> list[str]:
    """Extract /// and //! doc comments from Rust source."""
    chunks: list[str] = []
    doc_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("///") or stripped.startswith("//!"):
            doc_lines.append(stripped.lstrip("/!").strip())
        else:
            if len(doc_lines) >= 2:
                chunks.append("\n".join(doc_lines))
            doc_lines = []
    if len(doc_lines) >= 2:
        chunks.append("\n".join(doc_lines))
    return chunks


def _chunk_text(text: str, target_size: int = _CHUNK_TARGET_SIZE) -> list[str]:
    """Split text into chunks of approximately target_size characters."""
    if len(text) <= target_size:
        return [text] if text.strip() else []
    chunks: list[str] = []
    paragraphs = re.split(r"\n\n+", text)
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > target_size and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if c.strip()]


def index_file(file_path: Path, repo_name: str, repo_root: Path) -> list[ContentChunk]:
    """Index a single file into content chunks.

    Parameters
    ----------
    file_path : Path
        Absolute path to the file.
    repo_name : str
        Name of the repository.
    repo_root : Path
        Root directory of the repository.

    Returns
    -------
    list[ContentChunk]
        Extracted content chunks with provenance metadata.
    """
    ext = file_path.suffix.lower()
    weight = _EXTENSION_WEIGHTS.get(ext, 0.0)
    if weight == 0.0:
        return []

    try:
        stat = file_path.stat()
        if stat.st_size > _MAX_FILE_BYTES or stat.st_size == 0:
            return []
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        logger.debug("Skipping %s: %s", file_path, exc)
        return []

    rel_path = str(file_path.relative_to(repo_root))
    chunks: list[ContentChunk] = []

    if ext == ".py":
        doc_chunks = _extract_python_docstrings(text)
        for i, doc in enumerate(doc_chunks):
            chunks.append(
                ContentChunk(
                    repo_name=repo_name,
                    file_path=rel_path,
                    chunk_index=i,
                    text=doc,
                    content_type="docstring",
                    weight=weight * 1.5,
                )
            )
        # Also index the full code in larger chunks
        code_chunks = _chunk_text(text)
        for i, code in enumerate(code_chunks):
            chunks.append(
                ContentChunk(
                    repo_name=repo_name,
                    file_path=rel_path,
                    chunk_index=len(doc_chunks) + i,
                    text=code,
                    content_type="code",
                    weight=weight,
                )
            )
    elif ext == ".md":
        md_chunks = _chunk_text(text)
        for i, md in enumerate(md_chunks):
            chunks.append(
                ContentChunk(
                    repo_name=repo_name,
                    file_path=rel_path,
                    chunk_index=i,
                    text=md,
                    content_type="markdown",
                    weight=weight,
                )
            )
    elif ext in (".rs",):
        doc_chunks = _extract_rust_doc_comments(text)
        for i, doc in enumerate(doc_chunks):
            chunks.append(
                ContentChunk(
                    repo_name=repo_name,
                    file_path=rel_path,
                    chunk_index=i,
                    text=doc,
                    content_type="comment",
                    weight=weight * 1.3,
                )
            )
    else:
        text_chunks = _chunk_text(text)
        for i, tc in enumerate(text_chunks):
            chunks.append(
                ContentChunk(
                    repo_name=repo_name,
                    file_path=rel_path,
                    chunk_index=i,
                    text=tc,
                    content_type="code",
                    weight=weight,
                )
            )

    return chunks


def index_gotm_repo(
    repo_path: str | Path,
    repo_name: str | None = None,
) -> list[ContentChunk]:
    """Index an entire GOTM repository into content chunks.

    Parameters
    ----------
    repo_path : str or Path
        Path to the repository root.
    repo_name : str, optional
        Override repository name (default: directory name).

    Returns
    -------
    list[ContentChunk]
        All indexed chunks sorted by weight (descending).
    """
    repo_root = Path(repo_path)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repository not found: {repo_root}")
    if repo_name is None:
        repo_name = repo_root.name

    all_chunks: list[ContentChunk] = []
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Filter out skip directories in-place
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            chunks = index_file(fpath, repo_name, repo_root)
            all_chunks.extend(chunks)
            if chunks:
                file_count += 1

    all_chunks.sort(key=lambda c: c.weight, reverse=True)
    logger.info(
        "Indexed %s: %d files → %d chunks",
        repo_name,
        file_count,
        len(all_chunks),
    )
    return all_chunks


def embed_chunks(
    chunks: list[ContentChunk],
    n_dims: int = 32,
    seed: int = 42,
) -> np.ndarray[Any, Any]:
    """Convert content chunks to numerical vectors for neural input.

    Uses a lightweight deterministic hashing approach (not a neural
    embedding model) to produce fixed-size feature vectors.  Each
    dimension captures a different statistical property of the text:

    - Character frequency distribution (dims 0–25)
    - Text length features (dim 26–27)
    - Weight and content type (dim 28–29)
    - Hash-derived features (dim 30–31)

    Parameters
    ----------
    chunks : list[ContentChunk]
        Content chunks to embed.
    n_dims : int
        Output vector dimensionality (default 32).
    seed : int
        Random seed for hash-derived features.

    Returns
    -------
    np.ndarray[Any, Any]
        Shape ``(len(chunks), n_dims)``, values normalised to [0, 1].
    """
    rng = np.random.default_rng(seed)
    vectors = np.zeros((len(chunks), n_dims), dtype=np.float64)

    for i, chunk in enumerate(chunks):
        text = chunk.text.lower()
        n_chars = max(len(text), 1)

        # Dims 0–min(25, n_dims-1): character frequency (a–z)
        char_dims = min(26, n_dims)
        for c in text:
            idx = ord(c) - ord("a")
            if 0 <= idx < char_dims:
                vectors[i, idx] += 1.0
        if char_dims > 0:
            vectors[i, :char_dims] /= n_chars

        # Remaining features only if dims are available
        if n_dims > 26:
            vectors[i, 26] = min(np.log1p(n_chars) / 10.0, 1.0)
        if n_dims > 27:
            words = text.split()
            if words:
                vectors[i, 27] = len(set(words)) / len(words)
        if n_dims > 28:
            vectors[i, 28] = min(chunk.weight / 2.0, 1.0)
        if n_dims > 29:
            type_map = {
                "docstring": 0.9,
                "markdown": 0.8,
                "comment": 0.7,
                "code": 0.5,
                "metadata": 0.3,
            }
            vectors[i, 29] = type_map.get(chunk.content_type, 0.5)
        if n_dims > 31:
            h = int(chunk.sha256[:8], 16)
            vectors[i, 30] = (h & 0xFFFF) / 0xFFFF
            vectors[i, 31] = ((h >> 16) & 0xFFFF) / 0xFFFF

    return vectors


def embed_tfidf(
    chunks: list[ContentChunk],
    n_dims: int = 256,
    min_df: int = 2,
    max_df_ratio: float = 0.85,
) -> tuple[np.ndarray[Any, Any], dict[str, int]]:
    """Compute proper TF-IDF vectors from a corpus of chunks.

    Unlike ``embed_chunks()`` which uses character statistics, this
    computes true TF-IDF with corpus-wide Inverse Document Frequency:

        TF(t,d) = log(1 + count(t,d))
        IDF(t) = log(N / df(t))
        TF-IDF(t,d) = TF(t,d) × IDF(t)

    Parameters
    ----------
    chunks : list[ContentChunk]
        Corpus of content chunks.
    n_dims : int
        Number of top-IDF terms to use as feature dimensions.
    min_df : int
        Minimum document frequency for a term to be included.
    max_df_ratio : float
        Maximum document frequency ratio (terms in >85% of docs removed).

    Returns
    -------
    tuple[np.ndarray[Any, Any], dict[str, int]]
        - TF-IDF matrix of shape ``(len(chunks), n_dims)``, L2-normalised.
        - Vocabulary mapping {term: dimension_index}.
    """
    import math
    from collections import Counter

    N = len(chunks)
    if N == 0:
        return np.zeros((0, n_dims), dtype=np.float64), {}

    # Stopwords: common programming and English terms
    _STOPWORDS = frozenset(
        {
            "the",
            "and",
            "for",
            "that",
            "this",
            "with",
            "from",
            "are",
            "was",
            "not",
            "but",
            "has",
            "have",
            "had",
            "will",
            "can",
            "all",
            "been",
            "were",
            "they",
            "their",
            "which",
            "when",
            "what",
            "how",
            "who",
            "each",
            "than",
            "other",
            "into",
            "also",
            "its",
            "may",
            "use",
            # Programming stopwords
            "self",
            "none",
            "true",
            "false",
            "return",
            "import",
            "def",
            "class",
            "elif",
            "else",
            "pass",
            "raise",
            "yield",
            "lambda",
            "try",
            "except",
            "finally",
            "assert",
            "while",
            "break",
            "continue",
            "global",
            "str",
            "int",
            "float",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "type",
            "any",
            "args",
            "kwargs",
        }
    )

    def _stem(word: str) -> str:
        """Minimal suffix-stripping stemmer (Porter-like)."""
        if len(word) <= 4:
            return word
        for suffix in (
            "ation",
            "ment",
            "ness",
            "ting",
            "ing",
            "ies",
            "ous",
            "ive",
            "ful",
            "ble",
            "ed",
            "ly",
            "er",
            "es",
            "al",
        ):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    # Tokenise all documents with stopword removal and stemming
    _TOKEN_RE = re.compile(r"[a-z_][a-z0-9_]{2,}")
    doc_tokens = []
    for chunk in chunks:
        raw_tokens = _TOKEN_RE.findall(chunk.text.lower())
        tokens = [_stem(t) for t in raw_tokens if t not in _STOPWORDS]
        doc_tokens.append(tokens)

    # Compute document frequency for each term
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        for term in set(tokens):
            df[term] += 1

    # Filter by min_df and max_df
    max_df = int(N * max_df_ratio)
    vocab = {
        term: i
        for i, (term, freq) in enumerate(
            sorted(
                ((t, f) for t, f in df.items() if min_df <= f <= max_df),
                key=lambda x: -x[1],
            )[:n_dims]
        )
    }

    if not vocab:
        return np.zeros((N, n_dims), dtype=np.float64), {}

    actual_dims = len(vocab)

    # Compute TF-IDF matrix
    tfidf = np.zeros((N, actual_dims), dtype=np.float64)
    for doc_idx, tokens in enumerate(doc_tokens):
        tf_raw = Counter(tokens)
        for term, count in tf_raw.items():
            if term in vocab:
                dim = vocab[term]
                tf = math.log1p(count)
                idf = math.log(N / df[term])
                tfidf[doc_idx, dim] = tf * idf

    # L2-normalise each row
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf /= norms

    # Pad to n_dims if vocab is smaller
    if actual_dims < n_dims:
        padded = np.zeros((N, n_dims), dtype=np.float64)
        padded[:, :actual_dims] = tfidf
        tfidf = padded

    return tfidf, vocab


__all__ = [
    "ContentChunk",
    "index_file",
    "index_gotm_repo",
    "embed_chunks",
    "embed_tfidf",
]
