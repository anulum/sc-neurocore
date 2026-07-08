# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for GOTM Brain and content indexer

"""Tests for content_indexer.py and gotm_brain.py."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.content_indexer import (
    ContentChunk,
    _chunk_text,
    _extract_python_docstrings,
    _extract_rust_doc_comments,
    _should_skip_dir,
    embed_chunks,
    embed_tfidf,
    index_file,
    index_gotm_repo,
)
from sc_neurocore.quantum_cognition.gotm_brain import (
    GOTMBrain,
    LearningStep,
)
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF


_GOTM_MODULE = "sc_neurocore.quantum_cognition.gotm_brain"


class _GotmBrainModule(Protocol):
    """Typed view of a dynamically reloaded GOTM brain module."""

    HAS_LLM: bool
    GOTMBrain: type[GOTMBrain]
    _LLMEndpoint: type[object] | None


class _BlockingLLMFinder(importlib.abc.MetaPathFinder):
    """Import hook that makes the local ``llm`` module unavailable."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise on ``llm`` and ignore every other import."""
        if fullname == "llm":
            raise ModuleNotFoundError("blocked test llm module")
        return None


class _FixedSpikeNeuron:
    """Minimal neuron stub that returns a configured spike flag."""

    def __init__(self, spiked: bool) -> None:
        """Store the spike flag returned by ``step``."""
        self._spiked = spiked
        self.atp_level = 1.0

    def step(self, _current: float) -> tuple[float, bool]:
        """Return a deterministic voltage/spike pair."""
        return 0.0, self._spiked


# ───────── Fixtures ─────────


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal fake GOTM repository for testing."""
    src = tmp_path / "src" / "example"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('"""Example module docstring."""\n')
    (src / "core.py").write_text(
        '"""Core module.\n\n'
        "Implements the main algorithm.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "def compute(x):\n"
        '    """Compute the result."""\n'
        "    return x * 2\n"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "# Example Repository\n\n"
        "This is a test repository for the GOTM brain.\n\n"
        "## Features\n\n"
        "- Feature 1: quantum coupling\n"
        "- Feature 2: metabolic feedback\n"
    )
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "0.1.0"\n')
    rust_src = tmp_path / "src" / "lib.rs"
    rust_src.write_text(
        "/// Main computation function.\n"
        "/// Returns the doubled input.\n"
        "pub fn compute(x: f64) -> f64 {\n"
        "    x * 2.0\n"
        "}\n"
    )
    # Add a __pycache__ dir that should be skipped
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00" * 100)
    return tmp_path


# ───────── ContentChunk ─────────


class TestContentChunk:
    def test_create(self) -> None:
        c = ContentChunk(
            repo_name="TEST",
            file_path="a.py",
            chunk_index=0,
            text="hello world",
            content_type="code",
            weight=1.0,
        )
        assert c.repo_name == "TEST"
        assert len(c.sha256) == 16
        assert c.summary == "hello world"

    def test_to_dict(self) -> None:
        c = ContentChunk(
            repo_name="R",
            file_path="b.md",
            chunk_index=1,
            text="test content",
            content_type="markdown",
            weight=1.2,
        )
        d = c.to_dict()
        assert d["repo"] == "R"
        assert d["type"] == "markdown"
        assert d["length"] == 12


# ───────── Text extraction ─────────


class TestTextExtraction:
    def test_python_docstrings(self) -> None:
        code = '"""Module docstring that is long enough."""\ndef f():\n    """Function docstring text here."""\n    pass\n'
        docs = _extract_python_docstrings(code)
        assert len(docs) >= 1

    def test_python_single_quote_docstrings_and_comment_blocks(self) -> None:
        code = (
            "'''Single quoted module documentation that is long enough.'''\n"
            "# First scientific note\n"
            "# Second scientific note\n"
            "# Third scientific note\n"
            "value = 1\n"
            "# Terminal note one\n"
            "# Terminal note two\n"
            "# Terminal note three"
        )
        docs = _extract_python_docstrings(code)
        assert "Single quoted module documentation" in docs[0]
        assert any(
            "First scientific note\nSecond scientific note\nThird scientific note" in d
            for d in docs
        )
        assert any("Terminal note one\nTerminal note two\nTerminal note three" in d for d in docs)

    def test_rust_doc_comments(self) -> None:
        code = "/// First line of doc.\n/// Second line of doc.\nfn main() {}\n"
        docs = _extract_rust_doc_comments(code)
        assert len(docs) == 1
        assert "First line" in docs[0]

    def test_rust_inner_doc_comments_at_eof(self) -> None:
        code = "//! Module-level quantum cognition notes.\n//! Preserved at end of file."
        docs = _extract_rust_doc_comments(code)
        assert docs == ["Module-level quantum cognition notes.\nPreserved at end of file."]

    def test_chunk_text_short(self) -> None:
        chunks = _chunk_text("short text")
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_chunk_text_long(self) -> None:
        text = "\n\n".join([f"Paragraph {i} with some content." for i in range(50)])
        chunks = _chunk_text(text, target_size=200)
        assert len(chunks) > 1

    def test_skip_dir(self) -> None:
        assert _should_skip_dir("__pycache__")
        assert _should_skip_dir(".git")
        assert not _should_skip_dir("src")


# ───────── index_file ─────────


class TestIndexFile:
    def test_index_python(self, tmp_repo: Path) -> None:
        py_file = tmp_repo / "src" / "example" / "core.py"
        chunks = index_file(py_file, "TEST", tmp_repo)
        assert len(chunks) > 0
        types = {c.content_type for c in chunks}
        assert "docstring" in types or "code" in types

    def test_index_markdown(self, tmp_repo: Path) -> None:
        md_file = tmp_repo / "docs" / "README.md"
        chunks = index_file(md_file, "TEST", tmp_repo)
        assert len(chunks) > 0
        assert chunks[0].content_type == "markdown"

    def test_index_rust(self, tmp_repo: Path) -> None:
        rs_file = tmp_repo / "src" / "lib.rs"
        chunks = index_file(rs_file, "TEST", tmp_repo)
        assert len(chunks) > 0

    def test_skip_unknown_ext(self, tmp_repo: Path) -> None:
        unk = tmp_repo / "test.xyz"
        unk.write_text("unknown")
        chunks = index_file(unk, "TEST", tmp_repo)
        assert len(chunks) == 0

    def test_skips_empty_oversized_and_non_file_inputs(self, tmp_path: Path) -> None:
        """Indexer ignores unsupported filesystem payloads without poisoning a scan."""
        empty_md = tmp_path / "empty.md"
        empty_md.write_text("")
        assert index_file(empty_md, "TEST", tmp_path) == []

        oversized_py = tmp_path / "oversized.py"
        oversized_py.write_text('"""Large module docstring."""\n' + ("x = 1\n" * 60_000))
        assert index_file(oversized_py, "TEST", tmp_path) == []

        directory_with_supported_suffix = tmp_path / "not_a_file.md"
        directory_with_supported_suffix.mkdir()
        assert index_file(directory_with_supported_suffix, "TEST", tmp_path) == []

    def test_indexes_supported_metadata_and_hardware_files_as_code_chunks(
        self, tmp_path: Path
    ) -> None:
        """Non-doc source formats retain provenance and extension-specific weights."""
        payloads = {
            "Project.toml": 'name = "qc-indexer"\nversion = "1.0.0"\n',
            "config.yaml": "model: fisher_posner\nqubits: 12\n",
            "manifest.json": '{"repo": "SC-NEUROCORE", "pipeline": "quantum"}\n',
            "kernel.go": "package main\nfunc Step() {}\n",
            "proof.lean": "theorem posner_index : True := by trivial\n",
            "bridge.sv": "module bridge; endmodule\n",
        }
        for rel_path, text in payloads.items():
            path = tmp_path / rel_path
            path.write_text(text)
            chunks = index_file(path, "TEST", tmp_path)
            assert len(chunks) == 1
            assert chunks[0].repo_name == "TEST"
            assert chunks[0].file_path == rel_path
            assert chunks[0].content_type == "code"
            assert chunks[0].summary


# ───────── index_gotm_repo ─────────


class TestIndexRepo:
    def test_index_repo(self, tmp_repo: Path) -> None:
        chunks = index_gotm_repo(tmp_repo, "TEST-REPO")
        assert len(chunks) > 0
        repos = {c.repo_name for c in chunks}
        assert "TEST-REPO" in repos

    def test_skips_pycache(self, tmp_repo: Path) -> None:
        chunks = index_gotm_repo(tmp_repo)
        paths = {c.file_path for c in chunks}
        for p in paths:
            assert "__pycache__" not in p

    def test_nonexistent_repo(self) -> None:
        with pytest.raises(FileNotFoundError):
            index_gotm_repo("/nonexistent/path")

    def test_sorted_by_weight(self, tmp_repo: Path) -> None:
        chunks = index_gotm_repo(tmp_repo)
        weights = [c.weight for c in chunks]
        assert weights == sorted(weights, reverse=True)

    def test_skips_hidden_and_build_directories_during_walk(self, tmp_path: Path) -> None:
        visible = tmp_path / "src"
        visible.mkdir()
        (visible / "model.md").write_text("Visible quantum cognition notes.")

        hidden = tmp_path / ".cache"
        hidden.mkdir()
        (hidden / "secret.md").write_text("This hidden file must not be indexed.")

        build = tmp_path / "build"
        build.mkdir()
        (build / "artifact.md").write_text("This build artifact must not be indexed.")

        chunks = index_gotm_repo(tmp_path, "SCAN")
        paths = {chunk.file_path for chunk in chunks}
        assert paths == {"src/model.md"}


# ───────── embed_chunks ─────────


class TestEmbedChunks:
    def test_shape(self) -> None:
        chunks = [
            ContentChunk("R", "a.py", 0, "hello world test", "code", 1.0),
            ContentChunk("R", "b.md", 0, "documentation text", "markdown", 1.2),
        ]
        vectors = embed_chunks(chunks, n_dims=32)
        assert vectors.shape == (2, 32)

    def test_values_normalised(self) -> None:
        chunks = [ContentChunk("R", "a.py", 0, "test " * 100, "code", 1.0)]
        vectors = embed_chunks(chunks, n_dims=32)
        assert np.all(vectors >= 0.0)
        assert np.all(vectors <= 1.0)

    def test_deterministic(self) -> None:
        chunks = [ContentChunk("R", "a.py", 0, "deterministic test", "code", 1.0)]
        v1 = embed_chunks(chunks, seed=42)
        v2 = embed_chunks(chunks, seed=42)
        np.testing.assert_array_equal(v1, v2)

    def test_different_content_different_vectors(self) -> None:
        c1 = [ContentChunk("R", "a.py", 0, "aaaa" * 50, "code", 1.0)]
        c2 = [ContentChunk("R", "b.py", 0, "zzzz" * 50, "code", 1.0)]
        v1 = embed_chunks(c1)
        v2 = embed_chunks(c2)
        assert not np.allclose(v1, v2)

    def test_low_dimension_embeddings_preserve_requested_shape(self) -> None:
        chunk = ContentChunk("R", "empty.md", 0, "", "metadata", 0.3)
        zero_dim = embed_chunks([chunk], n_dims=0)
        assert zero_dim.shape == (1, 0)

        one_dim = embed_chunks([chunk], n_dims=1)
        assert one_dim.shape == (1, 1)
        assert np.all(one_dim == 0.0)

    def test_feature_dimensions_encode_weight_type_and_hash(self) -> None:
        chunks = [
            ContentChunk("R", "doc.py", 0, "alpha beta gamma", "docstring", 3.0),
            ContentChunk("R", "unknown.dat", 0, "alpha beta gamma", "custom", 1.0),
        ]
        vectors = embed_chunks(chunks, n_dims=32)
        assert vectors[0, 26] > 0.0
        assert vectors[0, 27] == pytest.approx(1.0)
        assert vectors[0, 28] == 1.0
        assert vectors[0, 29] == pytest.approx(0.9)
        assert vectors[1, 29] == pytest.approx(0.5)
        assert np.all((vectors[:, 30:32] >= 0.0) & (vectors[:, 30:32] <= 1.0))


# ───────── embed_tfidf ─────────


class TestEmbedTfidf:
    def test_empty_corpus_returns_empty_matrix_and_vocab(self) -> None:
        matrix, vocab = embed_tfidf([], n_dims=7)
        assert matrix.shape == (0, 7)
        assert vocab == {}

    def test_terms_filtered_out_returns_zero_matrix(self) -> None:
        chunks = [
            ContentChunk("R", "a.md", 0, "single unique alpha", "markdown", 1.0),
            ContentChunk("R", "b.md", 0, "another unique beta", "markdown", 1.0),
        ]
        matrix, vocab = embed_tfidf(chunks, n_dims=5, min_df=3)
        assert matrix.shape == (2, 5)
        assert vocab == {}
        assert np.all(matrix == 0.0)

    def test_corpus_tfidf_stems_stopwords_filters_and_l2_normalises(self) -> None:
        chunks = [
            ContentChunk(
                "R",
                "quantum_a.md",
                0,
                "the the fisher posner binding oscillation return",
                "markdown",
                1.0,
            ),
            ContentChunk(
                "R",
                "quantum_b.md",
                0,
                "fisher posner binding oscillations coherence",
                "markdown",
                1.0,
            ),
            ContentChunk(
                "R",
                "metabolic.md",
                0,
                "fisher atp metabolism coherence",
                "markdown",
                1.0,
            ),
        ]
        matrix, vocab = embed_tfidf(chunks, n_dims=8, min_df=2, max_df_ratio=0.85)

        assert matrix.shape == (3, 8)
        assert "the" not in vocab
        assert "return" not in vocab
        assert "fisher" not in vocab  # appears in every document, above max_df_ratio
        assert {"posn", "bind", "coherence"} <= set(vocab)

        row_norms = np.linalg.norm(matrix[:, : len(vocab)], axis=1)
        assert np.all(row_norms > 0.0)
        np.testing.assert_allclose(row_norms, np.ones_like(row_norms))


# ───────── GOTMBrain ─────────


class TestGOTMBrain:
    def test_init(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        assert brain.n_neurons == 8
        assert len(brain.neurons) == 8
        assert brain._total_steps == 0

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="n_neurons"):
            GOTMBrain(n_neurons=0)

    def test_get_llm_guidance_returns_valid(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        directive = brain.get_llm_guidance("test context")
        # Must always return a valid directive regardless of LLM availability
        assert directive in ("FOCUS", "EXPLORE", "STABILIZE")

    def test_process_content(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        vector = np.random.rand(8)
        spikes = brain.process_content(vector, "FOCUS")
        assert isinstance(spikes, list)
        for s in spikes:
            assert 0 <= s < 8

    def test_process_content_returns_indices_for_spiking_neurons(self) -> None:
        """Spiking neuron steps are returned as their stable indices."""
        brain = GOTMBrain(n_neurons=3, bridge_backend="emulated")
        brain.neurons = cast(
            list[HybridFisherPosnerLIF],
            [
                _FixedSpikeNeuron(False),
                _FixedSpikeNeuron(True),
                _FixedSpikeNeuron(True),
            ],
        )

        assert brain.process_content(np.ones(3), "STABILIZE") == [1, 2]

    def test_import_marks_missing_local_llm_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Importing without a local llm module leaves the fallback path enabled."""
        import sc_neurocore.quantum_cognition.gotm_brain as canonical_gotm

        monkeypatch.delitem(sys.modules, "llm", raising=False)
        sys.meta_path.insert(0, _BlockingLLMFinder())
        sys.modules.pop(_GOTM_MODULE, None)
        try:
            module = importlib.import_module(_GOTM_MODULE)
        finally:
            sys.meta_path = [
                finder for finder in sys.meta_path if not isinstance(finder, _BlockingLLMFinder)
            ]
            sys.modules[_GOTM_MODULE] = canonical_gotm

        gotm = cast(_GotmBrainModule, module)
        assert gotm.HAS_LLM is False
        assert gotm._LLMEndpoint is None

    def test_process_content_padding(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        short_vector = np.ones(4)
        spikes = brain.process_content(short_vector, "EXPLORE")
        assert isinstance(spikes, list)

    def test_process_content_truncation(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        long_vector = np.ones(16)
        spikes = brain.process_content(long_vector, "STABILIZE")
        assert isinstance(spikes, list)

    def test_learn_step(self) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        chunk = ContentChunk("R", "a.py", 0, "test content", "code", 1.0)
        vector = np.random.rand(8)
        step = brain.learn_step(chunk, vector)
        assert isinstance(step, LearningStep)
        assert step.step_index == 0
        assert step.directive in ("FOCUS", "EXPLORE", "STABILIZE")
        assert brain._total_steps == 1

    def test_learn_from_repo(self, tmp_repo: Path) -> None:
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        steps = brain.learn_from_repo(str(tmp_repo), max_chunks=5)
        assert len(steps) <= 5
        assert brain._total_steps == len(steps)

    def test_get_learning_state(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        chunk = ContentChunk("R", "a.py", 0, "state test", "code", 1.0)
        brain.learn_step(chunk, np.random.rand(4))
        state = brain.get_learning_state()
        assert state["n_neurons"] == 4
        assert state["total_steps"] == 1
        assert "pool_state" in state

    def test_get_history(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        chunk = ContentChunk("R", "a.py", 0, "history test", "code", 1.0)
        brain.learn_step(chunk, np.random.rand(4))
        history = brain.get_history()
        assert len(history) == 1
        assert history[0]["directive"] in ("FOCUS", "EXPLORE", "STABILIZE")

    def test_reset(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        brain.learn_step(
            ContentChunk("R", "a.py", 0, "reset test", "code", 1.0),
            np.random.rand(4),
        )
        brain.reset()
        assert brain._total_steps == 0
        assert len(brain._history) == 0

    def test_repr(self) -> None:
        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        r = repr(brain)
        assert "GOTMBrain" in r
        assert "n_neurons=4" in r

    def test_entanglement_evolves(self, tmp_repo: Path) -> None:
        """After learning, entanglement should have structure (non-uniform)."""
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        initial_ent = brain.pool.entanglement_map.copy()
        brain.learn_from_repo(str(tmp_repo), max_chunks=10)
        if brain._total_steps > 0 and sum(n._total_spikes for n in brain.neurons) > 0:
            assert not np.allclose(brain.pool.entanglement_map, initial_ent)


# ───────── LearningStep ─────────


class TestLearningStep:
    def test_to_dict(self) -> None:
        s = LearningStep(
            step_index=0,
            directive="FOCUS",
            target_coherence=0.8,
            n_spikes=5,
            avg_atp=0.95,
            avg_entanglement=0.125,
            chunk_summary="test",
            chunk_sha256="abc123",
        )
        d = s.to_dict()
        assert d["step"] == 0
        assert d["directive"] == "FOCUS"


# ───────── Package import ─────────


class TestPackageImport:
    def test_import_new_symbols(self) -> None:
        from sc_neurocore.quantum_cognition import (
            ContentChunk,
            GOTMBrain,
            embed_chunks,
            index_gotm_repo,
        )

        assert ContentChunk is not None
        assert GOTMBrain is not None
        assert embed_chunks is not None
        assert index_gotm_repo is not None


# ───────── v_deep persistence ─────────


class TestBrainPersistence:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """save_state → load_state preserves full brain state."""
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated", seed=42)
        chunk = ContentChunk("R", "a.py", 0, "persistence test content", "code", 1.0)
        vec = np.random.default_rng(42).random(8)
        brain.learn_step(chunk, vec)
        state_before = brain.get_learning_state()

        path = str(tmp_path / "brain.json")
        brain.save_state(path)

        brain2 = GOTMBrain(n_neurons=8, bridge_backend="emulated", seed=99)
        brain2.load_state(path)
        state_after = brain2.get_learning_state()

        assert state_before["total_steps"] == state_after["total_steps"]
        assert state_before["total_spikes"] == state_after["total_spikes"]
        assert state_after["history_length"] == 1

    def test_load_dimension_mismatch(self, tmp_path: Path) -> None:
        """load_state raises ValueError on neuron count mismatch."""
        brain = GOTMBrain(n_neurons=8, bridge_backend="emulated")
        path = str(tmp_path / "brain.json")
        brain.save_state(path)

        brain2 = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        with pytest.raises(ValueError, match="neurons"):
            brain2.load_state(path)

    def test_save_creates_valid_json(self, tmp_path: Path) -> None:
        """Saved file is valid JSON."""
        import json

        brain = GOTMBrain(n_neurons=4, bridge_backend="emulated")
        path = str(tmp_path / "brain.json")
        brain.save_state(path)

        with open(path) as f:
            data = json.load(f)
        assert data["n_neurons"] == 4
        assert "neuron_states" in data
        assert "pool_state" in data
        assert "history" in data


# ───────── HybridFisherPosnerLIF.v property ─────────


class TestVProperty:
    def test_v_reads_Vm(self) -> None:
        from sc_neurocore.quantum_cognition import SpinPoolMPS, HybridFisherPosnerLIF

        pool = SpinPoolMPS(n_sites=4)
        n = HybridFisherPosnerLIF(0, pool)
        assert n.v == n.Vm == -70.0

    def test_v_writes_Vm(self) -> None:
        from sc_neurocore.quantum_cognition import SpinPoolMPS, HybridFisherPosnerLIF

        pool = SpinPoolMPS(n_sites=4)
        n = HybridFisherPosnerLIF(0, pool)
        n.v = -55.0
        assert n.Vm == -55.0
        assert n.v == -55.0


# ───────── HybridFisherPosnerLIFNeuron (Population wrapper) ─────────


class TestPopulationWrapper:
    def test_step_returns_int(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        result = n.step(0.0)
        assert isinstance(result, int)
        assert result in (0, 1)

    def test_v_property(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        assert n.v == -70.0
        n.v = -55.0
        assert n.v == -55.0

    def test_spiking(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        total_spikes = 0
        for _ in range(200):
            total_spikes += n.step(50.0)
        assert total_spikes > 0

    def test_reset(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        n.step(50.0)
        n.reset()
        assert n.v == -70.0

    def test_get_state(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        state = n.get_state()
        assert "Vm" in state
        assert "atp_level" in state
