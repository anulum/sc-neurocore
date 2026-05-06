# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for GOTM Brain and content indexer

"""Tests for content_indexer.py and gotm_brain.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.content_indexer import (
    ContentChunk,
    _chunk_text,
    _extract_python_docstrings,
    _extract_rust_doc_comments,
    _should_skip_dir,
    embed_chunks,
    index_file,
    index_gotm_repo,
)
from sc_neurocore.quantum_cognition.gotm_brain import (
    GOTMBrain,
    LearningStep,
)


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
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "example"\nversion = "0.1.0"\n'
    )
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
            repo_name="TEST", file_path="a.py",
            chunk_index=0, text="hello world",
            content_type="code", weight=1.0,
        )
        assert c.repo_name == "TEST"
        assert len(c.sha256) == 16
        assert c.summary == "hello world"

    def test_to_dict(self) -> None:
        c = ContentChunk(
            repo_name="R", file_path="b.md",
            chunk_index=1, text="test content",
            content_type="markdown", weight=1.2,
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

    def test_rust_doc_comments(self) -> None:
        code = "/// First line of doc.\n/// Second line of doc.\nfn main() {}\n"
        docs = _extract_rust_doc_comments(code)
        assert len(docs) == 1
        assert "First line" in docs[0]

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
            step_index=0, directive="FOCUS",
            target_coherence=0.8, n_spikes=5,
            avg_atp=0.95, avg_entanglement=0.125,
            chunk_summary="test", chunk_sha256="abc123",
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
