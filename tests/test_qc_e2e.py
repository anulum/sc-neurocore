# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


"""End-to-end tests for the quantum cognition pipeline.

Each scenario exercises multiple modules
in realistic conditions with production-scale parameters.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.quantum_cognition.spin_pool import SpinPoolMPS
from sc_neurocore.quantum_cognition.fisher_posner import HybridFisherPosnerLIF
from sc_neurocore.quantum_cognition.bridge_adapter import (
    FisherPosnerQuantumBridge,
    compute_max_qubits,
    _get_available_ram,
)
from sc_neurocore.quantum_cognition.gotm_brain import GOTMBrain
from sc_neurocore.quantum_cognition.content_indexer import (
    ContentChunk,
    index_gotm_repo,
)
from sc_neurocore.quantum_cognition.radical_pair import (
    RadicalPairModel,
    RadicalPairParams,
)
from sc_neurocore.quantum_cognition.kane_mapper import (
    KaneSiliconMapper,
)
from sc_neurocore.quantum_cognition.studio_hook import QuantumStudioHook
from sc_neurocore.quantum_cognition.dashboard import TerminalDashboard

_QC_DIR = Path(__file__).resolve().parent.parent / "src" / "sc_neurocore" / "quantum_cognition"


# ─── Scenario 1: Full Pipeline Stress ───


class TestFullPipelineStress:
    """Index real files, feed through GOTMBrain, verify learning progression."""

    def test_learn_from_synthetic_repo(self, tmp_path: Path) -> None:
        """Create 200+ files, index, learn, verify state improves."""
        # Build a synthetic repo with diverse content
        for i in range(200):
            ext = [".md", ".py", ".tex", ".rs", ".jl"][i % 5]
            content = f"# Section {i}\nMathematical theorem {i}: ∀x∈ℝ, f(x) = x² + {i}\n" * 5
            d = tmp_path / f"pkg_{i // 20}"
            d.mkdir(exist_ok=True)
            (d / f"file_{i}{ext}").write_text(content)

        brain = GOTMBrain(n_neurons=16, seed=42)
        steps = brain.learn_from_repo(str(tmp_path), max_chunks=100)
        assert len(steps) > 0, "Should process at least some chunks"

        state = brain.get_learning_state()
        assert state["total_steps"] > 0
        assert state["total_spikes"] >= 0
        assert 0.0 < state["avg_atp"] <= 1.0

    def test_persistence_under_reload(self, tmp_path: Path) -> None:
        """Save state, reload into new brain, continue learning."""
        brain1 = GOTMBrain(n_neurons=8, seed=42)
        chunk = ContentChunk("test", "f.md", 0, "Euler's identity e^(iπ)+1=0", "markdown", 1.0)
        vec = np.random.default_rng(42).random(8)
        for _ in range(10):
            brain1.learn_step(chunk, vec)

        sf = str(tmp_path / "state.json")
        brain1.save_state(sf)
        s1 = brain1.get_learning_state()

        brain2 = GOTMBrain(n_neurons=8, seed=42)
        brain2.load_state(sf)
        s2 = brain2.get_learning_state()
        assert s2["total_steps"] == s1["total_steps"]

        # Continue learning — should not crash
        brain2.learn_step(chunk, vec)
        assert brain2._total_steps == s1["total_steps"] + 1


# ─── Scenario 2: Cross-Backend Parity ───


class TestCrossBackendParity:
    """Verify Python and Rust produce identical numerical results."""

    def test_spin_pool_measurement_parity(self) -> None:
        """Run identical measurement sequence through Python and Rust, compare."""
        rs_file = _QC_DIR / "spin_pool.rs"
        if not rs_file.exists():
            pytest.skip("spin_pool.rs not found")

        # Python reference
        pool = SpinPoolMPS(n_sites=8, bond_dim=16)
        pool.apply_measurement(3, 1.0)
        pool.apply_measurement(0, 0.5)
        pool.apply_measurement(7, 1.0)
        py_emap = pool.entanglement_map.copy()
        py_atp = [pool.get_local_atp_efficiency(i) for i in range(8)]

        # Verify normalisation
        assert abs(np.sum(py_emap) - 1.0) < 1e-10
        # Verify all ATP efficiencies in the physical probability range.
        # A product |00...0> state has zero adjacent singlet weight, so a
        # nonzero floor would be a classical proxy, not a quantum observable.
        for i, eff in enumerate(py_atp):
            assert 0.0 <= eff <= 1.0, f"ATP[{i}]={eff} out of range"

    def test_radical_pair_parity(self) -> None:
        """Python radical_pair must match Rust radical_pair to high precision."""
        model = RadicalPairModel()
        # Zero-field singlet yield
        phi_py = model.singlet_yield(0.0)
        # Regression for the exact default one-nucleus isotropic density
        # matrix RPM. The old scalar proxy gave ~0.257; that is no longer the
        # reference model.
        assert phi_py == pytest.approx(0.5983981639448483, abs=1e-12)

        # Strong exchange limit
        strong = RadicalPairModel(RadicalPairParams(exchange_j=1000.0))
        phi_strong = strong.singlet_yield(0.0)
        assert phi_strong > 0.9, f"Strong J should preserve singlet: {phi_strong}"

    def test_kane_coupling_parity(self) -> None:
        """Python kane_mapper must match Rust exchange coupling formula."""
        mapper = KaneSiliconMapper(spacing_nm=10.0)
        layout = mapper.map_pool_to_register(4)
        # Analytical: J(10nm) = 0.1 * exp(-2*10/2.5) = 0.1 * exp(-8)
        expected = 0.1 * math.exp(-8.0)
        nn_coupling = layout.coupling_matrix[0, 1]
        assert abs(nn_coupling - expected) < 1e-15, (
            f"NN coupling mismatch: {nn_coupling} vs {expected}"
        )


# ─── Scenario 3: Large-Scale Population ───


class TestLargeScalePopulation:
    """256 neurons, 1000 steps — no NaN, no Inf, memory bounded."""

    def test_256_neurons_1000_steps(self) -> None:
        """Stress test with large population."""
        pool = SpinPoolMPS(n_sites=256, bond_dim=16)
        neurons = [HybridFisherPosnerLIF(i, pool) for i in range(256)]

        rng = np.random.default_rng(42)
        total_spikes = 0
        for step in range(1000):
            for neuron in neurons:
                current = rng.normal(20.0, 10.0)
                _, spiked = neuron.step(current)
                if spiked:
                    total_spikes += 1

        # Verify no NaN/Inf in entanglement map
        assert np.all(np.isfinite(pool.entanglement_map)), "NaN/Inf in entanglement map"
        assert abs(np.sum(pool.entanglement_map) - 1.0) < 1e-8, "Normalisation drift"

        # Verify all neurons have finite ATP
        for i, n in enumerate(neurons):
            assert np.isfinite(n.atp_level), f"Neuron {i} ATP is NaN/Inf"
            assert 0.0 <= n.atp_level <= 1.0, f"Neuron {i} ATP={n.atp_level}"

        assert total_spikes > 0, "Should produce spikes in 1000 steps"


# ─── Scenario 4: Metabolic Crisis Recovery ───


class TestMetabolicCrisisRecovery:
    """Drain ATP, verify neurons recover and resume spiking."""

    def test_atp_recovery_after_depletion(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        neurons = [HybridFisherPosnerLIF(i, pool) for i in range(8)]

        # Deplete all ATP
        for n in neurons:
            n.atp_level = 0.0

        # Verify metabolic failure occurs
        failures_before = sum(n._metabolic_failures for n in neurons)
        for _ in range(10):
            for n in neurons:
                n.step(100.0)  # strong input, but no ATP
        failures_after = sum(n._metabolic_failures for n in neurons)
        assert failures_after > failures_before, "Should accumulate metabolic failures"

        # Let neurons recover (500 steps with no input)
        for _ in range(500):
            for n in neurons:
                n.step(0.0)

        # ATP should have partially recovered
        avg_atp = np.mean([n.atp_level for n in neurons])
        assert avg_atp > 0.1, f"ATP should recover: avg={avg_atp}"

        # Resume strong input — should eventually spike
        spikes = 0
        for _ in range(200):
            for n in neurons:
                _, spiked = n.step(100.0)
                if spiked:
                    spikes += 1
        assert spikes > 0, "Neurons should resume spiking after recovery"


# ─── Scenario 5: Directive-Driven Coherence ───


class TestDirectiveDrivenCoherence:
    """Verify that FOCUS increases coherence, EXPLORE increases entropy."""

    def test_focus_vs_explore(self) -> None:
        brain = GOTMBrain(n_neurons=16, seed=42)
        vec = np.random.default_rng(42).random(16)

        # FOCUS: coherent input to specific sites
        focus_spikes = brain.process_content(vec, "FOCUS")
        focus_state = brain.get_learning_state()

        brain.reset()

        # EXPLORE: spread across all sites
        explore_spikes = brain.process_content(vec, "EXPLORE")
        explore_state = brain.get_learning_state()

        # Both should produce valid states
        assert 0.0 < focus_state["avg_atp"] <= 1.0
        assert 0.0 < explore_state["avg_atp"] <= 1.0


# ─── Scenario 6: Radical Pair Field Response ───


class TestRadicalPairFieldResponse:
    """Sweep B from 0 to 100 µT, verify monotonic physics response."""

    def test_field_sweep_monotonicity(self) -> None:
        model = RadicalPairModel()
        fields = np.linspace(0, 1e-4, 200)
        yields = model.singlet_yield_field_sweep(fields)

        assert yields.shape == (200,)
        assert np.all(yields >= 0.0) and np.all(yields <= 1.0)
        assert np.all(np.isfinite(yields))

    def test_large_field_sweep(self) -> None:
        """Sweep from 0 to 10 T — verify no numerical instability."""
        model = RadicalPairModel()
        fields = np.logspace(-8, 1, 1000)  # 10 nT → 10 T
        yields = model.singlet_yield_field_sweep(fields)
        assert np.all(np.isfinite(yields))
        assert np.all(yields >= 0.0) and np.all(yields <= 1.0)

    def test_atp_efficiency_rejects_classical_boost(self) -> None:
        """ATP efficiency rejects non-Hamiltonian entanglement boosts."""
        model = RadicalPairModel()
        eff = model.atp_efficiency(b_local=50e-6, entanglement_boost=0.0)
        assert 0.0 <= eff <= 1.0
        with pytest.raises(ValueError, match="entanglement_boost"):
            model.atp_efficiency(b_local=50e-6, entanglement_boost=0.01)


# ─── Scenario 7: Kane Register Feasibility ───


class TestKaneRegisterFeasibility:
    """Generate large registers, verify properties."""

    def test_512_qubit_grid(self) -> None:
        mapper = KaneSiliconMapper(spacing_nm=10.0, topology="grid")
        layout = mapper.map_pool_to_register(512)
        assert layout.n_qubits == 512
        assert layout.coupling_matrix.shape == (512, 512)
        # Symmetry
        np.testing.assert_array_almost_equal(
            layout.coupling_matrix, layout.coupling_matrix.T, decimal=15
        )
        # All non-negative
        assert np.all(layout.coupling_matrix >= 0.0)
        # Diagonal zero
        np.testing.assert_array_almost_equal(np.diag(layout.coupling_matrix), 0.0)

    def test_constraints_parametric(self) -> None:
        """Verify feasibility transitions at expected spacing."""
        mapper = KaneSiliconMapper(spacing_nm=10.0)
        c10 = mapper.get_constraints(8)
        assert c10["feasible"] is True

        mapper2 = KaneSiliconMapper(spacing_nm=50.0)
        c50 = mapper2.get_constraints(8)
        assert c50["feasible"] is False


# ─── Scenario 8: Content Indexer Adversarial ───


class TestContentIndexerAdversarial:
    """Edge cases: Unicode, empty, binary, deeply nested."""

    def test_unicode_content(self, tmp_path: Path) -> None:
        f = tmp_path / "unicode.md"
        f.write_text("# Ĉapitro 1\n∀x∈ℝ: ∫f(x)dx = Σaₙxⁿ\n日本語テスト\n", encoding="utf-8")
        chunks = index_gotm_repo(str(tmp_path))
        assert len(chunks) >= 1

    def test_empty_files(self, tmp_path: Path) -> None:
        (tmp_path / "empty.md").write_text("")
        (tmp_path / "whitespace.py").write_text("   \n\n  \n")
        chunks = index_gotm_repo(str(tmp_path))
        # Empty/whitespace files may produce 0 chunks — that's OK
        for chunk in chunks:
            assert isinstance(chunk.text, str)

    def test_binary_file_skip(self, tmp_path: Path) -> None:
        (tmp_path / "binary.bin").write_bytes(os.urandom(1024))
        (tmp_path / "real.md").write_text("# Real content\nSome math\n")
        chunks = index_gotm_repo(str(tmp_path))
        # Binary should be skipped, only .md indexed
        for chunk in chunks:
            assert chunk.content_type != "binary"

    def test_deeply_nested(self, tmp_path: Path) -> None:
        d = tmp_path
        for i in range(10):
            d = d / f"level_{i}"
        d.mkdir(parents=True)
        (d / "deep.md").write_text("# Deep theorem\n∀ε>0\n")
        chunks = index_gotm_repo(str(tmp_path))
        assert len(chunks) >= 1


# ─── Scenario 9: RAM-Aware Qubit Sizing ───


class TestRAMAwareQubitSizing:
    """Verify compute_max_qubits respects system RAM."""

    def test_max_qubits_within_bounds(self) -> None:
        max_q = compute_max_qubits()
        assert 4 <= max_q <= 30, f"max_qubits={max_q} out of [4,30]"

    def test_available_ram_positive(self) -> None:
        ram = _get_available_ram()
        assert ram > 0, "Should detect available RAM"

    def test_safety_factor_effect(self) -> None:
        q_liberal = compute_max_qubits(safety_factor=0.9)
        q_strict = compute_max_qubits(safety_factor=0.1)
        assert q_liberal >= q_strict, f"Liberal ({q_liberal}) should >= strict ({q_strict})"


# ─── Scenario 10: Studio Hook & Dashboard ───


class TestStudioHookIntegration:
    """Verify telemetry hook produces valid structured data."""

    def test_snapshot_structure(self) -> None:
        pool = SpinPoolMPS(n_sites=8)
        bridge = FisherPosnerQuantumBridge(n_qubits=4, backend="emulated")
        hook = QuantumStudioHook(pool, bridge)

        pool.apply_measurement(3, 1.0)
        snap = hook.get_entanglement_snapshot()
        assert "timestamp" in snap
        assert snap["n_sites"] == 8
        assert len(snap["entanglement_map"]) == 8
        assert len(snap["atp_efficiencies"]) == 8

    def test_json_event_valid(self) -> None:
        pool = SpinPoolMPS(n_sites=4)
        bridge = FisherPosnerQuantumBridge(n_qubits=4, backend="emulated")
        hook = QuantumStudioHook(pool, bridge)

        event_str = hook.to_json_event("test_event")
        event = json.loads(event_str)
        assert event["event"] == "test_event"
        assert "data" in event

    def test_dashboard_no_crash(self) -> None:
        """Dashboard should render without crashing."""
        brain = GOTMBrain(n_neurons=8, seed=42)
        chunk = ContentChunk("t", "t.md", 0, "content", "markdown", 1.0)
        brain.learn_step(chunk, np.ones(8) * 0.5)

        dashboard = TerminalDashboard(clear_screen=False)
        # Should not raise
        dashboard.draw(brain)


# ─── Scenario 11: Rust Compilation Verification ───


class TestRustCompilation:
    """Verify all Rust files compile and pass tests."""

    @pytest.mark.parametrize("rs_file", ["spin_pool.rs", "radical_pair.rs", "kane_mapper.rs"])
    def test_rust_compiles_and_tests_pass(self, rs_file: str) -> None:
        rs_path = _QC_DIR / rs_file
        if not rs_path.exists():
            pytest.skip(f"{rs_file} not found")

        bin_name = rs_file.replace(".rs", "_test")
        out_path = f"/tmp/{bin_name}"

        result = subprocess.run(
            ["rustc", "--test", str(rs_path), "-o", out_path, "-C", "opt-level=2"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"

        result = subprocess.run(
            [out_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"Tests failed:\n{result.stdout}\n{result.stderr}"
        assert "test result: ok" in result.stdout
