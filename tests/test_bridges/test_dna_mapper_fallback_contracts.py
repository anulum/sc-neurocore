# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Fallback and bridge contracts for the DNA computing mapper."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest

import sc_neurocore.bridges.dna_mapper as dna_mapper
from sc_neurocore.bridges import dna_thermodynamics
from tests.module_reload import restore_module_namespace, snapshot_module_namespace


class _BlockingFinder(importlib.abc.MetaPathFinder):
    """Import hook that blocks one exact module name."""

    def __init__(self, blocked_name: str) -> None:
        self._blocked_name = blocked_name

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Raise ``ImportError`` for the configured module and ignore others."""
        if fullname == self._blocked_name:
            raise ImportError(f"blocked import for {fullname}")
        return None


def _save_modules(names: tuple[str, ...]) -> dict[str, ModuleType | None]:
    """Snapshot import-table entries before a reload mutation."""
    return {name: sys.modules.get(name) for name in names}


def _restore_modules(saved: dict[str, ModuleType | None]) -> None:
    """Restore import-table entries after a reload mutation."""
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_missing_rust_engine_import_installs_false_backend_probe() -> None:
    """Reload with the optional Rust DNA module absent and restore state."""
    saved = _save_modules(("sc_neurocore_engine.dna",))
    saved_namespace = snapshot_module_namespace(dna_mapper)
    finder = _BlockingFinder("sc_neurocore_engine.dna")
    sys.modules.pop("sc_neurocore_engine.dna", None)
    sys.meta_path.insert(0, finder)

    try:
        reloaded = importlib.reload(dna_mapper)
        probe = cast(Callable[[], bool], vars(reloaded)["has_full_dna_backend"])

        assert probe() is False
        assert vars(reloaded)["_HAS_RUST_DNA"] is False
    finally:
        sys.meta_path.remove(finder)
        _restore_modules(saved)
        restore_module_namespace(dna_mapper, saved_namespace)


def test_import_records_present_nupack_and_rust_probe_import_error() -> None:
    """Reload with a present NUPACK module and a failing Rust backend probe."""
    saved = _save_modules(("nupack", "sc_neurocore_engine", "sc_neurocore_engine.dna"))
    saved_namespace = snapshot_module_namespace(dna_mapper)
    fake_nupack = ModuleType("nupack")
    fake_engine = ModuleType("sc_neurocore_engine")
    fake_dna = ModuleType("sc_neurocore_engine.dna")

    def raising_backend_probe() -> bool:
        raise ImportError("backend probe unavailable")

    vars(fake_dna)["has_full_dna_backend"] = raising_backend_probe
    vars(fake_engine)["dna"] = fake_dna
    sys.modules["nupack"] = fake_nupack
    sys.modules["sc_neurocore_engine"] = fake_engine
    sys.modules["sc_neurocore_engine.dna"] = fake_dna

    try:
        reloaded = importlib.reload(dna_mapper)

        assert vars(reloaded)["_HAS_NUPACK"] is True
        assert vars(reloaded)["nupack"] is fake_nupack
        assert vars(reloaded)["_HAS_RUST_DNA"] is False
    finally:
        _restore_modules(saved)
        restore_module_namespace(dna_mapper, saved_namespace)


def test_bitstream_validate_delegates_to_thermodynamic_validator() -> None:
    """The high-level compiler exposes the NUPACK/fallback validation report."""
    compiler = dna_mapper.BitstreamToDNA(method="displacement", seed=42)
    design = compiler.compile_network(
        gates=[{"type": "AND", "inputs": ["a", "b"], "output": "out"}],
        input_names=["a", "b"],
        output_names=["out"],
        name="validate_surface",
    )

    report = compiler.validate(design)

    assert set(report) >= {"valid", "strand_results", "cross_hybridization", "warnings"}
    assert isinstance(report["strand_results"], dict)


def test_enzymatic_compiler_rejects_unsupported_gate_type() -> None:
    """The enzymatic backend fails closed on displacement-only gate requests."""
    compiler = dna_mapper.BitstreamToDNA(method="enzymatic", seed=42)

    with pytest.raises(ValueError, match="Unsupported enzymatic gate"):
        compiler.compile_network(
            gates=[{"type": "AND", "inputs": ["a", "b"], "output": "out"}],
            input_names=["a", "b"],
            output_names=["out"],
        )


def test_gf4_decode_preserves_trailing_partial_block() -> None:
    """Trailing sequences shorter than a full parity block remain data."""
    decoded, corrections = dna_mapper.GF4ErrorCorrection(n_parity=4, block_size=6).decode("ACGT")

    assert decoded == "ACGT"
    assert corrections == 0


def test_sc_network_bridge_emits_buffer_for_single_excitatory_source() -> None:
    """Single positive adjacency edges compile through the public buffer path."""
    adjacency = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    design = dna_mapper.SCNetworkBridge(seed=42).from_adjacency(
        adjacency,
        input_indices=[0],
        output_indices=[1],
        name="single_excitatory",
    )

    assert [gate.gate_type for gate in design.gates] == [dna_mapper.GateType.BUFFER]


def test_internal_nupack_provider_starts_fail_closed() -> None:
    """The thermodynamic module is safe before façade dependency injection."""
    assert dna_thermodynamics._fallback_nupack_backend() == (False, None)


def test_empty_topology_has_no_critical_path() -> None:
    """An empty molecular circuit has a stable zero-node topology report."""
    report = dna_mapper.TopologicalAnalyzer().analyze(dna_mapper.DNACircuitDesign())

    assert report == {
        "depth": 0,
        "fan_out": {},
        "has_feedback": False,
        "cycles": [],
        "topological_order": [],
        "critical_path": [],
        "n_nodes": 0,
    }


def test_dual_rail_fault_scan_ignores_unrelated_traces() -> None:
    """Non-rail metadata does not become a fault-detection signal."""
    result = {
        "time": np.array([0.0, 1.0]),
        "metadata": np.array([1.0, 1.0]),
    }

    assert dna_mapper.DualRailEncoder().check_faults(result) == []


def test_protocol_deduplicates_strands_by_name() -> None:
    """Repeated logical strand names produce one laboratory material row."""
    design = dna_mapper.DNACircuitDesign(
        input_strands=[
            dna_mapper.DNAStrand(name="duplicate", sequence="ACGT"),
            dna_mapper.DNAStrand(name="duplicate", sequence="TGCA"),
        ]
    )

    protocol = dna_mapper.generate_protocol(design)

    assert protocol.count("| duplicate |") == 1


def test_plate_layout_handles_empty_and_duplicate_sequences() -> None:
    """Plate planning omits empty and duplicate oligos without phantom plates."""
    layout = dna_mapper.PlateLayout()
    empty_result = layout.layout(dna_mapper.DNACircuitDesign())
    design = dna_mapper.DNACircuitDesign(
        input_strands=[
            dna_mapper.DNAStrand(name="empty", sequence=""),
            dna_mapper.DNAStrand(name="first", sequence="ACGT"),
            dna_mapper.DNAStrand(name="duplicate", sequence="ACGT"),
        ]
    )

    populated_result = layout.layout(design)

    assert empty_result["n_plates"] == 0
    assert empty_result["utilization_pct"] == 0.0
    assert populated_result["n_unique_oligos"] == 1


def _simulate_without_gate_outputs(
    _self: dna_mapper.KineticSimulator,
    _design: dna_mapper.DNACircuitDesign,
    _input_concentrations: dict[str, float],
    duration_s: float = 3600.0,
    dt: float = 1.0,
) -> dict[str, np.ndarray[Any, Any]]:
    """Return a deliberately incomplete simulator result for boundary tests."""
    return {"time": np.array([0.0, duration_s, dt])}


def _single_buffer_design() -> dna_mapper.DNACircuitDesign:
    """Build the smallest circuit with one expected output trace."""
    gate = dna_mapper.DNAGate(
        gate_id=0,
        gate_type=dna_mapper.GateType.BUFFER,
        input_names=["a"],
        output_name="out",
    )
    return dna_mapper.DNACircuitDesign(gates=[gate])


def test_noise_analysis_fails_closed_when_simulator_omits_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monte Carlo analysis rejects incomplete kinetic result surfaces."""
    monkeypatch.setattr(dna_mapper.KineticSimulator, "simulate", _simulate_without_gate_outputs)

    with pytest.raises(RuntimeError, match="omitted output trace: out"):
        dna_mapper.NoiseModel(n_trials=1).sensitivity_analysis(
            _single_buffer_design(),
            {"a": 200.0},
        )


def test_concentration_optimizer_fails_closed_when_expected_output_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concentration fitting rejects a truth table it cannot evaluate."""
    monkeypatch.setattr(dna_mapper.KineticSimulator, "simulate", _simulate_without_gate_outputs)

    with pytest.raises(RuntimeError, match="omitted expected output: out"):
        dna_mapper.ConcentrationOptimizer(n_evaluations=0).optimize(
            _single_buffer_design(),
            [{"inputs": {"a": 200.0}, "expected": {"out": "high"}}],
        )
