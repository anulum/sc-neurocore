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
from typing import cast

import numpy as np
import pytest

import sc_neurocore.bridges.dna_mapper as dna_mapper


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
        importlib.reload(dna_mapper)


def test_import_records_present_nupack_and_rust_probe_import_error() -> None:
    """Reload with a present NUPACK module and a failing Rust backend probe."""
    saved = _save_modules(("nupack", "sc_neurocore_engine", "sc_neurocore_engine.dna"))
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
        importlib.reload(dna_mapper)


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
