# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded estimator metrics protocol contract

"""Bind the folded area estimator to its structural metrics protocol.

The estimator reads seven architectural counts. It names them in
:class:`~sc_neurocore.energy.folded_estimator.FoldedMetrics` instead of importing the
compiler's ``FoldedResourceMetrics``, so the energy package needs no bridge module. These
tests keep the protocol, the estimator and the compiler's dataclass in step.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

from sc_neurocore.energy import estimate_folded_area
from sc_neurocore.energy.folded_estimator import FoldedMetrics
from sc_neurocore.nir_bridge.fpga_compilation_result import FoldedResourceMetrics

MODULE = (
    Path(__file__).resolve().parents[1] / "src" / "sc_neurocore" / "energy" / "folded_estimator.py"
)


def _protocol_members() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == FoldedMetrics.__name__
    )
    return {node.name for node in protocol.body if isinstance(node, ast.FunctionDef)}


def _attributes_read_from_metrics() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "metrics"
    }


def test_the_protocol_names_exactly_what_the_estimator_reads() -> None:
    """Every count the estimator reads is in the protocol, and the protocol holds no other."""
    assert _protocol_members() == _attributes_read_from_metrics()


def test_the_compiler_dataclass_provides_every_protocol_member_as_an_int() -> None:
    """``FoldedResourceMetrics`` has an ``int`` field for each protocol member."""
    fields = {field.name: field.type for field in dataclasses.fields(FoldedResourceMetrics)}
    for member in _protocol_members():
        assert fields.get(member) in (int, "int"), member


def test_the_estimator_module_imports_nothing_outside_the_energy_package() -> None:
    """No absolute ``sc_neurocore`` import, not even under ``TYPE_CHECKING``."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level > 0 or not (node.module or "").startswith("sc_neurocore"), node.module
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("sc_neurocore") for alias in node.names)


def test_an_fpga_compile_summary_is_accepted_as_folded_metrics() -> None:
    """A real ``FoldedResourceMetrics`` passes as ``FoldedMetrics`` and is estimated."""
    summary = FoldedResourceMetrics(
        neurons=8,
        state_vars_per_neuron=2,
        pe_instances=1,
        shared_multipliers=2,
        state_ram_bits=256,
        cycles_per_tick=9,
        direct_neuron_instances=8,
        populations=1,
        param_rom_bits=0,
    )
    metrics: FoldedMetrics = summary
    estimate = estimate_folded_area(metrics, target="ecp5")
    assert estimate.bram_bits == 256
    assert estimate.latency_cycles == 9
