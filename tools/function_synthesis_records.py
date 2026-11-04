# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Function synthesis cases and measured resource records

"""Describe function measurements and read the actual mapped Yosys statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

from sc_neurocore.compiler import (
    function_unit_functions,
    function_unit_programs,
    function_units,
    lut_accuracy,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88
from tools.function_synthesis_json import decode_native_json

MATRIX_FORMATS: tuple[tuple[int, int], ...] = (
    (16, 8),
    (16, 12),
    (32, 16),
    (18, 9),
    (32, 24),
    (16, 15),
)


@dataclass(frozen=True, slots=True)
class SynthesisCase:
    """Identify one canonical function, lowering and signed fixed-point format.

    Parameters
    ----------
    function : str
        Canonical function name; aliases are not separate measurements.
    data_width, fraction : int
        Word width and fractional bits, using existing function-unit admission.
    lowering : str, optional
        'v2' or 'lut-v1'. Unsupported legacy cases remain identifiable
        so that a complete report cannot silently omit them.

    """

    function: str
    data_width: int
    fraction: int
    lowering: str = "v2"

    def __post_init__(self) -> None:
        """Validate the existing numerical format and canonical function vocabulary."""
        Q88(
            data_width=self.data_width,
            fraction=self.fraction,
            function_lowering=self.lowering,
        )
        function_units.check_format(self.data_width, self.fraction)
        if self.function not in function_unit_functions():
            raise ValueError(f"Unknown canonical function: {self.function!r}")

    @property
    def key(self) -> str:
        """Return the lowering/function/Q-format identity used in measurement records."""
        return f"{self.lowering}/{self.function}/Q{self.data_width - self.fraction}.{self.fraction}"

    @property
    def supported(self) -> bool:
        """Return whether the selected lowering implements this function."""
        return self.function in function_unit_functions(self.lowering)

    @property
    def input_bits(self) -> int:
        """Return the combined width of the independently driven input ports."""
        return function_units.FUNCTION_ARITY[self.function] * self.data_width


def synthesis_cases() -> tuple[SynthesisCase, ...]:
    """Return the full comparison matrix, including unsupported legacy cases.

    Returns
    -------
    tuple of SynthesisCase
        Both lowerings for every canonical function at all six report formats.
        Membership does not establish that a measurement or proof has run.

    """
    return tuple(
        SynthesisCase(function, width, fraction, lowering)
        for function in function_unit_functions()
        for lowering in ("v2", "lut-v1")
        for width, fraction in MATRIX_FORMATS
    )


def declared_table_bits(case: SynthesisCase) -> int:
    """Count table bits declared by the standalone RTL before synthesis.

    Parameters
    ----------
    case : SynthesisCase
        A supported function measurement.

    Returns
    -------
    int
        Legacy table entries times word width, or 64 bits per emitted
        coefficient entry for every v2 table lookup. Repeated lookup tables
        are counted as emitted; unused tables in the program are excluded.
        This is source storage, not inferred BRAM capacity or mapped cell area.

    Raises
    ------
    ValueError
        If the selected legacy function has no implementation.

    """
    if not case.supported:
        raise ValueError(f"No standalone implementation for {case.key}")
    if case.lowering == "lut-v1":
        geometry = lut_accuracy.GEOMETRY.get(case.function)
        return 0 if geometry is None else geometry.size * case.data_width
    program = function_unit_programs.program(case.function, case.data_width, case.fraction)
    return sum(len(program.tables[step.table]) * 64 for step in program.steps if step.op == "table")


@dataclass(frozen=True, slots=True)
class SynthesisResources:
    """Hold statistics from the actual mapped top module.

    Parameters
    ----------
    total_cells : int
        Total mapped cells and primitive module instances in the standalone
        module; a direct wire may use zero.
    cell_types : tuple of (str, int)
        Sorted counts by the actual Yosys cell type.
    remaining_memory_bits : int
        Bits still represented as Yosys memories after mapping. Primitive
        RAM capacity is not inferred from this count.

    """

    total_cells: int
    cell_types: tuple[tuple[str, int], ...]
    remaining_memory_bits: int

    @property
    def dsp_blocks(self) -> int:
        """Return the actual DSP48E1 primitive count for the xc7 mapping target."""
        return dict(self.cell_types).get("DSP48E1", 0)

    def to_dict(self) -> dict[str, object]:
        """Return measured counts without converting missing measurements to zero.

        Returns
        -------
        dict of str to object
            Mapped cell counts and remaining Yosys memory bits. Logical source
            table bits are recorded separately by the caller.

        """
        return {
            "total_cells": self.total_cells,
            "cell_types": dict(self.cell_types),
            "dsp_blocks": self.dsp_blocks,
            "remaining_memory_bits": self.remaining_memory_bits,
        }


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _object(value: object, name: str) -> dict[str, object]:
    if not _is_object(value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _natural(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def read_synthesis_resources(path: Path) -> SynthesisResources:
    """Read checked resource counts from an actual Yosys stat -json file.

    Parameters
    ----------
    path : pathlib.Path
        Statistics for a mapped function_unit top module.

    Returns
    -------
    SynthesisResources
        Non-negative counts whose cell-type sum matches the complete reported
        total, including separately reported primitive module instances.

    Raises
    ------
    OSError
        If the report cannot be read.
    ValueError
        For invalid JSON, duplicate keys, non-finite values, missing top
        module, unresolved processes or inconsistent resource counts.

    """
    payload = decode_native_json(path.read_text(encoding="utf-8"))
    report = _object(payload, "Yosys statistics")
    modules = _object(report.get("modules"), "modules")
    module = _object(modules.get("\\function_unit"), "function_unit")
    processes = _natural(module.get("num_processes"), "num_processes")
    if processes:
        raise ValueError("Mapped function_unit still contains processes")
    total = _natural(module.get("num_cells"), "num_cells")
    total += _natural(module.get("num_submodules", 0), "num_submodules")
    counts = _object(module.get("num_cells_by_type"), "num_cells_by_type")
    cells = tuple(sorted((name, _natural(value, name)) for name, value in counts.items()))
    if sum(count for _, count in cells) != total:
        raise ValueError("Cell-type counts do not sum to num_cells")
    memory = _natural(module.get("num_memory_bits"), "num_memory_bits")
    return SynthesisResources(total, cells, memory)
