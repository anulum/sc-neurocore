# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — constants ledger regression tests

"""Regression tests for the named physical and model constants ledger."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from sc_neurocore import constants


REPO_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_ROOT: Final = REPO_ROOT / "src" / "sc_neurocore"
CONSTANTS_SOURCE: Final = SOURCE_ROOT / "constants.py"
EXCLUDED_SOURCE_DIRS: Final = frozenset(
    {
        ".git",
        ".hg",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".nox",
        ".pixi",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "site",
        "venv",
    }
)

EXPECTED_CONSTANT_VALUES: Final[dict[str, int | float]] = {
    "LIF_V_REST": 0.0,
    "LIF_V_RESET": 0.0,
    "LIF_V_THRESHOLD": 1.0,
    "LIF_TAU_MEM": 20.0,
    "LIF_RESISTANCE": 1.0,
    "LIF_DT": 1.0,
    "LIF_NOISE_STD": 0.0,
    "LIF_REFRACTORY_PERIOD": 0,
    "LIF_LAYER_NOISE_STD": 0.02,
    "IZH_A": 0.02,
    "IZH_B": 0.2,
    "IZH_C": -65.0,
    "IZH_D": 8.0,
    "IZH_SPIKE_THRESHOLD": 30.0,
    "HOMEOSTATIC_TARGET_RATE": 0.1,
    "HOMEOSTATIC_ADAPTATION_RATE": 0.01,
    "HOMEOSTATIC_TRACE_DECAY": 0.95,
    "HOMEOSTATIC_THRESHOLD_FLOOR": 0.1,
    "HOMEOSTATIC_THRESHOLD_CEILING_MULT": 10.0,
    "DENDRITIC_THRESHOLD": 0.5,
    "FP_DATA_WIDTH": 16,
    "FP_FRACTION": 8,
    "FP_V_THRESHOLD": 256,
    "FP_REFRACTORY_PERIOD": 2,
    "FP_LFSR_WIDTH": 16,
    "FP_LFSR_SEED": 0xACE1,
    "STDP_LEARNING_RATE": 0.01,
    "STDP_WINDOW_SIZE": 5,
    "STDP_LTD_RATIO": 0.5,
    "SYNAPSE_DEFAULT_LENGTH": 256,
    "SYNAPSE_DEFAULT_WEIGHT": 0.5,
    "RSTDP_TRACE_DECAY": 0.9,
    "RSTDP_ANTI_HEBBIAN_SCALE": 0.5,
    "LAYER_DEFAULT_LENGTH": 1024,
    "LAYER_CONV_LENGTH": 256,
    "DENSE_LAYER_LENGTH": 2048,
    "DENSE_Y_MIN": 0.0,
    "DENSE_Y_MAX": 0.1,
    "RESERVOIR_FEEDBACK_STRENGTH": 0.5,
    "RESERVOIR_INPUT_STRENGTH": 0.5,
    "RESERVOIR_SPECTRAL_RADIUS": 0.9,
    "MEMRISTIVE_STUCK_RATE": 0.01,
    "MEMRISTIVE_VARIABILITY": 0.05,
    "NEURON_SEED_OFFSET": 10_000,
}

EXPECTED_IMPORT_MAP: Final[dict[str, tuple[str, ...]]] = {
    "src/sc_neurocore/layers/fusion.py": ("LAYER_DEFAULT_LENGTH",),
    "src/sc_neurocore/layers/jax_dense_layer.py": (
        "LAYER_DEFAULT_LENGTH",
        "LIF_DT",
        "LIF_LAYER_NOISE_STD",
        "LIF_RESISTANCE",
        "LIF_TAU_MEM",
        "LIF_V_RESET",
        "LIF_V_REST",
        "LIF_V_THRESHOLD",
    ),
    "src/sc_neurocore/layers/memristive.py": (
        "MEMRISTIVE_STUCK_RATE",
        "MEMRISTIVE_VARIABILITY",
    ),
    "src/sc_neurocore/layers/recurrent.py": (
        "LAYER_DEFAULT_LENGTH",
        "RESERVOIR_FEEDBACK_STRENGTH",
        "RESERVOIR_INPUT_STRENGTH",
        "RESERVOIR_SPECTRAL_RADIUS",
    ),
    "src/sc_neurocore/layers/sc_conv_layer.py": ("LAYER_CONV_LENGTH",),
    "src/sc_neurocore/layers/sc_dense_layer.py": (
        "DENSE_LAYER_LENGTH",
        "DENSE_Y_MAX",
        "DENSE_Y_MIN",
        "LIF_DT",
        "LIF_LAYER_NOISE_STD",
        "LIF_RESISTANCE",
        "LIF_TAU_MEM",
        "LIF_V_RESET",
        "LIF_V_REST",
        "LIF_V_THRESHOLD",
        "NEURON_SEED_OFFSET",
    ),
    "src/sc_neurocore/layers/sc_learning_layer.py": (
        "LAYER_DEFAULT_LENGTH",
        "STDP_LEARNING_RATE",
        "STDP_LTD_RATIO",
    ),
    "src/sc_neurocore/layers/vectorized_layer.py": ("LAYER_DEFAULT_LENGTH",),
    "src/sc_neurocore/neurons/dendritic.py": ("DENDRITIC_THRESHOLD",),
    "src/sc_neurocore/neurons/fixed_point_lif.py": (
        "FP_DATA_WIDTH",
        "FP_FRACTION",
        "FP_LFSR_SEED",
        "FP_LFSR_WIDTH",
        "FP_REFRACTORY_PERIOD",
        "FP_V_THRESHOLD",
    ),
    "src/sc_neurocore/neurons/homeostatic_lif.py": (
        "HOMEOSTATIC_ADAPTATION_RATE",
        "HOMEOSTATIC_TARGET_RATE",
        "HOMEOSTATIC_THRESHOLD_CEILING_MULT",
        "HOMEOSTATIC_THRESHOLD_FLOOR",
        "HOMEOSTATIC_TRACE_DECAY",
    ),
    "src/sc_neurocore/neurons/sc_izhikevich.py": (
        "IZH_A",
        "IZH_B",
        "IZH_C",
        "IZH_D",
        "IZH_SPIKE_THRESHOLD",
        "LIF_DT",
    ),
    "src/sc_neurocore/neurons/stochastic_lif.py": (
        "LIF_DT",
        "LIF_NOISE_STD",
        "LIF_REFRACTORY_PERIOD",
        "LIF_RESISTANCE",
        "LIF_TAU_MEM",
        "LIF_V_RESET",
        "LIF_V_REST",
        "LIF_V_THRESHOLD",
    ),
    "src/sc_neurocore/synapses/r_stdp.py": (
        "RSTDP_ANTI_HEBBIAN_SCALE",
        "RSTDP_TRACE_DECAY",
    ),
    "src/sc_neurocore/synapses/sc_synapse.py": (
        "SYNAPSE_DEFAULT_LENGTH",
        "SYNAPSE_DEFAULT_WEIGHT",
    ),
    "src/sc_neurocore/synapses/stochastic_stdp.py": (
        "STDP_LEARNING_RATE",
        "STDP_LTD_RATIO",
        "STDP_WINDOW_SIZE",
    ),
}


def _public_constant_values() -> dict[str, int | float]:
    public_values: dict[str, int | float] = {}
    for name, value in vars(constants).items():
        if not name.isupper():
            continue
        assert type(value) in {int, float}, f"{name} has non-scalar value {value!r}"
        public_values[name] = value
    return public_values


def _has_excluded_parent(path: Path) -> bool:
    relative_parts = path.relative_to(SOURCE_ROOT).parts[:-1]
    return any(part in EXCLUDED_SOURCE_DIRS for part in relative_parts)


def _constants_imports_by_file() -> dict[str, tuple[str, ...]]:
    imports_by_file: dict[str, tuple[str, ...]] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if path.name == "constants.py" or _has_excluded_parent(path):
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if module == "sc_neurocore.constants" or (node.level > 0 and module == "constants"):
                imported.extend(alias.name for alias in node.names)

        if imported:
            imports_by_file[str(path.relative_to(REPO_ROOT))] = tuple(sorted(imported))
    return imports_by_file


def test_public_constant_ledger_matches_documented_values() -> None:
    """All public constants stay in the audited 44-value ledger."""
    assert _public_constant_values() == EXPECTED_CONSTANT_VALUES


def test_dimensionless_constants_remain_in_physical_ranges() -> None:
    """Normalised probabilities, rates, and hardware fields stay bounded."""
    bounded_unit_values = (
        constants.LIF_V_REST,
        constants.LIF_V_RESET,
        constants.LIF_V_THRESHOLD,
        constants.LIF_NOISE_STD,
        constants.LIF_LAYER_NOISE_STD,
        constants.HOMEOSTATIC_TARGET_RATE,
        constants.HOMEOSTATIC_ADAPTATION_RATE,
        constants.HOMEOSTATIC_TRACE_DECAY,
        constants.HOMEOSTATIC_THRESHOLD_FLOOR,
        constants.DENDRITIC_THRESHOLD,
        constants.STDP_LEARNING_RATE,
        constants.STDP_LTD_RATIO,
        constants.SYNAPSE_DEFAULT_WEIGHT,
        constants.RSTDP_TRACE_DECAY,
        constants.RSTDP_ANTI_HEBBIAN_SCALE,
        constants.DENSE_Y_MIN,
        constants.DENSE_Y_MAX,
        constants.RESERVOIR_FEEDBACK_STRENGTH,
        constants.RESERVOIR_INPUT_STRENGTH,
        constants.RESERVOIR_SPECTRAL_RADIUS,
        constants.MEMRISTIVE_STUCK_RATE,
        constants.MEMRISTIVE_VARIABILITY,
    )
    assert all(0.0 <= value <= 1.0 for value in bounded_unit_values)
    assert constants.LIF_TAU_MEM > 0.0
    assert constants.LIF_DT > 0.0
    assert constants.LIF_RESISTANCE > 0.0
    assert constants.HOMEOSTATIC_THRESHOLD_CEILING_MULT > 1.0
    assert constants.NEURON_SEED_OFFSET > 0


def test_fixed_point_q88_constants_are_self_consistent() -> None:
    """Q8.8 constants remain aligned with the fixed-point neuron model."""
    assert constants.FP_DATA_WIDTH == 16
    assert constants.FP_FRACTION == 8
    assert constants.FP_V_THRESHOLD == 1 << constants.FP_FRACTION
    assert 0 < constants.FP_LFSR_SEED < (1 << constants.FP_LFSR_WIDTH)
    assert constants.FP_REFRACTORY_PERIOD >= 0


def test_source_import_map_matches_current_adoption_boundary() -> None:
    """Maintained source imports from the constants ledger stay auditable."""
    assert _constants_imports_by_file() == EXPECTED_IMPORT_MAP


def test_module_docstring_states_current_adoption_boundary() -> None:
    """The constants module docstring must not overstate global adoption."""
    module_docstring = constants.__doc__
    assert module_docstring is not None
    assert "Modules import from here instead of using bare numeric literals" not in module_docstring
    assert "16 maintained source modules" in module_docstring
    assert "all 44 named constants" in module_docstring
    assert "remaining names are reserved vocabulary" not in module_docstring


def test_izhikevich_threshold_comment_identifies_spike_peak() -> None:
    """The Izhikevich threshold comment distinguishes detection from reset."""
    source_text = CONSTANTS_SOURCE.read_text(encoding="utf-8")
    assert "spike-detect peak" in source_text
    assert "not an adaptive membrane threshold" in source_text
