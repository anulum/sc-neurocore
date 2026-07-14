# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CMOS profile responsibility-boundary contracts

"""Contract and architecture tests for the CMOS profile registration facade."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sc_neurocore.compiler.platforms import _PROFILES, HardwareProfile, get_profile
from sc_neurocore.compiler.platforms import cmos_profiles

_PACKAGE_DIR = Path(cmos_profiles.__file__).parent
_CMOS_PROFILE_NAMES = (
    "spartan6",
    "artix7",
    "kintex7",
    "ultrascale",
    "ultrascale_plus",
    "versal",
    "cyclone_v",
    "cyclone_10",
    "arria10",
    "stratix10",
    "agilex",
    "ecp5",
    "crosslink_nx",
    "certuspro_nx",
    "gowin",
    "efinix",
    "polarfire",
    "smartfusion2",
    "achronix",
    "quicklogic",
    "ice40",
    "asic_16",
    "asic_32",
    "asic_custom",
    "sim_q88",
    "sim_q1616",
    "alveo",
    "nexus",
    "polarfire_soc",
    "avant",
    "tpu",
    "cerebras_wse",
    "graphcore_ipu",
    "tenstorrent",
    "ethos_u",
    "hexagon",
    "apple_ane",
    "sharc",
    "c6000",
    "ceva_xc",
    "photonic",
    "riscv_fpga",
    "in_memory",
    "quantum_hybrid",
    "nanoxplore",
    "rtg4",
    "kintex_us_rt",
    "hailo8",
    "kneron",
    "groq_tsp",
    "jetson",
    "habana_gaudi",
    "drp_ai",
    "speedcore",
    "eflx",
    "menta_efpga",
    "imx500",
    "samsung_npu",
    "samsung_cgra",
    "qualcomm_npu_cgra",
    "pact_xtensa",
    "tsmc_soic",
    "intel_foveros",
    "amd_3dv",
    "rp2040",
    "esp32_s3",
    "stm32h7",
    "nrf5340",
    "max78000",
    "sifive_x280",
    "qualcomm_ventana",
    "ainekko_rv",
)
_CMOS_PROFILE_SHA256 = "0d72e74b9a779dd3a98b70817a12fac902528e741cea3772a7dc6bf37be20fc9"
_MODULE_PROFILE_NAMES = {
    "_cmos_fpga_profiles": (
        "spartan6",
        "artix7",
        "kintex7",
        "ultrascale",
        "ultrascale_plus",
        "versal",
        "cyclone_v",
        "cyclone_10",
        "arria10",
        "stratix10",
        "agilex",
        "ecp5",
        "crosslink_nx",
        "certuspro_nx",
        "gowin",
        "efinix",
        "polarfire",
        "smartfusion2",
        "achronix",
        "quicklogic",
        "ice40",
        "alveo",
        "nexus",
        "polarfire_soc",
        "avant",
        "nanoxplore",
        "rtg4",
        "kintex_us_rt",
        "speedcore",
        "eflx",
        "menta_efpga",
    ),
    "_cmos_reference_profiles": (
        "asic_16",
        "asic_32",
        "asic_custom",
        "sim_q88",
        "sim_q1616",
    ),
    "_cmos_accelerator_profiles": (
        "tpu",
        "cerebras_wse",
        "graphcore_ipu",
        "tenstorrent",
        "ethos_u",
        "hexagon",
        "apple_ane",
        "hailo8",
        "kneron",
        "groq_tsp",
        "jetson",
        "habana_gaudi",
        "drp_ai",
        "imx500",
        "samsung_npu",
        "sifive_x280",
        "qualcomm_ventana",
        "ainekko_rv",
    ),
    "_cmos_processor_profiles": (
        "sharc",
        "c6000",
        "ceva_xc",
        "rp2040",
        "esp32_s3",
        "stm32h7",
        "nrf5340",
        "max78000",
    ),
    "_cmos_architecture_profiles": (
        "photonic",
        "riscv_fpga",
        "in_memory",
        "quantum_hybrid",
        "samsung_cgra",
        "qualcomm_npu_cgra",
        "pact_xtensa",
        "tsmc_soic",
        "intel_foveros",
        "amd_3dv",
    ),
}
_FACADE_REGISTRARS = (
    "_register_xilinx_fpga_profiles",
    "_register_intel_fpga_profiles",
    "_register_lattice_fpga_profiles",
    "_register_other_fpga_profiles",
    "_register_ice40_fpga_profiles",
    "_register_asic_profiles",
    "_register_simulation_profiles",
    "_register_additional_fpga_profiles",
    "_register_ai_accelerator_profiles",
    "_register_dsp_profiles",
    "_register_emerging_compute_profiles",
    "_register_radiation_hardened_fpga_profiles",
    "_register_edge_ai_accelerator_profiles",
    "_register_embedded_fpga_profiles",
    "_register_vision_sensor_profiles",
    "_register_cgra_profiles",
    "_register_stacked_3d_profiles",
    "_register_edge_mcu_profiles",
    "_register_riscv_ai_accelerator_profiles",
)
_MODULE_LINE_CEILINGS = {
    "cmos_profiles": 120,
    "_cmos_fpga_profiles": 650,
    "_cmos_accelerator_profiles": 400,
    "_cmos_architecture_profiles": 250,
    "_cmos_processor_profiles": 200,
    "_cmos_reference_profiles": 150,
}


def _module_tree(module_name: str) -> ast.Module:
    """Parse a platform-profile module into an abstract syntax tree."""
    source = (_PACKAGE_DIR / f"{module_name}.py").read_text(encoding="utf-8")
    return ast.parse(source)


def _declared_profile_names(module_name: str) -> tuple[str, ...]:
    """Return profile-name literals in source order for one module."""
    tree = _module_tree(module_name)
    profile_calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "HardwareProfile"
        ),
        key=lambda node: node.lineno,
    )
    names: list[str] = []
    for call in profile_calls:
        name_keyword = next(keyword for keyword in call.keywords if keyword.arg == "name")
        assert isinstance(name_keyword.value, ast.Constant)
        assert isinstance(name_keyword.value.value, str)
        names.append(name_keyword.value.value)
    return tuple(names)


def test_cmos_registry_values_and_historical_order_are_exact() -> None:
    """All 72 profile values and their registry insertion order remain exact."""
    expected_names = set(_CMOS_PROFILE_NAMES)
    registry_projection = tuple(name for name in _PROFILES if name in expected_names)
    payload = [asdict(get_profile(name)) for name in _CMOS_PROFILE_NAMES]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert registry_projection == _CMOS_PROFILE_NAMES
    assert hashlib.sha256(encoded).hexdigest() == _CMOS_PROFILE_SHA256


def test_responsibility_modules_own_exact_profile_families() -> None:
    """Each private module owns only its declared hardware responsibility."""
    declared = {
        module_name: _declared_profile_names(module_name) for module_name in _MODULE_PROFILE_NAMES
    }

    assert declared == _MODULE_PROFILE_NAMES
    assert sum(len(names) for names in declared.values()) == len(_CMOS_PROFILE_NAMES)
    assert set().union(*(set(names) for names in declared.values())) == set(_CMOS_PROFILE_NAMES)


def test_responsibility_modules_depend_only_on_registry() -> None:
    """Private profile partitions do not form lateral dependency edges."""
    for module_name in _MODULE_PROFILE_NAMES:
        imports = {
            (node.level, node.module)
            for node in ast.walk(_module_tree(module_name))
            if isinstance(node, ast.ImportFrom) and node.module != "__future__"
        }
        assert imports == {(1, "registry")}


def test_facade_composes_registrars_in_historical_order() -> None:
    """The facade pins the original registration sequence explicitly."""
    tree = _module_tree("cmos_profiles")
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PROFILE_REGISTRARS"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Tuple)
    registrar_names = tuple(
        element.id for element in assignment.value.elts if isinstance(element, ast.Name)
    )

    assert registrar_names == _FACADE_REGISTRARS


def test_facade_preserves_public_symbol_and_duplicate_guard() -> None:
    """The facade keeps its public class alias and rejects repeat registration."""
    public_names = {name for name in vars(cmos_profiles) if not name.startswith("_")}

    assert public_names == {"HardwareProfile", "annotations"}
    assert cmos_profiles.HardwareProfile is HardwareProfile
    with pytest.raises(
        ValueError,
        match="Duplicate hardware-profile registration for 'spartan6'",
    ):
        importlib.reload(cmos_profiles)


def test_private_module_reloads_have_no_registration_side_effects() -> None:
    """Reloading a private definition module leaves the live registry unchanged."""
    before = dict(_PROFILES)
    for module_name in _MODULE_PROFILE_NAMES:
        module = importlib.import_module(f"sc_neurocore.compiler.platforms.{module_name}")
        importlib.reload(module)

    assert before == _PROFILES


def test_profile_modules_remain_below_responsibility_ceilings() -> None:
    """The facade and private partitions remain bounded below GodFile size."""
    for module_name, ceiling in _MODULE_LINE_CEILINGS.items():
        line_count = len(
            (_PACKAGE_DIR / f"{module_name}.py").read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= ceiling, f"{module_name}.py has {line_count} lines"
