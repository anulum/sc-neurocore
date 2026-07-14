# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — neuromorphic profile responsibility-boundary contracts

"""Contract tests for the neuromorphic hardware-profile registration facade."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from sc_neurocore.compiler.platforms import (
    _PROFILES,
    HardwareProfile,
    get_profile,
    list_profiles,
)
from sc_neurocore.compiler.platforms import neuromorphic_profiles

_PACKAGE_DIR = Path(neuromorphic_profiles.__file__).parent
_PROFILE_NAMES = (
    "loihi2",
    "truenorth",
    "akida",
    "dynap_se2",
    "xylo",
    "loihi3",
    "northpole",
    "innatera_pulsar",
    "versal_ai_edge",
    "proasic3",
    "trion",
    "titanium",
    "gowin_arora_v",
    "intel_agilex5",
    "nvidia_dla",
    "mediatek_apu",
    "aws_inferentia",
    "qualcomm_nsp",
    "sambanova",
    "cambricon_mlu",
    "superconducting",
    "cim_sram",
    "analog_ai",
    "event_camera",
    "lightmatter_passage",
    "lightelligence_pace",
    "xanadu_x8",
    "ipronics_smartlight",
    "luminous_computing",
    "tenstorrent_blackhole",
    "cerebras_wse3",
    "intel_ponte_vecchio",
    "amd_mi300x",
    "ucie_generic",
    "upmem_pim",
    "samsung_hbm_pim",
    "sk_hynix_aim",
    "cxl_type3",
    "axdimm",
    "akida2",
    "spinnaker2",
    "dynapse2",
    "rain_neuromorphic",
    "brainscales2",
    "bae_rad750",
    "cobham_ut700",
    "mpfs250t_rt",
    "versal_xqrvc1902",
    "trenz_zynq_space",
    "mythic_m1076",
    "mobileye_eyeq6",
    "horizon_j6",
    "ambarella_cv72s",
    "hailo15",
    "syntiant_ndp120",
    "everspin_stt_mram",
    "samsung_sot_mram",
    "gf_fefet",
    "sk_hynix_feram",
    "aspinity_aml100",
    "renesas_analog_ai",
    "weebit_reram",
    "crossbar_rram",
    "adesto_cbram",
    "tsmc_cim_n7",
    "samsung_cim_sf3",
    "intel_horse_ridge",
    "google_cryo_ctrl",
    "microsoft_dna_store",
    "asu_dna_perovskite",
)
_PROFILE_SHA256 = "c21e2075bb98787db31afbaa0c41cb6300e6707663de49b6630d6b5dbb3d1d41"
_MODULE_PROFILE_NAMES = {
    "_event_driven_hardware_profiles": (
        "loihi2",
        "truenorth",
        "akida",
        "dynap_se2",
        "xylo",
        "loihi3",
        "northpole",
        "innatera_pulsar",
        "akida2",
        "spinnaker2",
        "dynapse2",
        "rain_neuromorphic",
        "brainscales2",
    ),
    "_programmable_aerospace_profiles": (
        "versal_ai_edge",
        "proasic3",
        "trion",
        "titanium",
        "gowin_arora_v",
        "intel_agilex5",
        "bae_rad750",
        "cobham_ut700",
        "mpfs250t_rt",
        "versal_xqrvc1902",
        "trenz_zynq_space",
    ),
    "_heterogeneous_accelerator_profiles": (
        "nvidia_dla",
        "mediatek_apu",
        "aws_inferentia",
        "qualcomm_nsp",
        "sambanova",
        "cambricon_mlu",
        "tenstorrent_blackhole",
        "cerebras_wse3",
        "intel_ponte_vecchio",
        "amd_mi300x",
        "ucie_generic",
        "mythic_m1076",
        "mobileye_eyeq6",
        "horizon_j6",
        "ambarella_cv72s",
        "hailo15",
        "syntiant_ndp120",
    ),
    "_memory_compute_profiles": (
        "upmem_pim",
        "samsung_hbm_pim",
        "sk_hynix_aim",
        "cxl_type3",
        "axdimm",
        "everspin_stt_mram",
        "samsung_sot_mram",
        "gf_fefet",
        "sk_hynix_feram",
        "weebit_reram",
        "crossbar_rram",
        "adesto_cbram",
        "tsmc_cim_n7",
        "samsung_cim_sf3",
    ),
    "_physical_compute_profiles": (
        "superconducting",
        "cim_sram",
        "analog_ai",
        "event_camera",
        "lightmatter_passage",
        "lightelligence_pace",
        "xanadu_x8",
        "ipronics_smartlight",
        "luminous_computing",
        "aspinity_aml100",
        "renesas_analog_ai",
        "intel_horse_ridge",
        "google_cryo_ctrl",
        "microsoft_dna_store",
        "asu_dna_perovskite",
    ),
}
_FACADE_REGISTRARS = (
    "_register_neuromorphic_chip_profiles",
    "_register_recent_neuromorphic_profiles",
    "_register_additional_fpga_profiles",
    "_register_ai_accelerator_profiles",
    "_register_emerging_compute_profiles",
    "_register_photonic_compute_profiles",
    "_register_chiplet_accelerator_profiles",
    "_register_processing_in_memory_profiles",
    "_register_additional_neuromorphic_profiles",
    "_register_aerospace_profiles",
    "_register_automotive_edge_profiles",
    "_register_spintronic_profiles",
    "_register_ferroelectric_profiles",
    "_register_analog_mixed_signal_profiles",
    "_register_rram_profiles",
    "_register_sram_cim_profiles",
    "_register_cryogenic_cmos_profiles",
    "_register_molecular_profiles",
)
_PLATFORM_CLASS_COUNTS = {
    "accelerator": 16,
    "analog_mixed": 2,
    "cryo_cmos": 2,
    "dna_molecular": 2,
    "emerging": 4,
    "ferroelectric": 2,
    "fpga": 11,
    "in_memory": 5,
    "neuromorphic": 14,
    "photonic": 5,
    "rram": 3,
    "spintronic": 2,
    "sram_cim": 2,
}
_MODULE_LINE_CEILINGS = {
    "neuromorphic_profiles": 100,
    "_event_driven_hardware_profiles": 250,
    "_programmable_aerospace_profiles": 250,
    "_heterogeneous_accelerator_profiles": 325,
    "_memory_compute_profiles": 275,
    "_physical_compute_profiles": 300,
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


def test_registry_values_and_historical_order_are_exact() -> None:
    """All 70 profile values and their registry insertion order remain exact."""
    expected_names = set(_PROFILE_NAMES)
    registry_projection = tuple(name for name in _PROFILES if name in expected_names)
    payload = [asdict(get_profile(name)) for name in _PROFILE_NAMES]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()

    assert registry_projection == _PROFILE_NAMES
    assert hashlib.sha256(encoded).hexdigest() == _PROFILE_SHA256


def test_responsibility_modules_own_exact_profile_families() -> None:
    """Each private module owns only its declared hardware responsibility."""
    declared = {
        module_name: _declared_profile_names(module_name) for module_name in _MODULE_PROFILE_NAMES
    }

    assert declared == _MODULE_PROFILE_NAMES
    assert sum(len(names) for names in declared.values()) == len(_PROFILE_NAMES)
    assert set().union(*(set(names) for names in declared.values())) == set(_PROFILE_NAMES)


def test_responsibility_modules_depend_only_on_registry() -> None:
    """Private profile partitions do not form lateral dependency edges."""
    for module_name in _MODULE_PROFILE_NAMES:
        imports = {
            (node.level, node.module)
            for node in ast.walk(_module_tree(module_name))
            if isinstance(node, ast.ImportFrom) and node.module != "__future__"
        }
        assert imports == {(1, "registry")}


def test_registrar_functions_are_fully_typed_and_documented() -> None:
    """Every private registrar has an explicit return type and docstring."""
    for module_name in _MODULE_PROFILE_NAMES:
        tree = _module_tree(module_name)
        assert ast.get_docstring(tree)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert functions
        for function in functions:
            assert function.name.startswith("_register_")
            assert not function.args.args
            assert function.returns is not None
            assert ast.get_docstring(function)


def test_facade_composes_registrars_in_historical_order() -> None:
    """The facade pins the original registration sequence explicitly."""
    tree = _module_tree("neuromorphic_profiles")
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
    public_names = {name for name in vars(neuromorphic_profiles) if not name.startswith("_")}
    before = dict(_PROFILES)

    assert public_names == {"HardwareProfile", "annotations"}
    assert neuromorphic_profiles.HardwareProfile is HardwareProfile
    with pytest.raises(
        ValueError,
        match="Duplicate hardware-profile registration for 'loihi2'",
    ):
        importlib.reload(neuromorphic_profiles)
    assert before == _PROFILES


def test_private_module_reloads_have_no_registration_side_effects() -> None:
    """Reloading a private definition module leaves the live registry unchanged."""
    before = dict(_PROFILES)
    for module_name in _MODULE_PROFILE_NAMES:
        module = importlib.import_module(f"sc_neurocore.compiler.platforms.{module_name}")
        importlib.reload(module)

    assert before == _PROFILES


def test_public_lookup_and_class_filters_cover_every_profile() -> None:
    """Public registry APIs expose every preserved profile under its class."""
    expected_names = set(_PROFILE_NAMES)
    resolved = [get_profile(name) for name in _PROFILE_NAMES]

    assert {profile.name for profile in list_profiles() if profile.name in expected_names} == (
        expected_names
    )
    assert Counter(profile.platform_class for profile in resolved) == _PLATFORM_CLASS_COUNTS
    for platform_class in _PLATFORM_CLASS_COUNTS:
        class_names = {profile.name for profile in list_profiles(platform_class=platform_class)}
        assert {
            profile.name for profile in resolved if profile.platform_class == platform_class
        } <= (class_names)


def test_profile_modules_remain_below_responsibility_ceilings() -> None:
    """The facade and private partitions remain bounded below GodFile size."""
    for module_name, ceiling in _MODULE_LINE_CEILINGS.items():
        line_count = len(
            (_PACKAGE_DIR / f"{module_name}.py").read_text(encoding="utf-8").splitlines()
        )
        assert line_count <= ceiling, f"{module_name}.py has {line_count} lines"
