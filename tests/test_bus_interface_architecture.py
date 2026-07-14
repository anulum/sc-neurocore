# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bus-interface responsibility and compatibility contracts

"""Guard the bus-interface module graph and parent-equivalent public outputs."""

from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

from sc_neurocore.compiler.live_control import MMIOUpdateSpec, ParameterBankSpec, TrapSpec
from sc_neurocore.hdl_gen import generate_live_parameter_bank as package_live_parameter_bank
from sc_neurocore.hdl_gen import bus_interface as bus_interface_module
from sc_neurocore.hdl_gen.bus_interface import (
    generate_bus_wrapper,
    generate_live_parameter_bank,
    generate_register_map,
)


_SOURCE_ROOT = Path(__file__).parents[1] / "src" / "sc_neurocore" / "hdl_gen"
_MODULE_LIMITS = {
    "bus_interface.py": 200,
    "_bus_wrappers.py": 450,
    "_live_parameter_bank.py": 650,
    "_pcie_live_parameter_bank.py": 225,
}
_EXPECTED_LOCAL_IMPORTS = {
    "bus_interface.py": {
        "_bus_wrappers",
        "_live_parameter_bank",
        "_pcie_live_parameter_bank",
    },
    "_bus_wrappers.py": set(),
    "_live_parameter_bank.py": set(),
    "_pcie_live_parameter_bank.py": {"_live_parameter_bank"},
}
_EXPECTED_SIGNATURES = {
    "generate_bus_wrapper": (
        "(inner_module: 'str', params: 'dict[str, int]', *, "
        "bus: 'BusProtocol' = 'axi_lite', data_width: 'int' = 16, "
        "addr_width: 'int' = 8, bus_data_width: 'int' = 32, "
        "base_address: 'int' = 0) -> 'str'"
    ),
    "generate_register_map": (
        "(params: 'dict[str, int]', *, base_address: 'int' = 0) -> 'dict[str, int]'"
    ),
    "generate_live_parameter_bank": (
        "(spec: 'MMIOUpdateSpec', *, module_name: 'str' = "
        "'sc_live_parameter_bank', addr_width: 'int | None' = None, "
        "bus_data_width: 'int' = 32, block_ram_threshold_bits: 'int' = 1024) -> 'str'"
    ),
}
_EXPECTED_PAYLOAD_HASHES = {
    "axi_wrapper": "2becc845fd5e3322f5c1937bc9add4a7a6aff0db6eec9f79c5ad1ec6bb302c1b",
    "wishbone_wrapper": "88c28e8df50bf0be33914df4166fbd74306a60c9ca85f9fb52b464b3935f3e8b",
    "empty_axi_wrapper": "8890e3da14007508436f5d5765c3206d9f5f726b5e7fac39c0fbe9963d3e2dac",
    "axi_live_parameter_bank": "a5f2f90cf10d9f8a66e619af6adc832407b912ef1305ed6eb360a7476a3d8e8b",
    "pcie_live_parameter_bank": "a924e88382310f50401ab436091d43ec3b53e72da1a4f87e661a1caa8e6e793d",
}


def _local_imports(path: Path) -> set[str]:
    """Return sibling-module imports declared by one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is not None
    }


def _sha256(source: str) -> str:
    """Return the UTF-8 SHA-256 digest for generated source."""
    return hashlib.sha256(source.encode()).hexdigest()


def _live_specs() -> tuple[MMIOUpdateSpec, MMIOUpdateSpec]:
    """Return representative AXI4-Lite and PCIe parent-fingerprint specs."""
    banks = (
        ParameterBankSpec(
            bank_name="weights",
            start_address_bytes=0x2000,
            parameter_count=4,
            parameter_names=("w0", "w1", "w2", "w3"),
            q_format="Q8.8",
        ),
        ParameterBankSpec(
            bank_name="kuramoto",
            start_address_bytes=0x3000,
            parameter_count=128,
            parameter_names=("k_mag",),
            q_format="Q16.16",
            reset_value=-1,
            writable=False,
        ),
        ParameterBankSpec(
            bank_name="wide48",
            start_address_bytes=0x4000,
            parameter_count=2,
            parameter_names=("a", "b"),
            q_format="Q24.24",
        ),
        ParameterBankSpec(
            bank_name="wide64",
            start_address_bytes=0x5000,
            parameter_count=2,
            parameter_names=("c", "d"),
            q_format="Q32.32",
        ),
    )
    axi = MMIOUpdateSpec(
        bus_protocol="axi4_lite",
        control_base_address_bytes=0x100,
        banks=banks,
        trap=TrapSpec(max_flags=8),
    )
    pcie = MMIOUpdateSpec(
        bus_protocol="pcie",
        control_base_address_bytes=0x100,
        banks=banks[:2],
        trap=TrapSpec(max_flags=8),
    )
    return axi, pcie


def test_bus_interface_modules_are_bounded_and_acyclic() -> None:
    """Each responsibility remains bounded and follows the one-way import graph."""
    for name, limit in _MODULE_LIMITS.items():
        path = _SOURCE_ROOT / name
        assert len(path.read_text(encoding="utf-8").splitlines()) <= limit
        assert _local_imports(path) == _EXPECTED_LOCAL_IMPORTS[name]


def test_public_functions_keep_historical_identity_and_signatures() -> None:
    """Public callables retain their historical import, module, and signature."""
    assert bus_interface_module.__all__ == [
        "BusProtocol",
        "generate_bus_wrapper",
        "generate_live_parameter_bank",
        "generate_register_map",
    ]
    assert (
        not {
            "render_axi_lite_wrapper",
            "render_wishbone_wrapper",
            "render_axi_live_parameter_bank",
            "render_pcie_live_parameter_bank",
        }
        & vars(bus_interface_module).keys()
    )
    functions = (
        generate_bus_wrapper,
        generate_register_map,
        generate_live_parameter_bank,
    )
    assert package_live_parameter_bank is generate_live_parameter_bank
    for function in functions:
        assert function.__module__ == "sc_neurocore.hdl_gen.bus_interface"
        assert str(inspect.signature(function)) == _EXPECTED_SIGNATURES[function.__name__]


def test_representative_outputs_match_parent_byte_for_byte() -> None:
    """Representative generated sources retain their exact parent digests."""
    params = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}
    axi_spec, pcie_spec = _live_specs()
    payloads = {
        "axi_wrapper": generate_bus_wrapper("sc_lif", params, bus="axi_lite"),
        "wishbone_wrapper": generate_bus_wrapper("sc_lif", params, bus="wishbone"),
        "empty_axi_wrapper": generate_bus_wrapper("sc_empty", {}, bus="axi_lite"),
        "axi_live_parameter_bank": generate_live_parameter_bank(
            axi_spec,
            module_name="sc_live_params",
        ),
        "pcie_live_parameter_bank": generate_live_parameter_bank(
            pcie_spec,
            module_name="sc_live_pcie_params",
        ),
    }
    assert {name: _sha256(source) for name, source in payloads.items()} == (
        _EXPECTED_PAYLOAD_HASHES
    )
    assert generate_register_map(params, base_address=0x1000) == {
        "CTRL": 0x1000,
        "I_T": 0x1004,
        "SPIKE_COUNT": 0x1008,
        "P_V_REST": 0x100C,
        "P_V_THRESH": 0x1010,
        "P_TAU_M": 0x1014,
    }
