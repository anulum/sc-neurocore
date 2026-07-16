# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM generator modular-architecture contracts

"""Lock UVM generator ownership, compatibility, and deterministic payloads."""

from __future__ import annotations

import ast
import enum
import hashlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Any

import sc_neurocore.uvm_gen as uvm_gen
from sc_neurocore.uvm_gen import uvm_gen as historical_facade

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATHS = {
    "__init__": _REPO_ROOT / "src/sc_neurocore/uvm_gen/__init__.py",
    "_benchmark": _REPO_ROOT / "src/sc_neurocore/uvm_gen/_benchmark.py",
    "_component_emitters": _REPO_ROOT / "src/sc_neurocore/uvm_gen/_component_emitters.py",
    "_config": _REPO_ROOT / "src/sc_neurocore/uvm_gen/_config.py",
    "_generator": _REPO_ROOT / "src/sc_neurocore/uvm_gen/_generator.py",
    "_harness_emitters": _REPO_ROOT / "src/sc_neurocore/uvm_gen/_harness_emitters.py",
    "_rtl": _REPO_ROOT / "src/sc_neurocore/uvm_gen/_rtl.py",
    "uvm_gen": _REPO_ROOT / "src/sc_neurocore/uvm_gen/uvm_gen.py",
}
_MODULE_LINE_CEILINGS = {
    "__init__": 80,
    "_benchmark": 100,
    "_component_emitters": 600,
    "_config": 150,
    "_generator": 175,
    "_harness_emitters": 350,
    "_rtl": 250,
    "uvm_gen": 100,
}
_EXPECTED_DEPENDENCIES = {
    "__init__": {"uvm_gen"},
    "_benchmark": set(),
    "_component_emitters": {"_config", "_rtl"},
    "_config": set(),
    "_generator": {
        "_benchmark",
        "_component_emitters",
        "_config",
        "_harness_emitters",
        "_rtl",
    },
    "_harness_emitters": {"_config", "_rtl"},
    "_rtl": set(),
    "uvm_gen": {"_benchmark", "_config", "_generator", "_rtl"},
}
_EXPECTED_CLASS_OWNERS = {
    "_benchmark": {"UVMBenchmark"},
    "_config": {
        "CoverageSpec",
        "FormalLink",
        "ScoreboardConfig",
        "SimTarget",
        "StimulusConfig",
    },
    "_generator": {"UVMGenerator"},
    "_rtl": {"ModuleParam", "ModulePort", "PortDirection", "PortType", "RTLModule"},
}
_EXPECTED_EXPORTS = [
    "CoverageSpec",
    "FormalLink",
    "ModuleParam",
    "ModulePort",
    "PortDirection",
    "PortType",
    "RTLModule",
    "SIM_TARGETS",
    "ScoreboardConfig",
    "SimTarget",
    "StimulusConfig",
    "UVMBenchmark",
    "UVMGenerator",
]
_EXPECTED_SIGNATURES = {
    "CoverageSpec": "(bitstream_density_bins: 'int' = 10, spike_rate_bins: 'int' = 5, "
    "scc_bins: 'int' = 8, cross_coverage: 'bool' = True, toggle_coverage: 'bool' = True, "
    "target_percent: 'float' = 95.0, formal_property_map: 'Dict[str, str]' = <factory>) -> None",
    "FormalLink": "(property_name: 'str', sby_module: 'str', assertion_sv: 'str', "
    "cover_sv: 'str') -> None",
    "ModuleParam": "(name: 'str', value: 'str', param_type: 'str' = 'int') -> None",
    "ModulePort": "(name: 'str', direction: 'PortDirection', port_type: 'PortType' = "
    "<PortType.LOGIC: 'logic'>, width: 'int' = 1, is_signed: 'bool' = False, "
    "is_array: 'bool' = False, array_size: 'int' = 0) -> None",
    "PortDirection": "(*values)",
    "PortType": "(*values)",
    "RTLModule": "(name: 'str', ports: 'List[ModulePort]', params: 'List[ModuleParam]' = "
    "<factory>, is_sc_module: 'bool' = True) -> None",
    "ScoreboardConfig": "(tolerance_bits: 'int' = 0, check_popcount: 'bool' = True, "
    "check_probability: 'bool' = True, check_spike_timing: 'bool' = True, "
    "check_golden_comparison: 'bool' = False, golden_model_type: 'str' = 'bit_true', "
    "golden_expressions: 'Dict[str, str]' = <factory>) -> None",
    "SimTarget": "(name: 'str', compile_cmd: 'str', run_cmd: 'str', coverage_cmd: 'str') -> None",
    "StimulusConfig": "(num_transactions: 'int' = 1000, bitstream_density_range: "
    "'Tuple[float, float]' = (0.1, 0.9), lfsr_seed_range: 'Tuple[int, int]' = "
    "(1, 65535), enable_corner_cases: 'bool' = True, max_consecutive_ones: 'int' = 32, "
    "max_consecutive_zeros: 'int' = 32) -> None",
    "UVMBenchmark": "(module_name: 'str', transaction_sv: 'str', sequence_sv: 'str', "
    "driver_sv: 'str', monitor_sv: 'str', scoreboard_sv: 'str', coverage_sv: 'str', "
    "agent_sv: 'str', env_sv: 'str', top_sv: 'str', sby_config: 'str', bind_sv: "
    "'str' = '', makefile: 'str' = '', regression_list: 'str' = '', filelist: "
    "'List[str]' = <factory>) -> None",
    "UVMGenerator": "(stimulus: 'Optional[StimulusConfig]' = None, coverage: "
    "'Optional[CoverageSpec]' = None, scoreboard: 'Optional[ScoreboardConfig]' = None)",
}
# ``PortDirection`` and ``PortType`` are ``enum.Enum`` subclasses. ``inspect``
# reports the CPython enum-metaclass call signature for them, and that string
# varies across interpreter versions (3.11 emits the full functional-API form,
# 3.12+ collapses it to ``(*values)``), so the stable public contract we pin is
# the member map rather than that interpreter-specific noise.
_EXPECTED_ENUM_MEMBERS = {
    "PortDirection": {"INPUT": "input", "OUTPUT": "output", "INOUT": "inout"},
    "PortType": {"LOGIC": "logic", "WIRE": "wire", "REG": "reg"},
}
_SOURCE_CASES = {
    "dense": """module dense(
    input logic i_clk,
    input logic reset_n,
    input logic [7:0] a,
    input logic [3:0] b,
    output logic [7:0] y
);
endmodule
""",
    "empty_data": "module tick(input logic clock, input logic reset); endmodule",
    "lif": """module sc_lif_neuron #(
    parameter DATA_WIDTH = 16,
    parameter FRACTION = 8,
    parameter V_THRESHOLD = 16'sd256
)(
    input wire clk,
    input wire rst_n,
    input wire signed [15:0] I_t,
    input wire signed [15:0] noise_in,
    output wire spike_out,
    output wire signed [15:0] v_out
);
endmodule
""",
}
_EXPECTED_PAYLOAD_DIGESTS = {
    "dense:configured": "86020a6939f567e7f89a1f2cf04229143bc56e85397196a0ccbb422176c1fa35",
    "dense:default": "0e1a12956504f5fd7c129bdba8b2225f7535b76c7a4ecdacc6594e74f00d4015",
    "empty_data:configured": "9cf1298252ea72417810c3c5d34b9cbc6701a396b84be1e227cf4077ecaeffeb",
    "empty_data:default": "e445cb6c81342ddd58617b5ba81c83c1da58ac4f064989a68d48857a19e53b74",
    "lif:configured": "99e0d3c81c6b3a013e4ccf52c1c41e7b0af972448f05177c9ba68a08e6416e30",
    "lif:default": "51fb22e0774419df7dc764d57d0046649cfa558f520f7c674243aee622a306cd",
}


def _dependencies(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dependencies: set[str] = set()
    prefix = "sc_neurocore.uvm_gen."
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.startswith(prefix):
                candidate = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
                if candidate in _MODULE_PATHS:
                    dependencies.add(candidate)
    return dependencies


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(module: str) -> None:
        if module in active:
            raise AssertionError(f"UVM generator import cycle reaches {module}")
        if module in visited:
            return
        active.add(module)
        for dependency in graph[module]:
            visit(dependency)
        active.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def _defined_classes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def _configured_generator() -> uvm_gen.UVMGenerator:
    return uvm_gen.UVMGenerator(
        stimulus=uvm_gen.StimulusConfig(
            num_transactions=17,
            bitstream_density_range=(0.25, 0.75),
            enable_corner_cases=False,
        ),
        coverage=uvm_gen.CoverageSpec(
            bitstream_density_bins=7,
            scc_bins=3,
            toggle_coverage=False,
            target_percent=98.5,
        ),
        scoreboard=uvm_gen.ScoreboardConfig(
            check_popcount=False,
            check_spike_timing=False,
        ),
    )


def _payload_digest(generator: uvm_gen.UVMGenerator, source: str) -> str:
    rtl = uvm_gen.RTLModule.from_verilog_source(source)
    benchmark = generator.generate(rtl)
    canonical_payload = json.dumps(
        {
            "artifacts": benchmark.to_dict(),
            "filelist": benchmark.filelist,
            "formal_links": [
                {
                    "assertion_sv": link.assertion_sv,
                    "cover_sv": link.cover_sv,
                    "property_name": link.property_name,
                    "sby_module": link.sby_module,
                }
                for link in generator.generate_formal_links(rtl)
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical_payload).hexdigest()


def test_uvm_modules_have_bounded_single_direction_dependencies() -> None:
    """UVM responsibilities remain bounded and form the intended DAG."""
    graph = {name: _dependencies(path) for name, path in _MODULE_PATHS.items()}

    assert graph == _EXPECTED_DEPENDENCIES
    _assert_acyclic(graph)
    for name, path in _MODULE_PATHS.items():
        assert len(path.read_text(encoding="utf-8").splitlines()) <= _MODULE_LINE_CEILINGS[name]


def test_uvm_public_classes_have_one_responsibility_owner() -> None:
    """Public definitions are implemented once outside the historical facade."""
    actual = {name: _defined_classes(_MODULE_PATHS[name]) for name in _EXPECTED_CLASS_OWNERS}
    assert actual == _EXPECTED_CLASS_OWNERS
    assert _defined_classes(_MODULE_PATHS["uvm_gen"]) == set()
    assert _defined_classes(_MODULE_PATHS["__init__"]) == set()


def test_uvm_facades_preserve_exports_signatures_and_pickle_identity() -> None:
    """Both import boundaries retain the exact historical public contract."""
    assert uvm_gen.__all__ == _EXPECTED_EXPORTS
    assert uvm_gen.__tier__ == "industrial"
    assert {
        name
        for name in vars(historical_facade)
        if not name.startswith("_") and name != "annotations"
    } == set(_EXPECTED_EXPORTS)

    for name, expected_signature in _EXPECTED_SIGNATURES.items():
        package_definition: Any = getattr(uvm_gen, name)
        facade_definition: Any = getattr(historical_facade, name)
        assert package_definition is facade_definition
        assert facade_definition.__module__ == "sc_neurocore.uvm_gen.uvm_gen"
        if isinstance(facade_definition, type) and issubclass(facade_definition, enum.Enum):
            assert {
                member.name: member.value for member in facade_definition
            } == _EXPECTED_ENUM_MEMBERS[name]
        else:
            assert str(inspect.signature(facade_definition)) == expected_signature
        assert pickle.loads(pickle.dumps(facade_definition)) is facade_definition

    assert uvm_gen.SIM_TARGETS is historical_facade.SIM_TARGETS


def test_uvm_generated_payloads_remain_parent_byte_exact() -> None:
    """Representative generated bundles retain their pre-split SHA-256 values."""
    actual: dict[str, str] = {}
    for source_name, source in _SOURCE_CASES.items():
        actual[f"{source_name}:default"] = _payload_digest(uvm_gen.UVMGenerator(), source)
        actual[f"{source_name}:configured"] = _payload_digest(_configured_generator(), source)

    assert actual == _EXPECTED_PAYLOAD_DIGESTS


def test_optional_benchmark_artifacts_are_omitted_when_empty() -> None:
    """The artifact contract preserves its historical empty-optional branches."""
    benchmark = uvm_gen.UVMBenchmark(
        module_name="empty",
        transaction_sv="transaction",
        sequence_sv="sequence",
        driver_sv="driver",
        monitor_sv="monitor",
        scoreboard_sv="scoreboard",
        coverage_sv="coverage",
        agent_sv="agent",
        env_sv="env",
        top_sv="top",
        sby_config="sby",
    )

    assert set(benchmark.to_dict()) == {
        "empty_agent.sv",
        "empty_coverage.sv",
        "empty_driver.sv",
        "empty_env.sv",
        "empty_monitor.sv",
        "empty_scoreboard.sv",
        "empty_sequence.sv",
        "empty_transaction.sv",
        "empty_verify.sby",
        "tb_empty_top.sv",
    }


def test_parser_retains_unsupported_header_fallbacks() -> None:
    """Unsupported entries remain ignored and a header without ports stays empty."""
    bare = uvm_gen.RTLModule.from_verilog_source("module bare; endmodule")
    mixed = uvm_gen.RTLModule.from_verilog_source(
        "module mixed(input logic valid, unsupported declaration); endmodule"
    )

    assert bare.ports == []
    assert [port.name for port in mixed.ports] == ["valid"]


def test_scalar_and_zero_width_metadata_retain_width_specific_fallbacks() -> None:
    """Parent behavior omits width-specific templates for widths below two bits."""
    rtl = uvm_gen.RTLModule(
        name="narrow",
        ports=[
            uvm_gen.ModulePort("clk", uvm_gen.PortDirection.INPUT),
            uvm_gen.ModulePort("rst_n", uvm_gen.PortDirection.INPUT),
            uvm_gen.ModulePort("enable", uvm_gen.PortDirection.INPUT),
            uvm_gen.ModulePort("empty_out", uvm_gen.PortDirection.OUTPUT, width=0),
        ],
    )
    benchmark = uvm_gen.UVMGenerator().generate(rtl)

    assert "c_enable_density" not in benchmark.transaction_sv
    assert "txn.enable = '0" not in benchmark.sequence_sv
    assert "enable_activity" not in benchmark.coverage_sv
    assert "empty_out_toggle" not in benchmark.coverage_sv
    assert "empty_out_bounded" not in benchmark.bind_sv
