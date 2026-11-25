# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense layers with a weight matrix bind sc_dense_matrix_layer

"""A 2-D ``weights`` matrix emits ``sc_dense_matrix_layer``; every other form keeps the shared core.

``sc_dense_layer_core`` runs one dot product and drives every neuron with the same current, so
a Dense layer emitted on it is a shared-weight population whose spike bits are one signal. A
``weights`` matrix with one row per neuron now emits ``sc_dense_matrix_layer``, seeded from
``input_seed_base`` and ``weight_seed_base`` and bounded by ``y_min_fp``/``y_max_fp``. A 1-D
``weights`` or one ``weight_fp`` keeps emitting the shared core byte for byte as before, and a
seed parameter that the shared core cannot honour is refused instead of being ignored. The Rust
safety mirror follows the same dispatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from sc_neurocore.hdl_gen.aer_emitter import AEREmitter
from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests/fixtures/hdl_gen_dense_shared_core_emission_93e8b87cc.json"
_PRIMITIVES = (
    "sc_bitstream_encoder.v",
    "sc_bitstream_synapse.v",
    "sc_dotproduct_to_current.v",
    "sc_lif_neuron.v",
    "sc_dense_matrix_layer.v",
)
_MATRIX = [[1000, 2000, 3000], [65535, 0, 30000]]


def _generate(layers: list[tuple[str, dict[str, Any]]], bus_width: int, mode: str = "sync") -> str:
    """Return the generator's output for ``layers``.

    Parameters
    ----------
    layers : list[tuple[str, dict[str, Any]]]
        ``(name, params)`` of each Dense layer, in order.
    bus_width : int
        Top-level input bus width.
    mode : str
        ``"sync"`` or ``"async_aer"``.

    Returns
    -------
    str
        Generated Verilog.
    """
    generator = VerilogGenerator(module_name="dense_probe", bus_width=bus_width)
    for name, params in layers:
        generator.add_layer("Dense", name, params)
    return generator.generate(mode)


@pytest.mark.parametrize("mode", ["sync", "async_aer"])
def test_the_shared_core_forms_emit_byte_for_byte_what_main_emitted(mode: str) -> None:
    """Each 1-D, ``weight_fp`` and default case hashes to its digest captured on 93e8b87cc."""
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        layers = [(name, params) for name, params in case["layers"]]
        emitted = _generate(layers, case["bus_width"], mode)
        assert "sc_dense_matrix_layer" not in emitted, case["name"]
        assert hashlib.sha256(emitted.encode()).hexdigest() == case[f"{mode}_sha256"], case["name"]


def test_a_weight_matrix_emits_the_matrix_layer_with_its_rows_seeds_and_range() -> None:
    """Rows go in row-major order with the last word first; seeds and current range are bound."""
    params = {
        "n_neurons": 2,
        "weights": _MATRIX,
        "input_seed_base": 0x1234,
        "weight_seed_base": 0x4321,
        "y_min_fp": -64,
        "y_max_fp": 1024,
    }
    emitted = _generate([("m", params)], 3)
    assert "    sc_dense_matrix_layer #(" in emitted
    assert "sc_dense_layer_core" not in emitted
    assert ".INPUT_SEED_BASE(16'h1234)," in emitted
    assert ".WEIGHT_SEED_BASE(16'h4321)" in emitted
    assert ".weight_fp({16'h7530, 16'h0000, 16'hFFFF, 16'h0BB8, 16'h07D0, 16'h03E8})," in emitted
    assert ".y_min_fp(16'hFFC0)," in emitted
    assert ".y_max_fp(16'h0400)," in emitted


def test_the_aer_emitter_binds_the_matrix_layer_too() -> None:
    """``AEREmitter`` takes the same dispatch: a matrix gives the matrix layer, seeds refused on 1-D."""
    emitter = AEREmitter(module_name="aer_probe", bus_width=3)
    emitter.add_layer(
        "Dense", "m", {"n_neurons": 2, "weights": _MATRIX, "weight_seed_base": 0x0101}
    )
    emitted = emitter.generate()
    assert "    sc_dense_matrix_layer #(" in emitted
    assert ".WEIGHT_SEED_BASE(16'h0101)" in emitted
    refused = AEREmitter(module_name="aer_probe", bus_width=3)
    refused.add_layer("Dense", "s", {"n_neurons": 2, "weight_seed_base": 0x0101})
    with pytest.raises(ValueError, match="weight_seed_base needs a 2-D weights matrix"):
        refused.generate()


def test_seeds_and_range_change_the_matrix_emission() -> None:
    """Different seeds or a different current range give different RTL; defaults are 0xACE1/0xBEEF."""
    base = {"n_neurons": 2, "weights": _MATRIX}
    default = _generate([("m", base)], 3)
    assert ".INPUT_SEED_BASE(16'hACE1)," in default and ".WEIGHT_SEED_BASE(16'hBEEF)" in default
    for change in ({"input_seed_base": 7}, {"weight_seed_base": 9}, {"y_max_fp": 300}):
        assert _generate([("m", {**base, **change})], 3) != default, change


@pytest.mark.parametrize("key", ["input_seed_base", "weight_seed_base"])
@pytest.mark.parametrize("weights", [None, [1, 2]])
def test_a_seed_on_the_shared_core_is_refused(key: str, weights: list[int] | None) -> None:
    """The shared core has fixed encoder seeds, so a seed parameter is refused, not ignored."""
    params: dict[str, Any] = {"n_neurons": 2, key: 5}
    if weights is not None:
        params["weights"] = weights
    with pytest.raises(ValueError, match=f"{key} needs a 2-D weights matrix"):
        _generate([("s", params)], 2)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"n_neurons": 3, "weights": _MATRIX}, "has 2 rows; n_neurons is 3"),
        ({"n_neurons": 2, "weights": [[1, 2, 3], [4, 5]]}, "share one non-zero width"),
        ({"n_neurons": 2, "weights": [[1, 2, 3], 4]}, "not a mix"),
        ({"n_neurons": 2, "weights": [[1, 2, 3], [4, 5, 65536]]}, r"weights\[1\]\[2\]"),
        ({"n_neurons": 2, "weights": _MATRIX, "input_seed_base": 0}, r"must be in \[1, 0xFFFF\]"),
        ({"n_neurons": 2, "weights": _MATRIX, "y_min_fp": 512}, "must be below y_max_fp"),
        ({"n_neurons": 2, "weights": _MATRIX, "input_width": 2}, "input_width is 2"),
    ],
)
def test_a_bad_weight_matrix_is_refused(params: dict[str, Any], message: str) -> None:
    """Row count, row widths, word range, seeds, range order and declared width are checked."""
    with pytest.raises(ValueError, match=message):
        _generate([("m", params)], 3)


def test_rows_wider_than_the_incoming_bus_are_refused() -> None:
    """A matrix whose rows do not match the bus that drives the layer is refused at emission."""
    with pytest.raises(ValueError, match=r"input_width \(4\) entries"):
        _generate([("m", {"n_neurons": 2, "weights": _MATRIX})], 4)


def test_equal_rows_in_the_matrix_layer_are_not_one_signal(tmp_path: Path) -> None:
    """With every row equal, Icarus shows the two neurons' spike bits differ at some cycle.

    Parameters
    ----------
    tmp_path : Path
        Scratch directory for the generated top, the bench and the simulation binary.
    """
    row = [40000, 40000, 40000]
    top = _generate([("m", {"n_neurons": 2, "weights": [row, row], "stream_len": 64})], 3)
    (tmp_path / "top.v").write_text(top, encoding="utf-8")
    bench = tmp_path / "bench.v"
    bench.write_text(
        "`timescale 1ns/1ps\n"
        "module bench;\n"
        "reg clk = 1'b0; reg rst_n = 1'b0; reg [2:0] input_bus = 3'b111; wire [1:0] output_bus;\n"
        "integer cycle; integer differ = 0; integer ones = 0;\n"
        "dense_probe dut(.clk(clk), .rst_n(rst_n), .input_bus(input_bus), .output_bus(output_bus));\n"
        "always #5 clk = ~clk;\n"
        "initial begin\n"
        "  #12 rst_n = 1'b1;\n"
        "  for (cycle = 0; cycle < 4000; cycle = cycle + 1) begin\n"
        "    @(posedge clk);\n"
        "    if (output_bus[0] !== output_bus[1]) differ = differ + 1;\n"
        "    if (output_bus != 2'b00) ones = ones + 1;\n"
        "  end\n"
        '  $display("DIFFER %0d ONES %0d", differ, ones);\n'
        "  $finish;\n"
        "end\n"
        "endmodule\n",
        encoding="utf-8",
    )
    primitives = [str(_REPO_ROOT / "hdl" / name) for name in _PRIMITIVES]
    binary = tmp_path / "sim.vvp"
    subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "bench",
            "-o",
            str(binary),
            str(bench),
            str(tmp_path / "top.v"),
            *primitives,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    run = subprocess.run(
        ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=120
    )
    line = next(text for text in run.stdout.splitlines() if text.startswith("DIFFER "))
    differ, ones = int(line.split()[1]), int(line.split()[3])
    assert ones > 0
    assert differ > 0


def test_the_rust_mirror_emits_the_matrix_layer_byte_for_byte(tmp_path: Path) -> None:
    """The Rust safety mirror and the Python generator emit identical matrix-layer RTL.

    Parameters
    ----------
    tmp_path : Path
        Scratch directory for the Rust driver and its output.
    """
    assert shutil.which("rustc") is not None
    rust_src = _REPO_ROOT / "src/sc_neurocore/accel/rust/safety/verilog_generator.rs"
    sync_out = tmp_path / "rust_sync.v"
    aer_out = tmp_path / "rust_aer.v"
    driver = tmp_path / "driver.rs"
    driver.write_text(
        f"""\
#[path = {json.dumps(str(rust_src))}]
mod verilog_generator;
use verilog_generator::{{DenseLayerParams, VerilogGenerator}};

fn params() -> DenseLayerParams {{
    DenseLayerParams {{
        weight_matrix: Some(vec![vec![1000, 2000, 3000], vec![0xFFFF, 0, 30000]]),
        input_seed_base: Some(0x1234),
        weight_seed_base: Some(0x4321),
        ..DenseLayerParams::new(2)
    }}
}}

fn main() {{
    let mut sync_gen = VerilogGenerator::try_new("dense_probe", 3).unwrap();
    sync_gen.add_dense_layer("m", params()).unwrap();
    sync_gen.add_dense_layer("tail", DenseLayerParams::new(1)).unwrap();
    std::fs::write({json.dumps(str(sync_out))}, sync_gen.generate("sync").unwrap()).unwrap();
    let mut aer_gen = VerilogGenerator::try_new("dense_probe", 3).unwrap();
    aer_gen.add_dense_layer("m", params()).unwrap();
    std::fs::write({json.dumps(str(aer_out))}, aer_gen.generate("async_aer").unwrap()).unwrap();
}}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "driver"
    built = subprocess.run(
        ["rustc", "--edition=2021", str(driver), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=False, timeout=30)
    assert ran.returncode == 0, ran.stderr
    params = {
        "n_neurons": 2,
        "weights": _MATRIX,
        "input_seed_base": 0x1234,
        "weight_seed_base": 0x4321,
    }
    assert sync_out.read_text(encoding="utf-8") == _generate(
        [("m", params), ("tail", {"n_neurons": 1})], 3
    )
    assert aer_out.read_text(encoding="utf-8") == _generate([("m", params)], 3, "async_aer")
