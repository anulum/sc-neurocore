# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Device-target SystemVerilog emission with its report

"""Emit SystemVerilog for a device through the native bridge with that device's report."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import (  # noqa: E402
    HeuristicResourceReport,
    Ice40ResourceReport,
    ScGraph,
    ScGraphBuilder,
)
from sc_neurocore.studio.synthesis_profiles import synthesis_target_profiles  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / "benchmarks" / "results" / "ice40_up5k_calibration" / "designs.json"
ESTIMATE_KEYS = {
    "block",
    "logic_cells_estimated",
    "critical_path_estimate_ns",
    "fits_logic_cells",
    "meets_timing_estimate",
}


def _weights(pattern: str, name: str, count: int) -> list[float]:
    """Return the recorded weight pattern of a calibration design.

    ``fit``: every weight non-zero, the measured logic-cell worst case.
    ``random``: random non-zero Q8 codes seeded by the design name.
    ``folded``: about half zero or negative, which synthesis folds away.
    """
    if pattern == "fit":
        return [0.05 + 0.9 * ((i * 7) % 11) / 10 for i in range(count)]
    if pattern == "random":
        rng = random.Random(name)
        return [(1 + rng.randrange(256)) / 256 for _ in range(count)]
    assert pattern == "folded"
    return [((i * 7) % 11 - 5) / 10 for i in range(count)]


def _dense(
    n_inputs: int, n_neurons: int, width: int = 16, weights: list[float] | None = None
) -> ScGraph:
    """Build the calibration's dense layer, by default with every weight non-zero."""
    builder = ScGraphBuilder(f"dense_{n_inputs}x{n_neurons}_w{width}")
    inputs = builder.input("inputs", f"vec<rate,{n_inputs}>")
    if weights is None:
        weights = _weights("fit", "", n_inputs * n_neurons)
    matrix = builder.constant_f64_vec(weights)
    leak = builder.constant_i64(16, f"fixed<{width},8>")
    gain = builder.constant_i64(256, f"fixed<{width},8>")
    result = builder.dense_forward(
        inputs,
        matrix,
        leak,
        gain,
        n_inputs=n_inputs,
        n_neurons=n_neurons,
        data_width=width,
        seed_base=0x1234,
        stream_length=7,
        y_min=0,
        y_max=300,
    )
    builder.output("spikes", result)
    graph = builder.build()
    assert graph.verify() is None
    return graph


def _lif(width: int) -> ScGraph:
    """Build the calibration's LIF neuron."""
    builder = ScGraphBuilder(f"lif_w{width}")
    ops = [
        builder.input(name, f"fixed<{width},8>") for name in ("current", "leak", "gain", "noise")
    ]
    builder.output("spike", builder.lif_step(*ops, data_width=width, refractory_period=7))
    return builder.build()


def _synapse(length: int) -> ScGraph:
    """Build the calibration's two-encoder synapse stream."""
    builder = ScGraphBuilder(f"synapse_{length}")
    x = builder.input("x_prob", "rate")
    w = builder.input("w_prob", "rate")
    product = builder.bitwise_and(
        builder.encode(x, length=length, seed=0xACE1), builder.encode(w, length=length, seed=0xBEEF)
    )
    builder.output("rate", builder.decode(product))
    return builder.build()


def _graph(design: dict[str, object]) -> ScGraph:
    """Rebuild one recorded calibration design."""
    params = design["parameters"]
    assert isinstance(params, dict)
    if design["family"] == "dense":
        count = params["n_inputs"] * params["n_neurons"]
        weights = _weights(str(design["weight_pattern"]), str(design["name"]), count)
        return _dense(params["n_inputs"], params["n_neurons"], params["data_width"], weights)
    if design["family"] == "lif":
        return _lif(params["data_width"])
    return _synapse(params["stream_length"])


def test_ice40_emission_adds_only_the_target_header_to_the_generic_rtl() -> None:
    """The iCE40 target names its device and leaves the generic logic untouched."""
    graph = _dense(4, 3)
    systemverilog, report = graph.emit_sv_with_report("ice40:UP5K", 12)
    header = "// Target: iCE40 UltraPlus UP5K, part=iCE40UP5K-SG48, clock=12 MHz\n\n"
    assert header in systemverilog
    assert systemverilog.replace(header, "", 1) == graph.emit_sv()
    assert report["target_name"] == "ice40:UP5K"
    assert report["device_part"] == "iCE40UP5K-SG48"
    assert "Yosys 0.70" in report["calibration"] and "0.6-3build5" in report["calibration"]


def test_ice40_budgets_equal_the_studio_capacity_of_the_same_device() -> None:
    """The engine and the Studio judge the UP5K against one capacity."""
    profile = next(p for p in synthesis_target_profiles() if p.target == "ice40")
    _, report = _dense(4, 3).emit_sv_with_report("ice40:UP5K", 12)
    assert profile.capacity_device == "iCE40 UP5K"
    assert report["logic_cell_budget"] == profile.capacity["luts"]
    assert report["ram_block_budget"] == profile.capacity["brams"]
    assert report["dsp_budget"] == profile.capacity["dsps"]


def test_report_keys_are_declared_and_an_unavailable_estimate_carries_no_number() -> None:
    """A word or out-of-domain graph gets the reason and none of the estimate keys."""
    declared = set(get_type_hints(Ice40ResourceReport))
    _, calibrated = _dense(4, 3).emit_sv_with_report("ice40:UP5K", 12)
    assert calibrated["static_estimate"] is True
    assert set(calibrated) == declared
    _, outside = _lif(10).emit_sv_with_report("ice40:UP5K", 12)
    assert outside["static_estimate"] is False
    assert outside["static_estimate_reason"] == "outside the calibrated domain"
    assert not ESTIMATE_KEYS & set(outside)
    assert set(outside) == declared - ESTIMATE_KEYS
    builder = ScGraphBuilder("word")
    a = builder.input("a", "i16")
    builder.output("doubled", builder.word_binary("add", a, a, "i16"))
    _, word = builder.build().emit_sv_with_report("ice40:UP5K", 12)
    assert (
        word["static_estimate_reason"]
        == "word-level operations have no static estimate on this target"
    )
    assert not ESTIMATE_KEYS & set(word)


def _dense_with_ports(port_weights: bool, port_leak_gain: bool) -> ScGraph:
    """Build a 2x2 w16 dense layer whose weights, or leak and gain, are input ports."""
    builder = ScGraphBuilder("dense_ports")
    inputs = builder.input("inputs", "vec<rate,2>")
    weights = (
        builder.input("weights", "vec<rate,4>")
        if port_weights
        else builder.constant_f64_vec(_weights("fit", "", 4))
    )
    if port_leak_gain:
        leak, gain = builder.input("leak", "fixed<16,8>"), builder.input("gain", "fixed<16,8>")
    else:
        leak = builder.constant_i64(16, "fixed<16,8>")
        gain = builder.constant_i64(256, "fixed<16,8>")
    result = builder.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        n_inputs=2,
        n_neurons=2,
        data_width=16,
        seed_base=0x1234,
        stream_length=7,
        y_min=0,
        y_max=300,
    )
    builder.output("spikes", result)
    return builder.build()


@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(lambda: _dense_with_ports(True, False), id="port weights"),
        pytest.param(lambda: _dense_with_ports(False, True), id="port leak and gain"),
        pytest.param(
            lambda: _dense(2, 2).to_text().replace("len=7", "len=9"), id="other stream length"
        ),
    ],
)
def test_operands_or_parameters_outside_the_calibration_get_no_estimate(graph: Any) -> None:
    """Synthesis follows constants and ports, so only the measured form has a bound."""
    from sc_neurocore_engine.ir import parse_ir

    built = graph()
    built = parse_ir(built) if isinstance(built, str) else built
    _, report = built.emit_sv_with_report("ice40:UP5K", 12)
    assert report["static_estimate"] is False
    assert report["static_estimate_reason"] == (
        "operand source, type or parameter outside the calibration"
    )
    assert not ESTIMATE_KEYS & set(report)
    _, calibrated = _dense_with_ports(False, False).emit_sv_with_report("ice40:UP5K", 12)
    assert calibrated["static_estimate"] is True


def test_engine_bound_covers_every_recorded_in_domain_design() -> None:
    """Every routed design inside its domain is within the engine's bound."""
    record = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    checked = 0
    for design in record["designs"]:
        if design["status"] != "routed":
            continue
        _, report = _graph(design).emit_sv_with_report("ice40:UP5K", 12)
        if not report["static_estimate"]:
            continue
        assert design["core_logic_cells"] <= report["logic_cells_estimated"], design["name"]
        assert max(design["critical_path_ns_per_seed"]) <= report["critical_path_estimate_ns"], (
            design["name"]
        )
        checked += 1
    assert checked >= 90


def test_every_harness_overflow_is_sized_and_bounded() -> None:
    """A design whose harness did not fit is bounded by its packed core, and judged by it."""
    record = json.loads(MEASUREMENTS.read_text(encoding="utf-8"))
    over = [d for d in record["designs"] if d["status"] == "over capacity"]
    judged = 0
    for design in over:
        _, report = _graph(design).emit_sv_with_report("ice40:UP5K", 12)
        if not report["static_estimate"]:
            continue
        core = design["packed_core_logic_cells"]
        assert core <= report["logic_cells_estimated"], design["name"]
        if core > report["logic_cell_budget"]:
            assert report["fits_logic_cells"] is False, design["name"]
        judged += 1
    assert judged >= 20
    fitting = sorted(d["name"] for d in over if d["packed_core_logic_cells"] <= 5280)
    assert fitting == ["fit_dense_12x4_w12", "fit_dense_8x8_w12", "folded_fit_dense_12x6_w24"]


def test_timing_verdict_compares_the_bound_with_the_clock_period() -> None:
    """The dense path bound meets 12 MHz and misses 100 MHz."""
    _, slow_clock = _dense(4, 3).emit_sv_with_report("ice40:UP5K", 12)
    _, fast_clock = _dense(4, 3).emit_sv_with_report("ice40:UP5K", 100)
    assert slow_clock["meets_timing_estimate"] is True
    assert fast_clock["meets_timing_estimate"] is False
    assert math.isclose(fast_clock["clock_period_ns"], 10.0)


def test_zynq_targets_keep_their_labelled_heuristic_report() -> None:
    """A Zynq target returns the uncalibrated heuristic report with its fold plan."""
    _, report = _dense(12, 8).emit_sv_with_report("zynq-ultrascale-plus:ZU3EG", 250)
    assert set(report) == set(get_type_hints(HeuristicResourceReport))
    assert report["calibration"] == "uncalibrated heuristic"
    assert report["device_part"] == "xczu3eg-sbva484-1-e"


@pytest.mark.parametrize("target", ["generic", "ice40", "ice40:up5k", "zynq7:", ""])
def test_unknown_or_deviceless_targets_raise_value_error(target: str) -> None:
    """Only device targets with a name the engine writes are accepted."""
    with pytest.raises(ValueError, match="unknown SystemVerilog target"):
        _dense(2, 2).emit_sv_with_report(target, 12)


@pytest.mark.parametrize("target", ["ice40:UP5K", "zynq-ultrascale-plus:ZU3EG"])
def test_clock_must_be_a_positive_32_bit_value(target: str) -> None:
    """A zero clock has no period; a negative one does not convert."""
    graph = _dense(2, 2)
    with pytest.raises(ValueError, match="clock_mhz must be positive"):
        graph.emit_sv_with_report(target, 0)
    with pytest.raises(OverflowError):
        graph.emit_sv_with_report(target, -1)
