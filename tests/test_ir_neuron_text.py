# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native neuron text through the Python bridge

"""Preserve configured native neuron graphs across the public Python boundary."""

from __future__ import annotations

import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder, parse_ir


@pytest.mark.parametrize("rest,reset", [(-64, -32), (31, -127), (0, 0)])
def test_lif_native_roundtrip_retains_rest_reset_and_emitted_hardware(
    rest: int, reset: int
) -> None:
    """Build a real native LIF and retain its state parameters after parsing."""
    builder = ScGraphBuilder("lif_text")
    operands = [builder.input(name, "fixed<16,8>") for name in ("current", "leak", "gain", "noise")]
    spike = builder.lif_step(*operands, v_rest=rest, v_reset=reset, refractory_period=7)
    builder.output("spike", spike)
    graph = builder.build()
    text = graph.to_text()
    assert f"vr={rest}, reset={reset}" in text
    parsed = parse_ir(text)
    assert parsed.verify() is None
    assert parsed.to_text() == text
    sv = parsed.emit_sv()
    assert sv == graph.emit_sv()
    assert f".V_REST({rest})" in sv
    assert f".V_RESET({reset})" in sv
    assert ".REFRACTORY_PERIOD(7)" in sv


def test_dense_native_roundtrip_retains_bridge_seed_width_and_current_words() -> None:
    """Keep the actual Python builder's complete dense parameter encoding."""
    builder = ScGraphBuilder("dense_text")
    inputs = builder.input("inputs", "vec<fixed<24,8>,3>")
    weights = builder.input("weights", "vec<fixed<24,8>,21>")
    leak = builder.input("leak", "fixed<24,8>")
    gain = builder.input("gain", "fixed<24,8>")
    spikes = builder.dense_forward(
        inputs,
        weights,
        leak,
        gain,
        data_width=24,
        stream_length=71,
        seed_base=0x1234,
        y_min=-100,
        y_max=300,
    )
    builder.output("spikes", spikes)
    graph = builder.build()
    text = graph.to_text()
    for field in ["dw=24", "input_seed=4660", "weight_seed=4661", "y_min=-100", "y_max=300"]:
        assert field in text
    parsed = parse_ir(text)
    assert parsed.verify() is None
    assert parsed.to_text() == text
    assert parsed.emit_sv() == graph.emit_sv()


_DCLS_TEXT = """sc.graph @dcls_text {
  %0 = sc.input "spike" : bool
  %1 = sc.input "weights" : vec<fixed<16,8>,3>
  %2 = sc.input "centre" : fixed<16,8>
  %3 = sc.input "sigma" : fixed<16,8>
  %4 = sc.dcls_layer %0, weights=%1, centre=%2, sigma=%3, taps=3, depth=64, dw=16, frac=8, ptr=6, offsets=[7;0;3] : fixed<16,8>
  sc.output "sum" %4
}
"""


def test_dcls_native_text_parses_through_python_and_emits_ordered_offsets() -> None:
    """Parse DCLS through PyO3 and preserve the exact packed tap order."""
    graph = parse_ir(_DCLS_TEXT)
    assert graph.verify() is None
    assert graph.to_text() == _DCLS_TEXT
    sv = graph.emit_sv()
    assert ".PTR_WIDTH(6)" in sv
    assert ".DELAY_DEPTH(64)" in sv
    assert ".tap_offsets({6'd3, 6'd0, 6'd7})" in sv


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("ptr=6", "ptr=invalid"),
        ("ptr=6", "ptr=6, ptr=5"),
        ("ptr=6", "unknown=6"),
        ("offsets=[7;0;3]", "offsets=[7;-1;3]"),
        (", offsets=[7;0;3]", ""),
    ],
)
def test_malformed_neuron_text_raises_native_value_error(field: str, replacement: str) -> None:
    """Reject corrupted persisted hardware parameters at the native boundary."""
    with pytest.raises(ValueError, match="line 6:"):
        parse_ir(_DCLS_TEXT.replace(field, replacement))
