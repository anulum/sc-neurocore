# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (escape_rate_and_licence) from former test_verilog_compiler_contracts.py

from __future__ import annotations

from tests.verilog_compiler_contracts_support import *  # noqa: F403


def test_generated_registered_and_folded_rtl_carry_licence_headers() -> None:
    """Both public emitters return HDL with the repository's seven-line header."""
    common = (
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "// Commercial license available",
        "// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "// © Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "// ORCID: 0009-0009-3560-0851",
        "// Contact: www.anulum.li | protoscience@anulum.li",
    )
    registered_header = "\n".join(
        (
            *common,
            "// SC-NeuroCore — Generated fixed-point RTL",
        )
    )
    folded_header = "\n".join(
        (
            *common,
            "// SC-NeuroCore — Generated folded fixed-point RTL",
        )
    )

    assert compile_to_verilog(_lif_without_threshold()).startswith(registered_header)
    assert compile_to_datapath(_lif_without_threshold()).startswith(folded_header)


def test_escape_rate_registered_rtl_owns_seeded_eight_advance_lfsr() -> None:
    """Registered RTL advances the canonical model-scoped RNG once per trial."""
    verilog = compile_to_verilog(
        _escape_rate_neuron(), module_name="sc_escape_rate", data_width=48, fraction=24
    )
    assert "parameter [15:0] RNG_SEED = 16'hace1" in verilog
    assert verilog.count("= _escape_advance(") == 8
    assert "wire [15:0] _escape_sample = _escape_sample_8;" in verilog
    assert "17'd65536" in verilog
    assert "{1'b0, _escape_sample} < _escape_threshold" in verilog
    assert "(RNG_SEED == 16'd0) ? 16'hace1 : RNG_SEED" in verilog


def test_escape_rate_folded_datapath_requires_caller_owned_rng_sample() -> None:
    """A folded population explicitly carries its per-neuron RNG state in BRAM."""
    verilog = compile_to_datapath(
        _escape_rate_neuron(), module_name="sc_escape_rate_pe", data_width=48, fraction=24
    )
    assert "input wire [15:0] rng_sample," in verilog
    assert "{1'b0, rng_sample} < _escape_threshold" in verilog
    assert "assign spike_out = _escape_spike;" in verilog


@pytest.mark.parametrize("folded", [False, True])
def test_escape_rate_emitted_rtl_passes_iverilog_syntax(folded: bool, tmp_path: Path) -> None:
    """Both registered and folded stochastic modules are real Verilog inputs."""
    if shutil.which("iverilog") is None:
        pytest.skip("Icarus Verilog not available")
    compiler = compile_to_datapath if folded else compile_to_verilog
    verilog = compiler(
        _escape_rate_neuron(),
        module_name="sc_escape_rate_syntax",
        data_width=48,
        fraction=24,
    )
    source = tmp_path / "sc_escape_rate_syntax.v"
    source.write_text(verilog, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-tnull", str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_escape_rate_rtl_rejects_unsigned_or_pipelined_contracts() -> None:
    """Unsupported stochastic datapaths fail closed instead of drifting."""
    neuron = _escape_rate_neuron()
    with pytest.raises(NotImplementedError, match="signed"):
        compile_to_verilog(neuron, signed=False)
    with pytest.raises(NotImplementedError, match="pipelining"):
        compile_to_verilog(neuron, pipeline_stages=1)
