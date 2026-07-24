# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (mode_rejects) from former test_verilog_compiler_contracts.py

from __future__ import annotations

from tests.verilog_compiler_contracts_support import *  # noqa: F403

def test_compile_to_verilog_rejects_unknown_overflow_mode() -> None:
    """The registered compiler rejects unsupported overflow policies."""
    neuron = _lif_without_threshold()

    with pytest.raises(ValueError, match="Unknown overflow mode"):
        compile_to_verilog(neuron, overflow="explode")


def test_public_compilers_reject_unknown_rounding_mode() -> None:
    """Both RTL surfaces validate rounding even when an equation has no multiply."""
    neuron = _lif_without_threshold()

    with pytest.raises(ValueError, match="Unknown rounding mode"):
        compile_to_verilog(neuron, rounding="dither-supreme")
    with pytest.raises(ValueError, match="Unknown rounding mode"):
        compile_to_datapath(neuron, rounding="dither-supreme")


def test_public_compilers_reject_unwired_stochastic_rounding() -> None:
    """Neither public RTL surface may emit references to an undeclared rounding LFSR."""
    neuron = _lif_without_threshold()

    with pytest.raises(NotImplementedError, match="no rounding LFSR"):
        compile_to_verilog(neuron, rounding="stochastic")
    with pytest.raises(NotImplementedError, match="no rounding LFSR"):
        compile_to_datapath(neuron, rounding="stochastic")


@pytest.mark.parametrize("pipeline_stages", [True, -1])
def test_registered_compiler_rejects_invalid_pipeline_stage_counts(
    pipeline_stages: int,
) -> None:
    """Boolean and negative pipeline counts fail instead of changing mode silently."""
    error = TypeError if pipeline_stages is True else ValueError

    with pytest.raises(error, match="pipeline_stages"):
        compile_to_verilog(_lif_without_threshold(), pipeline_stages=pipeline_stages)


def test_registered_compiler_validates_explicit_pipeline_points() -> None:
    """Explicit pipeline points require a unique string list and no global mode."""
    neuron = _lif_without_threshold()

    with pytest.raises(TypeError, match="list of strings"):
        compile_to_verilog(neuron, pipeline_points=cast(list[str], ("mul0",)))
    with pytest.raises(TypeError, match="entries must all be strings"):
        compile_to_verilog(neuron, pipeline_points=[cast(str, 1)])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        compile_to_verilog(neuron, pipeline_points=["mul0", "mul0"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        compile_to_verilog(neuron, pipeline_stages=1, pipeline_points=["mul0"])


def test_registered_and_folded_compilers_reject_unsigned_formats() -> None:
    """Both public emitters fail closed instead of producing mixed-signed RTL."""
    neuron = _lif_without_threshold()

    with pytest.raises(NotImplementedError, match="unsigned equation-to-Verilog"):
        compile_to_verilog(neuron, signed=False)
    with pytest.raises(NotImplementedError, match="unsigned equation-to-Verilog"):
        compile_to_datapath(neuron, signed=False)
