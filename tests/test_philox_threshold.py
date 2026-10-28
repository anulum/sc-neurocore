# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — threshold_rng="philox": a stochastic threshold on the counter-based draws, everywhere

"""A Philox stochastic threshold fires when ``rand(STOCHASTIC_SITE) < p`` on every target.

The 16-bit LFSR of ``detection="poisson"`` / ``"escape_rate"`` stays the default
(owner decision 2026-10-19: Philox as an optional mode). With
``threshold_rng="philox"`` the trial reads the neuron's own counter-based draw of
the reserved call site, and the compiler lowers it as the level threshold
``rand(S) < p`` — so the C and Rust kernels, which refuse the LFSR, compile it.
For a probability whose Q-format code is exact, the model, the RTL, the C
kernel and the Rust kernel fire on the same steps; an escape-rate neuron's RTL,
C and Rust agree word for word. Default LFSR emission is unchanged and the
option is refused where it has no meaning.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import pytest

from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.philox_threshold import lower_philox_threshold
from sc_neurocore.compiler.intelligence.bit_true_kernel import generate_bittrue_kernel_from_neuron
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_random import STOCHASTIC_SITE
from sc_neurocore.neurons.equation_refusals import EquationRefusal
from tests.bit_true_cosim_support import (
    HAS_COSIM,
    HAS_RUST,
    _c_trace,
    _rows,
    _rust_main,
    _verilog_trace,
)

_SEED = 2026


def test_lowering_refuses_a_removed_public_probability_expression() -> None:
    """A mutable Philox model cannot compile after its probability expression disappears."""
    neuron = _poisson(0.5)
    neuron.probability_expression = None
    with pytest.raises(ValueError, match="needs its probability or rate expression"):
        lower_philox_threshold(neuron)


def _poisson(probability: float, rng: str = "philox", stream: int = 0) -> EquationNeuron:
    """Return a stateless Poisson neuron firing with ``probability`` per step."""
    return EquationNeuron(
        equations={"v": "0.0"},
        state={"v": 0.0},
        parameters={"p": probability},
        dt=1.0,
        method="map",
        detection="poisson",
        probability_expression="p",
        rng_seed=_SEED,
        random_stream=stream,
        threshold_rng=rng,
    )


def test_the_trial_is_the_site_draw_below_p() -> None:
    """Each step fires exactly when word 0 of the reserved site's draw is below p·2**32."""
    neuron = _poisson(0.3)
    events = [neuron.step(0.0) for _ in range(3000)]
    expected = [
        int(ru.draw(STOCHASTIC_SITE, step, 0, _SEED)[0] / 2**32 < 0.3) for step in range(3000)
    ]
    assert events == expected
    assert 0.27 < sum(events) / 3000 < 0.33


def test_each_neuron_of_a_population_has_its_own_stream() -> None:
    """Streams 0 and 1 give different rasters at the same seed."""
    first = _poisson(0.5, stream=0)
    second = _poisson(0.5, stream=1)
    assert [first.step(0.0) for _ in range(64)] != [second.step(0.0) for _ in range(64)]


def test_the_option_is_refused_where_it_means_nothing() -> None:
    """A level threshold, or an unknown generator, refuses threshold_rng."""
    with pytest.raises(EquationRefusal, match="needs a stochastic threshold"):
        EquationNeuron(
            equations={"v": "-v"}, state={"v": 0.0}, threshold="v > 1", threshold_rng="philox"
        )
    with pytest.raises(EquationRefusal, match="threshold_rng must be"):
        _poisson(0.5, rng="mt19937")


def test_the_lfsr_default_is_unchanged() -> None:
    """The default emission still has the LFSR; Philox has none and draws at the site."""
    default = compile_to_verilog(_poisson(0.5, rng="lfsr16"), "m", 16, 8)
    philox = compile_to_verilog(_poisson(0.5), "m", 16, 8)
    assert "_escape_lfsr" in default and f"call site {STOCHASTIC_SITE}" not in default
    assert "_escape_lfsr" not in philox and f"call site {STOCHASTIC_SITE}" in philox


def _rust_trace(
    neuron: EquationNeuron, steps: int, width: int, fraction: int, tmp: Path
) -> list[list[str]]:
    """Return the Rust kernel's trace (a drawing kernel's state also holds ``rng_step``)."""
    kernel = generate_bittrue_kernel_from_neuron(
        neuron, "m", data_width=width, fraction=fraction, language="rust"
    )
    main = _rust_main(neuron, "m", 0, steps, width).replace(
        "spike_out: 0", "spike_out: 0, rng_step: 0", 1
    )
    (tmp / "m.rs").write_text("#![allow(warnings)]\n" + kernel + "\n" + main)
    subprocess.run(
        ["rustc", "-C", "overflow-checks=on", "-o", str(tmp / "r"), str(tmp / "m.rs")],
        check=True,
        capture_output=True,
        timeout=300,
    )
    return _rows(subprocess.run([str(tmp / "r")], capture_output=True, text=True).stdout)


def _targets(
    make: Callable[[], EquationNeuron], steps: int, width: int, fraction: int
) -> list[list[list[str]]]:
    """Return the RTL, C and (with rustc) Rust traces of fresh neurons."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        traces = [
            _verilog_trace(make(), "m", 0.0, steps, width, fraction, tmp),
            _c_trace(make(), "m", 0, steps, width, fraction, tmp),
        ]
        if HAS_RUST:
            traces.append(_rust_trace(make(), steps, width, fraction, tmp))
    return traces


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
@pytest.mark.parametrize(
    ("probability", "width", "fraction"),
    [(0.25, 16, 8), (0.5, 16, 8), (0.875, 16, 15), (0.0, 16, 8), (1.0, 16, 8)],
)
def test_model_rtl_c_and_rust_fire_on_the_same_steps(
    probability: float, width: int, fraction: int
) -> None:
    """A probability with an exact code: identical rasters on all four targets."""
    steps = 96
    model = _poisson(probability)
    expected = [str(model.step(0.0)) for _ in range(steps)]
    for trace in _targets(lambda: _poisson(probability), steps, width, fraction):
        assert [row[0] for row in trace] == expected


def _escape() -> EquationNeuron:
    """Return a driven escape-rate neuron with a state-dependent rate and a reset."""
    return EquationNeuron(
        equations={"v": "-v + 1.5"},
        state={"v": 0.0},
        dt=0.25,
        detection="escape_rate",
        rate_expression="0.5 + v",
        reset={"v": "0.0"},
        rng_seed=_SEED,
        threshold_rng="philox",
    )


@pytest.mark.skipif(not HAS_COSIM, reason="Icarus Verilog / gcc not available")
def test_an_escape_rate_neuron_is_equal_on_rtl_c_and_rust() -> None:
    """RTL = C = Rust word for word, and it fires (the rate is lowered through exp)."""
    traces = _targets(_escape, 80, 16, 8)
    assert all(trace == traces[0] for trace in traces)
    assert any(row[0] == "1" for row in traces[0])
