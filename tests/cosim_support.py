# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — shared Python↔Verilog co-simulation primitives

"""Compatibility surface for Python↔Verilog co-simulation references.

Generic Icarus/VVP process execution lives in :mod:`tests.cosim_runtime`.
The remaining model-specific reference and trace helpers retain their historical
imports here while they are decomposed into one-model owners.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, cast

from sc_neurocore.compiler.equation_compiler import generate_testbench
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog as compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron
from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

from tests.cosim_runtime import (
    HAS_IVERILOG as HAS_IVERILOG,
    _python_spike_count as _python_spike_count,
    _verilog_spike_count as _verilog_spike_count,
    simulate as simulate,
    spike_count_method as spike_count_method,
    verilog_spike_count_method as verilog_spike_count_method,
    verilog_spike_count_method_pipelined as verilog_spike_count_method_pipelined,
)

from tests.cosim_reference_adex import (
    _adex_subthreshold_euler_features as _adex_subthreshold_euler_features,
)
from tests.cosim_reference_conductance_rates import (
    _np_exp as _np_exp,
    _reference_exprel as _reference_exprel,
)
from tests.cosim_reference_dpi_neuron import (
    _dpi_neuron_driven_euler_features as _dpi_neuron_driven_euler_features,
    _dpi_neuron_hand_spike_count as _dpi_neuron_hand_spike_count,
    _dpi_neuron_verilog_q1616_trace as _dpi_neuron_verilog_q1616_trace,
)
from tests.cosim_reference_exp_if import _exp_if_rk4_features as _exp_if_rk4_features
from tests.cosim_reference_fitzhugh_nagumo import (
    _fitzhugh_nagumo_hand_spike_count as _fitzhugh_nagumo_hand_spike_count,
    _fitzhugh_nagumo_rk4_features as _fitzhugh_nagumo_rk4_features,
)
from tests.cosim_reference_fitzhugh_rinzel import (
    _fitzhugh_rinzel_hand_spike_count as _fitzhugh_rinzel_hand_spike_count,
    _fitzhugh_rinzel_rk4_features as _fitzhugh_rinzel_rk4_features,
)
from tests.cosim_reference_glif import (
    _glif_driven_rk4_features as _glif_driven_rk4_features,
    _glif_hand_spike_count as _glif_hand_spike_count,
)
from tests.cosim_reference_hindmarsh_rose import (
    _hindmarsh_rose_hand_spike_count as _hindmarsh_rose_hand_spike_count,
    _hindmarsh_rose_rk4_features as _hindmarsh_rose_rk4_features,
)
from tests.cosim_reference_hodgkin_huxley import (
    _hodgkin_huxley_hand_spike_count as _hodgkin_huxley_hand_spike_count,
    _hodgkin_huxley_macrostep_rk4_features as _hodgkin_huxley_macrostep_rk4_features,
)
from tests.cosim_reference_ibarz_tanaka import (
    _ibarz_tanaka_verilog_q1616_trace as _ibarz_tanaka_verilog_q1616_trace,
)
from tests.cosim_reference_izhikevich2007 import (
    _izhikevich2007_euler_features as _izhikevich2007_euler_features,
    _izhikevich2007_hand_euler_spike_count as _izhikevich2007_hand_euler_spike_count,
)
from tests.cosim_reference_izhikevich_rs import (
    _izhikevich_rs_euler_features as _izhikevich_rs_euler_features,
)
from tests.cosim_reference_mckean import (
    _MCKEAN_PARAMS as _MCKEAN_PARAMS,
    _mckean_hand_spike_count as _mckean_hand_spike_count,
    _mckean_rk4_features as _mckean_rk4_features,
)
from tests.cosim_reference_mihalas_niebur import (
    _MIHALAS_NIEBUR_PARAMS as _MIHALAS_NIEBUR_PARAMS,
    _mihalas_niebur_driven_rk4_features as _mihalas_niebur_driven_rk4_features,
    _mihalas_niebur_hand_spike_count as _mihalas_niebur_hand_spike_count,
)
from tests.cosim_reference_morris_lecar import (
    _morris_lecar_hand_spike_count as _morris_lecar_hand_spike_count,
    _morris_lecar_rk4_features as _morris_lecar_rk4_features,
)
from tests.cosim_reference_perfect_integrator import (
    _perfect_integrator_hand_spike_count as _perfect_integrator_hand_spike_count,
    _perfect_integrator_sawtooth_features as _perfect_integrator_sawtooth_features,
)
from tests.cosim_reference_pernarowski import (
    _pernarowski_hand_spike_count as _pernarowski_hand_spike_count,
    _pernarowski_rk4_features as _pernarowski_rk4_features,
)
from tests.cosim_reference_quadratic_if import (
    _quadratic_if_zero_current_features as _quadratic_if_zero_current_features,
)
from tests.cosim_reference_rulkov_map import (
    _rulkov_map_features as _rulkov_map_features,
    _rulkov_map_verilog_q1616_trace as _rulkov_map_verilog_q1616_trace,
)
from tests.cosim_reference_statistics import _summarise as _summarise
from tests.cosim_reference_terman_wang import (
    _terman_wang_hand_spike_count as _terman_wang_hand_spike_count,
    _terman_wang_rk4_features as _terman_wang_rk4_features,
)
from tests.cosim_reference_theta import (
    _theta_constant_current_features as _theta_constant_current_features,
)
from tests.cosim_reference_wilson_hr import (
    _wilson_hr_hand_spike_count as _wilson_hr_hand_spike_count,
    _wilson_hr_rk4_features as _wilson_hr_rk4_features,
)


def _connor_stevens_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Connor-Stevens macro-step (RK4, crossing) spike count.

    The maintained ``ConnorStevensNeuron.step`` is a 1 ms macro step of 100 inner
    four-stage RK4 sub-steps (``dt=0.01``) with a rising-edge ``v >= v_threshold`` crossing
    on the macro boundary and no reset. The bundled ``connor_stevens`` schema mirrors this
    exactly (``method="rk4"``, ``substeps=100``, ``detection="crossing"``), so one hand
    ``step()`` corresponds to one schema macro ``step()``.
    """
    neuron = ConnorStevensNeuron()
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _wang_buzsaki_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Wang-Buzsaki macro-step (Gauss-Seidel, crossing) spike count.

    ``WangBuzsakiNeuron.step`` is a 0.5 ms macro step of 50 inner sub-steps (``dt=0.01``)
    advanced sequentially (the gating variables ``h``/``n`` from the old voltage, then the
    membrane voltage ``v`` from the new gates), with a rising-edge ``v >= v_threshold``
    crossing on the macro boundary and no reset. The bundled ``wang_buzsaki`` schema mirrors
    that path exactly (``method="gauss_seidel"``, ``substeps=50``, state ordered ``h, n, v``,
    ``detection="crossing"``), so one hand ``step()`` corresponds to one schema macro
    ``step()``. The neuron is constructed once so the state accumulates across the train.
    """
    neuron = WangBuzsakiNeuron()
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _lif_schema_precision_values() -> dict[str, float]:
    """Return LIF schema values checked by the public precision CLI."""
    schema = UniversalNeuron.from_schema("lif").schema
    parameters = cast(Mapping[str, float], schema.get("parameters", {}))
    state = cast(Mapping[str, float], schema.get("state", {}))
    return {**parameters, **state}


def _verilog_spike_count_q412(model_name: str, n_steps: int, current: float) -> int:
    """Compile at Q4.12 precision and simulate, returning spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_q412"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=16,
        fraction=12,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=16,
        fraction=12,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _verilog_spike_count_q1616(model_name: str, n_steps: int, current: float) -> int:
    """Compile at Q16.16 precision (32-bit) and simulate, returning spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_q1616"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=32,
        fraction=16,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _neuron_verilog_spike_count_q1616(
    neuron: EquationNeuron, n_steps: int, current: float, module_name: str
) -> int:
    """Compile a raw ``EquationNeuron`` to Q16.16 RTL, simulate, return the spike count.

    Unlike :func:`_verilog_spike_count_q1616` this takes a constructed neuron directly (not a
    bundled schema name), so it can co-simulate an in-test configuration such as an artificial
    sub-step count on a polynomial oscillator.
    """
    verilog = compile_to_verilog(neuron, module_name=module_name, data_width=32, fraction=16)
    tb = generate_testbench(
        neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"
        rtl_path.write_text(verilog)
        tb_path.write_text(tb)
        compile_result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{compile_result.stderr}")
        run_result = subprocess.run(
            ["vvp", str(out_path)], capture_output=True, text=True, timeout=60
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{run_result.stderr}")
        match = re.search(r"(\d+) spikes", run_result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{run_result.stdout}")
        return int(match.group(1))


def _fitzhugh_nagumo_substep_neuron(substeps: int) -> EquationNeuron:
    """Build the faithful FitzHugh-Nagumo oscillator with an artificial sub-step count.

    FitzHugh-Nagumo is polynomial, so its Q16.16 datapath is bit-exact against float64; giving
    it ``substeps`` inner steps lets the macro-step lowering be validated on a model whose only
    residual would be a logic error (no look-up-table quantisation to confound the comparison).
    """
    return EquationNeuron(
        equations={
            "v": "v - v * v * v / 3.0 - w + I",
            "w": "epsilon * (v + a - b * w)",
        },
        parameters={"a": 0.7, "b": 0.8, "epsilon": 0.08, "v_threshold": 1.0},
        state={"v": -1.0, "w": -0.5},
        threshold="v >= v_threshold",
        dt=0.1,
        method="rk4",
        detection="crossing",
        substeps=substeps,
    )


def _verilog_spike_count_generic(
    model_name: str,
    n_steps: int,
    current: float,
    data_width: int,
    fraction: int,
) -> int:
    """Compile at arbitrary (data_width, fraction) and simulate, returning spike count.

    This is the universal co-simulation helper — all precision-specific
    helpers (_verilog_spike_count, _verilog_spike_count_q412, etc.) are
    special cases of this function.
    """
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    mode_tag = f"q{data_width - fraction}_{fraction}"
    module_name = f"sc_{model_name}_{mode_tag}"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=data_width,
        fraction=fraction,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _verilog_compiles(model_name: str) -> bool:
    """Return whether a model's generated Verilog is accepted by iverilog."""
    neuron = UniversalNeuron.from_schema(model_name)
    module_name = f"sc_{model_name}"
    verilog = neuron.to_verilog(module_name=module_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        out_path = Path(tmpdir) / f"{module_name}.out"
        rtl_path.write_text(verilog)
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0


def _closed_form_features(
    *,
    initial: float,
    steady: float,
    tau: float,
    dt: float,
    steps: int,
) -> dict[str, float]:
    values = [
        steady + (initial - steady) * math.exp(-(step * dt) / tau) for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _connor_stevens_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Connor-Stevens oscillator.

    The Connor-Stevens (1971) A-current model is the faithful representation of the
    maintained ``ConnorStevensNeuron`` (RK4, sub-stepped): each macro step advances
    ``substeps`` inner four-stage classical RK4 sub-steps of ``dt``, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary — matching the hand model's 100-sub-step-per-millisecond
    macro step and ``EquationNeuron.step``'s macro crossing, with **no reset**. The
    six-state membrane and Na/K/A-type gating rate functions are transcribed verbatim from
    the schema, reusing :func:`_np_exp` and :func:`_reference_exprel` (and the cube-root
    ``a``-gate) so the recurrence reproduces the schema runner bit-for-bit. The reference is
    an independent re-derivation of the committed driven-spiking trace, not a copy of the
    runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner RK4 sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``m``, ``h``, ``n``, ``a``, and ``b``
        state variables plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 20.0
    g_a = 47.7
    g_l = 0.3
    e_na = 55.0
    e_k = -72.0
    e_a = -75.0
    e_l = -17.0
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": [], "a": [], "b": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n, a, b = sv
        dv = (
            -g_na * m**3 * h * (v - e_na)
            - g_k * n**4 * (v - e_k)
            - g_a * a**3 * b * (v - e_a)
            - g_l * (v - e_l)
            + current
        ) / c_m
        dm = (
            3.8 / _reference_exprel(-(v + 29.7) / 10) * (1 - m)
            - 15.2 * _np_exp(-(v + 54.7) / 18) * m
        )
        dh = 0.266 * _np_exp(-(v + 48) / 20) * (1 - h) - 3.8 / (1 + _np_exp(-(v + 18) / 10)) * h
        dn = (
            0.2 / _reference_exprel(-(v + 45.7) / 10) * (1 - n)
            - 0.25 * _np_exp(-(v + 55.7) / 80) * n
        )
        da = (
            (0.0761 * _np_exp((v + 94.22) / 31.84) / (1 + _np_exp((v + 1.17) / 28.93)))
            ** (1.0 / 3.0)
            - a
        ) / (0.3632 + 1.158 / (1 + _np_exp((v + 55.96) / 20.12)))
        db = (1 / (1 + _np_exp((v + 53.3) / 14.54)) ** 4 - b) / (
            1.24 + 2.678 / (1 + _np_exp((v + 50) / 16.027))
        )
        return dv, dm, dh, dn, da, db

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(6))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(6))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(6))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(6))

    state: tuple[float, ...] = (-68.0, 0.01, 0.99, 0.1, 0.5, 0.1)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n", "a", "b")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)


def _wang_buzsaki_macrostep_gauss_seidel_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step Gauss-Seidel features for the driven Wang-Buzsaki oscillator.

    The Wang-Buzsaki (1996) fast-spiking interneuron is the faithful representation of the
    maintained ``WangBuzsakiNeuron``: each macro step advances ``substeps`` inner sequential
    (Gauss-Seidel) forward-Euler sub-steps of ``dt`` — the gating variables ``h`` and ``n``
    are updated from the old voltage first, then the membrane voltage ``v`` from the
    already-updated gates (the schema declares ``method="gauss_seidel"`` with state ordered
    ``h, n, v``). Sodium activation is instantaneous: ``m_inf = alpha_m/(alpha_m+beta_m)``
    with ``alpha_m = 1/exprel(-(v+35)/10)`` (the exprel rewrite of ``0.1*(v+35)/(1-exp(...))``)
    and ``beta_m = 4*exp(-(v+60)/18)``; the potassium rate ``alpha_n`` is likewise
    ``0.1/exprel(-(v+34)/10)``. The rising-edge ``v >= v_threshold`` crossing is evaluated
    only on the macro boundary against the condition at the previous macro boundary, with
    **no reset**. The rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` so the recurrence reproduces the schema
    runner bit-for-bit. The reference is an independent re-derivation of the committed
    driven-spiking trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner Gauss-Seidel sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``h``, ``n``, and ``v`` state variables plus
        spike-count and first-spike-step features.
    """
    phi = 5.0
    g_na = 35.0
    g_k = 9.0
    g_l = 0.1
    e_na = 55.0
    e_k = -90.0
    e_l = -65.0
    capacitance = 1.0
    v_threshold = -20.0
    h = 0.8
    n = 0.1
    v = -65.0
    recorded: dict[str, list[float]] = {"h": [], "n": [], "v": []}
    spikes: list[int] = []
    for _ in range(steps):
        v_prev = v
        for _ in range(substeps):
            # ``h`` (declared first): reads the old voltage and old ``h``.
            h = (
                h
                + phi
                * (0.07 * _np_exp(-(v + 58) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 28) / 10)) * h)
                * dt
            )
            # ``n`` (declared second): reads the old voltage and old ``n``.
            n = (
                n
                + phi
                * (
                    0.1 / _reference_exprel(-(v + 34) / 10) * (1 - n)
                    - 0.125 * _np_exp(-(v + 44) / 80) * n
                )
                * dt
            )
            # ``v`` (declared last): reads the already-updated ``h``/``n`` and old ``v``.
            inv_exprel = 1 / _reference_exprel(-(v + 35) / 10)
            m_inf = inv_exprel / (inv_exprel + 4 * _np_exp(-(v + 60) / 18))
            v = (
                v
                + (
                    -g_na * m_inf**3 * h * (v - e_na)
                    - g_k * n**4 * (v - e_k)
                    - g_l * (v - e_l)
                    + current
                )
                / capacitance
                * dt
            )
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        recorded["h"].append(h)
        recorded["n"].append(n)
        recorded["v"].append(v)

    return _summarise(recorded, spikes)
