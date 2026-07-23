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
from tests.cosim_reference_connor_stevens import (
    _connor_stevens_hand_spike_count as _connor_stevens_hand_spike_count,
    _connor_stevens_macrostep_rk4_features as _connor_stevens_macrostep_rk4_features,
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
    _fitzhugh_nagumo_substep_neuron as _fitzhugh_nagumo_substep_neuron,
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
from tests.cosim_reference_wang_buzsaki import (
    _wang_buzsaki_hand_spike_count as _wang_buzsaki_hand_spike_count,
    _wang_buzsaki_macrostep_gauss_seidel_features as _wang_buzsaki_macrostep_gauss_seidel_features,
)


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
