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
    _adex_published_euler_trace as _adex_published_euler_trace,
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
from tests.cosim_reference_exponential_relaxation import (
    _closed_form_features as _closed_form_features,
)
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
from tests.cosim_reference_lif import _lif_schema_precision_values as _lif_schema_precision_values
from tests.cosim_reference_mckean import (
    _MCKEAN_PARAMS as _MCKEAN_PARAMS,
    _mckean_hand_spike_count as _mckean_hand_spike_count,
    _mckean_rk4_features as _mckean_rk4_features,
)
from tests.cosim_reference_mihalas_niebur import (
    _features as _features,
    _MIHALAS_NIEBUR_PARAMS as _MIHALAS_NIEBUR_PARAMS,
    _mihalas_niebur_driven_rk4_features as _mihalas_niebur_driven_rk4_features,
    _mihalas_niebur_hand_spike_count as _mihalas_niebur_hand_spike_count,
    _rk4 as _rk4,
    _sc_scaled_reset_driven_rk4_features as _sc_scaled_reset_driven_rk4_features,
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
    _sc_resetting_wilson_hr_rk4_features as _sc_resetting_wilson_hr_rk4_features,
    _wilson_hr_hand_spike_count as _wilson_hr_hand_spike_count,
    _wilson_hr_rk4_features as _wilson_hr_rk4_features,
)
from tests.cosim_reference_wang_buzsaki import (
    _wang_buzsaki_hand_spike_count as _wang_buzsaki_hand_spike_count,
    _wang_buzsaki_macrostep_gauss_seidel_features as _wang_buzsaki_macrostep_gauss_seidel_features,
)
from tests.cosim_rtl_spike_execution import (
    _neuron_verilog_spike_count_q1616 as _neuron_verilog_spike_count_q1616,
    _verilog_compiles as _verilog_compiles,
    _verilog_spike_count_generic as _verilog_spike_count_generic,
    _verilog_spike_count_q1616 as _verilog_spike_count_q1616,
    _verilog_spike_count_q412 as _verilog_spike_count_q412,
    compile_to_verilog as compile_to_verilog,
)
