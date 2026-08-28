# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cosim_support_architecture.py

from __future__ import annotations


"""Pin generic simulator ownership outside the legacy model-reference surface."""


import ast


from pathlib import Path


from tests import (
    cosim_reference_adex,
    cosim_reference_conductance_rates,
    cosim_reference_connor_stevens,
    cosim_reference_dpi_neuron,
    cosim_reference_exp_if,
    cosim_reference_exponential_relaxation,
    cosim_reference_fitzhugh_nagumo,
    cosim_reference_fitzhugh_rinzel,
    cosim_reference_glif,
    cosim_reference_hindmarsh_rose,
    cosim_reference_hodgkin_huxley,
    cosim_reference_ibarz_tanaka,
    cosim_reference_izhikevich2007,
    cosim_reference_izhikevich_rs,
    cosim_reference_lif,
    cosim_reference_mckean,
    cosim_reference_mihalas_niebur,
    cosim_reference_morris_lecar,
    cosim_reference_perfect_integrator,
    cosim_reference_pernarowski,
    cosim_reference_quadratic_if,
    cosim_reference_rulkov_map,
    cosim_reference_statistics,
    cosim_reference_terman_wang,
    cosim_reference_theta,
    cosim_reference_wang_buzsaki,
    cosim_reference_wilson_hr,
    cosim_rtl_spike_execution,
    cosim_runtime,
    cosim_support,
)


_RTL_SPIKE_EXECUTION_NAMES = (
    "_neuron_verilog_spike_count_q1616",
    "_verilog_compiles",
    "_verilog_spike_count_generic",
    "_verilog_spike_count_q1616",
    "_verilog_spike_count_q412",
)


_RUNTIME_NAMES = (
    "HAS_IVERILOG",
    "_python_spike_count",
    "_verilog_spike_count",
    "simulate",
    "spike_count_method",
    "verilog_spike_count_method",
    "verilog_spike_count_method_pipelined",
)


_ADEX_NAMES = ("_adex_subthreshold_euler_features",)


_CONDUCTANCE_RATE_NAMES = (
    "_np_exp",
    "_reference_exprel",
)


_CONNOR_STEVENS_NAMES = (
    "_connor_stevens_hand_spike_count",
    "_connor_stevens_macrostep_rk4_features",
)


_DPI_NEURON_NAMES = (
    "_dpi_neuron_driven_euler_features",
    "_dpi_neuron_hand_spike_count",
    "_dpi_neuron_verilog_q1616_trace",
)


_EXP_IF_NAMES = ("_exp_if_rk4_features",)


_EXPONENTIAL_RELAXATION_NAMES = ("_closed_form_features",)


_FITZHUGH_NAGUMO_NAMES = (
    "_fitzhugh_nagumo_hand_spike_count",
    "_fitzhugh_nagumo_rk4_features",
    "_fitzhugh_nagumo_substep_neuron",
)


_FITZHUGH_RINZEL_NAMES = (
    "_fitzhugh_rinzel_hand_spike_count",
    "_fitzhugh_rinzel_rk4_features",
)


_GLIF_NAMES = (
    "_glif_driven_rk4_features",
    "_glif_hand_spike_count",
)


_HINDMARSH_ROSE_NAMES = (
    "_hindmarsh_rose_hand_spike_count",
    "_hindmarsh_rose_rk4_features",
)


_HODGKIN_HUXLEY_NAMES = (
    "_hodgkin_huxley_hand_spike_count",
    "_hodgkin_huxley_macrostep_rk4_features",
)


_IBARZ_TANAKA_NAMES = ("_ibarz_tanaka_verilog_q1616_trace",)


_IZHIKEVICH2007_NAMES = (
    "_izhikevich2007_euler_features",
    "_izhikevich2007_hand_euler_spike_count",
)


_IZHIKEVICH_RS_NAMES = ("_izhikevich_rs_euler_features",)


_LIF_NAMES = ("_lif_schema_precision_values",)


_MCKEAN_NAMES = (
    "_mckean_hand_spike_count",
    "_mckean_rk4_features",
)


_MIHALAS_NIEBUR_NAMES = (
    "_features",
    "_mihalas_niebur_driven_rk4_features",
    "_mihalas_niebur_hand_spike_count",
    "_rk4",
    "_sc_scaled_reset_driven_rk4_features",
)


_MORRIS_LECAR_NAMES = (
    "_morris_lecar_hand_spike_count",
    "_morris_lecar_rk4_features",
)


_PERFECT_INTEGRATOR_NAMES = (
    "_perfect_integrator_hand_spike_count",
    "_perfect_integrator_sawtooth_features",
)


_PERNAROWSKI_NAMES = (
    "_pernarowski_hand_spike_count",
    "_pernarowski_rk4_features",
)


_QUADRATIC_IF_NAMES = ("_quadratic_if_zero_current_features",)


_RULKOV_MAP_NAMES = (
    "_rulkov_map_features",
    "_rulkov_map_verilog_q1616_trace",
)


_THETA_NAMES = ("_theta_constant_current_features",)


_TERMAN_WANG_NAMES = (
    "_terman_wang_hand_spike_count",
    "_terman_wang_rk4_features",
)


_WANG_BUZSAKI_NAMES = (
    "_wang_buzsaki_hand_spike_count",
    "_wang_buzsaki_macrostep_gauss_seidel_features",
)


_WILSON_HR_NAMES = (
    "_sc_resetting_wilson_hr_rk4_features",
    "_wilson_hr_hand_spike_count",
    "_wilson_hr_rk4_features",
)


__all__ = [
    "ast",
    "Path",
    "cosim_reference_adex",
    "cosim_reference_conductance_rates",
    "cosim_reference_connor_stevens",
    "cosim_reference_dpi_neuron",
    "cosim_reference_exp_if",
    "cosim_reference_exponential_relaxation",
    "cosim_reference_fitzhugh_nagumo",
    "cosim_reference_fitzhugh_rinzel",
    "cosim_reference_glif",
    "cosim_reference_hindmarsh_rose",
    "cosim_reference_hodgkin_huxley",
    "cosim_reference_ibarz_tanaka",
    "cosim_reference_izhikevich2007",
    "cosim_reference_izhikevich_rs",
    "cosim_reference_lif",
    "cosim_reference_mckean",
    "cosim_reference_mihalas_niebur",
    "cosim_reference_morris_lecar",
    "cosim_reference_perfect_integrator",
    "cosim_reference_pernarowski",
    "cosim_reference_quadratic_if",
    "cosim_reference_rulkov_map",
    "cosim_reference_statistics",
    "cosim_reference_terman_wang",
    "cosim_reference_theta",
    "cosim_reference_wang_buzsaki",
    "cosim_reference_wilson_hr",
    "cosim_rtl_spike_execution",
    "cosim_runtime",
    "cosim_support",
    "_RTL_SPIKE_EXECUTION_NAMES",
    "_RUNTIME_NAMES",
    "_ADEX_NAMES",
    "_CONDUCTANCE_RATE_NAMES",
    "_CONNOR_STEVENS_NAMES",
    "_DPI_NEURON_NAMES",
    "_EXP_IF_NAMES",
    "_EXPONENTIAL_RELAXATION_NAMES",
    "_FITZHUGH_NAGUMO_NAMES",
    "_FITZHUGH_RINZEL_NAMES",
    "_GLIF_NAMES",
    "_HINDMARSH_ROSE_NAMES",
    "_HODGKIN_HUXLEY_NAMES",
    "_IBARZ_TANAKA_NAMES",
    "_IZHIKEVICH2007_NAMES",
    "_IZHIKEVICH_RS_NAMES",
    "_LIF_NAMES",
    "_MCKEAN_NAMES",
    "_MIHALAS_NIEBUR_NAMES",
    "_MORRIS_LECAR_NAMES",
    "_PERFECT_INTEGRATOR_NAMES",
    "_PERNAROWSKI_NAMES",
    "_QUADRATIC_IF_NAMES",
    "_RULKOV_MAP_NAMES",
    "_THETA_NAMES",
    "_TERMAN_WANG_NAMES",
    "_WANG_BUZSAKI_NAMES",
    "_WILSON_HR_NAMES",
]
