# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pre-synthesis ASIC resource estimation

"""Estimate pre-synthesis gate count, area, power, and timing."""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.asic_flow.pdk import PDKConfig


@dataclass
class DesignEstimate:
    """Uncalibrated pre-synthesis screening estimate for an SC module."""

    module_name: str
    gate_count: int
    area_um2: float
    dynamic_power_mw: float
    leakage_power_mw: float
    critical_path_ns: float
    max_frequency_mhz: float


class PreSynthEstimator:
    """Compute deterministic screening values before synthesis.

    The legacy coefficients are architectural scaling assumptions, not
    foundry-characterised PPA models:

    - Bitstream ops: ~10 gates/bit
    - LIF neuron: ~500 gates
    - STDP synapse: ~200 gates
    - AER router: ~100 gates/port

    Outputs support relative design screening only. They are not physical
    evidence and must not be presented as post-synthesis or signoff results.
    """

    GATES_PER_BIT = 10
    GATES_PER_LIF = 500
    GATES_PER_SYNAPSE = 200
    GATES_PER_AER_PORT = 100

    @classmethod
    def estimate(
        cls,
        n_neurons: int,
        n_synapses: int,
        bitstream_width: int,
        n_aer_ports: int,
        pdk: PDKConfig,
    ) -> DesignEstimate:
        """Estimate design metrics from architectural parameters."""
        gates = (
            n_neurons * cls.GATES_PER_LIF
            + n_synapses * cls.GATES_PER_SYNAPSE
            + bitstream_width * cls.GATES_PER_BIT
            + n_aer_ports * cls.GATES_PER_AER_PORT
        )

        # Legacy screening scale: 1 µm²/gate at 130 nm, quadratic by feature size.
        scale = (pdk.min_feature_nm / 130.0) ** 2
        area = gates * 1.0 * scale

        # Legacy screening scales: 1 µW/gate dynamic and 0.01 µW/gate leakage.
        freq_scale = 100.0 / (1000.0 / pdk.clock_period_ns)
        v_scale = (pdk.voltage_v / 1.8) ** 2
        dynamic = gates * 1e-3 * freq_scale * v_scale  # mW
        leakage = gates * 1e-5 * scale  # mW

        # Legacy critical-path screening scale, normalised to the 130 nm preset.
        cp = max(1.0, 10 + 0.01 * n_neurons) * (pdk.min_feature_nm / 130.0)
        max_freq = 1000.0 / cp

        return DesignEstimate(
            module_name="sc_neurocore_top",
            gate_count=gates,
            area_um2=area,
            dynamic_power_mw=dynamic,
            leakage_power_mw=leakage,
            critical_path_ns=cp,
            max_frequency_mhz=max_freq,
        )
