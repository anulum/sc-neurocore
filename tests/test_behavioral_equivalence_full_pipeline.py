# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFullPipeline from former test_behavioral_equivalence.py

"""Focused suite: TestFullPipeline from former test_behavioral_equivalence.py."""

from __future__ import annotations

from tests.behavioral_equivalence_support import *  # noqa: F403


class TestFullPipeline:
    """End-to-end stochastic pipeline matching HDL architecture."""

    def _run_pipeline(self, x_values, w_values, n_steps=512, leak_k=10, gain_k=256):
        """
        Software bit-true simulation of sc_dense_layer_core pipeline:
          input encoders -> weight encoders -> AND synapses -> popcount -> LIF
        """
        n_inputs = len(x_values)
        assert len(w_values) == n_inputs

        # Create decorrelated encoders (matching HDL SEED_INIT values)
        input_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1 + i * 7) for i in range(n_inputs)]
        weight_encs = [
            FixedPointBitstreamEncoder(seed_init=0xBEEF + i * 13) for i in range(n_inputs)
        ]

        neuron = FixedPointLIFNeuron(refractory_period=0)
        spikes = []

        for t in range(n_steps):
            # Encode
            pre_bits = [enc.step(x) for enc, x in zip(input_encs, x_values)]
            w_bits = [enc.step(w) for enc, w in zip(weight_encs, w_values)]

            # AND synapses
            post_bits = [p & w for p, w in zip(pre_bits, w_bits)]

            # Dot-product -> current (matching sc_dotproduct_to_current.v)
            count = sum(post_bits)
            # Map to fixed-point current range [y_min=0, y_max=256 (=1.0)]
            y_min = 0
            y_max = 256
            if n_inputs > 0:
                I_t = y_min + ((y_max - y_min) * count) // n_inputs
            else:
                I_t = 0

            spike, v = neuron.step(leak_k, gain_k, I_t, noise_in=0)
            spikes.append(spike)

        return spikes

    def test_pipeline_produces_spikes(self):
        """High input + high weight should produce spikes."""
        # x ~ 0.8, w ~ 0.8 -> post ~ 0.64 -> strong current
        x_vals = [52428, 52428, 52428]  # ~0.8 * 65535
        w_vals = [52428, 52428, 52428]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=1000)
        assert sum(spikes) > 0, "Pipeline should produce spikes with high input"

    def test_pipeline_no_spikes_zero_input(self):
        """Zero input should produce no spikes."""
        x_vals = [0, 0, 0]
        w_vals = [52428, 52428, 52428]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=500)
        assert sum(spikes) == 0, "Zero input should produce no spikes"

    def test_pipeline_no_spikes_zero_weight(self):
        """Zero weights should produce no spikes regardless of input."""
        x_vals = [65535, 65535, 65535]
        w_vals = [0, 0, 0]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=500)
        assert sum(spikes) == 0, "Zero weight should produce no spikes"

    def test_pipeline_decorrelation_matters(self):
        """
        With correlated encoders (same seed), AND-gate product is biased.
        With decorrelated encoders, result should differ.
        """
        # Run with decorrelated seeds (default pipeline)
        spikes_decorr = self._run_pipeline(
            [32768, 32768, 32768],
            [32768, 32768, 32768],
            n_steps=2000,
        )

        # Run with identical seeds (simulating the old bug)
        n_inputs = 3
        input_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1) for _ in range(n_inputs)]
        weight_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1) for _ in range(n_inputs)]
        neuron = FixedPointLIFNeuron(refractory_period=0)
        spikes_corr = []
        x_val, w_val = 32768, 32768
        for _ in range(2000):
            pre = [enc.step(x_val) for enc in input_encs]
            wbits = [enc.step(w_val) for enc in weight_encs]
            post = [p & w for p, w in zip(pre, wbits)]
            count = sum(post)
            I_t = (256 * count) // n_inputs
            spike, v = neuron.step(10, 256, I_t, 0)
            spikes_corr.append(spike)

        # Correlated encoders produce identical bitstreams when x == w,
        # so AND(x,x) = x -> higher current -> more spikes.
        # Decorrelated should have fewer spikes (p*w < p when independent).
        rate_corr = sum(spikes_corr) / len(spikes_corr)
        rate_decorr = sum(spikes_decorr) / len(spikes_decorr)

        assert rate_corr != rate_decorr, (
            f"Correlated ({rate_corr:.3f}) and decorrelated ({rate_decorr:.3f}) "
            "firing rates should differ, proving decorrelation matters"
        )
