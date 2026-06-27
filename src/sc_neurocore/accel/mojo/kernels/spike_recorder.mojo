# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for spike_recorder

def validate_spike(spike: Int) -> Bool:
    return spike == 0 or spike == 1


def total_spikes_from_pair(spike_a: Int, spike_b: Int) -> Int:
    if not validate_spike(spike_a) or not validate_spike(spike_b):
        return -1
    return spike_a + spike_b


def firing_rate_hz(spike_count: Int, sample_count: Int, dt_ms: Float64) -> Float64:
    if spike_count < 0 or sample_count <= 0 or dt_ms <= 0.0:
        return 0.0
    return Float64(spike_count) / (Float64(sample_count) * dt_ms / 1000.0)


def isi_ms(previous_index: Int, current_index: Int, dt_ms: Float64) -> Float64:
    if current_index <= previous_index or dt_ms < 0.0:
        return 0.0
    return Float64(current_index - previous_index) * dt_ms


def main() raises:
    if not validate_spike(1):
        raise Error("valid spike rejected")
    if validate_spike(2):
        raise Error("invalid spike accepted")
    if total_spikes_from_pair(1, 0) != 1:
        raise Error("pair spike count failed")
    if firing_rate_hz(3, 6, 1.0) != 500.0:
        raise Error("firing rate failed")
    if isi_ms(3, 5, 1.0) != 2.0:
        raise Error("ISI failed")
