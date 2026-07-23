// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner simulation results

//! Public simulation-result records and their lossless spike-event encoding.

pub struct SimResults {
    pub spike_counts: Vec<usize>,
    /// Per-population flat spike data: `(neuron_id << 32) | timestep` packed as `u64`.
    /// Supports up to 2^32 neurons and 2^32 timesteps.
    pub spike_data: Vec<Vec<u64>>,
    pub voltages: Vec<Vec<f64>>,
}

pub(super) fn pack_spike_event(neuron_id: usize, timestep: usize) -> u64 {
    assert!(
        timestep <= u32::MAX as usize,
        "spike event timestep exceeds 32-bit packing lane"
    );
    ((neuron_id as u64) << 32) | (timestep as u64)
}

#[cfg(test)]
mod tests {
    use super::pack_spike_event;

    #[test]
    fn spike_event_pack_preserves_high_neuron_and_timestep_bits() {
        let packed = pack_spike_event(100_000, 70_000);
        assert_eq!((packed >> 32) as usize, 100_000);
        assert_eq!((packed & u32::MAX as u64) as usize, 70_000);
    }

    #[test]
    fn spike_event_pack_rejects_timestep_overflow() {
        if usize::BITS > 32 {
            let result = std::panic::catch_unwind(|| {
                let _ = pack_spike_event(0, u32::MAX as usize + 1);
            });
            assert!(result.is_err());
        }
    }
}
