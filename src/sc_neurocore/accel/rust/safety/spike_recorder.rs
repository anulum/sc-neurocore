// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for spike_recorder

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct BitstreamSpikeRecorder {
    pub dt_ms: f64,
    pub spikes: Vec<u8>,
}

impl BitstreamSpikeRecorder {
    pub fn new() -> Self {
        Self {
            dt_ms: 1.0_f64,
            spikes: Vec::new(),
        }
    }

    pub fn with_dt(dt_ms: f64) -> Result<Self, &'static str> {
        if dt_ms < 0.0 {
            return Err("dt_ms must be non-negative");
        }
        Ok(Self {
            dt_ms,
            spikes: Vec::new(),
        })
    }

    pub fn record(&mut self, spike: u8) -> Result<(), &'static str> {
        if spike != 0 && spike != 1 {
            return Err("Spike must be 0 or 1");
        }
        self.spikes.push(spike);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.spikes.clear();
    }

    pub fn as_array(&self) -> Vec<u8> {
        self.spikes.clone()
    }

    pub fn total_spikes(&self) -> usize {
        self.spikes.iter().map(|&spike| spike as usize).sum()
    }

    pub fn firing_rate_hz(&self) -> f64 {
        let sample_count = self.spikes.len();
        if sample_count == 0 {
            return 0.0;
        }
        let duration_ms = sample_count as f64 * self.dt_ms;
        if duration_ms == 0.0 {
            return 0.0;
        }
        self.total_spikes() as f64 / (duration_ms / 1000.0)
    }

    pub fn isi_intervals_ms(&self) -> Vec<f64> {
        let spike_indices: Vec<usize> = self
            .spikes
            .iter()
            .enumerate()
            .filter_map(|(idx, &spike)| if spike == 1 { Some(idx) } else { None })
            .collect();

        spike_indices
            .windows(2)
            .map(|window| (window[1] - window[0]) as f64 * self.dt_ms)
            .collect()
    }
}

pub fn validate_spike_recorder(state: &BitstreamSpikeRecorder) -> bool {
    state.dt_ms >= 0.0 && state.spikes.iter().all(|&spike| spike == 0 || spike == 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_recorder_new() {
        let state = BitstreamSpikeRecorder::new();
        assert!(validate_spike_recorder(&state));
    }

    #[test]
    fn test_spike_recorder_statistics() {
        let mut state = BitstreamSpikeRecorder::with_dt(1.0).unwrap();
        for spike in [1, 0, 0, 1, 0, 1] {
            state.record(spike).unwrap();
        }
        assert_eq!(state.as_array(), vec![1, 0, 0, 1, 0, 1]);
        assert_eq!(state.total_spikes(), 3);
        assert_eq!(state.firing_rate_hz(), 500.0);
        assert_eq!(state.isi_intervals_ms(), vec![3.0, 2.0]);
        assert!(validate_spike_recorder(&state));
    }

    #[test]
    fn test_spike_recorder_validation() {
        let mut state = BitstreamSpikeRecorder::new();
        assert!(state.record(2).is_err());
        assert!(BitstreamSpikeRecorder::with_dt(-1.0).is_err());
        state.record(1).unwrap();
        state.reset();
        assert!(state.spikes.is_empty());
    }
}
