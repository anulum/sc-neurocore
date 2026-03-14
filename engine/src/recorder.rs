// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spike recording and statistics.

/// Buffered spike recorder with firing rate and ISI statistics.
#[derive(Clone, Debug)]
pub struct SpikeRecorder {
    pub buffer: Vec<u8>,
    pub dt_ms: f64,
}

impl SpikeRecorder {
    pub fn new(dt_ms: f64) -> Self {
        Self {
            buffer: Vec::new(),
            dt_ms,
        }
    }

    pub fn record(&mut self, spike: u8) {
        debug_assert!(spike <= 1);
        self.buffer.push(spike);
    }

    pub fn total_spikes(&self) -> u64 {
        self.buffer.iter().map(|&s| s as u64).sum()
    }

    pub fn firing_rate_hz(&self) -> f64 {
        let t = self.buffer.len();
        if t == 0 {
            return 0.0;
        }
        let duration_s = (t as f64 * self.dt_ms) / 1000.0;
        if duration_s <= 0.0 {
            return 0.0;
        }
        self.total_spikes() as f64 / duration_s
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_spikes_rate() {
        let mut r = SpikeRecorder::new(1.0);
        for _ in 0..1000 {
            r.record(1);
        }
        assert_eq!(r.total_spikes(), 1000);
        assert!((r.firing_rate_hz() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn no_spikes_rate() {
        let mut r = SpikeRecorder::new(1.0);
        for _ in 0..100 {
            r.record(0);
        }
        assert_eq!(r.total_spikes(), 0);
        assert!(r.firing_rate_hz().abs() < 1e-12);
    }

    #[test]
    fn half_spikes() {
        let mut r = SpikeRecorder::new(1.0);
        for i in 0..100 {
            r.record((i % 2) as u8);
        }
        assert_eq!(r.total_spikes(), 50);
        assert!((r.firing_rate_hz() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn empty_recorder() {
        let r = SpikeRecorder::new(1.0);
        assert!(r.firing_rate_hz().abs() < 1e-12);
        assert!(r.is_empty());
    }

    #[test]
    fn reset_clears() {
        let mut r = SpikeRecorder::new(1.0);
        r.record(1);
        r.reset();
        assert!(r.is_empty());
        assert_eq!(r.total_spikes(), 0);
    }
}
