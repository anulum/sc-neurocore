// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Runtime Telemetry (no_std)

//! On-device spike-rate and utilization counters for runtime monitoring.
//!
//! Zero-allocation, fixed-size ring buffers for tracking spike rates,
//! activity, and health metrics without heap or external dependencies.

/// Fixed-size ring buffer for telemetry samples.
pub struct RingBuffer<const N: usize> {
    buf: [u32; N],
    write_idx: usize,
    count: usize,
}

impl<const N: usize> RingBuffer<N> {
    pub const fn new() -> Self {
        Self {
            buf: [0; N],
            write_idx: 0,
            count: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, val: u32) {
        self.buf[self.write_idx] = val;
        self.write_idx = (self.write_idx + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    #[inline]
    pub fn mean(&self) -> u32 {
        if self.count == 0 {
            return 0;
        }
        let mut sum: u64 = 0;
        for i in 0..self.count {
            sum += self.buf[i] as u64;
        }
        (sum / self.count as u64) as u32
    }

    #[inline]
    pub fn max(&self) -> u32 {
        let mut m: u32 = 0;
        for i in 0..self.count {
            if self.buf[i] > m {
                m = self.buf[i];
            }
        }
        m
    }

    #[inline]
    pub fn min(&self) -> u32 {
        if self.count == 0 {
            return 0;
        }
        let mut m: u32 = u32::MAX;
        for i in 0..self.count {
            if self.buf[i] < m {
                m = self.buf[i];
            }
        }
        m
    }

    pub const fn len(&self) -> usize {
        self.count
    }

    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn clear(&mut self) {
        self.write_idx = 0;
        self.count = 0;
    }

    pub fn last(&self) -> Option<u32> {
        if self.count == 0 {
            return None;
        }
        let idx = if self.write_idx == 0 {
            N - 1
        } else {
            self.write_idx - 1
        };
        Some(self.buf[idx])
    }
}

impl<const N: usize> Default for RingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime telemetry collector for a tinySC network.
///
/// Tracks spike counts, layer utilization, and timing over a sliding window.
pub struct Telemetry<const W: usize> {
    pub total_ticks: u64,
    pub total_spikes: u64,
    pub spike_history: RingBuffer<W>,
    pub tick_cycles: RingBuffer<W>,
    pub peak_spikes: u32,
    pub min_spikes: u32,
}

impl<const W: usize> Telemetry<W> {
    pub const fn new() -> Self {
        Self {
            total_ticks: 0,
            total_spikes: 0,
            spike_history: RingBuffer::new(),
            tick_cycles: RingBuffer::new(),
            peak_spikes: 0,
            min_spikes: u32::MAX,
        }
    }

    /// Record one tick's spike count and cycle count.
    #[inline]
    pub fn record(&mut self, spikes: u32, cycles: u32) {
        self.total_ticks += 1;
        self.total_spikes += spikes as u64;
        self.spike_history.push(spikes);
        self.tick_cycles.push(cycles);

        if spikes > self.peak_spikes {
            self.peak_spikes = spikes;
        }
        if spikes < self.min_spikes {
            self.min_spikes = spikes;
        }
    }

    /// Mean spike rate over the sliding window.
    pub fn mean_spike_rate(&self) -> u32 {
        self.spike_history.mean()
    }

    /// Mean cycles per tick over the sliding window.
    pub fn mean_cycles(&self) -> u32 {
        self.tick_cycles.mean()
    }

    /// Overall spike rate (total spikes / total ticks).
    pub fn overall_spike_rate(&self) -> f32 {
        if self.total_ticks == 0 {
            return 0.0;
        }
        self.total_spikes as f32 / self.total_ticks as f32
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.total_ticks = 0;
        self.total_spikes = 0;
        self.spike_history.clear();
        self.tick_cycles.clear();
        self.peak_spikes = 0;
        self.min_spikes = u32::MAX;
    }
}

impl<const W: usize> Default for Telemetry<W> {
    fn default() -> Self {
        Self::new()
    }
}

/// Heartbeat counter for watchdog feeding.
pub struct Heartbeat {
    pub interval_ticks: u64,
    last_beat: u64,
    pub beats: u64,
    started: bool,
}

impl Heartbeat {
    pub const fn new(interval_ticks: u64) -> Self {
        Self {
            interval_ticks,
            last_beat: 0,
            beats: 0,
            started: false,
        }
    }

    /// Check if heartbeat is due at the given tick count.
    #[inline]
    pub fn check(&mut self, current_tick: u64) -> bool {
        if !self.started || current_tick.wrapping_sub(self.last_beat) >= self.interval_ticks {
            self.last_beat = current_tick;
            self.beats += 1;
            self.started = true;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = RingBuffer::<4>::new();
        assert!(rb.is_empty());
        rb.push(10);
        rb.push(20);
        assert_eq!(rb.len(), 2);
        assert_eq!(rb.mean(), 15);
        assert_eq!(rb.max(), 20);
        assert_eq!(rb.min(), 10);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut rb = RingBuffer::<4>::new();
        for i in 0..8 {
            rb.push(i);
        }
        assert_eq!(rb.len(), 4);
        assert_eq!(rb.last(), Some(7));
    }

    #[test]
    fn test_ring_buffer_last() {
        let mut rb = RingBuffer::<4>::new();
        assert_eq!(rb.last(), None);
        rb.push(42);
        assert_eq!(rb.last(), Some(42));
    }

    #[test]
    fn test_telemetry_record() {
        let mut t = Telemetry::<16>::new();
        t.record(5, 100);
        t.record(3, 80);
        t.record(7, 120);
        assert_eq!(t.total_ticks, 3);
        assert_eq!(t.total_spikes, 15);
        assert_eq!(t.peak_spikes, 7);
        assert_eq!(t.min_spikes, 3);
    }

    #[test]
    fn test_telemetry_rates() {
        let mut t = Telemetry::<16>::new();
        for _ in 0..10 {
            t.record(4, 100);
        }
        assert_eq!(t.mean_spike_rate(), 4);
        assert_eq!(t.mean_cycles(), 100);
        assert!((t.overall_spike_rate() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_telemetry_reset() {
        let mut t = Telemetry::<8>::new();
        t.record(5, 100);
        t.reset();
        assert_eq!(t.total_ticks, 0);
        assert_eq!(t.total_spikes, 0);
    }

    #[test]
    fn test_heartbeat() {
        let mut hb = Heartbeat::new(10);
        assert!(hb.check(0));
        assert!(!hb.check(5));
        assert!(hb.check(10));
        assert_eq!(hb.beats, 2);
    }

    #[test]
    fn test_heartbeat_every_tick() {
        let mut hb = Heartbeat::new(1);
        for i in 0..5 {
            assert!(hb.check(i));
        }
        assert_eq!(hb.beats, 5);
    }
}
