// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sliding-window bitstream probability estimator

/// Sliding-window bitstream probability estimator.
///
/// Mirrors Python's `BitstreamAverager`.
#[derive(Clone, Debug)]
pub struct BitstreamAverager {
    buffer: Vec<u8>,
    index: usize,
    filled: bool,
    running_sum: u64,
}

impl BitstreamAverager {
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        Self {
            buffer: vec![0; window],
            index: 0,
            filled: false,
            running_sum: 0,
        }
    }

    pub fn push(&mut self, bit: u8) {
        debug_assert!(bit <= 1, "bit must be 0 or 1");
        let old = self.buffer[self.index];
        self.buffer[self.index] = bit;

        if self.filled {
            self.running_sum = self.running_sum - old as u64 + bit as u64;
        } else {
            self.running_sum += bit as u64;
        }

        self.index += 1;
        if self.index == self.buffer.len() {
            self.index = 0;
            self.filled = true;
        }
    }

    pub fn estimate(&self) -> f64 {
        if !self.filled {
            if self.index == 0 {
                return 0.0;
            }
            return self.running_sum as f64 / self.index as f64;
        }
        self.running_sum as f64 / self.buffer.len() as f64
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.index = 0;
        self.filled = false;
        self.running_sum = 0;
    }

    pub fn window(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::BitstreamAverager;

    #[test]
    fn all_ones_estimate_one() {
        let mut avg = BitstreamAverager::new(100);
        for _ in 0..100 {
            avg.push(1);
        }
        assert!((avg.estimate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn all_zeros_estimate_zero() {
        let mut avg = BitstreamAverager::new(50);
        for _ in 0..50 {
            avg.push(0);
        }
        assert!(avg.estimate().abs() < 1e-12);
    }

    #[test]
    fn alternating_bits_estimate_half() {
        let mut avg = BitstreamAverager::new(100);
        for i in 0..100 {
            avg.push((i % 2) as u8);
        }
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sliding_window_replaces_oldest_bits() {
        let mut avg = BitstreamAverager::new(4);
        for &bit in &[1_u8, 1, 0, 0] {
            avg.push(bit);
        }
        assert!((avg.estimate() - 0.5).abs() < 1e-12);

        avg.push(1);
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
        avg.push(1);
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
        avg.push(1);
        assert!((avg.estimate() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn partial_window_uses_observed_count() {
        let mut avg = BitstreamAverager::new(100);
        avg.push(1);
        avg.push(0);
        assert!((avg.estimate() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_window_estimate_is_zero() {
        let avg = BitstreamAverager::new(10);
        assert!(avg.estimate().abs() < 1e-12);
    }

    #[test]
    fn reset_clears_window_state() {
        let mut avg = BitstreamAverager::new(10);
        for _ in 0..10 {
            avg.push(1);
        }
        avg.reset();
        assert!(avg.estimate().abs() < 1e-12);
    }
}
