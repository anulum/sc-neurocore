// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parallel spiking neuron model

/// Parallel Spiking Neuron — convolution-based filter. Fang et al. 2023.
#[derive(Clone, Debug)]
pub struct ParallelSpikingNeuron {
    pub kernel: Vec<f64>,
    pub buffer: Vec<f64>,
    pub v_threshold: f64,
    ptr: usize,
}

impl ParallelSpikingNeuron {
    pub fn new(kernel_size: usize, v_threshold: f64) -> Self {
        let k = 1.0 / kernel_size as f64;
        Self {
            kernel: vec![k; kernel_size],
            buffer: vec![0.0; kernel_size],
            v_threshold,
            ptr: 0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let ks = self.buffer.len();
        self.buffer[self.ptr % ks] = current;
        self.ptr += 1;
        let n = self.ptr.min(ks);
        let score: f64 = self.kernel[..n]
            .iter()
            .zip(self.buffer[..n].iter())
            .map(|(&w, &b)| w * b)
            .sum();
        if score >= self.v_threshold {
            self.buffer.fill(0.0);
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.ptr = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psn_fires() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        let t: i32 = (0..20).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn psn_reset() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        for _ in 0..20 {
            n.step(1.0);
        }
        n.reset();
    }

    #[test]
    fn psn_nan_no_panic() {
        ParallelSpikingNeuron::new(4, 0.5).step(f64::NAN);
    }
}
