// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker LIF Neuron Emulator

/// SpiNNaker LIF — ARM Cortex-M4 digital LIF with refractory. Furber et al. 2014.
#[derive(Clone, Debug)]
pub struct SpiNNakerLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub i_offset: f64,
    pub tau_refrac: f64,
    pub refrac_count: f64,
    pub dt: f64,
}

impl SpiNNakerLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            i_offset: 0.0,
            tau_refrac: 2.0,
            refrac_count: 0.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if self.refrac_count > 0.0 {
            self.refrac_count -= self.dt;
            return 0;
        }
        self.v += (-(self.v - self.v_rest) + current + self.i_offset) / self.tau_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.tau_refrac;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0.0;
    }
}
impl Default for SpiNNakerLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinnaker_fires() {
        let mut n = SpiNNakerLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn spinnaker_silent() {
        let mut n = SpiNNakerLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn spinnaker_reset() {
        let mut n = SpiNNakerLIFNeuron::new();
        for _ in 0..50 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn spinnaker_bounded() {
        let mut n = SpiNNakerLIFNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn spinnaker_nan_no_panic() {
        SpiNNakerLIFNeuron::new().step(f64::NAN);
    }
}
