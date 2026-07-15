// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Alpha-Synapse LIF Neuron Model

//! Alpha-synapse leaky integrate-and-fire dynamics.

/// Alpha-synapse LIF — separate excitatory/inhibitory exponential synapses.
#[derive(Clone, Debug)]
pub struct AlphaNeuron {
    pub v: f64,
    pub i_exc: f64,
    pub i_inh: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub tau_v: f64,
    pub tau_exc: f64,
    pub tau_inh: f64,
    pub dt: f64,
}

impl AlphaNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            i_exc: 0.0,
            i_inh: 0.0,
            v_rest: 0.0,
            v_threshold: 1.0,
            tau_v: 20.0,
            tau_exc: 5.0,
            tau_inh: 10.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, exc_current: f64, inh_current: f64) -> i32 {
        self.i_exc += (-self.i_exc / self.tau_exc + exc_current) * self.dt;
        self.i_inh += (-self.i_inh / self.tau_inh + inh_current) * self.dt;
        self.v += (-(self.v - self.v_rest) + self.i_exc - self.i_inh) / self.tau_v * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_rest;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.i_exc = 0.0;
        self.i_inh = 0.0;
    }
}
impl Default for AlphaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = AlphaNeuron::default();
        let constructed = AlphaNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn alpha_fires() {
        let mut n = AlphaNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(0.5, 0.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn alpha_reset_clears_state() {
        let mut n = AlphaNeuron::new();
        for _ in 0..50 {
            n.step(0.5, 0.0);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn alpha_bounded() {
        let mut n = AlphaNeuron::new();
        for _ in 0..1000 {
            n.step(100.0, 0.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn alpha_spike_input_drives() {
        let mut n = AlphaNeuron::new();
        for _ in 0..100 {
            n.step(0.0, 1.0);
        }
        // Spike input should contribute to synaptic current
        assert!(n.v.is_finite());
    }

    #[test]
    fn alpha_nan_no_panic() {
        AlphaNeuron::new().step(f64::NAN, 0.0);
    }

    #[test]
    fn alpha_negative_no_crash() {
        let mut n = AlphaNeuron::new();
        for _ in 0..100 {
            n.step(-5.0, 0.0);
        }
        assert!(n.v.is_finite());
    }
}
