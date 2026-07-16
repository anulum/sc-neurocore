// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Inhibitory LIF Neuron

/// Inhibitory LIF — LIF with self-inhibition trace.
#[derive(Clone, Debug)]
pub struct InhibitoryLIFNeuron {
    pub v: f64,
    pub inh_trace: f64,
    pub alpha_m: f64,
    pub alpha_inh: f64,
    pub v_threshold: f64,
    pub inh_strength: f64,
}

impl InhibitoryLIFNeuron {
    pub fn new(tau_m: f64, tau_inh: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            inh_trace: 0.0,
            alpha_m: (-dt / tau_m).exp(),
            alpha_inh: (-dt / tau_inh).exp(),
            v_threshold: 1.0,
            inh_strength: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.inh_trace *= self.alpha_inh;
        self.v = self.alpha_m * self.v + current - self.inh_strength * self.inh_trace;
        if self.v >= self.v_threshold {
            self.v = 0.0;
            self.inh_trace += 1.0;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.inh_trace = 0.0;
    }
}

impl Default for InhibitoryLIFNeuron {
    fn default() -> Self {
        Self::new(10.0, 5.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilif_self_inhibits() {
        let mut n = InhibitoryLIFNeuron::default();
        let total: i32 = (0..100).map(|_| n.step(0.8)).sum();
        assert!(total > 0);
    }
    #[test]
    fn ilif_silent_without_input() {
        let mut n = InhibitoryLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn ilif_reset_clears_state() {
        let mut n = InhibitoryLIFNeuron::default();
        for _ in 0..50 {
            n.step(0.8);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
        assert!((n.inh_trace - 0.0).abs() < 1e-10);
    }
    #[test]
    fn ilif_bounded() {
        let mut n = InhibitoryLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn ilif_nan_no_panic() {
        InhibitoryLIFNeuron::default().step(f64::NAN);
    }
}
