// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — BrainScaleS AdEx Neuron Emulator

/// BrainScaleS AdEx — Heidelberg analog wafer-scale. Schemmel et al. 2010.
#[derive(Clone, Debug)]
pub struct BrainScaleSAdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub delta_t: f64,
    pub v_rh: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub hw_speedup: f64,
    pub dt: f64,
}

impl BrainScaleSAdExNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            delta_t: 2.0,
            v_rh: -55.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            hw_speedup: 1000.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        self.v += dv;
        self.w += dw;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.w += self.b;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}
impl Default for BrainScaleSAdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brainscales_fires() {
        let mut n = BrainScaleSAdExNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn brainscales_silent() {
        let mut n = BrainScaleSAdExNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn brainscales_reset() {
        let mut n = BrainScaleSAdExNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn brainscales_bounded() {
        let mut n = BrainScaleSAdExNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn brainscales_nan_no_panic() {
        BrainScaleSAdExNeuron::new().step(f64::NAN);
    }
}
