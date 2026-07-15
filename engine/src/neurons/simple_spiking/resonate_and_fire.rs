// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Resonate-and-Fire Neuron Model

//! Resonate-and-Fire oscillator dynamics.

/// Resonate-and-Fire — damped oscillator with threshold. Izhikevich 2001.
#[derive(Clone, Debug)]
pub struct ResonateAndFireNeuron {
    pub x: f64,
    pub y: f64,
    pub b: f64,
    pub omega: f64,
    pub threshold: f64,
    pub dt: f64,
}

impl ResonateAndFireNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            b: -0.1,
            omega: 1.0,
            threshold: 1.0,
            dt: 0.05,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        // Izhikevich 2001: simultaneous Euler (both derivatives use old state)
        let dx = (self.b * self.x - self.omega * self.y + current) * self.dt;
        let dy = (self.omega * self.x + self.b * self.y) * self.dt;
        self.x += dx;
        self.y += dy;
        let r = (self.x * self.x + self.y * self.y).sqrt();
        if r >= self.threshold {
            self.x = 0.0;
            self.y = 0.0;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}
impl Default for ResonateAndFireNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = ResonateAndFireNeuron::default();
        let constructed = ResonateAndFireNeuron::new();
        assert_eq!(default.x, constructed.x);
    }

    #[test]
    fn rnf_fires() {
        let mut n = ResonateAndFireNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn rnf_reset_clears_state() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..500 {
            n.step(3.0);
        }
        n.reset();
        assert!((n.x - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rnf_bounded() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.x.is_finite());
    }

    #[test]
    fn rnf_nan_no_panic() {
        ResonateAndFireNeuron::new().step(f64::NAN);
    }

    #[test]
    fn rnf_negative_no_crash() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.x.is_finite());
    }

    #[test]
    fn rnf_subthreshold_oscillation() {
        let mut n = ResonateAndFireNeuron::new();
        for _ in 0..100 {
            n.step(0.5);
        }
        assert!(n.x.abs() > 0.0 || n.y.abs() > 0.0);
    }
}
