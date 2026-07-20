// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wendling neural-mass model

//! Wendling extended Jansen-Rit dynamics with slow GABA_B inhibition.

/// Wendling et al. 2002 — extended Jansen-Rit with slow GABA_B.
#[derive(Clone, Debug)]
pub struct WendlingNeuron {
    pub y: [f64; 10],
    pub a_exc: f64,
    pub b_fast: f64,
    pub g_slow: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub g_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl WendlingNeuron {
    pub fn new() -> Self {
        Self {
            y: [0.0; 10],
            a_exc: 3.25,
            b_fast: 22.0,
            g_slow: 10.0,
            a_rate: 100.0,
            b_rate: 500.0,
            g_rate: 20.0,
            c: 135.0,
            e0: 2.5,
            v0: 6.0,
            r: 0.56,
            dt: 0.001,
        }
    }
    fn sigmoid(&self, x: f64) -> f64 {
        2.0 * self.e0 / (1.0 + (self.r * (self.v0 - x)).exp())
    }
    pub fn step(&mut self, p_ext: f64) -> f64 {
        let sig_main = self.sigmoid(self.y[1] - self.y[2] - self.y[3]);
        let sig_0 = self.sigmoid(self.c * 0.8 * self.y[0]);
        let sig_fast = self.sigmoid(self.c * 0.25 * self.y[0]);
        let sig_slow = self.sigmoid(self.c * 0.1 * self.y[0]);
        let a = self.a_rate;
        let b = self.b_rate;
        let g = self.g_rate;
        let dy0 = self.y[5];
        let dy5 = self.a_exc * a * sig_main - 2.0 * a * self.y[5] - a * a * self.y[0];
        let dy1 = self.y[6];
        let dy6 = self.a_exc * a * (p_ext + self.c * 0.8 * sig_0)
            - 2.0 * a * self.y[6]
            - a * a * self.y[1];
        let dy2 = self.y[7];
        let dy7 =
            self.b_fast * b * self.c * 0.25 * sig_fast - 2.0 * b * self.y[7] - b * b * self.y[2];
        let dy3 = self.y[8];
        let dy8 =
            self.g_slow * g * self.c * 0.1 * sig_slow - 2.0 * g * self.y[8] - g * g * self.y[3];
        self.y[0] += dy0 * self.dt;
        self.y[5] += dy5 * self.dt;
        self.y[1] += dy1 * self.dt;
        self.y[6] += dy6 * self.dt;
        self.y[2] += dy2 * self.dt;
        self.y[7] += dy7 * self.dt;
        self.y[3] += dy3 * self.dt;
        self.y[8] += dy8 * self.dt;
        self.y[1] - self.y[2] - self.y[3]
    }
    pub fn reset(&mut self) {
        self.y = [0.0; 10];
    }
}

impl Default for WendlingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driven_state_becomes_nonzero() {
        let mut n = WendlingNeuron::new();
        let mut nonzero_steps = 0;
        for _ in 0..5000 {
            if n.step(220.0).abs() > 0.001 {
                nonzero_steps += 1;
            }
        }
        assert!(nonzero_steps > 0);
    }

    #[test]
    fn reset_clears_all_state_variables() {
        let mut n = WendlingNeuron::new();
        for _ in 0..1000 {
            n.step(220.0);
        }
        n.reset();
        assert!(n.y.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn state_remains_finite_under_large_input() {
        let mut n = WendlingNeuron::new();
        for _ in 0..5000 {
            n.step(1e3);
        }
        assert!(n.y.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn nan_input_does_not_panic() {
        WendlingNeuron::new().step(f64::NAN);
    }
}
