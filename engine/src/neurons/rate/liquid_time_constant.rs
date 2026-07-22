// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Liquid time-constant neuron model

/// Liquid Time-Constant neuron — input-dependent time constant. Hasani et al. 2021.
#[derive(Clone, Debug)]
pub struct LiquidTimeConstantNeuron {
    pub x: f64,
    pub tau_base: f64,
    pub w_tau: f64,
    pub w_x: f64,
    pub w_in: f64,
    pub bias: f64,
    pub v_threshold: f64,
    pub dt: f64,
}

impl LiquidTimeConstantNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            tau_base: 10.0,
            w_tau: -0.5,
            w_x: 0.8,
            w_in: 1.0,
            bias: 0.0,
            v_threshold: 1.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let sigma_tau = 1.0 / (1.0 + (-(self.w_tau * current + self.bias)).exp());
        let tau = (self.tau_base * sigma_tau).max(0.1);
        let f_target = (self.w_x * self.x + self.w_in * current).tanh();
        self.x += self.dt / tau * (-self.x + f_target);
        if self.x >= self.v_threshold {
            self.x = 0.0;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
    }
}
impl Default for LiquidTimeConstantNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltc_fires() {
        let mut n = LiquidTimeConstantNeuron {
            v_threshold: 0.9,
            ..LiquidTimeConstantNeuron::new()
        };
        let t: i32 = (0..100).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn ltc_reset() {
        let mut n = LiquidTimeConstantNeuron {
            v_threshold: 0.9,
            ..LiquidTimeConstantNeuron::new()
        };
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.x - 0.0).abs() < 1e-10);
    }

    #[test]
    fn ltc_nan_no_panic() {
        LiquidTimeConstantNeuron::new().step(f64::NAN);
    }
}
