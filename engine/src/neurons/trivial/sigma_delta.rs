// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sigma-Delta Neuron

/// Sigma-Delta neuron — first-order delta modulation.
#[derive(Clone, Debug)]
pub struct SigmaDeltaNeuron {
    pub sigma: f64,
    pub v_threshold: f64,
}

impl SigmaDeltaNeuron {
    pub fn new(v_threshold: f64) -> Self {
        Self {
            sigma: 0.0,
            v_threshold,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.sigma += current;
        if self.sigma >= self.v_threshold {
            self.sigma -= self.v_threshold;
            1
        } else if self.sigma <= -self.v_threshold {
            self.sigma += self.v_threshold;
            -1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.sigma = 0.0;
    }
}

impl Default for SigmaDeltaNeuron {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_delta_encodes() {
        let mut n = SigmaDeltaNeuron::default();
        let total: i32 = (0..10).map(|_| n.step(0.3)).sum();
        assert!(total > 0);
    }
    #[test]
    fn sd_reset_clears_state() {
        let mut n = SigmaDeltaNeuron::default();
        for _ in 0..10 {
            n.step(0.3);
        }
        n.reset();
        assert!((n.sigma - 0.0).abs() < 1e-10);
    }
    #[test]
    fn sd_bounded() {
        let mut n = SigmaDeltaNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.sigma.is_finite());
    }
    #[test]
    fn sd_nan_no_panic() {
        SigmaDeltaNeuron::default().step(f64::NAN);
    }
}
