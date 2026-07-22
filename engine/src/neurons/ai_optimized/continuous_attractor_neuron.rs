// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Continuous-attractor neuron model

/// Ring attractor for continuous working memory.
/// Mexican hat connectivity; holds a continuous value in persistent activity.
#[derive(Clone, Debug)]
pub struct ContinuousAttractorNeuron {
    pub u: Vec<f64>,
    pub tau: f64,
    pub dt: f64,
    weights: Vec<Vec<f64>>,
    n_units: usize,
}

impl ContinuousAttractorNeuron {
    pub fn new(n_units: usize) -> Self {
        let sigma_e: f64 = 1.0;
        let excitation: f64 = 4.0;
        let inhibition: f64 = 0.5;
        let mut weights = vec![vec![0.0; n_units]; n_units];
        for i in 0..n_units {
            for j in 0..n_units {
                let d = (i as f64 - j as f64)
                    .abs()
                    .min((n_units as f64) - (i as f64 - j as f64).abs());
                weights[i][j] =
                    excitation * (-d * d / (2.0 * sigma_e * sigma_e)).exp() - inhibition;
            }
        }
        Self {
            u: vec![0.0; n_units],
            tau: 10.0,
            dt: 1.0,
            weights,
            n_units,
        }
    }

    fn activation(x: f64) -> f64 {
        let r = x.max(0.0);
        r * r / (1.0 + r * r)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let mut new_u = vec![0.0; self.n_units];
        for i in 0..self.n_units {
            let mut recurrent = 0.0;
            for j in 0..self.n_units {
                recurrent += self.weights[i][j] * Self::activation(self.u[j]);
            }
            new_u[i] = self.u[i] + (-self.u[i] + recurrent + current) / self.tau * self.dt;
        }
        self.u = new_u;
        let peak = self.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if peak > 1.0 {
            1
        } else {
            0
        }
    }

    pub fn bump_position(&self) -> usize {
        self.u
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

impl Default for ContinuousAttractorNeuron {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_attractor_fires() {
        let mut n = ContinuousAttractorNeuron::new(16);
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn continuous_attractor_bump_forms() {
        let mut n = ContinuousAttractorNeuron::new(16);
        for _ in 0..200 {
            n.step(2.0);
        }
        let peak = n.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(peak > 0.0);
    }

    #[test]
    fn continuous_attractor_reset() {
        let mut n = ContinuousAttractorNeuron::new(16);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!(n.u.iter().all(|&x| x == 0.0));
    }
}
