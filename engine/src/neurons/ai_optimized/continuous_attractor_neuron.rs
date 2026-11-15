// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Continuous-attractor neuron model

#![deny(missing_docs)]

/// Ring attractor for continuous working memory.
/// Mexican hat connectivity; a bump set in `u` persists at its position.
/// The scalar current drives every unit alike, so it cannot place or move a bump.
#[derive(Clone, Debug)]
pub struct ContinuousAttractorNeuron {
    /// Activity of each ring unit; one entry per unit, and the bump's position over them is the value the attractor holds (dimensionless).
    pub u: Vec<f64>,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Step (ms).
    pub dt: f64,
    weights: Vec<Vec<f64>>,
    n_units: usize,
}

impl ContinuousAttractorNeuron {
    /// Create a ring of `n_units` units at rest, `tau` = 10 ms and `dt` = 1 ms.
    ///
    /// The weights are `4 * exp(-d^2 / 2) - 0.5`, with `d` the distance around the ring.
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

    /// Advance one forward-Euler step with `current` added to every unit; return 1 when the
    /// largest activity exceeds 1, else 0.
    ///
    /// Each unit gains `(-u + recurrent + current) / tau * dt`. The recurrent input sums the
    /// weights times `r^2 / (1 + r^2)` with `r = max(u, 0)`. All units update from the old
    /// state. Nothing is reset by an event. The input is not validated.
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

    /// Return the index of the most active unit, the first of equal maxima as in the Python
    /// model, and 0 for an empty ring.
    ///
    /// A unit takes the position only when it is strictly more active than the one holding it,
    /// so a NaN activity never takes it, and holds it only as unit 0.
    pub fn bump_position(&self) -> usize {
        let mut position = 0;
        for (index, &activity) in self.u.iter().enumerate().skip(1) {
            if activity > self.u[position] {
                position = index;
            }
        }
        position
    }

    /// Set every unit to 0; the weights and parameters are kept.
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
    fn the_bump_is_the_first_of_equal_maxima() {
        let mut n = ContinuousAttractorNeuron::new(16);
        assert_eq!(n.bump_position(), 0, "every unit at rest is a tie");
        n.u = vec![0.0, 3.0, 1.0, 3.0];
        assert_eq!(n.bump_position(), 1);
        n.u = vec![1.0, 2.0, 0.5, 2.0, 1.5];
        assert_eq!(n.bump_position(), 1);
        n.u = vec![0.2; 16];
        assert_eq!(n.bump_position(), 0);
    }

    #[test]
    fn a_nan_activity_never_takes_the_bump() {
        let mut n = ContinuousAttractorNeuron::new(4);
        n.u = vec![1.0, f64::NAN, 2.0, 0.0];
        assert_eq!(n.bump_position(), 2);
        n.u = vec![f64::NAN, 1.0, 2.0, 0.0];
        assert_eq!(
            n.bump_position(),
            0,
            "nothing is more active than unit 0's NaN"
        );
    }

    #[test]
    fn an_empty_ring_puts_the_bump_at_zero() {
        assert_eq!(ContinuousAttractorNeuron::new(0).bump_position(), 0);
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
