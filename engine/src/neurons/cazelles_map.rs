// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cazelles-Courbage-Rabinovich piecewise-linear map

//! Source-faithful scalar map from Cazelles et al. (2001).

/// Four-branch map with the source's Figure-1 defaults.
#[derive(Clone, Debug)]
pub struct CazellesMapNeuron {
    pub x: f64,
    pub alpha: f64,
    pub exponent: u8,
    pub x0: f64,
    pub x1: f64,
    pub x2: f64,
    pub x3: f64,
    pub x4: f64,
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub a4: f64,
    pub b1: f64,
    pub b2: f64,
    pub b3: f64,
    pub b4: f64,
}

impl CazellesMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.1,
            alpha: 0.0,
            exponent: 2,
            x0: 0.0,
            x1: 0.4,
            x2: 0.6,
            x3: 0.7,
            x4: 1.0,
            a1: 0.0,
            a2: 1.5,
            a3: -0.9,
            a4: 1.4,
            b1: 1.05,
            b2: -1.25,
            b3: 1.5,
            b4: -1.0,
        }
    }

    fn parameters_are_valid(&self) -> bool {
        [
            self.x, self.alpha, self.x0, self.x1, self.x2, self.x3, self.x4, self.a1, self.a2,
            self.a3, self.a4, self.b1, self.b2, self.b3, self.b4,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (0.0..1.0).contains(&self.alpha)
            && matches!(self.exponent, 1 | 2)
            && self.x0 < self.x1
            && self.x1 < self.x2
            && self.x2 < self.x3
            && self.x3 < self.x4
            && (self.x0..=self.x4).contains(&self.x)
    }

    fn candidate(&self, current: f64) -> Result<f64, &'static str> {
        let base = if self.x < self.x1 {
            self.a1 + self.b1 * self.x
        } else if self.x < self.x2 {
            self.a2 + self.b2 * self.x
        } else if self.x < self.x3 {
            self.a3 + self.b3 * self.x
        } else {
            self.a4 + self.b4 * self.x
        };
        let power = if self.exponent == 1 {
            self.x
        } else {
            self.x * self.x
        };
        let mut candidate = base + self.alpha * power + current;
        if !candidate.is_finite() {
            return Err("Cazelles candidate became non-finite");
        }
        let tolerance = 8.0 * f64::EPSILON * self.x0.abs().max(self.x4.abs()).max(1.0);
        if candidate < self.x0 && candidate >= self.x0 - tolerance {
            candidate = self.x0;
        } else if candidate > self.x4 && candidate <= self.x4 + tolerance {
            candidate = self.x4;
        }
        if !(self.x0..=self.x4).contains(&candidate) {
            return Err("Cazelles candidate left its configured domain");
        }
        Ok(candidate)
    }

    /// Checked update; rejected input leaves state unchanged.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.parameters_are_valid() {
            return Err("invalid Cazelles runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Cazelles current");
        }
        let candidate = self.candidate(current)?;
        let event = i32::from(self.x >= self.x1 && candidate < self.x1);
        self.x = candidate;
        Ok(event)
    }

    /// Infallible engine-class adapter; the checked batch API reports errors.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(self.try_step(current)?);
            trace.push(self.x);
        }
        Ok((trace, events))
    }

    pub fn reset(&mut self) {
        if self.parameters_are_valid() || self.x0 < self.x4 {
            self.x = if (self.x0..=self.x4).contains(&0.1) {
                0.1
            } else {
                self.x0
            };
        }
    }
}

impl Default for CazellesMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_figure_one_orbit() {
        let mut neuron = CazellesMapNeuron::new();
        let (trace, events) = neuron.simulate(200, 0.0).unwrap();
        assert_eq!(events, 2);
        assert_eq!(trace[0], 0.105_000_000_000_000_01);
        assert!(trace.iter().all(|x| (0.0..=1.0).contains(x)));
    }

    #[test]
    fn breakpoint_convention_is_right_continuous() {
        let mut neuron = CazellesMapNeuron {
            x: 0.4,
            ..CazellesMapNeuron::new()
        };
        neuron.try_step(0.0).unwrap();
        assert_eq!(neuron.x, 1.0);
    }

    #[test]
    fn invalid_candidate_is_atomic() {
        let mut neuron = CazellesMapNeuron::new();
        let before = neuron.x;
        assert!(neuron.try_step(2.0).is_err());
        assert_eq!(neuron.x, before);
    }
}
