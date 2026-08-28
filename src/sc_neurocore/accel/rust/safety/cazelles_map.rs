// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Safety kernel for the Cazelles four-branch map

#[derive(Debug, Clone)]
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

    fn candidate(&self, current: f64) -> Option<f64> {
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
        let tolerance = 8.0 * f64::EPSILON * self.x0.abs().max(self.x4.abs()).max(1.0);
        if candidate < self.x0 && candidate >= self.x0 - tolerance {
            candidate = self.x0;
        } else if candidate > self.x4 && candidate <= self.x4 + tolerance {
            candidate = self.x4;
        }
        (candidate.is_finite() && (self.x0..=self.x4).contains(&candidate)).then_some(candidate)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_cazelles_map(self) || !current.is_finite() {
            return 0;
        }
        let Some(candidate) = self.candidate(current) else {
            return 0;
        };
        let event = i32::from(self.x >= self.x1 && candidate < self.x1);
        self.x = candidate;
        event
    }

    pub fn reset(&mut self) {
        self.x = if (self.x0..=self.x4).contains(&0.1) {
            0.1
        } else {
            self.x0
        };
    }
}

impl Default for CazellesMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_cazelles_map(state: &CazellesMapNeuron) -> bool {
    [
        state.x,
        state.alpha,
        state.x0,
        state.x1,
        state.x2,
        state.x3,
        state.x4,
        state.a1,
        state.a2,
        state.a3,
        state.a4,
        state.b1,
        state.b2,
        state.b3,
        state.b4,
    ]
    .iter()
    .all(|value| value.is_finite())
        && (0.0..1.0).contains(&state.alpha)
        && matches!(state.exponent, 1 | 2)
        && state.x0 < state.x1
        && state.x1 < state.x2
        && state.x2 < state.x3
        && state.x3 < state.x4
        && (state.x0..=state.x4).contains(&state.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_orbit_and_event_count() {
        let mut neuron = CazellesMapNeuron::new();
        let events: i32 = (0..200).map(|_| neuron.step(0.0)).sum();
        assert_eq!(events, 2);
        assert!((0.0..=1.0).contains(&neuron.x));
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut neuron = CazellesMapNeuron::new();
        let before = neuron.x;
        assert_eq!(neuron.step(2.0), 0);
        assert_eq!(neuron.x, before);
    }
}
