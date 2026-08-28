// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Courbage-Nekorkin-Vdovin source map

//! Source-faithful Courbage-Nekorkin-Vdovin map with Figure-4 defaults.

#[derive(Clone, Debug)]
pub struct CourageNekorkinMapNeuron {
    pub x: f64,
    pub y: f64,
    pub m0: f64,
    pub m1: f64,
    pub a: f64,
    pub d: f64,
    pub j: f64,
    pub beta: f64,
    pub eps: f64,
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            m0: 0.4,
            m1: 0.65,
            a: 0.2,
            d: 0.3,
            j: 0.13,
            beta: 0.25,
            eps: 0.002,
            x_threshold: 0.3,
        }
    }

    fn breakpoints(&self) -> (f64, f64) {
        let am1 = self.a * self.m1;
        let denominator = self.m0 + self.m1;
        (am1 / denominator, (self.m0 + am1) / denominator)
    }

    fn parameters_are_valid(&self) -> bool {
        [
            self.x,
            self.y,
            self.m0,
            self.m1,
            self.a,
            self.d,
            self.j,
            self.beta,
            self.eps,
            self.x_threshold,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.m0 > 0.0
            && self.m0 < 1.0
            && self.m1 > 0.0
            && self.a > 0.0
            && self.a < 1.0
            && self.d > 0.0
            && self.beta > 0.0
            && self.eps > 0.0
            && self.j > 0.0
            && self.j < self.d
            && {
                let (j_min, j_max) = self.breakpoints();
                j_min < self.d && self.d < j_max
            }
    }

    /// Checked simultaneous update. Rejected input leaves state unchanged.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.parameters_are_valid() {
            return Err("invalid Courbage runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Courbage current");
        }
        let x_previous = self.x;
        let (j_min, j_max) = self.breakpoints();
        let field = if self.x <= j_min {
            -self.m0 * self.x
        } else if self.x < j_max {
            self.m1 * (self.x - self.a)
        } else {
            -self.m0 * (self.x - 1.0)
        };
        let heaviside = if self.x >= self.d { 1.0 } else { 0.0 };
        let x_new = self.x + field - self.y - self.beta * heaviside + current;
        let y_new = self.y + self.eps * (self.x - self.j);
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("Courbage candidate became non-finite");
        }
        let event = i32::from(x_new >= self.x_threshold && x_previous < self.x_threshold);
        self.x = x_new;
        self.y = y_new;
        Ok(event)
    }

    /// Infallible engine adapter; checked batch simulation reports errors.
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
        self.x = 0.0;
        self.y = 0.0;
    }
}

impl Default for CourageNekorkinMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figure_four_profile_and_orbit_are_pinned() {
        let mut neuron = CourageNekorkinMapNeuron::new();
        let (trace, events) = neuron.simulate(1_000, 0.0).unwrap();
        assert_eq!(events, 21);
        assert_eq!(trace[0], 0.0);
        assert!((neuron.x - 0.116_738_843_564_412_33).abs() < 1.0e-15);
        assert!((neuron.y + 0.046_762_693_421_842_21).abs() < 1.0e-15);
    }

    #[test]
    fn all_source_branches_are_reachable() {
        let base = CourageNekorkinMapNeuron::new();
        let (j_min, j_max) = base.breakpoints();
        for x in [j_min - 0.01, 0.5 * (j_min + j_max), j_max + 0.01] {
            let mut neuron = CourageNekorkinMapNeuron { x, ..base.clone() };
            assert!(neuron.try_step(0.0).is_ok());
        }
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut neuron = CourageNekorkinMapNeuron::new();
        let before = (neuron.x, neuron.y);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.x, neuron.y), before);
        neuron.m0 = 0.0;
        assert!(neuron.try_step(0.0).is_err());
        assert_eq!((neuron.x, neuron.y), before);
    }
}
