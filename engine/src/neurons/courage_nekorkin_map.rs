// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Courbage-Nekorkin-Vdovin source map

//! Source-faithful Courbage-Nekorkin-Vdovin map with Figure-4 defaults.

#![deny(missing_docs)]

/// Courbage-Nekorkin-Vdovin piecewise-linear map with the source's Figure-4 defaults. The
/// type name keeps the historical spelling `CourageNekorkin`.
#[derive(Clone, Debug)]
pub struct CourageNekorkinMapNeuron {
    /// Fast variable (dimensionless).
    pub x: f64,
    /// Slow variable (dimensionless).
    pub y: f64,
    /// Slope of the resting branch of the piecewise fast map (dimensionless).
    pub m0: f64,
    /// Slope of the spike-generating branch of the piecewise fast map (dimensionless).
    pub m1: f64,
    /// Breakpoint location of the piecewise fast map (dimensionless).
    pub a: f64,
    /// Threshold of the slow regenerative (bursting) term (dimensionless).
    pub d: f64,
    /// Set-point of the fast variable in the slow-variable update (dimensionless).
    pub j: f64,
    /// Strength of the slow regenerative (bursting) reinjection (dimensionless).
    pub beta: f64,
    /// Slow-variable timescale; small values give slow bursting (dimensionless).
    pub eps: f64,
    /// Level on x at which a spike is registered (dimensionless).
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    /// Create the map at `x` = `y` = 0 with the Figure-4 defaults and an observation level
    /// of 0.3.
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
    ///
    /// `x' = x + F(x) - y - beta * H(x - d) + current`, where `F` is the three-branch
    /// piecewise-linear field and `H` the Heaviside step, 1 at `x = d`, and
    /// `y' = y + eps * (x - j)`. Both use the old state. Returns 1 when `x` crosses
    /// `x_threshold` upwards, else 0.
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

    /// Run `n_steps` checked updates with a constant `current`; return `x` after each step
    /// and the number of events.
    ///
    /// This is not atomic: an error stops the run and is returned, the steps already taken
    /// stay in the state, and the partial trace is dropped.
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

    /// Set `x` and `y` to 0; the parameters are kept.
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
