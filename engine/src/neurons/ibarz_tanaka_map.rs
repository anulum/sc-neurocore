// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ibarz-Tanaka discrete map neuron

//! Ibarz-Tanaka discrete map neuron.

/// Ibarz-Tanaka (2007) four-branch Rulkov map.
#[derive(Clone, Debug)]
pub struct IbarzTanakaMapNeuron {
    pub v: f64,
    pub u: f64,
    pub alpha: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl IbarzTanakaMapNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0,
            u: -0.1,
            alpha: 1.0,
            mu: 0.001,
            sigma: 0.1,
        }
    }

    fn parameters_are_valid(&self) -> bool {
        self.alpha.is_finite()
            && self.mu.is_finite()
            && self.sigma.is_finite()
            && self.alpha > 0.0
            && self.mu > 0.0
    }

    fn candidate(&self, current: f64) -> Result<(f64, f64, i32), &'static str> {
        let lower = -1.0 - self.alpha / 2.0;
        let upper = 1.0 + current + self.u;
        let v_next = if self.v < lower {
            -(self.alpha * self.alpha) / 4.0 - self.alpha + current + self.u
        } else if self.v <= 0.0 {
            self.alpha * self.v + (self.v + 1.0) * (self.v + 1.0) + current + self.u
        } else if self.v < upper {
            upper
        } else {
            -1.0
        };
        let u_next = self.u - self.mu * (self.v + 1.0 - self.sigma);
        if !v_next.is_finite() || !u_next.is_finite() {
            return Err("invalid Ibarz-Tanaka map candidate");
        }
        Ok((v_next, u_next, i32::from(self.v >= upper)))
    }

    /// Checked source-derived update; a rejected step leaves the state intact.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.v.is_finite() || !self.u.is_finite() || !self.parameters_are_valid() {
            return Err("invalid Ibarz-Tanaka runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Ibarz-Tanaka current");
        }
        let (v_next, u_next, event) = self.candidate(current)?;
        self.v = v_next;
        self.u = u_next;
        Ok(event)
    }

    /// Legacy infallible engine-class update; invalid input emits no event.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run checked Eq. 2-3 iterations and return the post-step `v` trace.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(self.try_step(current)?);
            trace.push(self.v);
        }
        Ok((trace, events))
    }

    pub fn reset(&mut self) {
        self.v = -1.0;
        self.u = -0.1;
    }
}
impl Default for IbarzTanakaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ibarz_fires() {
        let mut n = IbarzTanakaMapNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
}
