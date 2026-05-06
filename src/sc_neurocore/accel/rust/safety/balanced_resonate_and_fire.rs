// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for balanced_resonate_and_fire

#[derive(Debug, Clone)]
pub struct BalancedResonateAndFireNeuron {
    pub x: f64,
    pub y: f64,
    pub q: f64,
    pub omega: f64,
    pub b_offset: f64,
    pub threshold: f64,
    pub gamma: f64,
    pub dt: f64,
}

pub fn sustain_oscillation_boundary(omega: f64, dt: f64) -> Result<f64, &'static str> {
    if dt <= 0.0 {
        return Err("dt must be positive");
    }
    if omega <= 0.0 {
        return Err("omega must be positive");
    }
    let scaled = dt * omega;
    if scaled > 1.0 {
        return Err("dt * omega must be <= 1");
    }
    Ok((-1.0 + (1.0 - scaled * scaled).max(0.0).sqrt()) / dt)
}

impl BalancedResonateAndFireNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            q: 0.0,
            omega: 10.0,
            b_offset: 1.0,
            threshold: 1.0,
            gamma: 0.9,
            dt: 0.01,
        }
    }

    pub fn p_omega(&self) -> Result<f64, &'static str> {
        sustain_oscillation_boundary(self.omega, self.dt)
    }

    pub fn damping(&self) -> Result<f64, &'static str> {
        Ok(self.p_omega()? - self.b_offset - self.q)
    }

    pub fn dynamic_threshold(&self) -> f64 {
        self.threshold + self.q
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        validate_balanced_resonate_and_fire(self)?;
        let b_t = self.damping()?;
        let theta_t = self.dynamic_threshold();
        let x_prev = self.x;
        let y_prev = self.y;

        self.x = x_prev + self.dt * (b_t * x_prev - self.omega * y_prev + current);
        self.y = y_prev + self.dt * (self.omega * x_prev + b_t * y_prev);

        let spike = if self.x >= theta_t { 1 } else { 0 };
        self.q = self.gamma * self.q + spike as f64;
        Ok(spike)
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.q = 0.0;
    }
}

pub fn validate_balanced_resonate_and_fire(
    state: &BalancedResonateAndFireNeuron,
) -> Result<(), &'static str> {
    if !(state.dt.is_finite() && state.dt > 0.0) {
        return Err("dt must be finite and positive");
    }
    if !(state.omega.is_finite() && state.omega > 0.0) {
        return Err("omega must be finite and positive");
    }
    if state.dt * state.omega > 1.0 {
        return Err("dt * omega must be <= 1");
    }
    if !(state.b_offset.is_finite() && state.b_offset > 0.0) {
        return Err("b_offset must be finite and positive");
    }
    if !(state.threshold.is_finite() && state.threshold > 0.0) {
        return Err("threshold must be finite and positive");
    }
    if !(state.gamma.is_finite() && state.gamma >= 0.0 && state.gamma < 1.0) {
        return Err("gamma must satisfy 0 <= gamma < 1");
    }
    if !(state.x.is_finite() && state.y.is_finite() && state.q.is_finite()) {
        return Err("state variables must be finite");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_matches_algorithm() {
        let p = sustain_oscillation_boundary(10.0, 0.01).unwrap();
        let expected = (-1.0 + (1.0_f64 - 0.1_f64 * 0.1_f64).sqrt()) / 0.01;
        assert!((p - expected).abs() < 1e-12);
    }

    #[test]
    fn step_updates_refractory_after_spike() {
        let mut neuron = BalancedResonateAndFireNeuron::new();
        let spike = neuron.step(200.0).unwrap();
        assert_eq!(spike, 1);
        assert!(neuron.q > 0.0);
        assert!(neuron.dynamic_threshold() > neuron.threshold);
    }

    #[test]
    fn invalid_boundary_rejected() {
        assert!(sustain_oscillation_boundary(200.0, 0.01).is_err());
    }
}
