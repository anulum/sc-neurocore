// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for clif

#![allow(dead_code)]

const V_MAX: f64 = 1.0e12;

#[derive(Debug, Clone)]
pub struct ComplementaryLIFNeuron {
    pub v_pos: f64,
    pub v_neg: f64,
    pub tau: f64,
    pub v_threshold: f64,
    pub dt: f64,
    pub alpha: f64,
}

impl ComplementaryLIFNeuron {
    pub fn new() -> Self {
        let tau = 10.0_f64;
        let dt = 1.0_f64;
        Self {
            v_pos: 0.0,
            v_neg: 0.0,
            tau,
            v_threshold: 1.0,
            dt,
            alpha: (-dt / tau).exp(),
        }
    }

    fn finite(value: f64) -> bool {
        value.is_finite()
    }

    fn validated_alpha(&self) -> Result<f64, &'static str> {
        if !Self::finite(self.tau) || self.tau <= 0.0 {
            return Err("tau must be positive");
        }
        if !Self::finite(self.dt) || self.dt <= 0.0 {
            return Err("dt must be positive");
        }
        let ratio = -self.dt / self.tau;
        let alpha = if ratio < -700.0 { 0.0 } else { ratio.exp() };
        if !Self::finite(alpha) || !(0.0..1.0).contains(&alpha) {
            return Err("alpha must be in [0, 1)");
        }
        Ok(alpha)
    }

    fn validate(&self) -> Result<f64, &'static str> {
        if !Self::finite(self.v_pos)
            || !Self::finite(self.v_neg)
            || self.v_pos.abs() > V_MAX
            || self.v_neg.abs() > V_MAX
        {
            return Err("CLIF membrane paths outside safety envelope");
        }
        if !Self::finite(self.v_threshold) || self.v_threshold <= 0.0 {
            return Err("threshold must be positive");
        }
        self.validated_alpha()
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("current must be finite");
        }
        let alpha = self.validate()?;
        let inp_pos = i_ext.max(0.0);
        let inp_neg = (-i_ext).max(0.0);
        let v_pos_next = alpha * self.v_pos + inp_pos;
        let v_neg_next = alpha * self.v_neg + inp_neg;
        let diff = v_pos_next - v_neg_next;
        if !v_pos_next.is_finite()
            || !v_neg_next.is_finite()
            || !diff.is_finite()
            || v_pos_next.abs() > V_MAX
            || v_neg_next.abs() > V_MAX
        {
            return Err("CLIF membrane candidate outside safety envelope");
        }
        self.alpha = alpha;
        if diff >= self.v_threshold {
            self.v_pos = 0.0;
            self.v_neg = 0.0;
            return Ok(1);
        }
        if diff <= -self.v_threshold {
            self.v_pos = 0.0;
            self.v_neg = 0.0;
            return Ok(-1);
        }
        self.v_pos = v_pos_next;
        self.v_neg = v_neg_next;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v_pos = 0.0;
        self.v_neg = 0.0;
    }
}

pub fn validate_clif(state: &ComplementaryLIFNeuron) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_spikes_reset_both_paths() {
        let mut positive = ComplementaryLIFNeuron::new();
        let mut negative = ComplementaryLIFNeuron::new();
        assert_eq!(positive.step(1.5).unwrap(), 1);
        assert_eq!(negative.step(-1.5).unwrap(), -1);
        assert_eq!((positive.v_pos, positive.v_neg), (0.0, 0.0));
        assert_eq!((negative.v_pos, negative.v_neg), (0.0, 0.0));
    }

    #[test]
    fn invalid_runtime_state_is_rejected_without_mutation() {
        let mut state = ComplementaryLIFNeuron::new();
        state.v_pos = f64::NAN;
        let before = (state.v_pos, state.v_neg, state.alpha);
        assert!(state.step(0.1).is_err());
        assert!(state.v_pos.is_nan() && before.0.is_nan());
        assert_eq!((state.v_neg, state.alpha), (before.1, before.2));
    }

    #[test]
    fn runtime_tau_mutation_recomputes_alpha() {
        let mut state = ComplementaryLIFNeuron::new();
        state.v_pos = 1.0;
        state.v_threshold = 10.0;
        state.tau = 100.0;
        state.dt = 2.0;
        assert_eq!(state.step(0.0).unwrap(), 0);
        let expected = (-2.0_f64 / 100.0_f64).exp();
        assert!((state.alpha - expected).abs() < 1e-12);
        assert!((state.v_pos - expected).abs() < 1e-12);
    }
}
