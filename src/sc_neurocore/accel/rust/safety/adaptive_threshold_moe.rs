// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for adaptive_threshold_moe

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AdaptiveThresholdMoENeuron {
    pub k: f64,
    pub ema_alpha: f64,
    pub v: f64,
    pub v_th: f64,
    pub _mean_abs_x: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveThresholdMoEError {
    InvalidInput,
    InvalidState,
    NonFiniteOutput,
}

impl AdaptiveThresholdMoENeuron {
    pub fn new() -> Self {
        Self {
            k: 4.0_f64,
            ema_alpha: 0.1_f64,
            v: 0.0_f64,
            v_th: 1.0_f64,
            _mean_abs_x: 0.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i64, AdaptiveThresholdMoEError> {
        if !i_ext.is_finite() {
            return Err(AdaptiveThresholdMoEError::InvalidInput);
        }
        if !validate_adaptive_threshold_moe(self) {
            return Err(AdaptiveThresholdMoEError::InvalidState);
        }
        let next_mean_abs_x =
            (1.0 - self.ema_alpha) * self._mean_abs_x + self.ema_alpha * i_ext.abs();
        let next_v_th = threshold_from_mean(next_mean_abs_x, self.k)?;
        let next_v = self.v + i_ext;
        if !next_v.is_finite() {
            return Err(AdaptiveThresholdMoEError::NonFiniteOutput);
        }
        let ratio = next_v / next_v_th;
        if !ratio.is_finite() {
            return Err(AdaptiveThresholdMoEError::NonFiniteOutput);
        }
        let mut spikes = ratio.round_ties_even() as i64;
        if spikes < 0 {
            spikes = 0;
        }
        let residual = if spikes != 0 {
            next_v - next_v_th * spikes as f64
        } else {
            next_v
        };
        if !residual.is_finite() {
            return Err(AdaptiveThresholdMoEError::NonFiniteOutput);
        }
        self._mean_abs_x = next_mean_abs_x;
        self.v_th = next_v_th;
        self.v = residual;
        Ok(spikes)
    }

    pub fn step_collapsed(&mut self, activation: f64) -> Result<i64, AdaptiveThresholdMoEError> {
        if !activation.is_finite() {
            return Err(AdaptiveThresholdMoEError::InvalidInput);
        }
        if !validate_adaptive_threshold_moe(self) {
            return Err(AdaptiveThresholdMoEError::InvalidState);
        }
        let next_mean_abs_x =
            (1.0 - self.ema_alpha) * self._mean_abs_x + self.ema_alpha * activation.abs();
        let next_v_th = threshold_from_mean(next_mean_abs_x, self.k)?;
        let ratio = activation / next_v_th;
        if !ratio.is_finite() {
            return Err(AdaptiveThresholdMoEError::NonFiniteOutput);
        }
        let mut spikes = ratio.round_ties_even() as i64;
        if spikes < 0 {
            spikes = 0;
        }
        self._mean_abs_x = next_mean_abs_x;
        self.v_th = next_v_th;
        Ok(spikes)
    }

    pub fn sparsity(&self) -> Result<f64, AdaptiveThresholdMoEError> {
        if !validate_adaptive_threshold_moe(self) {
            return Err(AdaptiveThresholdMoEError::InvalidState);
        }
        Ok(if self.v.abs() < self.v_th { 1.0 } else { 0.0 })
    }

    pub fn reset(&mut self) {
        // self.v = 0.0
        // self._mean_abs_x = 0.0
        // self.v_th = 1.0
        self.v = 0.0_f64;
        self.v_th = 1.0_f64;
        self._mean_abs_x = 0.0_f64;
    }
}

pub fn validate_adaptive_threshold_moe(state: &AdaptiveThresholdMoENeuron) -> bool {
    state.k.is_finite()
        && state.k > 0.0
        && state.ema_alpha.is_finite()
        && state.ema_alpha > 0.0
        && state.ema_alpha <= 1.0
        && state.v.is_finite()
        && state.v_th.is_finite()
        && state.v_th > 0.0
        && state._mean_abs_x.is_finite()
        && state._mean_abs_x >= 0.0
}

fn threshold_from_mean(mean_abs_x: f64, k: f64) -> Result<f64, AdaptiveThresholdMoEError> {
    if !mean_abs_x.is_finite() || mean_abs_x < 0.0 || !k.is_finite() || k <= 0.0 {
        return Err(AdaptiveThresholdMoEError::NonFiniteOutput);
    }
    let v_th = if mean_abs_x > 1e-12 {
        mean_abs_x / k
    } else {
        1.0
    };
    if !v_th.is_finite() || v_th <= 0.0 {
        return Err(AdaptiveThresholdMoEError::NonFiniteOutput);
    }
    Ok(v_th)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_moe_new() {
        let state = AdaptiveThresholdMoENeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_adaptive_threshold_moe(&state));
    }

    #[test]
    fn test_adaptive_threshold_moe_step() {
        let mut state = AdaptiveThresholdMoENeuron::new();
        state.ema_alpha = 1.0;
        assert_eq!(state.step(2.0), Ok(4));
        assert_eq!(state.v, 0.0);
        assert_eq!(state.v_th, 0.5);
    }

    #[test]
    fn test_adaptive_threshold_moe_rejects_invalid_input_without_mutation() {
        let mut state = AdaptiveThresholdMoENeuron::new();
        let before = (state.v, state.v_th, state._mean_abs_x);
        assert_eq!(
            state.step(f64::INFINITY),
            Err(AdaptiveThresholdMoEError::InvalidInput)
        );
        assert_eq!((state.v, state.v_th, state._mean_abs_x), before);
    }

    #[test]
    fn test_adaptive_threshold_moe_rejects_nonfinite_threshold_without_mutation() {
        let mut state = AdaptiveThresholdMoENeuron::new();
        state.k = 1.0e-308;
        let before = (state.v, state.v_th, state._mean_abs_x);
        assert_eq!(
            state.step(1.0e308),
            Err(AdaptiveThresholdMoEError::NonFiniteOutput)
        );
        assert_eq!((state.v, state.v_th, state._mean_abs_x), before);
    }
}
