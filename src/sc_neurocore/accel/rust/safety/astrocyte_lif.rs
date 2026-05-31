// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for astrocyte_lif

#![allow(dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AstrocyteLIFNeuron {
    pub tau_m: f64,
    pub tau_ca: f64,
    pub e_l: f64,
    pub theta: f64,
    pub v_reset: f64,
    pub ca_delta: f64,
    pub ca_thresh: f64,
    pub g_glio: f64,
    pub dt: f64,
    pub v: f64,
    pub ca: f64,
}

impl AstrocyteLIFNeuron {
    pub fn new() -> Self {
        Self {
            tau_m: 20.0,
            tau_ca: 500.0,
            e_l: -65.0,
            theta: -50.0,
            v_reset: -65.0,
            ca_delta: 0.1,
            ca_thresh: 0.5,
            g_glio: 2.0,
            dt: 0.1,
            v: -65.0,
            ca: 0.0,
        }
    }

    pub fn validate(&self) -> bool {
        [self.tau_m, self.tau_ca, self.dt]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0)
            && [self.e_l, self.theta, self.v_reset, self.v]
                .iter()
                .all(|x| x.is_finite())
            && self.theta > self.v_reset
            && [self.ca_delta, self.ca_thresh, self.g_glio, self.ca]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
    }

    pub fn step_with_pre(&mut self, i_ext: f64, pre_spike: bool) -> Result<i32, &'static str> {
        if !self.validate() || !i_ext.is_finite() {
            return Err("invalid astrocyte LIF state or input");
        }
        let mut dca = -self.ca / self.tau_ca;
        if pre_spike {
            dca += self.ca_delta / self.dt;
        }
        let ca_next = (self.ca + dca * self.dt).max(0.0);
        if !ca_next.is_finite() || ca_next < 0.0 {
            return Err("invalid astrocyte calcium candidate");
        }
        let i_glio = if ca_next > self.ca_thresh {
            self.g_glio
        } else {
            0.0
        };
        if !i_glio.is_finite() {
            return Err("invalid gliotransmitter current");
        }
        let dv = (-(self.v - self.e_l) + i_ext + i_glio) / self.tau_m;
        let v_next = self.v + dv * self.dt;
        if !v_next.is_finite() {
            return Err("invalid membrane candidate");
        }
        self.ca = ca_next;
        if v_next >= self.theta {
            self.v = self.v_reset;
            Ok(1)
        } else {
            self.v = v_next;
            Ok(0)
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        self.step_with_pre(i_ext, false)
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.ca = 0.0;
    }
}

pub fn validate_astrocyte_lif(state: &AstrocyteLIFNeuron) -> bool {
    state.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astrocyte_lif_new() {
        let state = AstrocyteLIFNeuron::new();
        assert!(validate_astrocyte_lif(&state));
    }

    #[test]
    fn test_astrocyte_lif_step() {
        let mut state = AstrocyteLIFNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
        assert!(state.v.is_finite());
        assert!(state.ca >= 0.0);
    }

    #[test]
    fn test_astrocyte_lif_rejects_invalid_runtime_state() {
        let mut state = AstrocyteLIFNeuron::new();
        state.ca = f64::INFINITY;
        let before = state.v;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.v, before);
    }
}
