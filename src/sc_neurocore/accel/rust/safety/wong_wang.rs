// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wong_wang

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct WongWangUnit {
    pub s1: f64,
    pub s2: f64,
    pub tau_s: f64,
    pub gamma: f64,
    pub j_n: f64,
    pub j_cross: f64,
    pub i_0: f64,
    pub sigma: f64,
    pub dt: f64,
}

impl WongWangUnit {
    pub fn new() -> Self {
        Self {
            s1: 0.1_f64,
            s2: 0.1_f64,
            tau_s: 0.1_f64,
            gamma: 0.641_f64,
            j_n: 0.2609_f64,
            j_cross: 0.0497_f64,
            i_0: 0.3255_f64,
            sigma: 0.02_f64,
            dt: 0.001_f64,
        }
    }

    pub fn _phi(&self, i_syn: f64) -> f64 {
        let a = 270.0_f64;
        let b = 108.0_f64;
        let d = 0.154_f64;
        let x = a * i_syn - b;
        if x.abs() < 1e-6 {
            return 1.0 / d;
        }
        let exponent = -d * x;
        if exponent > 700.0 {
            return 0.0;
        }
        x / (1.0 - exponent.exp())
    }

    pub fn step(
        &mut self,
        stim1: f64,
        stim2: f64,
        xi1: f64,
        xi2: f64,
    ) -> Result<(f64, f64), &'static str> {
        if !validate_wong_wang(self) {
            return Err("invalid Wong-Wang runtime state");
        }
        if !stim1.is_finite() || !stim2.is_finite() || !xi1.is_finite() || !xi2.is_finite() {
            return Err("invalid Wong-Wang stimulus or noise");
        }

        let i1 = self.j_n * self.s1 - self.j_cross * self.s2 + self.i_0 + stim1 + self.sigma * xi1;
        let i2 = self.j_n * self.s2 - self.j_cross * self.s1 + self.i_0 + stim2 + self.sigma * xi2;
        let r1 = self._phi(i1);
        let r2 = self._phi(i2);
        if !r1.is_finite() || r1 < 0.0 || !r2.is_finite() || r2 < 0.0 {
            return Err("invalid Wong-Wang transfer response");
        }

        let next_s1 =
            self.s1 + (-self.s1 / self.tau_s + (1.0 - self.s1) * self.gamma * r1) * self.dt;
        let next_s2 =
            self.s2 + (-self.s2 / self.tau_s + (1.0 - self.s2) * self.gamma * r2) * self.dt;
        if !next_s1.is_finite() || !next_s2.is_finite() {
            return Err("invalid Wong-Wang candidate state");
        }
        self.s1 = next_s1.clamp(0.0, 1.0);
        self.s2 = next_s2.clamp(0.0, 1.0);
        Ok((r1, r2))
    }

    pub fn reset(&mut self) {
        // self.s1, self.s2 = 0.1, 0.1
        self.s1 = 0.1_f64;
        self.s2 = 0.1_f64;
        self.tau_s = 0.1_f64;
        self.gamma = 0.641_f64;
        self.j_n = 0.2609_f64;
    }
}

pub fn validate_wong_wang(state: &WongWangUnit) -> bool {
    finite_gate(state.s1)
        && finite_gate(state.s2)
        && state.tau_s.is_finite()
        && state.tau_s > 0.0
        && state.gamma.is_finite()
        && state.gamma > 0.0
        && state.j_n.is_finite()
        && state.j_n >= 0.0
        && state.j_cross.is_finite()
        && state.j_cross >= 0.0
        && state.i_0.is_finite()
        && state.sigma.is_finite()
        && state.sigma >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

fn finite_gate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wong_wang_new() {
        let state = WongWangUnit::new();
        assert!(validate_wong_wang(&state));
    }

    #[test]
    fn test_wong_wang_step() {
        let mut state = WongWangUnit::new();
        let rates = state.step(0.1, 0.0, 0.0, 0.0).unwrap();
        assert!(rates.0.is_finite() && rates.1.is_finite());
    }

    #[test]
    fn test_wong_wang_rejects_invalid_runtime_state() {
        let mut state = WongWangUnit::new();
        state.s1 = 1.5;
        assert!(state.step(0.1, 0.0, 0.0, 0.0).is_err());
    }
}
