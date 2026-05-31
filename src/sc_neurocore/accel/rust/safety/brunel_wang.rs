// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for brunel_wang

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct BrunelWangNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_ref: f64,
    pub tau_ampa: f64,
    pub tau_nmda_rise: f64,
    pub tau_nmda_decay: f64,
    pub tau_gaba: f64,
    pub g_ampa_ext: f64,
    pub g_ampa_rec: f64,
    pub g_nmda: f64,
    pub g_gaba: f64,
    pub v_ampa: f64,
    pub v_nmda: f64,
    pub v_gaba: f64,
    pub C_m: f64,
    pub mg_conc: f64,
    pub dt: f64,
    pub ref_remaining: f64,
}

impl BrunelWangNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            v_rest: -70.0_f64,
            v_reset: -55.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 20.0_f64,
            tau_ref: 2.0_f64,
            tau_ampa: 2.0_f64,
            tau_nmda_rise: 2.0_f64,
            tau_nmda_decay: 100.0_f64,
            tau_gaba: 5.0_f64,
            g_ampa_ext: 2.1_f64,
            g_ampa_rec: 0.05_f64,
            g_nmda: 0.165_f64,
            g_gaba: 1.3_f64,
            v_ampa: 0.0_f64,
            v_nmda: 0.0_f64,
            v_gaba: -70.0_f64,
            C_m: 0.5_f64,
            mg_conc: 1.0_f64,
            dt: 0.1_f64,
            ref_remaining: 0.0_f64,
        }
    }

    pub fn _nmda_voltage_dep(&self, v: f64) -> f64 {
        let exponent = -0.062 * v;
        if exponent > 700.0 {
            return 0.0;
        }
        1.0 / (1.0 + self.mg_conc / 3.57 * exponent.exp())
    }

    pub fn step(
        &mut self,
        i_ampa_ext: f64,
        s_ampa_rec: f64,
        s_nmda_rec: f64,
        s_gaba: f64,
    ) -> Result<i32, &'static str> {
        if !validate_brunel_wang(self) {
            return Err("invalid Brunel-Wang runtime state");
        }
        if !nonnegative(i_ampa_ext) || !gate(s_ampa_rec) || !gate(s_nmda_rec) || !gate(s_gaba) {
            return Err("invalid Brunel-Wang synaptic input");
        }

        if self.ref_remaining > 0.0 {
            self.ref_remaining = (self.ref_remaining - self.dt).max(0.0);
            return Ok(0);
        }

        let i_ampa = -self.g_ampa_ext * (self.v - self.v_ampa) * i_ampa_ext
            - self.g_ampa_rec * (self.v - self.v_ampa) * s_ampa_rec;
        let i_nmda =
            -self.g_nmda * self._nmda_voltage_dep(self.v) * (self.v - self.v_nmda) * s_nmda_rec;
        let i_gaba = -self.g_gaba * (self.v - self.v_gaba) * s_gaba;
        let i_leak = -(self.v - self.v_rest) / self.tau_m;
        let dv = (i_leak + (i_ampa + i_nmda + i_gaba) / self.C_m) * self.dt;
        let next_v = self.v + dv;
        if !i_ampa.is_finite()
            || !i_nmda.is_finite()
            || !i_gaba.is_finite()
            || !i_leak.is_finite()
            || !dv.is_finite()
            || !next_v.is_finite()
        {
            return Err("invalid Brunel-Wang membrane candidate");
        }

        self.v = next_v;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.ref_remaining = self.tau_ref;
            return Ok(1);
        }
        Ok(0)
    }

    pub fn reset(&mut self) {
        // self.v = self.v_rest
        // self._s_ampa = 0.0
        // self._s_nmda = 0.0
        // self._x_nmda = 0.0
        // self._s_gaba = 0.0
        // self._ref_remaining = 0.0
        self.v = -70.0_f64;
        self.v_rest = -70.0_f64;
        self.v_reset = -55.0_f64;
        self.v_threshold = -50.0_f64;
        self.tau_m = 20.0_f64;
        self.ref_remaining = 0.0_f64;
    }

    pub fn get_state(&self) -> (f64, f64) {
        (self.v, self.ref_remaining)
    }
}

pub fn validate_brunel_wang(state: &BrunelWangNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && positive(state.tau_m)
        && positive(state.tau_ref)
        && positive(state.tau_ampa)
        && positive(state.tau_nmda_rise)
        && positive(state.tau_nmda_decay)
        && positive(state.tau_gaba)
        && nonnegative(state.g_ampa_ext)
        && nonnegative(state.g_ampa_rec)
        && nonnegative(state.g_nmda)
        && nonnegative(state.g_gaba)
        && state.v_ampa.is_finite()
        && state.v_nmda.is_finite()
        && state.v_gaba.is_finite()
        && positive(state.C_m)
        && nonnegative(state.mg_conc)
        && positive(state.dt)
        && nonnegative(state.ref_remaining)
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn gate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brunel_wang_new() {
        let state = BrunelWangNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_brunel_wang(&state));
    }

    #[test]
    fn test_brunel_wang_step() {
        let mut state = BrunelWangNeuron::new();
        let spike = state.step(1.0, 0.0, 0.0, 0.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_brunel_wang_rejects_invalid_runtime_state() {
        let mut state = BrunelWangNeuron::new();
        state.v = f64::INFINITY;
        assert!(state.step(1.0, 0.0, 0.0, 0.0).is_err());
    }
}
