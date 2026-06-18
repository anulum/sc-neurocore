// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for coba_lif

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const G_MAX: f64 = 1.0e9;

#[derive(Debug, Clone)]
pub struct COBALIFNeuron {
    pub v: f64,
    pub g_e: f64,
    pub g_i: f64,
    pub c_m: f64,
    pub g_l: f64,
    pub e_l: f64,
    pub e_e: f64,
    pub e_i: f64,
    pub tau_e: f64,
    pub tau_i: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl COBALIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            g_e: 0.0,
            g_i: 0.0,
            c_m: 200.0,
            g_l: 10.0,
            e_l: -65.0,
            e_e: 0.0,
            e_i: -80.0,
            tau_e: 5.0,
            tau_i: 10.0,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }

    fn finite(value: f64) -> bool {
        value.is_finite()
    }
    fn nonnegative(value: f64) -> bool {
        value.is_finite() && value >= 0.0
    }

    fn validate(&self) -> Result<(), &'static str> {
        if !Self::finite(self.v) || !(V_MIN..=V_MAX).contains(&self.v) {
            return Err("v outside COBA LIF safety envelope");
        }
        if !Self::nonnegative(self.g_e)
            || !Self::nonnegative(self.g_i)
            || self.g_e > G_MAX
            || self.g_i > G_MAX
        {
            return Err("conductance outside COBA LIF safety envelope");
        }
        for value in [self.c_m, self.tau_e, self.tau_i, self.dt] {
            if !Self::finite(value) || value <= 0.0 {
                return Err("positive COBA LIF parameter invalid");
            }
        }
        if !Self::nonnegative(self.g_l) {
            return Err("leak conductance invalid");
        }
        for value in [self.e_l, self.e_e, self.e_i, self.v_threshold, self.v_reset] {
            if !Self::finite(value) {
                return Err("finite COBA LIF parameter invalid");
            }
        }
        if !(V_MIN..=V_MAX).contains(&self.v_reset) {
            return Err("v_reset outside COBA LIF safety envelope");
        }
        Ok(())
    }

    fn derivatives(&self, v: f64, g_e: f64, g_i: f64, i_ext: f64) -> (f64, f64, f64) {
        let i_syn = g_e * (v - self.e_e) + g_i * (v - self.e_i);
        let dv = (-self.g_l * (v - self.e_l) - i_syn + i_ext) / self.c_m;
        (dv, -g_e / self.tau_e, -g_i / self.tau_i)
    }

    fn rk4_candidate(&self, v: f64, g_e: f64, g_i: f64, i_ext: f64) -> (f64, f64, f64) {
        let (k1v, k1e, k1i) = self.derivatives(v, g_e, g_i, i_ext);
        let (k2v, k2e, k2i) = self.derivatives(
            v + 0.5 * self.dt * k1v,
            g_e + 0.5 * self.dt * k1e,
            g_i + 0.5 * self.dt * k1i,
            i_ext,
        );
        let (k3v, k3e, k3i) = self.derivatives(
            v + 0.5 * self.dt * k2v,
            g_e + 0.5 * self.dt * k2e,
            g_i + 0.5 * self.dt * k2i,
            i_ext,
        );
        let (k4v, k4e, k4i) = self.derivatives(
            v + self.dt * k3v,
            g_e + self.dt * k3e,
            g_i + self.dt * k3i,
            i_ext,
        );
        (
            v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            g_e + (self.dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
            g_i + (self.dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
        )
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        self.step_with_conductance(i_ext, 0.0, 0.0)
    }

    pub fn step_with_conductance(
        &mut self,
        i_ext: f64,
        delta_ge: f64,
        delta_gi: f64,
    ) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !Self::nonnegative(delta_ge) || !Self::nonnegative(delta_gi) {
            return Err("invalid COBA LIF step input");
        }
        self.validate()?;
        let ge_pre = self.g_e + delta_ge;
        let gi_pre = self.g_i + delta_gi;
        if ge_pre > G_MAX || gi_pre > G_MAX {
            return Err("conductance candidate outside COBA LIF safety envelope");
        }
        let i_syn = ge_pre * (self.v - self.e_e) + gi_pre * (self.v - self.e_i);
        let (v_candidate, ge_candidate, gi_candidate) =
            self.rk4_candidate(self.v, ge_pre, gi_pre, i_ext);
        for value in [i_syn, v_candidate, ge_candidate, gi_candidate] {
            if !value.is_finite() {
                return Err("COBA LIF candidate must be finite");
            }
        }
        if !(V_MIN..=V_MAX).contains(&v_candidate) {
            return Err("voltage candidate outside COBA LIF safety envelope");
        }
        if ge_candidate < 0.0 || gi_candidate < 0.0 {
            return Err("conductance candidate must remain non-negative");
        }
        if v_candidate >= self.v_threshold {
            self.v = self.v_reset;
            self.g_e = ge_candidate;
            self.g_i = gi_candidate;
            return Ok(1);
        }
        self.v = v_candidate;
        self.g_e = ge_candidate;
        self.g_i = gi_candidate;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.g_e = 0.0;
        self.g_i = 0.0;
    }
}

pub fn validate_coba_lif(state: &COBALIFNeuron) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conductance_injection_uses_coupled_rk4_candidate() {
        let mut state = COBALIFNeuron::new();
        let (v_candidate, ge_candidate, gi_candidate) = state.rk4_candidate(state.v, 5.0, 3.0, 0.0);
        assert_eq!(state.step_with_conductance(0.0, 5.0, 3.0).unwrap(), 0);
        assert!((state.v - v_candidate).abs() < 1.0e-12);
        assert!((state.g_e - ge_candidate).abs() < 1.0e-12);
        assert!((state.g_i - gi_candidate).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_runtime_state_is_rejected_without_mutation() {
        let mut state = COBALIFNeuron::new();
        state.g_e = -1.0;
        let before = (state.v, state.g_e, state.g_i);
        assert!(state.step(0.0).is_err());
        assert_eq!((state.v, state.g_e, state.g_i), before);
    }

    #[test]
    fn suprathreshold_drive_resets_voltage() {
        let mut state = COBALIFNeuron::new();
        state.v = -51.0;
        assert_eq!(state.step_with_conductance(1.0e5, 5.0, 0.0).unwrap(), 1);
        assert_eq!(state.v, state.v_reset);
        assert!(state.g_e > 0.0);
    }
}
