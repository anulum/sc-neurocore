// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for compte_wm

#![allow(dead_code)]

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const GATE_MAX: f64 = 1.0e6;
const GABA_TAU: f64 = 5.0;

#[derive(Debug, Clone)]
pub struct CompteWMNeuron {
    pub v: f64,
    pub s_ampa: f64,
    pub s_nmda: f64,
    pub x_nmda: f64,
    pub s_gaba: f64,
    pub g_l: f64,
    pub g_ampa: f64,
    pub g_nmda: f64,
    pub g_gaba: f64,
    pub e_l: f64,
    pub e_exc: f64,
    pub e_inh: f64,
    pub c_m: f64,
    pub mg: f64,
    pub tau_ampa: f64,
    pub tau_nmda: f64,
    pub tau_x: f64,
    pub alpha_nmda: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl CompteWMNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            s_ampa: 0.0,
            s_nmda: 0.0,
            x_nmda: 0.0,
            s_gaba: 0.0,
            g_l: 0.025,
            g_ampa: 0.005,
            g_nmda: 0.165,
            g_gaba: 0.013,
            e_l: -70.0,
            e_exc: 0.0,
            e_inh: -70.0,
            c_m: 0.5,
            mg: 1.0,
            tau_ampa: 2.0,
            tau_nmda: 100.0,
            tau_x: 2.0,
            alpha_nmda: 0.5,
            v_threshold: -50.0,
            v_reset: -55.0,
            dt: 0.1,
        }
    }

    fn gate(value: f64) -> bool {
        value.is_finite() && (0.0..=GATE_MAX).contains(&value)
    }
    fn nonnegative(value: f64) -> bool {
        value.is_finite() && value >= 0.0
    }

    fn decay(&self, tau: f64) -> Result<f64, &'static str> {
        let ratio = -self.dt / tau;
        let decay = if ratio < -700.0 { 0.0 } else { ratio.exp() };
        if !decay.is_finite() || !(0.0..1.0).contains(&decay) {
            return Err("decay must be in [0, 1)");
        }
        Ok(decay)
    }

    fn validate(&self) -> Result<(f64, f64, f64), &'static str> {
        if !self.v.is_finite() || !(V_MIN..=V_MAX).contains(&self.v) {
            return Err("v outside Compte WM safety envelope");
        }
        if !Self::gate(self.s_ampa)
            || !Self::gate(self.s_nmda)
            || !Self::gate(self.x_nmda)
            || !Self::gate(self.s_gaba)
            || self.s_nmda > 1.0
        {
            return Err("synaptic gate outside Compte WM safety envelope");
        }
        for value in [
            self.g_l,
            self.g_ampa,
            self.g_nmda,
            self.g_gaba,
            self.mg,
            self.alpha_nmda,
        ] {
            if !Self::nonnegative(value) {
                return Err("non-negative Compte parameter invalid");
            }
        }
        for value in [self.c_m, self.tau_ampa, self.tau_nmda, self.tau_x, self.dt] {
            if !value.is_finite() || value <= 0.0 {
                return Err("positive Compte parameter invalid");
            }
        }
        for value in [
            self.e_l,
            self.e_exc,
            self.e_inh,
            self.v_threshold,
            self.v_reset,
        ] {
            if !value.is_finite() {
                return Err("finite Compte parameter invalid");
            }
        }
        if !(V_MIN..=V_MAX).contains(&self.v_reset) {
            return Err("v_reset outside Compte WM safety envelope");
        }
        Ok((
            self.decay(self.tau_ampa)?,
            self.decay(self.tau_x)?,
            self.decay(GABA_TAU)?,
        ))
    }

    pub fn mg_block(&self, v: f64) -> Result<f64, &'static str> {
        if !v.is_finite() {
            return Err("v must be finite");
        }
        let exponent = -0.062 * v;
        let exp_value = if exponent < -700.0 {
            0.0
        } else {
            exponent.min(700.0).exp()
        };
        let denominator = 1.0 + self.mg / 3.57 * exp_value;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("Mg block denominator invalid");
        }
        let block = 1.0 / denominator;
        if !(0.0..=1.0).contains(&block) {
            return Err("Mg block outside [0, 1]");
        }
        Ok(block)
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        self.step_with_spike(i_ext, false)
    }

    pub fn step_with_spike(&mut self, i_ext: f64, spike_in: bool) -> Result<i32, &'static str> {
        if !i_ext.is_finite() {
            return Err("current must be finite");
        }
        let (decay_ampa, decay_x, decay_gaba) = self.validate()?;
        let spike_increment = if spike_in { 1.0 } else { 0.0 };
        let s_ampa_pre = self.s_ampa + spike_increment;
        let x_nmda_pre = self.x_nmda + spike_increment;
        if s_ampa_pre > GATE_MAX || x_nmda_pre > GATE_MAX {
            return Err("spike input gate candidate outside Compte safety envelope");
        }
        let s_ampa_candidate = s_ampa_pre * decay_ampa;
        let s_nmda_candidate = self.s_nmda
            + (-self.s_nmda / self.tau_nmda + self.alpha_nmda * x_nmda_pre * (1.0 - self.s_nmda))
                * self.dt;
        let x_nmda_candidate = x_nmda_pre * decay_x;
        let s_gaba_candidate = self.s_gaba * decay_gaba;
        for value in [
            s_ampa_candidate,
            s_nmda_candidate,
            x_nmda_candidate,
            s_gaba_candidate,
        ] {
            if !value.is_finite() || value < 0.0 || value > GATE_MAX {
                return Err("gate candidate outside Compte safety envelope");
            }
        }
        if s_nmda_candidate > 1.0 {
            return Err("NMDA gate candidate must remain bounded by 1");
        }
        let block = self.mg_block(self.v)?;
        let i_l = self.g_l * (self.v - self.e_l);
        let i_ampa = self.g_ampa * s_ampa_candidate * (self.v - self.e_exc);
        let i_nmda = self.g_nmda * block * s_nmda_candidate * (self.v - self.e_exc);
        let i_gaba = self.g_gaba * s_gaba_candidate * (self.v - self.e_inh);
        let dv = (-i_l - i_ampa - i_nmda - i_gaba + i_ext) / self.c_m * self.dt;
        let v_candidate = self.v + dv;
        for value in [i_l, i_ampa, i_nmda, i_gaba, dv, v_candidate] {
            if !value.is_finite() {
                return Err("Compte current candidate must be finite");
            }
        }
        if !(V_MIN..=V_MAX).contains(&v_candidate) {
            return Err("voltage candidate outside Compte WM safety envelope");
        }
        if v_candidate >= self.v_threshold {
            let gaba_after_spike = s_gaba_candidate + 1.0;
            if gaba_after_spike > GATE_MAX {
                return Err("GABA spike candidate outside Compte safety envelope");
            }
            self.v = self.v_reset;
            self.s_ampa = s_ampa_candidate;
            self.s_nmda = s_nmda_candidate;
            self.x_nmda = x_nmda_candidate;
            self.s_gaba = gaba_after_spike;
            return Ok(1);
        }
        self.v = v_candidate;
        self.s_ampa = s_ampa_candidate;
        self.s_nmda = s_nmda_candidate;
        self.x_nmda = x_nmda_candidate;
        self.s_gaba = s_gaba_candidate;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.s_ampa = 0.0;
        self.s_nmda = 0.0;
        self.x_nmda = 0.0;
        self.s_gaba = 0.0;
    }
}

pub fn validate_compte_wm(state: &CompteWMNeuron) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spike_input_activates_nmda_pathway() {
        let mut state = CompteWMNeuron::new();
        assert_eq!(state.step_with_spike(0.0, true).unwrap(), 0);
        assert!(state.s_ampa > 0.0 && state.x_nmda > 0.0 && state.s_nmda > 0.0);
    }

    #[test]
    fn invalid_gate_state_is_rejected_without_mutation() {
        let mut state = CompteWMNeuron::new();
        state.s_nmda = 1.1;
        let before = (
            state.v,
            state.s_ampa,
            state.s_nmda,
            state.x_nmda,
            state.s_gaba,
        );
        assert!(state.step(0.0).is_err());
        assert_eq!(
            (
                state.v,
                state.s_ampa,
                state.s_nmda,
                state.x_nmda,
                state.s_gaba
            ),
            before
        );
    }

    #[test]
    fn mg_block_is_voltage_dependent() {
        let state = CompteWMNeuron::new();
        let low = state.mg_block(-80.0).unwrap();
        let high = state.mg_block(0.0).unwrap();
        assert!(low < high && high <= 1.0);
    }
}
