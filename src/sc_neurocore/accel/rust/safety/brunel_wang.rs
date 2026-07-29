// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — dependency-free Brunel-Wang 2001 safety mirror

/// Complete dynamic/configuration state for the source pyramidal-cell boundary.
#[derive(Debug, Clone)]
pub struct BrunelWangNeuron {
    /// Membrane potential in mV.
    pub v: f64,
    /// Leak reversal potential in mV.
    pub v_rest: f64,
    /// Post-event reset potential in mV.
    pub v_reset: f64,
    /// Sampled firing threshold in mV.
    pub v_threshold: f64,
    /// Membrane time constant in ms.
    pub tau_m: f64,
    /// Absolute refractory interval in ms.
    pub tau_ref: f64,
    /// External AMPA conductance in nS.
    pub g_ampa_ext: f64,
    /// Recurrent AMPA conductance in nS.
    pub g_ampa_rec: f64,
    /// Recurrent NMDA conductance in nS.
    pub g_nmda: f64,
    /// Recurrent GABA conductance in nS.
    pub g_gaba: f64,
    /// Excitatory reversal potential in mV.
    pub v_ampa: f64,
    /// NMDA reversal potential in mV.
    pub v_nmda: f64,
    /// Inhibitory reversal potential in mV.
    pub v_gaba: f64,
    /// Membrane capacitance in nF.
    pub c_m: f64,
    /// Extracellular magnesium concentration in mM.
    pub mg_conc: f64,
    /// Integration interval in ms.
    pub dt: f64,
    /// Remaining refractory interval in ms.
    pub ref_remaining: f64,
}

impl BrunelWangNeuron {
    /// Construct Brunel and Wang's pyramidal-cell defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -55.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            tau_ref: 2.0,
            g_ampa_ext: 2.08,
            g_ampa_rec: 0.104,
            g_nmda: 0.327,
            g_gaba: 1.25,
            v_ampa: 0.0,
            v_nmda: 0.0,
            v_gaba: -70.0,
            c_m: 0.5,
            mg_conc: 1.0,
            dt: 0.1,
            ref_remaining: 0.0,
        }
    }

    /// Advance one atomic midpoint-RK2 step over aggregate channel gates.
    pub fn step(
        &mut self,
        s_ampa_ext: f64,
        s_ampa_rec: f64,
        s_nmda_rec: f64,
        s_gaba: f64,
    ) -> Result<i32, &'static str> {
        if !validate_brunel_wang(self)
            || ![s_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba]
                .iter()
                .all(|value| nonnegative(*value))
        {
            return Err("invalid Brunel-Wang configuration or aggregate gate");
        }
        if self.ref_remaining > 0.0 {
            self.v = self.v_reset;
            self.ref_remaining = (self.ref_remaining - self.dt).max(0.0);
            return Ok(0);
        }
        let v = self.v;
        let k1 = self.derivative(v, s_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba);
        let midpoint = v + 0.5 * self.dt * k1;
        let k2 = self.derivative(midpoint, s_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba);
        let candidate = v + self.dt * k2;
        if !k1.is_finite() || !midpoint.is_finite() || !k2.is_finite() || !candidate.is_finite() {
            return Err("non-finite Brunel-Wang RK2 candidate");
        }
        self.v = candidate;
        if candidate >= self.v_threshold {
            self.v = self.v_reset;
            self.ref_remaining = self.tau_ref;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    fn derivative(&self, v: f64, ext: f64, ampa: f64, nmda: f64, gaba: f64) -> f64 {
        let i_ampa =
            -self.g_ampa_ext * (v - self.v_ampa) * ext - self.g_ampa_rec * (v - self.v_ampa) * ampa;
        let block = 1.0 / (1.0 + self.mg_conc / 3.57 * (-0.062 * v).exp());
        let i_nmda = -self.g_nmda * block * (v - self.v_nmda) * nmda;
        let i_gaba = -self.g_gaba * (v - self.v_gaba) * gaba;
        -(v - self.v_rest) / self.tau_m + (i_ampa + i_nmda + i_gaba) / self.c_m
    }

    /// Reset only dynamic state; configuration is preserved.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ref_remaining = 0.0;
    }

    /// Return the complete dynamic state.
    #[must_use]
    pub fn get_state(&self) -> (f64, f64) {
        (self.v, self.ref_remaining)
    }
}

impl Default for BrunelWangNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate all mutable configuration and dynamic state fields.
#[must_use]
pub fn validate_brunel_wang(state: &BrunelWangNeuron) -> bool {
    [
        state.v,
        state.v_rest,
        state.v_reset,
        state.v_threshold,
        state.tau_m,
        state.tau_ref,
        state.g_ampa_ext,
        state.g_ampa_rec,
        state.g_nmda,
        state.g_gaba,
        state.v_ampa,
        state.v_nmda,
        state.v_gaba,
        state.c_m,
        state.mg_conc,
        state.dt,
        state.ref_remaining,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.tau_m > 0.0
        && state.tau_ref > 0.0
        && state.c_m > 0.0
        && state.dt > 0.0
        && state.g_ampa_ext >= 0.0
        && state.g_ampa_rec >= 0.0
        && state.g_nmda >= 0.0
        && state.g_gaba >= 0.0
        && state.mg_conc >= 0.0
        && state.ref_remaining >= 0.0
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_source_values() {
        let state = BrunelWangNeuron::new();
        assert_eq!((state.g_ampa_ext, state.g_ampa_rec), (2.08, 0.104));
        assert_eq!((state.g_nmda, state.g_gaba), (0.327, 1.25));
    }

    #[test]
    fn failure_is_atomic() {
        let mut state = BrunelWangNeuron::new();
        let before = state.get_state();
        assert!(state.step(f64::NAN, 0.0, 0.0, 0.0).is_err());
        assert_eq!(state.get_state(), before);
    }

    #[test]
    fn reset_preserves_configuration() {
        let mut state = BrunelWangNeuron::new();
        state.g_nmda = 0.4;
        state.step(0.2, 0.1, 0.3, 0.0).unwrap();
        state.reset();
        assert_eq!(state.g_nmda, 0.4);
        assert_eq!(state.get_state(), (-70.0, 0.0));
    }
}
