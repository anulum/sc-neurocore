// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Compte et al. 2000 pyramidal cell

//! Source-bounded Compte pyramidal-cell and incoming channel dynamics.

const V_MIN: f64 = -200.0;
const V_MAX: f64 = 100.0;
const GATE_MAX: f64 = 1.0e6;

/// Complete Compte pyramidal-cell state and source control-set configuration.
///
/// Conductances are microSiemens, voltages millivolts, current nanoamps,
/// capacitance nanofarads, and time milliseconds. Incoming event flags map to
/// recurrent NMDA, external AMPA, and inhibitory GABAA pathways respectively.
#[derive(Clone, Debug)]
pub struct CompteWMNeuron {
    /// Membrane potential in mV.
    pub v: f64,
    /// External AMPA open-channel accumulator.
    pub s_ampa: f64,
    /// Recurrent NMDA open fraction.
    pub s_nmda: f64,
    /// Recurrent NMDA rise precursor.
    pub x_nmda: f64,
    /// Incoming GABAA open-channel accumulator.
    pub s_gaba: f64,
    /// Remaining absolute refractory duration in ms.
    pub ref_remaining: f64,
    /// Leak conductance in microSiemens.
    pub g_l: f64,
    /// External pyramidal AMPA conductance in microSiemens.
    pub g_ampa: f64,
    /// Recurrent pyramidal NMDA conductance in microSiemens.
    pub g_nmda: f64,
    /// Interneuron-to-pyramidal GABAA conductance in microSiemens.
    pub g_gaba: f64,
    /// Leak reversal in mV.
    pub e_l: f64,
    /// Excitatory reversal in mV.
    pub e_exc: f64,
    /// Inhibitory reversal in mV.
    pub e_inh: f64,
    /// Membrane capacitance in nF.
    pub c_m: f64,
    /// Extracellular magnesium concentration in mM.
    pub mg: f64,
    /// AMPA decay constant in ms.
    pub tau_ampa: f64,
    /// NMDA decay constant in ms.
    pub tau_nmda: f64,
    /// NMDA rise-precursor decay constant in ms.
    pub tau_x: f64,
    /// GABAA decay constant in ms.
    pub tau_gaba: f64,
    /// NMDA saturation rate in inverse ms.
    pub alpha_nmda: f64,
    /// Sampled firing threshold in mV.
    pub v_threshold: f64,
    /// Post-spike and refractory voltage in mV.
    pub v_reset: f64,
    /// Absolute refractory duration in ms.
    pub tau_ref: f64,
    /// Midpoint-RK2 integration step in ms.
    pub dt: f64,
}

impl CompteWMNeuron {
    /// Construct the Compte (2000) source control-set pyramidal defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            v: -70.0,
            s_ampa: 0.0,
            s_nmda: 0.0,
            x_nmda: 0.0,
            s_gaba: 0.0,
            ref_remaining: 0.0,
            g_l: 0.025,
            g_ampa: 0.0031,
            g_nmda: 0.000_381,
            g_gaba: 0.001_336,
            e_l: -70.0,
            e_exc: 0.0,
            e_inh: -70.0,
            c_m: 0.5,
            mg: 1.0,
            tau_ampa: 2.0,
            tau_nmda: 100.0,
            tau_x: 2.0,
            tau_gaba: 10.0,
            alpha_nmda: 0.5,
            v_threshold: -50.0,
            v_reset: -60.0,
            tau_ref: 2.0,
            dt: 0.02,
        }
    }

    /// Return whether all mutable state and configuration invariants hold.
    #[must_use]
    pub fn validate(&self) -> bool {
        let finite = [
            self.v,
            self.s_ampa,
            self.s_nmda,
            self.x_nmda,
            self.s_gaba,
            self.ref_remaining,
            self.g_l,
            self.g_ampa,
            self.g_nmda,
            self.g_gaba,
            self.e_l,
            self.e_exc,
            self.e_inh,
            self.c_m,
            self.mg,
            self.tau_ampa,
            self.tau_nmda,
            self.tau_x,
            self.tau_gaba,
            self.alpha_nmda,
            self.v_threshold,
            self.v_reset,
            self.tau_ref,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite());
        finite
            && (V_MIN..=V_MAX).contains(&self.v)
            && (V_MIN..=V_MAX).contains(&self.v_reset)
            && [self.s_ampa, self.x_nmda, self.s_gaba]
                .iter()
                .all(|value| (0.0..=GATE_MAX).contains(value))
            && (0.0..=1.0).contains(&self.s_nmda)
            && self.ref_remaining >= 0.0
            && [
                self.g_l,
                self.g_ampa,
                self.g_nmda,
                self.g_gaba,
                self.mg,
                self.alpha_nmda,
            ]
            .iter()
            .all(|value| *value >= 0.0)
            && [
                self.c_m,
                self.tau_ampa,
                self.tau_nmda,
                self.tau_x,
                self.tau_gaba,
                self.tau_ref,
                self.dt,
            ]
            .iter()
            .all(|value| *value > 0.0)
    }

    fn mg_block(&self, v: f64) -> Option<f64> {
        let exponent = -0.062 * v;
        let block = if exponent > 700.0 {
            0.0
        } else {
            1.0 / (1.0 + self.mg / 3.57 * exponent.exp())
        };
        (block.is_finite() && (0.0..=1.0).contains(&block)).then_some(block)
    }

    fn derivatives(
        &self,
        state: [f64; 5],
        current: f64,
        membrane_active: bool,
    ) -> Option<[f64; 5]> {
        let [v, s_ampa, s_nmda, x_nmda, s_gaba] = state;
        let d_ampa = -s_ampa / self.tau_ampa;
        let d_nmda = -s_nmda / self.tau_nmda + self.alpha_nmda * x_nmda * (1.0 - s_nmda);
        let d_x = -x_nmda / self.tau_x;
        let d_gaba = -s_gaba / self.tau_gaba;
        let d_v = if membrane_active {
            let i_l = self.g_l * (v - self.e_l);
            let i_ampa = self.g_ampa * s_ampa * (v - self.e_exc);
            let i_nmda = self.g_nmda * self.mg_block(v)? * s_nmda * (v - self.e_exc);
            let i_gaba = self.g_gaba * s_gaba * (v - self.e_inh);
            (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m
        } else {
            0.0
        };
        let result = [d_v, d_ampa, d_nmda, d_x, d_gaba];
        result
            .iter()
            .all(|value| value.is_finite())
            .then_some(result)
    }

    /// Advance one atomic source-level midpoint-RK2 step.
    ///
    /// The three flags apply recurrent-NMDA, external-AMPA, and
    /// inhibitory-GABAA event jumps before the continuous flow. Threshold
    /// detection is sampled; within-step firing-time interpolation is not
    /// claimed. An error leaves all public state unchanged.
    pub fn step_events(
        &mut self,
        current: f64,
        recurrent_event: bool,
        external_event: bool,
        inhibitory_event: bool,
    ) -> Result<i32, &'static str> {
        if !self.validate() || !current.is_finite() {
            return Err("invalid Compte state, configuration, or current");
        }
        let initial = [
            self.v,
            self.s_ampa + if external_event { 1.0 } else { 0.0 },
            self.s_nmda,
            self.x_nmda + if recurrent_event { 1.0 } else { 0.0 },
            self.s_gaba + if inhibitory_event { 1.0 } else { 0.0 },
        ];
        if !initial[1..]
            .iter()
            .all(|value| value.is_finite() && (0.0..=GATE_MAX).contains(value))
        {
            return Err("Compte event candidate outside gate envelope");
        }
        let active = self.ref_remaining <= 0.0;
        let k1 = self
            .derivatives(initial, current, active)
            .ok_or("non-finite Compte RK2 first stage")?;
        let midpoint = std::array::from_fn(|index| initial[index] + 0.5 * self.dt * k1[index]);
        let k2 = self
            .derivatives(midpoint, current, active)
            .ok_or("non-finite Compte RK2 midpoint stage")?;
        let mut candidate: [f64; 5] =
            std::array::from_fn(|index| initial[index] + self.dt * k2[index]);
        if !candidate.iter().all(|value| value.is_finite())
            || !(V_MIN..=V_MAX).contains(&candidate[0])
            || !candidate[1..]
                .iter()
                .all(|value| (0.0..=GATE_MAX).contains(value))
            || candidate[2] > 1.0
        {
            return Err("Compte RK2 candidate outside safety envelope");
        }
        let mut event = 0;
        let mut ref_remaining = (self.ref_remaining - self.dt).max(0.0);
        if !active {
            candidate[0] = self.v_reset;
        } else if candidate[0] >= self.v_threshold {
            candidate[0] = self.v_reset;
            ref_remaining = self.tau_ref;
            event = 1;
        }
        self.v = candidate[0];
        self.s_ampa = candidate[1];
        self.s_nmda = candidate[2];
        self.x_nmda = candidate[3];
        self.s_gaba = candidate[4];
        self.ref_remaining = ref_remaining;
        Ok(event)
    }

    /// Retain the generic two-argument engine adapter.
    ///
    /// The boolean retains its historical meaning as a recurrent excitatory
    /// NMDA event. Invalid state returns zero without mutation; callers that
    /// require explicit errors use the complete step_events method.
    pub fn step(&mut self, current: f64, recurrent_event: bool) -> i32 {
        self.step_events(current, recurrent_event, false, false)
            .unwrap_or(0)
    }

    /// Reset all dynamic state while preserving configuration.
    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.s_ampa = 0.0;
        self.s_nmda = 0.0;
        self.x_nmda = 0.0;
        self.s_gaba = 0.0;
        self.ref_remaining = 0.0;
    }

    /// Return the complete dynamic state in public trace order.
    #[must_use]
    pub fn get_state(&self) -> [f64; 6] {
        [
            self.v,
            self.s_ampa,
            self.s_nmda,
            self.x_nmda,
            self.s_gaba,
            self.ref_remaining,
        ]
    }
}

impl Default for CompteWMNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_pathways_are_separate() {
        let mut n = CompteWMNeuron::new();
        assert_eq!(n.step_events(0.0, true, false, false), Ok(0));
        assert_eq!(n.s_ampa, 0.0);
        assert!(n.s_nmda > 0.0 && n.x_nmda > 0.0);
        assert_eq!(n.s_gaba, 0.0);
    }

    #[test]
    fn failure_is_atomic() {
        let mut n = CompteWMNeuron::new();
        let before = n.get_state();
        assert!(n.step_events(f64::NAN, false, false, false).is_err());
        assert_eq!(n.get_state(), before);
    }

    #[test]
    fn reset_preserves_configuration() {
        let mut n = CompteWMNeuron::new();
        n.dt = 0.01;
        n.step_events(1.0, true, true, true).unwrap();
        n.reset();
        assert_eq!(n.get_state(), [-70.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(n.dt, 0.01);
    }
}
