// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wang 1999 NMDA-autapse pyramidal neuron

#![deny(missing_docs)]

/// Wang (1999) pyramidal LIF neuron with two-stage NMDA-autapse kinetics.
#[derive(Clone, Debug)]
pub struct NMDANeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// NMDA transmitter variable (dimensionless); it decays with `tau_x` and gains
    /// `kinetic_scale * alpha_x` at each emitted spike.
    pub x_nmda: f64,
    /// NMDA gate in `[0, 1]`, opened by `x_nmda` with saturation and decaying with `tau_s`.
    pub s_nmda: f64,
    /// Calcium variable (dimensionless) driving the AHP current; it decays with `tau_ca` and
    /// gains `alpha_ca` at each spike.
    pub ca: f64,
    /// Refractory time left (ms), in `[0, refractory_period]`.
    pub refractory_remaining: f64,
    /// Pyramidal-cell membrane capacitance (nF).
    pub c_m: f64,
    /// Leak conductance (uS).
    pub g_l: f64,
    /// Leak reversal potential (mV).
    pub v_l: f64,
    /// NMDA autapse conductance used by the source single-cell experiment (uS).
    pub g_nmda: f64,
    /// NMDA reversal potential (mV).
    pub e_nmda: f64,
    /// Extracellular magnesium concentration (mM).
    pub mg_conc: f64,
    /// Presynaptic NMDA transmitter increment per emitted event (dimensionless).
    pub alpha_x: f64,
    /// NMDA transmitter decay time constant (ms).
    pub tau_x: f64,
    /// Saturating NMDA gate opening factor (ms^-1).
    pub alpha_s: f64,
    /// NMDA gate decay time constant (ms).
    pub tau_s: f64,
    /// Explicit scaling of the NMDA kinetic equations (dimensionless).
    pub kinetic_scale: f64,
    /// Optional calcium-activated potassium adaptation conductance (uS).
    pub g_ahp: f64,
    /// Potassium reversal potential (mV).
    pub v_k: f64,
    /// Calcium increment per emitted event (dimensionless).
    pub alpha_ca: f64,
    /// Calcium decay time constant (ms).
    pub tau_ca: f64,
    /// Step (ms), in `(0, 0.05]`; each call takes one midpoint (RK2) step.
    pub dt: f64,
    /// Pyramidal-cell spike threshold (mV).
    pub v_threshold: f64,
    /// Post-spike reset potential (mV).
    pub v_reset: f64,
    /// Absolute refractory period (ms).
    pub refractory_period: f64,
}

impl NMDANeuron {
    /// Create a neuron at rest (`v = v_l`, no NMDA, calcium or refractory activity) with the
    /// descriptor defaults and `dt` = 0.05 ms.
    pub fn new() -> Self {
        Self {
            v: -70.0,
            x_nmda: 0.0,
            s_nmda: 0.0,
            ca: 0.0,
            refractory_remaining: 0.0,
            c_m: 0.5,
            g_l: 0.025,
            v_l: -70.0,
            g_nmda: 0.1,
            e_nmda: 0.0,
            mg_conc: 1.0,
            alpha_x: 1.0,
            tau_x: 2.0,
            alpha_s: 1.0,
            tau_s: 80.0,
            kinetic_scale: 1.0,
            g_ahp: 0.0,
            v_k: -85.0,
            alpha_ca: 0.2,
            tau_ca: 80.0,
            dt: 0.05,
            v_threshold: -52.0,
            v_reset: -59.0,
            refractory_period: 2.0,
        }
    }

    fn valid(&self) -> bool {
        let finite = [
            self.v,
            self.x_nmda,
            self.s_nmda,
            self.ca,
            self.refractory_remaining,
            self.c_m,
            self.g_l,
            self.v_l,
            self.g_nmda,
            self.e_nmda,
            self.mg_conc,
            self.alpha_x,
            self.tau_x,
            self.alpha_s,
            self.tau_s,
            self.kinetic_scale,
            self.g_ahp,
            self.v_k,
            self.alpha_ca,
            self.tau_ca,
            self.dt,
            self.v_threshold,
            self.v_reset,
            self.refractory_period,
        ]
        .into_iter()
        .all(f64::is_finite);
        finite
            && (-120.0..=80.0).contains(&self.v)
            && self.x_nmda >= 0.0
            && (0.0..=1.0).contains(&self.s_nmda)
            && self.ca >= 0.0
            && (0.0..=self.refractory_period).contains(&self.refractory_remaining)
            && (0.01..=10.0).contains(&self.c_m)
            && (0.0..=1.0).contains(&self.g_l)
            && (-100.0..=-40.0).contains(&self.v_l)
            && (0.0..=2.0).contains(&self.g_nmda)
            && (-10.0..=10.0).contains(&self.e_nmda)
            && (0.0..=5.0).contains(&self.mg_conc)
            && (0.0..=10.0).contains(&self.alpha_x)
            && (0.01..=100.0).contains(&self.tau_x)
            && (0.0..=10.0).contains(&self.alpha_s)
            && (1.0..=1000.0).contains(&self.tau_s)
            && (0.01..=100.0).contains(&self.kinetic_scale)
            && (0.0..=10.0).contains(&self.g_ahp)
            && (-120.0..=-40.0).contains(&self.v_k)
            && (0.0..=10.0).contains(&self.alpha_ca)
            && (1.0..=1000.0).contains(&self.tau_ca)
            && self.dt > 0.0
            && self.dt <= 0.05
            && (-80.0..=-30.0).contains(&self.v_threshold)
            && self.v_reset >= -100.0
            && self.v_reset < self.v_threshold
            && (0.0..=20.0).contains(&self.refractory_period)
    }

    fn derivatives(&self, v: f64, x_nmda: f64, s_nmda: f64, ca: f64, current: f64) -> [f64; 4] {
        let mg_block = 1.0 / (1.0 + self.mg_conc * (-0.062 * v).exp() / 3.57);
        let i_l = self.g_l * (v - self.v_l);
        let i_ahp = self.g_ahp * ca * (v - self.v_k);
        let i_nmda = self.g_nmda * s_nmda * mg_block * (v - self.e_nmda);
        [
            (-i_l - i_ahp - i_nmda + current) / self.c_m,
            self.kinetic_scale * (-x_nmda / self.tau_x),
            self.kinetic_scale * (self.alpha_s * x_nmda * (1.0 - s_nmda) - s_nmda / self.tau_s),
            -ca / self.tau_ca,
        ]
    }

    fn rk2_candidate(&self, v: f64, current: f64) -> [f64; 4] {
        let state = [v, self.x_nmda, self.s_nmda, self.ca];
        let k1 = self.derivatives(state[0], state[1], state[2], state[3], current);
        let half_dt = 0.5 * self.dt;
        let midpoint = [
            state[0] + half_dt * k1[0],
            state[1] + half_dt * k1[1],
            state[2] + half_dt * k1[2],
            state[3] + half_dt * k1[3],
        ];
        let k2 = self.derivatives(midpoint[0], midpoint[1], midpoint[2], midpoint[3], current);
        [
            state[0] + self.dt * k2[0],
            state[1] + self.dt * k2[1],
            state[2] + self.dt * k2[2],
            state[3] + self.dt * k2[3],
        ]
    }

    /// Advance one source-grid step atomically.
    ///
    /// One midpoint (RK2) step of the membrane, NMDA and calcium equations, with the
    /// magnesium block `1 / (1 + mg_conc * exp(-0.062 * v) / 3.57)`. While refractory, the
    /// step starts from and holds `v_reset`. Otherwise reaching `v_threshold` is a spike: `v`
    /// is set to `v_reset`, the refractory period starts, `x_nmda` gains
    /// `kinetic_scale * alpha_x` and `ca` gains `alpha_ca`. Returns 1 on a spike, else 0.
    ///
    /// Returns an error, with the state unchanged, if `current` is non-finite, a state or
    /// parameter is non-finite or outside its public bound, or the candidate is
    /// non-finite.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !self.valid() {
            return Err("NMDA state and parameters must satisfy the public bounds");
        }
        let held = self.refractory_remaining > 0.0;
        let voltage = if held { self.v_reset } else { self.v };
        let mut next = self.rk2_candidate(voltage, current);
        let mut refractory = (self.refractory_remaining - self.dt).max(0.0);
        let mut fired = 0;
        if held {
            next[0] = self.v_reset;
        } else if next[0] >= self.v_threshold {
            fired = 1;
            next[0] = self.v_reset;
            refractory = self.refractory_period;
            next[1] += self.kinetic_scale * self.alpha_x;
            next[3] += self.alpha_ca;
        }
        if !next.into_iter().chain([refractory]).all(f64::is_finite) {
            return Err("NMDA candidate state became non-finite");
        }
        self.v = next[0].clamp(-120.0, 80.0);
        self.x_nmda = next[1].max(0.0);
        self.s_nmda = next[2].clamp(0.0, 1.0);
        self.ca = next[3].max(0.0);
        self.refractory_remaining = refractory;
        Ok(fired)
    }

    /// Legacy fail-closed NetworkRunner surface.
    ///
    /// Returns 1 on a spike and 0 otherwise; an error also returns 0 and leaves the state
    /// unchanged.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Return `v` to `v_l` and clear the NMDA, calcium and refractory state; the parameters
    /// are kept.
    pub fn reset(&mut self) {
        self.v = self.v_l;
        self.x_nmda = 0.0;
        self.s_nmda = 0.0;
        self.ca = 0.0;
        self.refractory_remaining = 0.0;
    }
}

impl Default for NMDANeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_anchor_and_atomic_failure() {
        let mut state = NMDANeuron::new();
        assert_eq!(state.try_step(0.3), Ok(0));
        assert!((state.v - -69.970_037_5).abs() < 1.0e-12);
        let before = state.clone();
        assert!(state.try_step(f64::NAN).is_err());
        assert_eq!(state.v, before.v);
        assert_eq!(state.s_nmda, before.s_nmda);
    }
}
