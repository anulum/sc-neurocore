// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sherman-Rinzel-Keizer Neuron Model

//! Sherman-Rinzel-Keizer pancreatic beta-cell dynamics.

#![deny(missing_docs)]

/// Sherman-Rinzel-Keizer 1988 — pancreatic beta cell (reduced).
#[derive(Clone, Debug)]
pub struct ShermanRinzelKeizerNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Fast potassium activation; relaxes with a fixed 9.09 ms time constant, and the
    /// potassium current is linear in it.
    pub n: f64,
    /// Slow potassium activation; relaxes with `tau_s`.
    pub s: f64,
    /// Maximal calcium conductance (mS/cm^2).
    pub g_ca: f64,
    /// Maximal potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Maximal conductance of the slow current (mS/cm^2).
    pub g_s: f64,
    /// Calcium reversal potential (mV).
    pub e_ca: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Time constant of the slow gating variable (ms).
    pub tau_s: f64,
    /// Forward-Euler step (ms); one per [`ShermanRinzelKeizerNeuron::step`] call.
    pub dt: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
}

impl ShermanRinzelKeizerNeuron {
    /// Create a neuron with the default parameters and initial state of its model descriptor.
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            s: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s: 4.0,
            e_ca: 25.0,
            e_k: -75.0,
            tau_s: 5000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }
    /// Advance the neuron by one fourth-order Runge-Kutta step of `dt` with the injected
    /// current `current`.
    ///
    /// The voltage equation has unit capacitance and no leak, and calcium activation is
    /// taken at its steady state. The steady-state sigmoids clamp their argument to
    /// [-80, 80], and the four slopes and their combination are evaluated in the same order
    /// of operations as the Python model, so the engine follows it bit for bit. Returns 1 when
    /// the membrane potential starts below `v_threshold` and ends at or above it, else 0. The
    /// input and the state are not validated, so a non-finite current propagates into the
    /// state.
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let (next_v, next_n, next_s) = self.rk4_candidate(current);
        self.v = next_v;
        self.n = next_n;
        self.s = next_s;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    fn sigmoid(arg: f64) -> f64 {
        let clamped = arg.clamp(-80.0, 80.0);
        1.0 / (1.0 + (-clamped).exp())
    }

    fn derivatives(&self, v: f64, n_gate: f64, s_gate: f64, current: f64) -> (f64, f64, f64) {
        let m_inf = Self::sigmoid((v + 20.0) / 12.0);
        let n_inf = Self::sigmoid((v + 16.0) / 5.0);
        let s_inf = Self::sigmoid((v + 35.0) / 10.0);
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * n_gate * (v - self.e_k);
        let i_s = self.g_s * s_gate * (v - self.e_k);
        let dv = -i_ca - i_k - i_s + current;
        let dn = (n_inf - n_gate) / 9.09;
        let ds = (s_inf - s_gate) / self.tau_s;
        (dv, dn, ds)
    }

    fn rk4_candidate(&self, current: f64) -> (f64, f64, f64) {
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(self.v, self.n, self.s, current);
        let k2 = self.derivatives(
            self.v + half_dt * k1.0,
            self.n + half_dt * k1.1,
            self.s + half_dt * k1.2,
            current,
        );
        let k3 = self.derivatives(
            self.v + half_dt * k2.0,
            self.n + half_dt * k2.1,
            self.s + half_dt * k2.2,
            current,
        );
        let k4 = self.derivatives(
            self.v + self.dt * k3.0,
            self.n + self.dt * k3.1,
            self.s + self.dt * k3.2,
            current,
        );
        (
            self.v + self.dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            self.n + self.dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
            self.s + self.dt * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0,
        )
    }
    /// Return the membrane potential and the gates to their initial values; the parameters
    /// are kept.
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.s = 0.1;
    }
}
impl Default for ShermanRinzelKeizerNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srk_rk4_matches_the_python_model_bit_for_bit() {
        // Python's ShermanRinzelKeizerNeuron after 400 steps of 0.0 from its defaults: seven
        // upward crossings, v = -62.16970656932478, n = 0.018485292025607728 and
        // s = 0.10156391845310253.
        let mut n = ShermanRinzelKeizerNeuron::new();
        let crossings: i32 = (0..400).map(|_| n.step(0.0)).sum();
        assert_eq!(crossings, 7);
        assert_eq!(n.v, -62.16970656932478);
        assert_eq!(n.n, 0.018485292025607728);
        assert_eq!(n.s, 0.10156391845310253);
    }

    #[test]
    fn default_matches_constructor_state() {
        let default = ShermanRinzelKeizerNeuron::default();
        let constructed = ShermanRinzelKeizerNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn srk_fires() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn srk_reset_clears_state() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..1000 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn srk_bounded() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn srk_nan_no_panic() {
        ShermanRinzelKeizerNeuron::new().step(f64::NAN);
    }

    #[test]
    fn srk_negative_no_crash() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
