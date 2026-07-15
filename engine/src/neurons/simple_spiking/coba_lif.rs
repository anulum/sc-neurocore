// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Conductance-Based LIF Neuron Model

//! Conductance-based leaky integrate-and-fire dynamics.

/// Conductance-based LIF (COBA). Brette et al. 2007.
#[derive(Clone, Debug)]
pub struct COBALIFNeuron {
    pub v: f64,
    pub g_e: f64,
    pub g_i: f64,
    pub refractory_time: f64,
    pub c_m: f64,
    pub g_l: f64,
    pub e_l: f64,
    pub e_e: f64,
    pub e_i: f64,
    pub tau_e: f64,
    pub tau_i: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refractory_period: f64,
    pub dt: f64,
}

impl COBALIFNeuron {
    const V_MIN: f64 = -200.0;
    const V_MAX: f64 = 100.0;
    const G_MAX: f64 = 1.0e9;

    pub fn new() -> Self {
        Self {
            v: -60.0,
            g_e: 0.0,
            g_i: 0.0,
            refractory_time: 0.0,
            c_m: 200.0,
            g_l: 10.0,
            e_l: -60.0,
            e_e: 0.0,
            e_i: -80.0,
            tau_e: 5.0,
            tau_i: 10.0,
            v_threshold: -50.0,
            v_reset: -60.0,
            refractory_period: 5.0,
            dt: 0.1,
        }
    }

    fn valid(&self) -> bool {
        self.v.is_finite()
            && (Self::V_MIN..=Self::V_MAX).contains(&self.v)
            && self.g_e.is_finite()
            && self.g_e >= 0.0
            && self.g_e <= Self::G_MAX
            && self.g_i.is_finite()
            && self.g_i >= 0.0
            && self.g_i <= Self::G_MAX
            && self.refractory_time.is_finite()
            && self.refractory_time >= 0.0
            && self.c_m.is_finite()
            && self.c_m > 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_l.is_finite()
            && self.e_e.is_finite()
            && self.e_i.is_finite()
            && self.tau_e.is_finite()
            && self.tau_e > 0.0
            && self.tau_i.is_finite()
            && self.tau_i > 0.0
            && self.v_threshold.is_finite()
            && self.v_reset.is_finite()
            && (Self::V_MIN..=Self::V_MAX).contains(&self.v_reset)
            && self.refractory_period.is_finite()
            && self.refractory_period > 0.0
            && self.refractory_time <= self.refractory_period
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.refractory_period >= self.dt
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.valid() {
            Ok(())
        } else {
            Err("invalid COBA LIF state or parameter contract")
        }
    }

    fn derivatives(&self, v: f64, g_e: f64, g_i: f64, current: f64) -> (f64, f64, f64) {
        let i_syn = g_e * (v - self.e_e) + g_i * (v - self.e_i);
        (
            (-self.g_l * (v - self.e_l) - i_syn + current) / self.c_m,
            -g_e / self.tau_e,
            -g_i / self.tau_i,
        )
    }

    fn rk4_candidate(&self, v: f64, g_e: f64, g_i: f64, current: f64) -> (f64, f64, f64) {
        let (k1v, k1e, k1i) = self.derivatives(v, g_e, g_i, current);
        let (k2v, k2e, k2i) = self.derivatives(
            v + 0.5 * self.dt * k1v,
            g_e + 0.5 * self.dt * k1e,
            g_i + 0.5 * self.dt * k1i,
            current,
        );
        let (k3v, k3e, k3i) = self.derivatives(
            v + 0.5 * self.dt * k2v,
            g_e + 0.5 * self.dt * k2e,
            g_i + 0.5 * self.dt * k2i,
            current,
        );
        let (k4v, k4e, k4i) = self.derivatives(
            v + self.dt * k3v,
            g_e + self.dt * k3e,
            g_i + self.dt * k3i,
            current,
        );
        (
            v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            g_e + (self.dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
            g_i + (self.dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
        )
    }

    fn conductance_candidates(&self, g_e: f64, g_i: f64) -> (f64, f64) {
        let decay = |value: f64, tau: f64| {
            let k1 = -value / tau;
            let k2 = -(value + 0.5 * self.dt * k1) / tau;
            let k3 = -(value + 0.5 * self.dt * k2) / tau;
            let k4 = -(value + self.dt * k3) / tau;
            value + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        };
        (decay(g_e, self.tau_e), decay(g_i, self.tau_i))
    }

    pub fn try_step(
        &mut self,
        current: f64,
        delta_ge: f64,
        delta_gi: f64,
    ) -> Result<i32, &'static str> {
        if !self.valid()
            || !current.is_finite()
            || !delta_ge.is_finite()
            || delta_ge < 0.0
            || !delta_gi.is_finite()
            || delta_gi < 0.0
        {
            return Err("invalid COBA LIF state or step input");
        }
        let ge_pre = self.g_e + delta_ge;
        let gi_pre = self.g_i + delta_gi;
        if !ge_pre.is_finite()
            || ge_pre > Self::G_MAX
            || !gi_pre.is_finite()
            || gi_pre > Self::G_MAX
        {
            return Err("conductance candidate outside COBA LIF safety envelope");
        }

        let (v_candidate, ge_candidate, gi_candidate, refractory_candidate, spiked) =
            if self.refractory_time > 0.0 {
                let (ge, gi) = self.conductance_candidates(ge_pre, gi_pre);
                (
                    self.v_reset,
                    ge,
                    gi,
                    if self.refractory_time <= self.dt * (1.0 + 1.0e-12) {
                        0.0
                    } else {
                        self.refractory_time - self.dt
                    },
                    false,
                )
            } else {
                let (v, ge, gi) = self.rk4_candidate(self.v, ge_pre, gi_pre, current);
                if !v.is_finite() || !(Self::V_MIN..=Self::V_MAX).contains(&v) {
                    return Err("voltage candidate outside COBA LIF safety envelope");
                }
                let spiked = v >= self.v_threshold;
                (
                    if spiked { self.v_reset } else { v },
                    ge,
                    gi,
                    if spiked { self.refractory_period } else { 0.0 },
                    spiked,
                )
            };
        if !v_candidate.is_finite()
            || !(Self::V_MIN..=Self::V_MAX).contains(&v_candidate)
            || !ge_candidate.is_finite()
            || ge_candidate < 0.0
            || !gi_candidate.is_finite()
            || gi_candidate < 0.0
            || !refractory_candidate.is_finite()
            || refractory_candidate < 0.0
        {
            return Err("COBA LIF candidate outside safety envelope");
        }
        self.v = v_candidate;
        self.g_e = ge_candidate;
        self.g_i = gi_candidate;
        self.refractory_time = refractory_candidate;
        Ok(i32::from(spiked))
    }

    pub fn step(&mut self, current: f64, delta_ge: f64, delta_gi: f64) -> i32 {
        self.try_step(current, delta_ge, delta_gi).unwrap_or(-1)
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.g_e = 0.0;
        self.g_i = 0.0;
        self.refractory_time = 0.0;
    }
}
impl Default for COBALIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = COBALIFNeuron::default();
        let constructed = COBALIFNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn validate_accepts_default_and_rejects_invalid_dt() {
        assert!(COBALIFNeuron::new().validate().is_ok());
        let invalid = COBALIFNeuron {
            dt: 0.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn coba_fires() {
        let mut n = COBALIFNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(500.0, 0.0, 0.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn coba_factory_matches_brette_benchmark_one() {
        let n = COBALIFNeuron::new();
        assert_eq!(n.v, -60.0);
        assert_eq!(n.g_e, 0.0);
        assert_eq!(n.g_i, 0.0);
        assert_eq!(n.refractory_time, 0.0);
        assert_eq!(n.c_m, 200.0);
        assert_eq!(n.g_l, 10.0);
        assert_eq!(n.e_l, -60.0);
        assert_eq!(n.e_e, 0.0);
        assert_eq!(n.e_i, -80.0);
        assert_eq!(n.tau_e, 5.0);
        assert_eq!(n.tau_i, 10.0);
        assert_eq!(n.v_threshold, -50.0);
        assert_eq!(n.v_reset, -60.0);
        assert_eq!(n.refractory_period, 5.0);
        assert_eq!(n.dt, 0.1);
    }

    #[test]
    fn coba_reset_clears_state() {
        let mut n = COBALIFNeuron::new();
        for _ in 0..100 {
            n.step(500.0, 0.0, 0.0);
        }
        n.reset();
        assert_eq!(n.v, n.e_l);
        assert_eq!(n.g_e, 0.0);
        assert_eq!(n.g_i, 0.0);
        assert_eq!(n.refractory_time, 0.0);
    }

    #[test]
    fn coba_refractory_interval_clamps_without_float_residue() {
        let mut n = COBALIFNeuron::new();
        n.v = -51.0;
        n.e_l = -65.0;
        assert_eq!(n.try_step(1.0e5, 0.0, 0.0).unwrap(), 1);
        for _ in 0..50 {
            assert_eq!(n.try_step(0.0, 0.0, 0.0).unwrap(), 0);
            assert_eq!(n.v, n.v_reset);
        }
        assert_eq!(n.refractory_time, 0.0);
        assert_eq!(n.try_step(0.0, 0.0, 0.0).unwrap(), 0);
        assert!(n.v < n.v_reset);
    }

    #[test]
    fn coba_raw_voltage_candidate_rejects_before_reset() {
        let mut n = COBALIFNeuron::new();
        n.v = 90.0;
        let before = (n.v, n.g_e, n.g_i, n.refractory_time);
        assert!(n.try_step(1.0e8, 0.0, 0.0).is_err());
        assert_eq!((n.v, n.g_e, n.g_i, n.refractory_time), before);
    }

    #[test]
    fn coba_bounded() {
        let mut n = COBALIFNeuron::new();
        for _ in 0..2000 {
            n.step(1e5, 0.0, 0.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn coba_inhibition_suppresses() {
        let mut n_exc = COBALIFNeuron::new();
        let mut n_inh = COBALIFNeuron::new();
        let t_exc: i32 = (0..2000).map(|_| n_exc.step(500.0, 0.0, 0.0)).sum();
        let t_inh: i32 = (0..2000).map(|_| n_inh.step(500.0, 0.0, 5.0)).sum();
        assert!(t_inh <= t_exc, "inhibition should reduce spiking");
    }

    #[test]
    fn coba_nan_no_panic() {
        COBALIFNeuron::new().step(f64::NAN, 0.0, 0.0);
    }

    #[test]
    fn coba_negative_no_crash() {
        let mut n = COBALIFNeuron::new();
        for _ in 0..500 {
            n.step(-100.0, 0.0, 0.0);
        }
        assert!(n.v.is_finite());
    }
}
