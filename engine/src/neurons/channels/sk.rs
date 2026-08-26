// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SK calcium-activated potassium channel neuron

use crate::neurons::biophysical::safe_rate;

/// SK channel neuron — WB base + Ca2+-only-dependent K+ current.
///
/// SK (KCa2.x) channels are activated solely by intracellular Ca2+
/// (no voltage dependence). They have slower kinetics than BK and produce
/// the medium afterhyperpolarisation (mAHP) lasting 50-200 ms after spikes.
///
/// Key mechanism for:
/// - Medium AHP (mAHP): limits sustained firing rate
/// - Spike frequency adaptation: Ca2+ builds → SK activates → firing slows
/// - Rhythmic firing patterns: SK-mediated pauses create regular ISIs
/// - Synaptic plasticity gating: SK in dendritic spines regulates NMDA currents
///
/// Stocker, Nat Rev Neurosci 5:758, 2004; Wang & Buzsáki, J Neurosci
/// 16:6402, 1996. The threshold-reset event, the spike-triggered Ca2+
/// increment, and the Hill constants are repository-specific
/// specialisations of that review material, not a publication-exact
/// recurrence.
#[derive(Clone, Debug)]
pub struct SKNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_sk: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for SKNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl SKNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            ca: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_sk: 2.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_ca: 150.0, // Slower Ca2+ decay than BK → longer mAHP
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    fn valid(&self) -> bool {
        let finite = [
            self.v,
            self.h,
            self.n,
            self.ca,
            self.g_na,
            self.g_k,
            self.g_sk,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_l,
            self.c_m,
            self.phi,
            self.tau_ca,
            self.dt,
            self.v_threshold,
            self.gain,
        ]
        .into_iter()
        .all(f64::is_finite);
        finite
            && (-100.0..=60.0).contains(&self.v)
            && [self.h, self.n]
                .into_iter()
                .all(|gate| (0.0..=1.0).contains(&gate))
            && self.ca >= 0.0
            && (0.0..=200.0).contains(&self.g_na)
            && (0.0..=100.0).contains(&self.g_k)
            && (0.0..=50.0).contains(&self.g_sk)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
            && (-80.0..=-40.0).contains(&self.e_l)
            && (0.5..=2.0).contains(&self.c_m)
            && (0.5..=10.0).contains(&self.phi)
            && (10.0..=2000.0).contains(&self.tau_ca)
            && self.dt > 0.0
            && self.dt <= 1.0
            && (-20.0..=20.0).contains(&self.v_threshold)
            && (0.0..=10.0).contains(&self.gain)
    }

    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() {
            return Err("current must be finite");
        }
        if !self.valid() {
            return Err("SK state and parameters must satisfy the public bounds");
        }

        let mut candidate = self.clone();
        let input = candidate.gain * current;
        let sub_steps = 50;
        let sub_dt = candidate.dt / sub_steps as f64;
        let mut fired = 0i32;

        for _ in 0..sub_steps {
            let v = candidate.v;

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // SK activation: purely Ca2+-dependent (Hill function, n=2)
            let ca2 = candidate.ca * candidate.ca;
            let sk_inf = ca2 / (ca2 + 0.25); // Half-activation at [Ca2+]=0.5

            // Ca2+ decay
            candidate.ca += sub_dt * (-candidate.ca / candidate.tau_ca);

            candidate.h +=
                sub_dt * candidate.phi * (alpha_h * (1.0 - candidate.h) - beta_h * candidate.h);
            candidate.n +=
                sub_dt * candidate.phi * (alpha_n * (1.0 - candidate.n) - beta_n * candidate.n);

            let i_na = candidate.g_na * m_inf.powi(3) * candidate.h * (v - candidate.e_na);
            let i_k = candidate.g_k * candidate.n.powi(4) * (v - candidate.e_k);
            let i_sk = candidate.g_sk * sk_inf * (v - candidate.e_k);
            let i_l = candidate.g_l * (v - candidate.e_l);

            let dv = (-i_na - i_k - i_sk - i_l + input) / candidate.c_m;
            candidate.v += sub_dt * dv;
            if ![candidate.v, candidate.h, candidate.n, candidate.ca]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err("SK candidate state became non-finite");
            }

            if candidate.v >= candidate.v_threshold {
                fired = 1;
                candidate.v = -65.0;
                candidate.ca += 0.2;
            }
        }

        candidate.v = candidate.v.clamp(-100.0, 60.0);
        candidate.h = candidate.h.clamp(0.0, 1.0);
        candidate.n = candidate.n.clamp(0.0, 1.0);
        candidate.ca = candidate.ca.max(0.0);
        *self = candidate;

        Ok(fired)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.6;
        self.n = 0.32;
        self.ca = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- SK Neuron tests --

    #[test]
    fn sk_fires_with_input() {
        let mut n = SKNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(spikes > 5, "SK neuron must fire with input, got {spikes}");
    }

    #[test]
    fn sk_silent_without_input() {
        let mut n = SKNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "SK neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn sk_nominal_step_matches_reference_anchor() {
        let mut n = SKNeuron::new();
        assert_eq!(n.try_step(5.0), Ok(0));
        assert!((n.v - -63.180_064_213_072_19).abs() < 1.0e-12);
        assert!((n.h - 0.648_122_835_749_998_1).abs() < 1.0e-12);
        assert!((n.n - 0.237_186_365_946_861_5).abs() < 1.0e-12);
        assert_eq!(n.ca, 0.0);
    }

    #[test]
    fn sk_adaptation() {
        // SK causes spike frequency adaptation
        let mut n = SKNeuron::new();
        let input = 5.0;
        let mut early = 0;
        for _ in 0..2000 {
            early += n.step(input);
        }
        let mut late = 0;
        for _ in 0..2000 {
            late += n.step(input);
        }
        assert!(
            early >= late,
            "SK should cause adaptation: early={early}, late={late}"
        );
    }

    #[test]
    fn sk_ca_dependent_only() {
        // SK at rest (ca=0) should contribute zero current
        let n = SKNeuron::new();
        let ca2 = n.ca * n.ca;
        let sk_inf = ca2 / (ca2 + 0.25);
        assert!(
            sk_inf < 0.001,
            "SK must be inactive at ca=0, sk_inf={sk_inf}"
        );
    }

    #[test]
    fn sk_reduces_firing_rate() {
        let mut with_sk = SKNeuron::new();
        let mut no_sk = SKNeuron::new();
        no_sk.g_sk = 0.0;

        let input = 3.0;
        let mut spikes_sk = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_sk += with_sk.step(input);
            spikes_no += no_sk.step(input);
        }
        assert!(
            spikes_no >= spikes_sk,
            "SK should reduce firing: SK={spikes_sk} vs none={spikes_no}"
        );
    }

    #[test]
    fn sk_negative_input_no_crash() {
        let mut n = SKNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn sk_nan_input_is_rejected_atomically() {
        let mut n = SKNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.ca, before.ca);
    }

    #[test]
    fn sk_infinite_input_is_rejected_atomically() {
        let mut n = SKNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::INFINITY).is_err());
        assert!(n.try_step(f64::NEG_INFINITY).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.ca, before.ca);
    }

    #[test]
    fn sk_invalid_configuration_is_rejected_atomically() {
        let mut n = SKNeuron::new();
        n.c_m = 0.0;
        let before = n.clone();
        assert!(n.try_step(1.0).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.c_m, before.c_m);
    }

    #[test]
    fn sk_extreme_input_bounded() {
        let mut n = SKNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn sk_reset_clears_state() {
        let mut n = SKNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.ca, 0.0);
    }

    #[test]
    fn sk_reset_preserves_parameters() {
        let mut n = SKNeuron::new();
        n.g_sk = 4.0;
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.g_sk, 4.0);
    }

    #[test]
    fn sk_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = SKNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }
}
