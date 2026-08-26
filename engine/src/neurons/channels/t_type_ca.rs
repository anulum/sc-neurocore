// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — T-type calcium channel neuron

use crate::neurons::biophysical::safe_rate;

/// T-type Ca2+ (IT) neuron — WB base + low-voltage-activated Ca2+ current.
///
/// IT activates at subthreshold voltages (-65 to -50 mV) and inactivates
/// slowly. When de-inactivated by hyperpolarisation, IT produces a
/// low-threshold spike (LTS) — a broad Ca2+ depolarisation that can
/// trigger a burst of Na+ action potentials riding on top.
///
/// Key mechanism for:
/// - Rebound bursting in thalamocortical relay neurons
/// - Sleep spindle generation (thalamic reticular nucleus)
/// - Low-threshold calcium spikes in cortical layer V pyramidal cells
/// - Rhythmic bursting in inferior olive neurons
///
/// Huguenard, Annu Rev Physiol 58:329, 1996; Destexhe et al., J Neurophysiol 76:2049, 1996.
#[derive(Clone, Debug)]
pub struct TTypeCaNeuron {
    pub v: f64,
    pub h: f64, // Na+ inactivation
    pub n: f64, // Kdr activation
    pub s: f64, // T-type Ca2+ inactivation (slow)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_t: f64, // T-type Ca2+
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for TTypeCaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl TTypeCaNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            s: 0.9, // De-inactivated at rest (-65 mV)
            g_na: 35.0,
            g_k: 9.0,
            g_t: 0.1, // Reduced to avoid window current at rest
            g_l: 0.2,
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
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
            self.s,
            self.g_na,
            self.g_k,
            self.g_t,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_l,
            self.c_m,
            self.phi,
            self.dt,
            self.v_threshold,
            self.gain,
        ]
        .into_iter()
        .all(f64::is_finite);
        finite
            && (-100.0..=60.0).contains(&self.v)
            && [self.h, self.n, self.s]
                .into_iter()
                .all(|gate| (0.0..=1.0).contains(&gate))
            && (0.0..=200.0).contains(&self.g_na)
            && (0.0..=100.0).contains(&self.g_k)
            && (0.0..=20.0).contains(&self.g_t)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
            && (60.0..=150.0).contains(&self.e_ca)
            && (-80.0..=-40.0).contains(&self.e_l)
            && (0.5..=2.0).contains(&self.c_m)
            && (0.5..=10.0).contains(&self.phi)
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
            return Err("T-type state and parameters must satisfy the public bounds");
        }

        let mut candidate = self.clone();
        let input = candidate.gain * current;
        let sub_steps = 50;
        let sub_dt = candidate.dt / sub_steps as f64;
        let mut fired = 0i32;

        for _ in 0..sub_steps {
            let v = candidate.v;

            // WB alpha/beta rates
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // T-type Ca2+ gating
            let m_t_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).exp());
            let s_inf = 1.0 / (1.0 + ((v + 81.0) / 4.0).exp());
            let tau_s = 30.0 + 100.0 / (1.0 + ((v + 75.0) / 10.0).exp());

            // Gate updates
            candidate.h +=
                sub_dt * candidate.phi * (alpha_h * (1.0 - candidate.h) - beta_h * candidate.h);
            candidate.n +=
                sub_dt * candidate.phi * (alpha_n * (1.0 - candidate.n) - beta_n * candidate.n);
            candidate.s += sub_dt * (s_inf - candidate.s) / tau_s;

            // Currents
            let i_na = candidate.g_na * m_inf.powi(3) * candidate.h * (v - candidate.e_na);
            let i_k = candidate.g_k * candidate.n.powi(4) * (v - candidate.e_k);
            let i_t = candidate.g_t * m_t_inf.powi(2) * candidate.s * (v - candidate.e_ca);
            let i_l = candidate.g_l * (v - candidate.e_l);

            let dv = (-i_na - i_k - i_t - i_l + input) / candidate.c_m;
            candidate.v += sub_dt * dv;
            if ![candidate.v, candidate.h, candidate.n, candidate.s]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err("T-type candidate state became non-finite");
            }

            if candidate.v >= candidate.v_threshold {
                fired = 1;
                candidate.v = -65.0;
                candidate.s *= 0.3; // Spike inactivates T-type strongly
            }
        }

        candidate.v = candidate.v.clamp(-100.0, 60.0);
        candidate.h = candidate.h.clamp(0.0, 1.0);
        candidate.n = candidate.n.clamp(0.0, 1.0);
        candidate.s = candidate.s.clamp(0.0, 1.0);
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
        self.s = 0.9;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- T-type Ca2+ Neuron tests --

    #[test]
    fn ttype_fires_with_input() {
        let mut n = TTypeCaNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(
            spikes > 5,
            "T-type neuron must fire with input, got {spikes}"
        );
    }

    #[test]
    fn ttype_silent_without_input() {
        let mut n = TTypeCaNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "T-type neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn ttype_rebound_burst() {
        // Hyperpolarise → de-inactivate T-type → rebound burst on release
        let mut n = TTypeCaNeuron::new();
        // Hyperpolarise
        for _ in 0..4000 {
            n.step(-3.0);
        }
        assert!(
            n.s > 0.3,
            "T-type must de-inactivate during hyperpolarisation, s={}",
            n.s
        );

        // Release with mild input
        let mut rebound_spikes = 0;
        for _ in 0..500 {
            rebound_spikes += n.step(1.5);
        }

        // Compare with pre-inactivated neuron
        let mut n2 = TTypeCaNeuron::new();
        n2.s = 0.05;
        let mut direct_spikes = 0;
        for _ in 0..500 {
            direct_spikes += n2.step(1.5);
        }
        assert!(
            rebound_spikes >= direct_spikes,
            "Rebound should facilitate firing: rebound={rebound_spikes} vs inact={direct_spikes}"
        );
    }

    #[test]
    fn ttype_s_gate_de_inactivates_at_hyperpolarised() {
        let mut n = TTypeCaNeuron::new();
        n.v = -85.0;
        n.s = 0.1; // Start inactivated
                   // s_inf at -85 = 1/(1+exp((-85+81)/4)) = 1/(1+exp(-1)) = 1/1.37 = 0.73
        for _ in 0..5000 {
            n.step(-5.0);
        }
        assert!(
            n.s > 0.5,
            "s must de-inactivate at hyperpolarised potentials, s={}",
            n.s
        );
    }

    #[test]
    fn ttype_spike_inactivates_t_channel() {
        let mut n = TTypeCaNeuron::new();
        let s_before_spiking = n.s;
        // Drive until spike
        let mut spiked = false;
        for _ in 0..2000 {
            if n.step(3.0) > 0 {
                spiked = true;
                break;
            }
        }
        if spiked {
            assert!(
                n.s < s_before_spiking,
                "Spike must inactivate T-type: before={s_before_spiking}, after={}",
                n.s
            );
        }
    }

    #[test]
    fn ttype_negative_input_no_crash() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn ttype_nan_input_is_rejected_atomically() {
        let mut n = TTypeCaNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.s, before.s);
    }

    #[test]
    fn ttype_infinite_input_is_rejected_atomically() {
        let mut n = TTypeCaNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::INFINITY).is_err());
        assert!(n.try_step(f64::NEG_INFINITY).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.s, before.s);
    }

    #[test]
    fn ttype_invalid_configuration_is_rejected_atomically() {
        let mut n = TTypeCaNeuron::new();
        n.c_m = 0.0;
        let before = n.clone();
        assert!(n.try_step(1.0).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.c_m, before.c_m);
    }

    #[test]
    fn ttype_nominal_step_matches_reference_anchor() {
        let mut n = TTypeCaNeuron::new();
        assert_eq!(n.try_step(5.0), Ok(0));
        assert!((n.v - -63.168_136_340_251_8).abs() < 1.0e-12);
        assert!((n.h - 0.648_043_259_776_001_7).abs() < 1.0e-12);
        assert!((n.n - 0.237_216_896_172_727_87).abs() < 1.0e-12);
        assert!((n.s - 0.892_025_427_204_723_3).abs() < 1.0e-12);
    }

    #[test]
    fn ttype_reset_preserves_parameters() {
        let mut n = TTypeCaNeuron::new();
        n.g_t = 0.5;
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.s, 0.9);
        assert_eq!(n.g_t, 0.5);
    }

    #[test]
    fn ttype_extreme_input_bounded() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn ttype_reset_clears_state() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.s, 0.9);
    }

    #[test]
    fn ttype_gates_bounded() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.s >= 0.0 && n.s <= 1.0);
    }

    #[test]
    fn ttype_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = TTypeCaNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(2.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }
}
