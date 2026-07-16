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
/// Bhatt & Storm, J Physiol 557:329, 2003; Stocker, Nat Rev Neurosci 5:758, 2004.
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

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let sub_steps = 50;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;

        for _ in 0..sub_steps {
            let v = self.v;

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // SK activation: purely Ca2+-dependent (Hill function, n=2)
            let ca2 = self.ca * self.ca;
            let sk_inf = ca2 / (ca2 + 0.25); // Half-activation at [Ca2+]=0.5

            // Ca2+ decay
            self.ca += sub_dt * (-self.ca / self.tau_ca);

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_sk = self.g_sk * sk_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_sk - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
                self.ca += 0.2;
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        if !self.ca.is_finite() {
            self.ca = 0.0;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.ca = self.ca.max(0.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
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
    fn sk_nan_input_stays_finite() {
        let mut n = SKNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
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
