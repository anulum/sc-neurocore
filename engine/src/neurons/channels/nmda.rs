// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NMDA receptor-gated channel neuron

use crate::neurons::biophysical::safe_rate;

/// NMDA receptor neuron — WB base + NMDA-type glutamate receptor current.
///
/// NMDA receptors require both glutamate binding (modelled as input current)
/// AND membrane depolarisation (Mg2+ block removal) for activation. The
/// Mg2+ block is voltage-dependent: at rest (-65 mV) channels are blocked,
/// but depolarisation to -40 mV relieves ~80% of the block.
///
/// Key mechanism for:
/// - Coincidence detection: requires presynaptic (glutamate) + postsynaptic
///   (depolarisation) signals simultaneously
/// - Synaptic plasticity: Ca2+ influx through NMDA triggers LTP/LTD
/// - Working memory: NMDA-mediated recurrent excitation sustains persistent
///   activity in prefrontal cortex
/// - Slow synaptic integration: rise ~10 ms, decay ~100 ms
///
/// Jahr & Stevens, J Neurosci 10(9):3178, 1990; Wang, Neuron 22:409, 1999.
/// The WB spiking base follows Wang & Buzsáki, J Neurosci 16:6402, 1996;
/// the threshold-reset event and input-driven s_NMDA drive are
/// repository-specific specialisations, not publication-exact claims.
#[derive(Clone, Debug)]
pub struct NMDANeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub s_nmda: f64, // NMDA synaptic variable (slow rise/decay)
    pub g_na: f64,
    pub g_k: f64,
    pub g_nmda: f64, // NMDA conductance
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_nmda: f64, // NMDA reversal (0 mV, mixed cation)
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub mg_conc: f64,   // Extracellular Mg2+ (mM), typically 1.0
    pub tau_rise: f64,  // NMDA rise time (ms)
    pub tau_decay: f64, // NMDA decay time (ms)
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for NMDANeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl NMDANeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            s_nmda: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_nmda: 0.5,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_nmda: 0.0, // Mixed cation reversal
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            mg_conc: 1.0,
            tau_rise: 10.0,
            tau_decay: 100.0,
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
            self.s_nmda,
            self.g_na,
            self.g_k,
            self.g_nmda,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_nmda,
            self.e_l,
            self.c_m,
            self.phi,
            self.mg_conc,
            self.tau_rise,
            self.tau_decay,
            self.dt,
            self.v_threshold,
            self.gain,
        ]
        .into_iter()
        .all(f64::is_finite);
        finite
            && (-100.0..=60.0).contains(&self.v)
            && [self.h, self.n, self.s_nmda]
                .into_iter()
                .all(|gate| (0.0..=1.0).contains(&gate))
            && (0.0..=200.0).contains(&self.g_na)
            && (0.0..=100.0).contains(&self.g_k)
            && (0.0..=20.0).contains(&self.g_nmda)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
            && (-10.0..=10.0).contains(&self.e_nmda)
            && (-80.0..=-40.0).contains(&self.e_l)
            && (0.5..=2.0).contains(&self.c_m)
            && (0.5..=10.0).contains(&self.phi)
            && (0.0..=5.0).contains(&self.mg_conc)
            && (0.1..=20.0).contains(&self.tau_rise)
            && (10.0..=500.0).contains(&self.tau_decay)
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
            return Err("NMDA state and parameters must satisfy the public bounds");
        }

        let mut candidate = self.clone();
        let input = candidate.gain * current;
        let sub_steps = 50;
        let sub_dt = candidate.dt / sub_steps as f64;
        let mut fired = 0i32;

        // NMDA synaptic variable: driven by input (as proxy for glutamate)
        let drive = if input > 0.0 {
            input / (input + 5.0)
        } else {
            0.0
        };
        let ds = (drive - candidate.s_nmda)
            / if drive > candidate.s_nmda {
                candidate.tau_rise
            } else {
                candidate.tau_decay
            };
        candidate.s_nmda += candidate.dt * ds;
        candidate.s_nmda = candidate.s_nmda.clamp(0.0, 1.0);

        for _ in 0..sub_steps {
            let v = candidate.v;

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Mg2+ block: B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))
            // Jahr & Stevens 1990
            let mg_block = 1.0 / (1.0 + (candidate.mg_conc / 3.57) * (-0.062 * v).exp());

            candidate.h +=
                sub_dt * candidate.phi * (alpha_h * (1.0 - candidate.h) - beta_h * candidate.h);
            candidate.n +=
                sub_dt * candidate.phi * (alpha_n * (1.0 - candidate.n) - beta_n * candidate.n);

            let i_na = candidate.g_na * m_inf.powi(3) * candidate.h * (v - candidate.e_na);
            let i_k = candidate.g_k * candidate.n.powi(4) * (v - candidate.e_k);
            let i_nmda =
                candidate.g_nmda * candidate.s_nmda * mg_block * (v - candidate.e_nmda);
            let i_l = candidate.g_l * (v - candidate.e_l);

            let dv = (-i_na - i_k - i_nmda - i_l + input) / candidate.c_m;
            candidate.v += sub_dt * dv;
            if ![candidate.v, candidate.h, candidate.n]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err("NMDA candidate state became non-finite");
            }

            if candidate.v >= candidate.v_threshold {
                fired = 1;
                candidate.v = -65.0;
            }
        }

        candidate.v = candidate.v.clamp(-100.0, 60.0);
        candidate.h = candidate.h.clamp(0.0, 1.0);
        candidate.n = candidate.n.clamp(0.0, 1.0);
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
        self.s_nmda = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- NMDA Neuron tests --

    #[test]
    fn nmda_fires_with_input() {
        let mut n = NMDANeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(3.0);
        }
        assert!(spikes > 5, "NMDA neuron must fire with input, got {spikes}");
    }

    #[test]
    fn nmda_silent_without_input() {
        let mut n = NMDANeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "NMDA neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn nmda_nominal_step_matches_reference_anchor() {
        let mut n = NMDANeuron::new();
        assert_eq!(n.try_step(5.0), Ok(0));
        assert!((n.v - -63.155_663_780_395_78).abs() < 1.0e-12);
        assert!((n.h - 0.648_031_194_399_744_1).abs() < 1.0e-12);
        assert!((n.n - 0.237_221_887_163_776).abs() < 1.0e-12);
        assert!((n.s_nmda - 0.025).abs() < 1.0e-15);
    }

    #[test]
    fn nmda_mg_block_at_rest() {
        // At -65 mV: B = 1/(1 + 1/3.57 * exp(0.062*65)) = 1/(1 + 0.28 * 56.3) = 1/16.8 = 0.06
        let n = NMDANeuron::new();
        let mg_block = 1.0 / (1.0 + (n.mg_conc / 3.57) * (-0.062 * n.v).exp());
        assert!(
            mg_block < 0.1,
            "Mg2+ block must be strong at rest, B={mg_block}"
        );
    }

    #[test]
    fn nmda_mg_relief_at_depolarised() {
        // At -20 mV: B = 1/(1 + 0.28 * exp(0.062*20)) = 1/(1 + 0.28*3.45) = 1/1.97 = 0.51
        let mg_block = 1.0 / (1.0 + (1.0 / 3.57) * (-0.062 * (-20.0_f64)).exp());
        assert!(
            mg_block > 0.4,
            "Mg2+ block must be relieved at -20 mV, B={mg_block}"
        );
    }

    #[test]
    fn nmda_s_builds_with_input() {
        let mut n = NMDANeuron::new();
        assert_eq!(n.s_nmda, 0.0);
        for _ in 0..2000 {
            n.step(5.0);
        }
        assert!(
            n.s_nmda > 0.0,
            "s_nmda must build with input, s={}",
            n.s_nmda
        );
    }

    #[test]
    fn nmda_s_decays_without_input() {
        let mut n = NMDANeuron::new();
        // Build up s
        for _ in 0..2000 {
            n.step(5.0);
        }
        let s_peak = n.s_nmda;
        // Remove input
        for _ in 0..2000 {
            n.step(0.0);
        }
        assert!(n.s_nmda < s_peak, "s_nmda must decay after input removal");
    }

    #[test]
    fn nmda_zero_mg_increases_current() {
        // Without Mg2+ block, NMDA should contribute more
        let mut with_mg = NMDANeuron::new();
        let mut no_mg = NMDANeuron::new();
        no_mg.mg_conc = 0.0;

        let input = 2.0;
        let mut spikes_mg = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_mg += with_mg.step(input);
            spikes_no += no_mg.step(input);
        }
        assert!(
            spikes_no >= spikes_mg,
            "No Mg2+ should increase NMDA current: no_mg={spikes_no} vs mg={spikes_mg}"
        );
    }

    #[test]
    fn nmda_negative_input_no_crash() {
        let mut n = NMDANeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn nmda_nan_input_is_rejected_atomically() {
        let mut n = NMDANeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.s_nmda, before.s_nmda);
    }

    #[test]
    fn nmda_infinite_input_is_rejected_atomically() {
        let mut n = NMDANeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::INFINITY).is_err());
        assert!(n.try_step(f64::NEG_INFINITY).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.s_nmda, before.s_nmda);
    }

    #[test]
    fn nmda_invalid_configuration_is_rejected_atomically() {
        let mut n = NMDANeuron::new();
        n.c_m = 0.0;
        let before = n.clone();
        assert!(n.try_step(1.0).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.c_m, before.c_m);
    }

    #[test]
    fn nmda_extreme_input_bounded() {
        let mut n = NMDANeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn nmda_reset_clears_state() {
        let mut n = NMDANeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.s_nmda, 0.0);
    }

    #[test]
    fn nmda_reset_preserves_parameters() {
        let mut n = NMDANeuron::new();
        n.g_nmda = 1.5;
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.g_nmda, 1.5);
    }

    #[test]
    fn nmda_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = NMDANeuron::new();
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
