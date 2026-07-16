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
/// Jahr & Stevens, J Neurosci 10:1830, 1990; Wang, Neuron 22:409, 1999.
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

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let sub_steps = 50;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;

        // NMDA synaptic variable: driven by input (as proxy for glutamate)
        let drive = if input > 0.0 {
            input / (input + 5.0)
        } else {
            0.0
        };
        let ds = (drive - self.s_nmda)
            / if drive > self.s_nmda {
                self.tau_rise
            } else {
                self.tau_decay
            };
        self.s_nmda += self.dt * ds;
        self.s_nmda = self.s_nmda.clamp(0.0, 1.0);

        for _ in 0..sub_steps {
            let v = self.v;

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Mg2+ block: B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))
            // Jahr & Stevens 1990
            let mg_block = 1.0 / (1.0 + (self.mg_conc / 3.57) * (-0.062 * v).exp());

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_nmda = self.g_nmda * self.s_nmda * mg_block * (v - self.e_nmda);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_nmda - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        if !self.s_nmda.is_finite() {
            self.s_nmda = 0.0;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
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
    fn nmda_nan_input_stays_finite() {
        let mut n = NMDANeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
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
