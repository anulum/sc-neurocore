// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hyperpolarisation-activated channel neuron

use crate::neurons::biophysical::safe_rate;

/// Ih (hyperpolarisation-activated cation current) neuron — WB base + HCN.
///
/// Ih activates upon hyperpolarisation (opposite to most voltage-gated
/// channels) and conducts a mixed Na+/K+ current with reversal ~-40 mV.
/// Key mechanism for:
/// - Voltage sag: during hyperpolarisation, Ih activates and depolarises
///   the cell back towards rest (sag potential)
/// - Rebound excitation: Ih accumulated during inhibition depolarises
///   the cell after inhibition ends, triggering rebound spikes
/// - Pacemaker oscillations: interplay of Ih and T-type Ca2+ in thalamic
///   relay neurons drives rhythmic bursting
///
/// Robinson & Bhatt, Neuron 11:953, 1993; Pape, Annu Rev Physiol 58:299, 1996.
#[derive(Clone, Debug)]
pub struct IhNeuron {
    pub v: f64,
    pub h: f64, // Na+ inactivation
    pub n: f64, // Kdr activation
    pub r: f64, // Ih activation (activates on hyperpolarisation)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_h: f64, // Ih conductance
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_h: f64, // Ih reversal (~-40 mV, mixed cation)
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for IhNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl IhNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            r: 0.1,
            g_na: 35.0,
            g_k: 9.0,
            g_h: 0.15,
            g_l: 0.2,
            e_na: 55.0,
            e_k: -90.0,
            e_h: -40.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
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

            // WB alpha/beta rates
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Ih gating: activates on hyperpolarisation
            // Half-activation ~-80 mV, slow kinetics (100-300 ms)
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());

            // Gate updates
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.r += sub_dt * (r_inf - self.r) / tau_r;

            // Currents
            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_h = self.g_h * self.r * (v - self.e_h);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_h - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
            }
        }

        // Safety bounds
        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.r = self.r.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Ih Neuron tests --

    #[test]
    fn ih_fires_with_input() {
        let mut n = IhNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(spikes > 5, "Ih neuron must fire with input, got {spikes}");
    }

    #[test]
    fn ih_silent_without_input() {
        let mut n = IhNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Ih neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn ih_sag_potential() {
        // Hyperpolarising input should produce sag (voltage rebounds towards rest)
        let mut with_ih = IhNeuron::new();
        let mut no_ih = IhNeuron::new();
        no_ih.g_h = 0.0;

        // Apply hyperpolarising step
        for _ in 0..4000 {
            with_ih.step(-3.0);
            no_ih.step(-3.0);
        }
        // With Ih, voltage should be less hyperpolarised (sag back)
        assert!(
            with_ih.v > no_ih.v,
            "Ih sag must depolarise from hyperpolarisation: Ih={:.1} vs no_Ih={:.1}",
            with_ih.v,
            no_ih.v
        );
    }

    #[test]
    fn ih_r_gate_activates_on_hyperpolarisation() {
        let mut n = IhNeuron::new();
        let r_before = n.r;
        // Hyperpolarise
        for _ in 0..4000 {
            n.step(-5.0);
        }
        assert!(
            n.r > r_before,
            "r gate must increase during hyperpolarisation, r={}",
            n.r
        );
    }

    #[test]
    fn ih_rebound_excitation() {
        // After hyperpolarisation, Ih should help reach threshold
        let mut n = IhNeuron::new();
        // Hyperpolarise to build up Ih
        for _ in 0..4000 {
            n.step(-3.0);
        }
        let r_after_hyp = n.r;
        assert!(
            r_after_hyp > 0.2,
            "r must build up during hyperpolarisation, r={r_after_hyp}"
        );

        // Release — count spikes during rebound period
        let mut rebound_spikes = 0;
        for _ in 0..500 {
            rebound_spikes += n.step(1.5);
        }

        // Compare with neuron that was not hyperpolarised
        let mut n2 = IhNeuron::new();
        let mut direct_spikes = 0;
        for _ in 0..500 {
            direct_spikes += n2.step(1.5);
        }

        assert!(
            rebound_spikes >= direct_spikes,
            "Rebound should facilitate firing: rebound={rebound_spikes} vs direct={direct_spikes}"
        );
    }

    #[test]
    fn ih_negative_input_no_crash() {
        let mut n = IhNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn ih_nan_input_stays_finite() {
        let mut n = IhNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn ih_extreme_input_bounded() {
        let mut n = IhNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn ih_reset_clears_state() {
        let mut n = IhNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.r, 0.1);
    }

    #[test]
    fn ih_gates_bounded() {
        let mut n = IhNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.r >= 0.0 && n.r <= 1.0);
    }

    #[test]
    fn ih_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = IhNeuron::new();
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
