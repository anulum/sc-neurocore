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
/// Biological context: Robinson & Siegelbaum, Annu Rev Physiol 65:453, 2003;
/// Pape, Annu Rev Physiol 58:299, 1996. The repository-specific WB+HCN
/// recurrence is an experimental composite, not a publication-exact model.
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

    fn valid(&self) -> bool {
        let finite = [
            self.v,
            self.h,
            self.n,
            self.r,
            self.g_na,
            self.g_k,
            self.g_h,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_h,
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
            && [self.h, self.n, self.r]
                .into_iter()
                .all(|gate| (0.0..=1.0).contains(&gate))
            && (0.0..=200.0).contains(&self.g_na)
            && (0.0..=100.0).contains(&self.g_k)
            && (0.0..=5.0).contains(&self.g_h)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
            && (-50.0..=0.0).contains(&self.e_h)
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
            return Err("Ih state and parameters must satisfy the public bounds");
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

            // Ih gating: activates on hyperpolarisation
            // Half-activation ~-80 mV, slow kinetics (100-300 ms)
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());

            // Gate updates
            candidate.h +=
                sub_dt * candidate.phi * (alpha_h * (1.0 - candidate.h) - beta_h * candidate.h);
            candidate.n +=
                sub_dt * candidate.phi * (alpha_n * (1.0 - candidate.n) - beta_n * candidate.n);
            candidate.r += sub_dt * (r_inf - candidate.r) / tau_r;

            // Currents
            let i_na = candidate.g_na * m_inf.powi(3) * candidate.h * (v - candidate.e_na);
            let i_k = candidate.g_k * candidate.n.powi(4) * (v - candidate.e_k);
            let i_h = candidate.g_h * candidate.r * (v - candidate.e_h);
            let i_l = candidate.g_l * (v - candidate.e_l);

            let dv = (-i_na - i_k - i_h - i_l + input) / candidate.c_m;
            candidate.v += sub_dt * dv;
            if ![candidate.v, candidate.h, candidate.n, candidate.r]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err("Ih candidate state became non-finite");
            }

            if candidate.v >= candidate.v_threshold {
                fired = 1;
                candidate.v = -65.0;
            }
        }

        candidate.v = candidate.v.clamp(-100.0, 60.0);
        candidate.h = candidate.h.clamp(0.0, 1.0);
        candidate.n = candidate.n.clamp(0.0, 1.0);
        candidate.r = candidate.r.clamp(0.0, 1.0);
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
        self.r = 0.1;
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
    fn ih_nan_input_is_rejected_atomically() {
        let mut n = IhNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.r, before.r);
    }

    #[test]
    fn ih_invalid_configuration_is_rejected_atomically() {
        let mut n = IhNeuron::new();
        n.c_m = 0.0;
        let before = n.clone();
        assert!(n.try_step(1.0).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.c_m, before.c_m);
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
        n.g_h = 0.3;
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.r, 0.1);
        assert_eq!(n.g_h, 0.3);
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
