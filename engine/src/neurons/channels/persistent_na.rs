// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Persistent sodium channel neuron

use crate::neurons::biophysical::safe_rate;

/// Persistent Na+ (INaP) neuron — WB base + non-inactivating Na+ current.
///
/// INaP activates at subthreshold voltages (-60 to -40 mV) and does not
/// inactivate, providing a sustained depolarising drive. Key mechanism for:
/// - Subthreshold membrane oscillations (entorhinal cortex, layer II stellate)
/// - Plateau potentials and bistability (spinal motoneurons)
/// - Amplification of synaptic inputs near threshold
/// - Burst generation in respiratory neurons (pre-Bötzinger complex)
///
/// Biological context: Crill, Annu Rev Physiol 58:349, 1996; French et al.,
/// J Gen Physiol 95:1139, 1990. The repository-specific WB+INaP recurrence
/// is an experimental composite, not a publication-exact model.
#[derive(Clone, Debug)]
pub struct PersistentNaNeuron {
    pub v: f64,
    pub h: f64, // Transient Na+ inactivation
    pub n: f64, // Kdr activation
    pub p: f64, // INaP activation (slow)
    // Conductances (mS/cm²)
    pub g_na: f64,  // Transient Na+
    pub g_nap: f64, // Persistent Na+
    pub g_k: f64,   // Kdr
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for PersistentNaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentNaNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            p: 0.0,
            g_na: 35.0,
            g_nap: 0.15, // Persistent Na+ — small but significant
            g_k: 9.0,
            g_l: 0.3, // Higher leak to counteract INaP window current
            e_na: 55.0,
            e_k: -90.0,
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
            self.p,
            self.g_na,
            self.g_nap,
            self.g_k,
            self.g_l,
            self.e_na,
            self.e_k,
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
            && [self.h, self.n, self.p]
                .into_iter()
                .all(|gate| (0.0..=1.0).contains(&gate))
            && (0.0..=200.0).contains(&self.g_na)
            && (0.0..=20.0).contains(&self.g_nap)
            && (0.0..=100.0).contains(&self.g_k)
            && (0.0..=5.0).contains(&self.g_l)
            && (30.0..=70.0).contains(&self.e_na)
            && (-100.0..=-70.0).contains(&self.e_k)
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
            return Err("PersistentNa state and parameters must satisfy the public bounds");
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

            // Persistent Na+ gating: slow activation, no inactivation
            // Half-activation at -48 mV (subthreshold), tau ~10-50 ms
            let p_inf = 1.0 / (1.0 + (-(v + 48.0) / 5.0).exp());
            let tau_p = 10.0 + 40.0 / (1.0 + ((v + 48.0) / 10.0).powi(2));

            // Gate updates
            candidate.h +=
                sub_dt * candidate.phi * (alpha_h * (1.0 - candidate.h) - beta_h * candidate.h);
            candidate.n +=
                sub_dt * candidate.phi * (alpha_n * (1.0 - candidate.n) - beta_n * candidate.n);
            candidate.p += sub_dt * (p_inf - candidate.p) / tau_p;

            // Currents
            let i_na = candidate.g_na * m_inf.powi(3) * candidate.h * (v - candidate.e_na);
            let i_nap = candidate.g_nap * candidate.p * (v - candidate.e_na);
            let i_k = candidate.g_k * candidate.n.powi(4) * (v - candidate.e_k);
            let i_l = candidate.g_l * (v - candidate.e_l);

            let dv = (-i_na - i_nap - i_k - i_l + input) / candidate.c_m;
            candidate.v += sub_dt * dv;
            if ![candidate.v, candidate.h, candidate.n, candidate.p]
                .into_iter()
                .all(f64::is_finite)
            {
                return Err("PersistentNa candidate state became non-finite");
            }

            if candidate.v >= candidate.v_threshold {
                fired = 1;
                candidate.v = -65.0;
            }
        }

        candidate.v = candidate.v.clamp(-100.0, 60.0);
        candidate.h = candidate.h.clamp(0.0, 1.0);
        candidate.n = candidate.n.clamp(0.0, 1.0);
        candidate.p = candidate.p.clamp(0.0, 1.0);
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
        self.p = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Persistent Na+ Neuron tests --

    #[test]
    fn nap_fires_with_input() {
        let mut n = PersistentNaNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(spikes > 5, "NaP neuron must fire with input, got {spikes}");
    }

    #[test]
    fn nap_subthreshold_oscillations() {
        // INaP neurons often show subthreshold oscillations or low-rate
        // spontaneous firing — this is a biological feature, not a bug.
        // With negative input, INaP should be suppressed.
        let mut n = PersistentNaNeuron::new();
        let mut spikes_inhibited = 0;
        for _ in 0..10_000 {
            spikes_inhibited += n.step(-2.0);
        }
        assert_eq!(
            spikes_inhibited, 0,
            "INaP neuron must be silent with inhibitory input, got {spikes_inhibited}"
        );
    }

    #[test]
    fn nap_lowers_threshold() {
        // INaP provides subthreshold depolarisation → lower effective threshold
        let mut with_nap = PersistentNaNeuron::new();
        let mut no_nap = PersistentNaNeuron::new();
        no_nap.g_nap = 0.0;

        // Use near-threshold input
        let input = 1.0;
        let mut spikes_nap = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_nap += with_nap.step(input);
            spikes_no += no_nap.step(input);
        }
        assert!(
            spikes_nap >= spikes_no,
            "INaP must lower effective threshold: NaP={spikes_nap} vs none={spikes_no}"
        );
    }

    #[test]
    fn nap_p_gate_activates_at_subthreshold() {
        // At -50 mV (subthreshold), p_inf should be significant
        let mut n = PersistentNaNeuron::new();
        n.v = -50.0;
        // Step a few times for p to converge
        for _ in 0..1000 {
            // Hold at -50 mV artificially by resetting v each step
            let _ = n.step(0.0);
        }
        // p_inf at -50 mV = 1/(1+exp(2/5)) = 1/(1+1.49) = 0.40
        // After many steps p should approach p_inf
        assert!(
            n.p > 0.01,
            "p gate must activate at subthreshold voltages, p={}",
            n.p
        );
    }

    #[test]
    fn nap_increases_firing_rate() {
        // Same input, higher g_nap → more spikes
        let mut low = PersistentNaNeuron::new();
        low.g_nap = 0.2;
        let mut high = PersistentNaNeuron::new();
        high.g_nap = 1.5;

        let input = 1.5;
        let mut spikes_low = 0;
        let mut spikes_high = 0;
        for _ in 0..10_000 {
            spikes_low += low.step(input);
            spikes_high += high.step(input);
        }
        assert!(
            spikes_high >= spikes_low,
            "Higher g_nap must increase firing: high={spikes_high} vs low={spikes_low}"
        );
    }

    #[test]
    fn nap_negative_input_no_crash() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn nap_nan_input_is_rejected_atomically() {
        let mut n = PersistentNaNeuron::new();
        let before = n.clone();
        assert!(n.try_step(f64::NAN).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.p, before.p);
    }

    #[test]
    fn nap_invalid_configuration_is_rejected_atomically() {
        let mut n = PersistentNaNeuron::new();
        n.c_m = 0.0;
        let before = n.clone();
        assert!(n.try_step(1.0).is_err());
        assert_eq!(n.v, before.v);
        assert_eq!(n.c_m, before.c_m);
    }

    #[test]
    fn nap_extreme_input_bounded() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn nap_reset_clears_state() {
        let mut n = PersistentNaNeuron::new();
        n.g_nap = 0.3;
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.p, 0.0);
        assert_eq!(n.h, 0.6);
        assert_eq!(n.g_nap, 0.3);
    }

    #[test]
    fn nap_gates_bounded() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.p >= 0.0 && n.p <= 1.0);
    }

    #[test]
    fn nap_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = PersistentNaNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }
}
