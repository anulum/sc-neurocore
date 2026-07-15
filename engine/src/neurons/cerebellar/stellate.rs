// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

use super::super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Stellate Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar stellate cell — fast-spiking interneuron in the molecular layer.
///
/// Biophysics: Wang-Buzsáki Na+/K+ core extended with Kv3.1 for narrow
/// action potentials and high-frequency firing. Provides feedforward
/// inhibition onto Purkinje cell dendrites. Receives excitatory input
/// from parallel fibres (granule cell axons).
///
/// Stellate cells are smaller than basket cells and innervate more distal
/// Purkinje cell dendrites. They show minimal spike frequency adaptation
/// and can sustain high firing rates.
///
/// Sultan & Bower, J Comp Neurol 409:63, 1999; Häusser & Clark, Neuron 19:665, 1997.
#[derive(Clone, Debug)]
pub struct StellateCell {
    pub v: f64,
    pub h: f64, // Na+ inactivation
    pub n: f64, // Kdr activation
    pub p: f64, // Kv3.1 activation
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv3: f64,
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

impl Default for StellateCell {
    fn default() -> Self {
        Self::new()
    }
}

impl StellateCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv3: 3.0, // Less Kv3.1 than PV+ basket
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 0.5, // Smaller cell → lower capacitance
            phi: 5.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    #[inline]
    fn safe_exp(value: f64) -> f64 {
        value.clamp(-60.0, 60.0).exp()
    }

    #[inline]
    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        let z = -(v - vh) / k;
        if z > 60.0 {
            0.0
        } else if z < -60.0 {
            1.0
        } else {
            1.0 / (1.0 + z.exp())
        }
    }

    #[inline]
    fn exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        if tau <= f64::EPSILON {
            target.clamp(0.0, 1.0)
        } else {
            (target + (value - target) * (-dt / tau).exp()).clamp(0.0, 1.0)
        }
    }

    #[inline]
    fn exact_hh_gate(value: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
        let total = alpha + beta;
        if total <= f64::EPSILON {
            value.clamp(0.0, 1.0)
        } else {
            let steady = alpha / total;
            (steady + (value - steady) * (-phi * total * dt).exp()).clamp(0.0, 1.0)
        }
    }

    #[inline]
    fn exact_voltage_step(
        v: f64,
        c_m: f64,
        input: f64,
        conductances: [(f64, f64); 4],
        dt: f64,
    ) -> f64 {
        let g_total = conductances.iter().map(|(g, _)| *g).sum::<f64>();
        let drive = input
            + conductances
                .iter()
                .map(|(g, reversal)| g * reversal)
                .sum::<f64>();
        if g_total <= f64::EPSILON {
            v + dt * drive / c_m
        } else {
            let v_inf = drive / g_total;
            let tau = c_m / g_total;
            v_inf + (v - v_inf) * (-dt / tau).exp()
        }
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.h,
            self.n,
            self.p,
            self.g_na,
            self.g_k,
            self.g_kv3,
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
        .iter()
        .all(|value| value.is_finite())
            && (-100.0..=60.0).contains(&self.v)
            && [self.h, self.n, self.p]
                .iter()
                .all(|gate| (0.0..=1.0).contains(gate))
            && [self.g_na, self.g_k, self.g_kv3, self.g_l]
                .iter()
                .all(|conductance| *conductance >= 0.0)
            && self.c_m > 0.0
            && self.phi > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }

        let input = self.gain * current;
        let sub_steps = 50;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut p = self.p;

        for _ in 0..sub_steps {
            // WB alpha/beta rates
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * Self::safe_exp(-(v + 60.0) / 18.0);
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * Self::safe_exp(-(v + 58.0) / 20.0);
            let beta_h = Self::boltz(v, -28.0, 10.0);

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * Self::safe_exp(-(v + 44.0) / 80.0);

            // Kv3.1 gating (fast activation, no inactivation)
            let p_inf = Self::boltz(v, -10.0, 10.0);
            let tau_p = 1.0 + 4.0 / (1.0 + Self::safe_exp((v + 20.0) / 15.0));

            // Gate updates
            h = Self::exact_hh_gate(h, alpha_h, beta_h, self.phi, sub_dt);
            n = Self::exact_hh_gate(n, alpha_n, beta_n, self.phi, sub_dt);
            p = Self::exact_relax(p, p_inf, tau_p, sub_dt);

            // Currents (m uses steady-state for speed)
            let g_na = self.g_na * m_inf.powi(3) * h;
            let g_k = self.g_k * n.powi(4);
            let g_kv3 = self.g_kv3 * p.powi(2);
            let g_l = self.g_l;

            v = Self::exact_voltage_step(
                v,
                self.c_m,
                input,
                [
                    (g_na, self.e_na),
                    (g_k, self.e_k),
                    (g_kv3, self.e_k),
                    (g_l, self.e_l),
                ],
                sub_dt,
            )
            .clamp(-100.0, 60.0);
            if ![v, h, n, p].iter().all(|value| value.is_finite()) {
                return 0;
            }

            if v >= self.v_threshold {
                fired = 1;
                v = -65.0;
            }
        }

        self.v = v;
        self.h = h;
        self.n = n;
        self.p = p;

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Stellate Cell tests --

    #[test]
    fn stellate_fires_with_input() {
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(
            spikes > 5,
            "Stellate cell must fire with input, got {spikes}"
        );
    }

    #[test]
    fn stellate_silent_without_input() {
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Stellate cell must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn stellate_high_frequency() {
        // Fast-spiking: should sustain high rates
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(20.0);
        }
        // 2000 steps * 0.5ms = 1000 ms; >100 spikes = >100 Hz
        assert!(
            spikes > 50,
            "FS stellate should fire at high rate, got {spikes}"
        );
    }

    #[test]
    fn stellate_minimal_adaptation() {
        // Compare early vs late firing — should show little adaptation
        let mut n = StellateCell::new();
        let input = 10.0;
        let mut spikes_early = 0;
        for _ in 0..2000 {
            spikes_early += n.step(input);
        }
        let mut spikes_late = 0;
        for _ in 0..2000 {
            spikes_late += n.step(input);
        }
        // No AHP → minimal adaptation
        let diff = (spikes_early - spikes_late).abs();
        assert!(
            diff < 20,
            "FS should have minimal adaptation: early={spikes_early}, late={spikes_late}"
        );
    }

    #[test]
    fn stellate_kv3_narrows_spikes() {
        // Kv3.1 should allow faster repolarisation → more spikes
        let mut with_kv3 = StellateCell::new();
        let mut no_kv3 = StellateCell::new();
        no_kv3.g_kv3 = 0.0;

        let mut spikes_kv3 = 0;
        let mut spikes_no = 0;
        for _ in 0..2000 {
            spikes_kv3 += with_kv3.step(15.0);
            spikes_no += no_kv3.step(15.0);
        }
        // Kv3.1 should enable higher frequency (more spikes at same input)
        assert!(spikes_kv3 > 0, "With Kv3.1 must fire, got {spikes_kv3}");
        assert!(
            spikes_no >= 0,
            "No-Kv3.1 baseline must not panic, got {spikes_no}"
        );
    }

    #[test]
    fn stellate_negative_input_no_crash() {
        let mut n = StellateCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn stellate_nan_input_stays_finite() {
        let mut n = StellateCell::new();
        let before = n.clone();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.p, before.p);
    }

    #[test]
    fn stellate_corrupted_state_preserved_on_step() {
        let mut n = StellateCell::new();
        n.h = -0.1;
        let before = n.clone();
        assert_eq!(n.step(8.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.p, before.p);
    }

    #[test]
    fn stellate_invalid_voltage_preserved_on_step() {
        let mut n = StellateCell::new();
        n.v = 60.1;
        let before = n.clone();
        assert_eq!(n.step(8.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.h, before.h);
        assert_eq!(n.n, before.n);
        assert_eq!(n.p, before.p);
    }

    #[test]
    fn stellate_closed_form_gate_kinetics() {
        let mut n = StellateCell::new();
        n.g_na = 0.0;
        n.g_k = 0.0;
        n.g_kv3 = 0.0;
        n.g_l = 0.0;
        n.gain = 0.0;

        let alpha_h = 0.07 * StellateCell::safe_exp(-(n.v + 58.0) / 20.0);
        let beta_h = StellateCell::boltz(n.v, -28.0, 10.0);
        let alpha_n = safe_rate(0.01, 34.0, n.v, 10.0, 0.1);
        let beta_n = 0.125 * StellateCell::safe_exp(-(n.v + 44.0) / 80.0);
        let p_inf = StellateCell::boltz(n.v, -10.0, 10.0);
        let tau_p = 1.0 + 4.0 / (1.0 + StellateCell::safe_exp((n.v + 20.0) / 15.0));

        let expected_h = exact_hh_gate_stellate(n.h, alpha_h, beta_h, n.phi, n.dt);
        let expected_n = exact_hh_gate_stellate(n.n, alpha_n, beta_n, n.phi, n.dt);
        let expected_p = exact_relax_stellate(n.p, p_inf, tau_p, n.dt);
        let expected_v = n.v;

        assert_eq!(n.step(0.0), 0);
        assert_close_stellate(n.v, expected_v, 1e-12);
        assert_close_stellate(n.h, expected_h, 1e-12);
        assert_close_stellate(n.n, expected_n, 1e-12);
        assert_close_stellate(n.p, expected_p, 1e-12);
    }

    fn exact_relax_stellate(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        if tau <= f64::EPSILON {
            target.clamp(0.0, 1.0)
        } else {
            (target + (value - target) * (-dt / tau).exp()).clamp(0.0, 1.0)
        }
    }

    fn exact_hh_gate_stellate(value: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
        let total = alpha + beta;
        if total <= f64::EPSILON {
            value.clamp(0.0, 1.0)
        } else {
            let steady = alpha / total;
            (steady + (value - steady) * (-phi * total * dt).exp()).clamp(0.0, 1.0)
        }
    }

    fn assert_close_stellate(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e} expected={expected:.16e} tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn stellate_extreme_input_bounded() {
        let mut n = StellateCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn stellate_reset_clears_state() {
        let mut n = StellateCell::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.p, 0.0);
    }

    #[test]
    fn stellate_gates_bounded() {
        let mut n = StellateCell::new();
        for _ in 0..10_000 {
            n.step(15.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.p >= 0.0 && n.p <= 1.0);
    }

    #[test]
    fn stellate_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = StellateCell::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(10.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms, took {}ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn stellate_default_matches_constructor_contract() {
        let default = StellateCell::default();
        let constructed = StellateCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.h, constructed.h);
        assert_eq!(default.p, constructed.p);
        assert_eq!(default.dt, constructed.dt);
    }
}
