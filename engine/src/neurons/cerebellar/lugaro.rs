// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Lugaro Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar Lugaro cell — rare fusiform interneuron in the granular layer.
///
/// Biophysics: LIF with adaptation for regular spiking, serotonin modulation
/// (5-HT increases gain), and a depolarised leak for spontaneous firing.
/// Inhibits Golgi cells and molecular layer interneurons (stellate, basket).
///
/// Lugaro cells are distinguished by their horizontal axonal projection,
/// large fusiform soma, and sensitivity to serotonergic afferents from
/// the brainstem raphe nuclei.
///
/// Dieudonné & Bhatt, J Physiol 548:97, 2003; Lainé & Bhatt, Front Syst Neurosci 1:4, 2007.
#[derive(Clone, Debug)]
pub struct LugaroCell {
    pub v: f64,
    pub adapt: f64, // Adaptation current
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64, // Adaptation coupling strength
    pub gain: f64,
    pub serotonin: f64, // 5-HT modulation factor [0, 1]
    pub dt: f64,
}

impl Default for LugaroCell {
    fn default() -> Self {
        Self::new()
    }
}

impl LugaroCell {
    pub fn new() -> Self {
        Self {
            v: -55.0,
            adapt: 0.0,
            v_rest: -55.0, // Depolarised rest for spontaneous firing
            v_reset: -65.0,
            v_threshold: -48.0,
            tau_m: 10.0,
            tau_adapt: 150.0,
            a_adapt: 0.05,
            gain: 2.0,
            serotonin: 0.0, // No 5-HT modulation by default
            dt: 0.5,
        }
    }

    /// Create with serotonin modulation active.
    pub fn with_serotonin(serotonin_level: f64) -> Self {
        let mut n = Self::new();
        n.serotonin = serotonin_level.clamp(0.0, 1.0);
        n
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.adapt,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau_m,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.serotonin,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.tau_m > 0.0
            && self.tau_adapt > 0.0
            && self.dt > 0.0
            && self.a_adapt >= 0.0
            && self.gain >= 0.0
            && (-100.0..=60.0).contains(&self.v)
            && (0.0..=1.0).contains(&self.serotonin)
            && self.adapt >= 0.0
            && self.v_threshold > self.v_reset
            && self.v_threshold > self.v_rest
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }

        // 5-HT modulation: increases effective gain
        let effective_gain = self.gain * (1.0 + 0.5 * self.serotonin);
        let input = effective_gain * current;

        // LIF dynamics with closed-form first-order relaxation.
        let v_inf = self.v_rest + input - self.adapt;
        let v_next = v_inf + (self.v - v_inf) * (-self.dt / self.tau_m).exp();

        // Adaptation dynamics with non-negative hyperpolarising current.
        let adapt_inf = (self.a_adapt * (v_next - self.v_rest).max(0.0)).max(0.0);
        let adapt_next =
            (adapt_inf + (self.adapt - adapt_inf) * (-self.dt / self.tau_adapt).exp()).max(0.0);
        if !v_next.is_finite() || !adapt_next.is_finite() {
            return 0;
        }

        // Spike detection
        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.adapt = adapt_next + 1.0; // Spike-triggered adaptation increment
            return 1;
        }

        // Safety bounds
        self.v = v_next.clamp(-100.0, 60.0);
        self.adapt = adapt_next;

        0
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Lugaro Cell tests --

    #[test]
    fn lugaro_fires_with_input() {
        let mut n = LugaroCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 10,
            "Lugaro must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn lugaro_low_threshold() {
        // Near-threshold rest → fires easily with moderate input
        let mut n = LugaroCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(4.0);
        }
        assert!(
            spikes > 10,
            "Lugaro should fire easily with moderate input, got {spikes}"
        );
    }

    #[test]
    fn lugaro_adaptation() {
        let mut n = LugaroCell::new();
        let input = 10.0;
        let mut spikes_early = 0;
        for _ in 0..2000 {
            spikes_early += n.step(input);
        }
        let mut spikes_late = 0;
        for _ in 0..2000 {
            spikes_late += n.step(input);
        }
        assert!(
            spikes_early >= spikes_late,
            "Adaptation should slow firing: early={spikes_early}, late={spikes_late}"
        );
    }

    #[test]
    fn lugaro_serotonin_increases_firing() {
        let mut no_5ht = LugaroCell::new();
        let mut with_5ht = LugaroCell::with_serotonin(1.0);

        let input = 3.0;
        let mut spikes_no = 0;
        let mut spikes_5ht = 0;
        for _ in 0..10_000 {
            spikes_no += no_5ht.step(input);
            spikes_5ht += with_5ht.step(input);
        }
        assert!(
            spikes_5ht >= spikes_no,
            "5-HT must increase firing: 5HT={spikes_5ht} vs none={spikes_no}"
        );
    }

    #[test]
    fn lugaro_negative_input_no_crash() {
        let mut n = LugaroCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn lugaro_nan_input_stays_finite() {
        let mut n = LugaroCell::new();
        let before = n.clone();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert_eq!(n.v, before.v);
        assert_eq!(n.adapt, before.adapt);
    }

    #[test]
    fn lugaro_corrupted_state_preserved_on_step() {
        let mut n = LugaroCell::new();
        n.adapt = f64::NAN;
        let before = n.clone();
        assert_eq!(n.step(5.0), 0);
        assert_eq!(n.v, before.v);
        assert!(n.adapt.is_nan());
    }

    #[test]
    fn lugaro_invalid_voltage_preserved_on_step() {
        let mut n = LugaroCell::new();
        n.v = 60.1;
        let before = n.clone();
        assert_eq!(n.step(5.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.adapt, before.adapt);
    }

    #[test]
    fn lugaro_closed_form_membrane_and_adaptation_relaxation() {
        let mut n = LugaroCell::new();
        n.v = -56.0;
        n.adapt = 0.2;
        n.gain = 0.0;

        let v_inf = n.v_rest - n.adapt;
        let expected_v = exact_relax_lugaro(n.v, v_inf, n.tau_m, n.dt);
        let adapt_inf = (n.a_adapt * (expected_v - n.v_rest).max(0.0)).max(0.0);
        let expected_adapt = exact_relax_lugaro(n.adapt, adapt_inf, n.tau_adapt, n.dt).max(0.0);

        assert_eq!(n.step(0.0), 0);
        assert_close_lugaro(n.v, expected_v, 1e-12);
        assert_close_lugaro(n.adapt, expected_adapt, 1e-12);
    }

    fn exact_relax_lugaro(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn assert_close_lugaro(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e} expected={expected:.16e} tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn lugaro_extreme_input_bounded() {
        let mut n = LugaroCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn lugaro_reset_clears_state() {
        let mut n = LugaroCell::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -55.0);
        assert_eq!(n.adapt, 0.0);
        assert_eq!(n.serotonin, 0.0);
    }

    #[test]
    fn lugaro_adapt_increases_during_spiking() {
        let mut n = LugaroCell::new();
        let initial = n.adapt;
        for _ in 0..5000 {
            n.step(10.0);
        }
        assert!(
            n.adapt > initial,
            "Adaptation must increase during spiking, adapt={}",
            n.adapt
        );
    }

    #[test]
    fn lugaro_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = LugaroCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps must complete in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn lugaro_default_matches_constructor_contract() {
        let default = LugaroCell::default();
        let constructed = LugaroCell::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.adapt, constructed.adapt);
        assert_eq!(default.serotonin, constructed.serotonin);
        assert_eq!(default.dt, constructed.dt);
    }
}
