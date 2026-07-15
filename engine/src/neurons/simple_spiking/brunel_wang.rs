// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel-Wang Neuron Model

//! Brunel-Wang conductance-based LIF dynamics.

// ═══════════════════════════════════════════════════════════════════
// Brunel-Wang LIF with NMDA/AMPA/GABA
// ═══════════════════════════════════════════════════════════════════

/// Brunel-Wang 2001 — LIF with NMDA (Mg²⁺ block), AMPA, and GABA synaptic
/// conductances for decision-making and working memory circuits.
///
/// Key feature: voltage-dependent NMDA conductance via Mg²⁺ block factor
/// `1 / (1 + [Mg²⁺]/3.57 · exp(-0.062·V))`. This creates positive feedback
/// that sustains persistent activity in recurrent circuits.
///
/// The single-current interface routes external input to i_ampa_ext; recurrent
/// AMPA/NMDA/GABA inputs are zero (use the multi-arg `step_full()` for
/// network simulations).
///
/// Brunel, N. & Wang, X.J., J Comput Neurosci 11:63, 2001.
#[derive(Clone, Debug)]
pub struct BrunelWangNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_ref: f64,
    pub g_ampa_ext: f64,
    pub g_ampa_rec: f64,
    pub g_nmda: f64,
    pub g_gaba: f64,
    pub v_ampa: f64,
    pub v_nmda: f64,
    pub v_gaba: f64,
    pub c_m: f64,
    pub mg_conc: f64,
    pub dt: f64,
    pub ref_remaining: f64,
    pub gain: f64,
}

impl BrunelWangNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -55.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            tau_ref: 2.0,
            g_ampa_ext: 2.1,
            g_ampa_rec: 0.05,
            g_nmda: 0.165,
            g_gaba: 1.3,
            v_ampa: 0.0,
            v_nmda: 0.0,
            v_gaba: -70.0,
            c_m: 0.5,
            mg_conc: 1.0,
            dt: 0.1,
            ref_remaining: 0.0,
            gain: 1.0,
        }
    }

    /// Mg²⁺ block factor (Jahr & Stevens 1990).
    #[inline]
    fn nmda_mg_block(&self, v: f64) -> f64 {
        1.0 / (1.0 + self.mg_conc / 3.57 * (-0.062 * v).exp())
    }

    /// Full step with all 4 synaptic inputs (for network simulations).
    pub fn step_full(
        &mut self,
        i_ampa_ext: f64,
        s_ampa_rec: f64,
        s_nmda_rec: f64,
        s_gaba: f64,
    ) -> i32 {
        if self.ref_remaining > 0.0 {
            self.ref_remaining -= self.dt;
            return 0;
        }

        let v = self.v;

        // Synaptic currents (conductance-based)
        let i_ampa = -self.g_ampa_ext * (v - self.v_ampa) * i_ampa_ext
            - self.g_ampa_rec * (v - self.v_ampa) * s_ampa_rec;
        let i_nmda = -self.g_nmda * self.nmda_mg_block(v) * (v - self.v_nmda) * s_nmda_rec;
        let i_gaba = -self.g_gaba * (v - self.v_gaba) * s_gaba;

        // Membrane dynamics (LIF)
        let i_leak = -(v - self.v_rest) / self.tau_m;
        let dv = (i_leak + (i_ampa + i_nmda + i_gaba) / self.c_m) * self.dt;
        self.v += dv;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.ref_remaining = self.tau_ref;
            1
        } else {
            0
        }
    }

    /// Single-current interface: routes input to external AMPA drive.
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_full(self.gain * current, 0.0, 0.0, 0.0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ref_remaining = 0.0;
    }
}

impl Default for BrunelWangNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = BrunelWangNeuron::default();
        let constructed = BrunelWangNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn brunel_wang_fires_with_ampa_ext() {
        let mut n = BrunelWangNeuron::new();
        let mut spikes = 0;
        for _ in 0..5000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "Must fire with external AMPA drive, got {spikes}"
        );
    }

    #[test]
    fn brunel_wang_silent_without_input() {
        let mut n = BrunelWangNeuron::new();
        let spikes: i32 = (0..10_000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0, "Must be silent without input");
    }

    #[test]
    fn brunel_wang_nmda_mg_block() {
        let n = BrunelWangNeuron::new();
        // At resting potential (-70 mV), Mg²⁺ block should be strong
        let block_rest = n.nmda_mg_block(-70.0);
        // At depolarised potential (0 mV), block should be weak
        let block_depol = n.nmda_mg_block(0.0);
        assert!(
            block_depol > block_rest,
            "Mg²⁺ block should weaken with depolarisation: rest={block_rest:.3} depol={block_depol:.3}"
        );
        // At -70 mV, block factor should be small (< 0.1)
        assert!(
            block_rest < 0.1,
            "Block at -70 mV should be < 0.1, got {block_rest:.3}"
        );
        // At 0 mV, block factor should be close to 1
        assert!(
            block_depol > 0.5,
            "Block at 0 mV should be > 0.5, got {block_depol:.3}"
        );
    }

    #[test]
    fn brunel_wang_full_step_nmda_drive() {
        // NMDA recurrent input should drive firing via positive feedback
        let mut n = BrunelWangNeuron::new();
        n.v = -55.0; // near threshold, Mg²⁺ block partially relieved
        let mut spikes = 0;
        for _ in 0..1000 {
            spikes += n.step_full(0.0, 0.0, 1.0, 0.0); // pure NMDA drive
        }
        assert!(
            spikes > 0,
            "NMDA drive at depolarised V should cause spikes"
        );
    }

    #[test]
    fn brunel_wang_gaba_suppresses() {
        let mut with_gaba = BrunelWangNeuron::new();
        let mut no_gaba = BrunelWangNeuron::new();
        let mut spikes_gaba = 0;
        let mut spikes_no = 0;
        for _ in 0..5000 {
            spikes_gaba += with_gaba.step_full(3.0, 0.0, 0.0, 1.0); // GABA on
            spikes_no += no_gaba.step_full(3.0, 0.0, 0.0, 0.0);
        }
        assert!(
            spikes_no >= spikes_gaba,
            "GABA should suppress: no_gaba={spikes_no}, with_gaba={spikes_gaba}"
        );
    }

    #[test]
    fn brunel_wang_refractory() {
        let mut n = BrunelWangNeuron::new();
        // Drive to spike
        while n.step(10.0) == 0 {}
        // Immediately after spike, should be in refractory
        assert!(n.ref_remaining > 0.0, "Should be refractory after spike");
        // Should not spike during refractory
        assert_eq!(
            n.step(100.0),
            0,
            "Should not spike during refractory period"
        );
    }

    #[test]
    fn brunel_wang_reset() {
        let mut n = BrunelWangNeuron::new();
        for _ in 0..1000 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, n.v_rest);
        assert_eq!(n.ref_remaining, 0.0);
    }

    #[test]
    fn brunel_wang_voltage_bounded() {
        let mut n = BrunelWangNeuron::new();
        for _ in 0..10_000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite(), "V must stay finite");
        // V should stay near v_reset during sustained spiking (refractory resets)
        assert!(
            n.v <= n.v_threshold,
            "V should be at or below threshold (reset clamp)"
        );
    }

    #[test]
    fn brunel_wang_nan_input() {
        let mut n = BrunelWangNeuron::new();
        n.step(f64::NAN);
        // NaN input → V becomes NaN. Check we don't panic.
    }

    #[test]
    fn brunel_wang_performance() {
        let start = std::time::Instant::now();
        let mut n = BrunelWangNeuron::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }
}
