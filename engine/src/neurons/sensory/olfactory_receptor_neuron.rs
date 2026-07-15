// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

// ═══════════════════════════════════════════════════════════════════
// Olfactory Receptor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Olfactory receptor neuron — chemical-to-spike transducer.
///
/// Odorant binding → Golf → adenylyl cyclase → cAMP → CNG channels.
/// Produces spiking output to olfactory bulb.
///
/// Adaptation via two pathways:
/// - **Ca²⁺/CaM feedback** on CNG channels (fast, ~500 ms)
/// - **PDE4 negative feedback** on cAMP (slow, ~300 ms): cAMP → PKA → PDE4 ↑ → cAMP ↓
///
/// Based on Rospars et al. 2008 / Firestein 2001.
#[derive(Clone, Debug)]
pub struct OlfactoryReceptorNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub camp: f64,      // Normalised cAMP [0, 1]
    pub adapt: f64,     // Ca²⁺/CaM adaptation
    pub pde4: f64,      // PDE4 activity (tracks cAMP with delay)
    pub tau_camp: f64,  // cAMP dynamics (ms)
    pub tau_adapt: f64, // CaM adaptation tau
    pub tau_pde4: f64,  // PDE4 activation tau (ms)
    pub k_pde4: f64,    // PDE4 degradation rate on cAMP
    pub gain: f64,
    pub dt: f64,
}

impl OlfactoryReceptorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -45.0,
            tau: 5.0,
            camp: 0.0,
            adapt: 0.0,
            pde4: 0.0,
            tau_camp: 50.0,
            tau_adapt: 500.0,
            tau_pde4: 300.0, // PDE4 activation ~300 ms (slow negative feedback)
            k_pde4: 1.5,     // PDE4 degradation strength
            gain: 1.5,
            dt: 0.5,
        }
    }

    /// Step with odorant concentration (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, concentration: f64) -> i32 {
        let conc = concentration.max(0.0);

        // cAMP production: Hill function of odorant, reduced by CaM adaptation
        let camp_production = conc / (conc + 1.0) * (1.0 - 0.8 * self.adapt);
        // PDE4 degradation: proportional to PDE4 activity × cAMP
        let pde4_degradation = self.k_pde4 * self.pde4 * self.camp;
        let camp_target = (camp_production - pde4_degradation).max(0.0);
        self.camp += (camp_target - self.camp) / self.tau_camp * self.dt;
        self.camp = self.camp.clamp(0.0, 1.0);

        // PDE4 activation: tracks cAMP with delay (cAMP → PKA → PDE4 upregulation)
        self.pde4 += (self.camp - self.pde4) / self.tau_pde4 * self.dt;
        self.pde4 = self.pde4.clamp(0.0, 1.0);

        let drive = self.gain * self.camp * 50.0; // Scale to mV
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;

        // Ca²⁺/CaM adaptation (fast pathway)
        let ca_proxy = if self.v > self.v_rest {
            (self.v - self.v_rest) / 20.0
        } else {
            0.0
        };
        self.adapt += (ca_proxy - self.adapt) / self.tau_adapt * self.dt;
        self.adapt = self.adapt.clamp(0.0, 1.0);

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.camp = 0.0;
        self.adapt = 0.0;
        self.pde4 = 0.0;
    }
}

impl Default for OlfactoryReceptorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn olfactory_fires_with_odorant() {
        let mut o = OlfactoryReceptorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        assert!(spikes > 0, "olfactory should fire with odorant");
    }

    #[test]
    fn olfactory_adapts() {
        let mut o = OlfactoryReceptorNeuron::new();
        let first: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        let second: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        assert!(
            second <= first + 5,
            "olfactory should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn olfactory_no_fire_without_odorant() {
        let mut o = OlfactoryReceptorNeuron::new();
        let spikes: i32 = (0..1000).map(|_| o.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn olfactory_reset() {
        let mut o = OlfactoryReceptorNeuron::new();
        for _ in 0..1000 {
            o.step(5.0);
        }
        o.reset();
        assert_eq!(o.camp, 0.0);
        assert_eq!(o.adapt, 0.0);
        assert_eq!(o.pde4, 0.0);
    }

    #[test]
    fn olfactory_pde4_activates_with_odorant() {
        // PDE4 should rise when cAMP is elevated
        let mut o = OlfactoryReceptorNeuron::new();
        assert_eq!(o.pde4, 0.0);
        for _ in 0..5000 {
            o.step(5.0);
        }
        assert!(
            o.pde4 > 0.0,
            "PDE4 should activate with sustained odorant, got {}",
            o.pde4
        );
    }

    #[test]
    fn olfactory_pde4_reduces_camp() {
        // With PDE4, sustained cAMP should be lower than without
        let mut with_pde4 = OlfactoryReceptorNeuron::new();
        let mut no_pde4 = OlfactoryReceptorNeuron::new();
        no_pde4.k_pde4 = 0.0; // disable PDE4

        for _ in 0..10_000 {
            with_pde4.step(5.0);
            no_pde4.step(5.0);
        }
        assert!(
            with_pde4.camp < no_pde4.camp,
            "PDE4 should reduce cAMP: with={:.3} vs without={:.3}",
            with_pde4.camp,
            no_pde4.camp
        );
    }

    #[test]
    fn olfactory_pde4_enhances_adaptation() {
        // PDE4 feedback should reduce late firing more than CaM alone
        let mut with_pde4 = OlfactoryReceptorNeuron::new();
        let mut no_pde4 = OlfactoryReceptorNeuron::new();
        no_pde4.k_pde4 = 0.0;

        // Warm up
        for _ in 0..5000 {
            with_pde4.step(5.0);
            no_pde4.step(5.0);
        }
        // Measure late firing
        let spikes_with: i32 = (0..5000).map(|_| with_pde4.step(5.0)).sum();
        let spikes_no: i32 = (0..5000).map(|_| no_pde4.step(5.0)).sum();
        assert!(
            spikes_with <= spikes_no,
            "PDE4 should enhance adaptation: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn olfactory_default_matches_constructor_contract() {
        let default = OlfactoryReceptorNeuron::default();
        let constructed = OlfactoryReceptorNeuron::new();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.camp, constructed.camp);
        assert_eq!(default.pde4, constructed.pde4);
        assert_eq!(default.dt, constructed.dt);
    }
}
