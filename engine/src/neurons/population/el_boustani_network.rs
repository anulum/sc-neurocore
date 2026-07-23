// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — El Boustani Network Model

/// El Boustani & Bhatt 2009 — E/I network with NMDA-mediated bistability.
///
/// Two-population (E/I) mean-field with separate fast (AMPA) and slow
/// (NMDA) excitatory components. NMDA provides the recurrent excitation
/// needed for working memory persistent activity (bistability).
///
/// tau_e * dr_e/dt = -r_e + phi(J_ampa*r_e + J_nmda*s + I - J_ei*r_i)
/// tau_i * dr_i/dt = -r_i + phi(J_ie*r_e - J_ii*r_i)
/// tau_s * ds/dt = -s + gamma * r_e * (1 - s)
///
/// El Boustani & Bhatt, J Comput Neurosci 26:313, 2009.
#[derive(Clone, Debug)]
pub struct ElBoustaniNetwork {
    pub r_e: f64,
    pub r_i: f64,
    pub s: f64, // NMDA synaptic gating variable
    pub tau_e: f64,
    pub tau_i: f64,
    pub tau_s: f64,  // NMDA decay (~100 ms)
    pub j_ampa: f64, // Fast E→E (AMPA)
    pub j_nmda: f64, // Slow E→E (NMDA)
    pub j_ei: f64,
    pub j_ie: f64,
    pub j_ii: f64,
    pub gamma: f64, // NMDA saturation rate
    pub threshold: f64,
    pub gain_phi: f64,
    pub dt: f64,
    pub r_threshold: f64,
    pub gain: f64,
}

impl Default for ElBoustaniNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl ElBoustaniNetwork {
    pub fn new() -> Self {
        Self {
            r_e: 0.1,
            r_i: 0.1,
            s: 0.0,
            tau_e: 20.0,
            tau_i: 10.0,
            tau_s: 100.0,
            j_ampa: 0.1,
            j_nmda: 0.5,
            j_ei: 0.8,
            j_ie: 0.5,
            j_ii: 0.2,
            gamma: 0.641, // NMDA saturation parameter
            threshold: 0.0,
            gain_phi: 1.0,
            dt: 0.1,
            r_threshold: 1.0,
            gain: 1.0,
        }
    }

    fn phi(&self, x: f64) -> f64 {
        if x > self.threshold {
            self.gain_phi * (x - self.threshold)
        } else {
            0.0
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_e_prev = self.r_e;

        // NMDA gating dynamics
        let ds = (-self.s + self.gamma * self.r_e * (1.0 - self.s)) / self.tau_s;
        self.s += self.dt * ds;

        // E and I rate dynamics
        let drive_e = self.j_ampa * self.r_e + self.j_nmda * self.s - self.j_ei * self.r_i + input;
        let drive_i = self.j_ie * self.r_e - self.j_ii * self.r_i;

        let dr_e = (-self.r_e + self.phi(drive_e)) / self.tau_e;
        let dr_i = (-self.r_i + self.phi(drive_i)) / self.tau_i;

        self.r_e += self.dt * dr_e;
        self.r_i += self.dt * dr_i;

        // Bounds
        self.r_e = self.r_e.clamp(0.0, 200.0);
        self.r_i = self.r_i.clamp(0.0, 200.0);
        self.s = self.s.clamp(0.0, 1.0);
        if !self.r_e.is_finite() {
            self.r_e = 0.1;
        }
        if !self.r_i.is_finite() {
            self.r_i = 0.1;
        }
        if !self.s.is_finite() {
            self.s = 0.0;
        }

        if self.r_e >= self.r_threshold && r_e_prev < self.r_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elboustani_fires_with_input() {
        let mut n = ElBoustaniNetwork::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 0,
            "ElBoustani must produce bursts with input, got {spikes}"
        );
    }

    #[test]
    fn elboustani_silent_without_input() {
        let mut n = ElBoustaniNetwork::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "ElBoustani must be quiescent without input, got {spikes}"
        );
    }

    #[test]
    fn elboustani_nmda_builds_with_activity() {
        // NMDA gating variable s should increase with sustained E activity
        let mut n = ElBoustaniNetwork::new();
        let s0 = n.s;
        for _ in 0..10_000 {
            n.step(5.0);
        }
        assert!(
            n.s > s0,
            "NMDA gating should increase with activity: s0={s0}, s_now={}",
            n.s
        );
    }

    #[test]
    fn elboustani_ei_balance() {
        // Inhibition should keep E rate bounded
        let mut n = ElBoustaniNetwork::new();
        for _ in 0..50_000 {
            n.step(3.0);
        }
        assert!(
            n.r_e < 50.0,
            "E/I balance should keep r_e bounded, r_e={}",
            n.r_e
        );
        assert!(n.r_i >= 0.0, "r_i must be non-negative");
    }

    #[test]
    fn elboustani_nmda_enhances_excitation() {
        // With NMDA (j_nmda > 0), E rate should be higher than without
        let mut with_nmda = ElBoustaniNetwork::new();
        let mut no_nmda = ElBoustaniNetwork::new();
        no_nmda.j_nmda = 0.0;
        for _ in 0..20_000 {
            with_nmda.step(3.0);
            no_nmda.step(3.0);
        }
        assert!(
            with_nmda.r_e >= no_nmda.r_e,
            "NMDA should enhance excitation: with={:.3} vs without={:.3}",
            with_nmda.r_e,
            no_nmda.r_e
        );
    }

    #[test]
    fn elboustani_nmda_bounded() {
        // NMDA gating s must stay in [0, 1]
        let mut n = ElBoustaniNetwork::new();
        for _ in 0..50_000 {
            n.step(10.0);
        }
        assert!(
            n.s >= 0.0 && n.s <= 1.0,
            "NMDA gating must be in [0,1], s={}",
            n.s
        );
    }

    #[test]
    fn elboustani_rates_non_negative() {
        let mut n = ElBoustaniNetwork::new();
        for _ in 0..50_000 {
            n.step(-10.0);
        }
        assert!(n.r_e >= 0.0);
        assert!(n.r_i >= 0.0);
    }

    #[test]
    fn elboustani_nan_input_stays_finite() {
        let mut n = ElBoustaniNetwork::new();
        n.step(f64::NAN);
        assert!(n.r_e.is_finite());
        assert!(n.r_i.is_finite());
        assert!(n.s.is_finite());
    }

    #[test]
    fn elboustani_extreme_input_bounded() {
        let mut n = ElBoustaniNetwork::new();
        for _ in 0..10_000 {
            n.step(1e6);
        }
        assert!(n.r_e.is_finite() && n.r_e <= 200.0);
        assert!(n.s >= 0.0 && n.s <= 1.0);
    }

    #[test]
    fn elboustani_reset_clears_state() {
        let mut n = ElBoustaniNetwork::new();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.r_e, 0.1);
        assert_eq!(n.r_i, 0.1);
        assert_eq!(n.s, 0.0);
    }
}
