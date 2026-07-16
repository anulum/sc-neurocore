// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Energy-aware LIF Neuron

/// Energy-aware LIF — metabolic cost modulates gain. Sengupta et al. 2013.
#[derive(Clone, Debug)]
pub struct EnergyLIFNeuron {
    pub v: f64,
    pub epsilon: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_e: f64,
    pub alpha: f64,
    pub epsilon_0: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl EnergyLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            epsilon: 1.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_e: 500.0,
            alpha: 0.1,
            epsilon_0: 1.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let effective_r = self.resistance * self.epsilon;
        self.v += (-(self.v - self.v_rest) + effective_r * current) / self.tau_m * self.dt;
        self.epsilon += (self.epsilon_0 - self.epsilon) / self.tau_e * self.dt;
        if self.v >= self.v_threshold && self.epsilon > 0.1 {
            self.v = self.v_reset;
            self.epsilon -= self.alpha;
            1
        } else if self.v >= self.v_threshold {
            self.v = self.v_threshold;
            0
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.epsilon = self.epsilon_0;
    }
}

impl Default for EnergyLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_lif_fires() {
        let mut n = EnergyLIFNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
    }
    #[test]
    fn energy_lif_silent_without_input() {
        let mut n = EnergyLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn energy_lif_reset_clears_state() {
        let mut n = EnergyLIFNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn energy_lif_bounded() {
        let mut n = EnergyLIFNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn energy_lif_nan_no_panic() {
        EnergyLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn energy_lif_epsilon_depletes() {
        let mut n = EnergyLIFNeuron::new();
        let e0 = n.epsilon;
        for _ in 0..200 {
            n.step(30.0);
        }
        assert!(n.epsilon < e0, "energy should deplete during spiking");
    }
}
