// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DPI Neuron Circuit Emulator

/// DPI current-mode adaptive integrate-and-fire circuit.
///
/// Implements Indiveri, Stefanini & Chicca (2010), Eqs. (2)–(3), with a
/// simultaneous explicit-Euler macro-step and a spike-driven refractory pulse.
#[derive(Clone, Debug)]
pub struct DPINeuron {
    pub i_mem: f64,
    pub i_ahp: f64,
    pub refractory_time: f64,
    pub i_threshold: f64,
    pub i_reset: f64,
    pub i_rest: f64,
    pub i_tau: f64,
    pub i_g: f64,
    pub i_tau_ahp: f64,
    pub i_ga: f64,
    pub i_spike: f64,
    pub i_0: f64,
    pub kappa: f64,
    pub alpha: f64,
    pub tau: f64,
    pub tau_ahp: f64,
    pub refractory_period: f64,
    pub dt: f64,
}

impl DPINeuron {
    pub fn new() -> Self {
        Self {
            i_mem: 0.01,
            i_ahp: 0.01,
            refractory_time: 0.0,
            i_threshold: 1.0,
            i_reset: 0.01,
            i_rest: 0.1,
            i_tau: 1.0,
            i_g: 1.0,
            i_tau_ahp: 0.1,
            i_ga: 1.0,
            i_spike: 5.0,
            i_0: 0.01,
            kappa: 0.7,
            alpha: 10.0,
            tau: 20.0,
            tau_ahp: 100.0,
            refractory_period: 2.0,
            dt: 0.1,
        }
    }

    fn valid(&self) -> bool {
        self.i_mem.is_finite()
            && self.i_mem > 0.0
            && self.i_ahp.is_finite()
            && self.i_ahp >= 0.0
            && self.refractory_time.is_finite()
            && self.refractory_time >= 0.0
            && self.i_threshold.is_finite()
            && self.i_threshold > 0.0
            && self.i_reset.is_finite()
            && self.i_reset > 0.0
            && self.i_reset < self.i_threshold
            && self.i_rest.is_finite()
            && self.i_rest >= 0.0
            && self.i_tau.is_finite()
            && self.i_tau > 0.0
            && self.i_g.is_finite()
            && self.i_g > 0.0
            && self.i_tau_ahp.is_finite()
            && self.i_tau_ahp > 0.0
            && self.i_ga.is_finite()
            && self.i_ga > 0.0
            && self.i_spike.is_finite()
            && self.i_spike > 0.0
            && self.i_0.is_finite()
            && self.i_0 > 0.0
            && self.kappa.is_finite()
            && self.kappa > 0.0
            && self.alpha.is_finite()
            && self.alpha > 0.0
            && self.tau.is_finite()
            && self.tau > 0.0
            && self.tau_ahp.is_finite()
            && self.tau_ahp > 0.0
            && self.refractory_period.is_finite()
            && self.refractory_period > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.refractory_period >= self.dt
    }

    fn sigmoid(value: f64) -> f64 {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exponential = value.exp();
            exponential / (1.0 + exponential)
        }
    }

    fn feedback_current(&self) -> f64 {
        let log_current = (self.i_0.ln() + self.kappa * self.i_mem.ln()) / (self.kappa + 1.0);
        log_current.exp() * Self::sigmoid(self.alpha * (self.i_mem - self.i_threshold))
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let total_input = self.i_rest + current;
        if !total_input.is_finite() || total_input < 0.0 {
            return 0;
        }

        let spike_active = self.refractory_time > 0.0;
        let spike_current = if spike_active { self.i_spike } else { 0.0 };
        let d_i_ahp = self.i_ahp / (self.tau_ahp * self.i_tau_ahp)
            * (spike_current / (1.0 + self.i_ahp / self.i_ga) - self.i_tau_ahp);
        let next_i_ahp = self.i_ahp + self.dt * d_i_ahp;

        let (next_i_mem, next_refractory, spiked) = if spike_active {
            (
                self.i_reset,
                (self.refractory_time - self.dt).max(0.0),
                false,
            )
        } else {
            let i_fb = self.feedback_current();
            let d_i_mem = self.i_mem / (self.tau * self.i_tau)
                * (total_input / (1.0 + self.i_mem / self.i_g) - self.i_tau + i_fb - self.i_ahp);
            let candidate = self.i_mem + self.dt * d_i_mem;
            if !candidate.is_finite() || candidate <= 0.0 {
                return 0;
            }
            if candidate >= self.i_threshold {
                (self.i_reset, self.refractory_period, true)
            } else {
                (candidate, 0.0, false)
            }
        };

        if !next_i_mem.is_finite()
            || !next_i_ahp.is_finite()
            || !next_refractory.is_finite()
            || next_i_mem <= 0.0
            || next_i_ahp < 0.0
            || next_refractory < 0.0
        {
            return 0;
        }

        self.i_mem = next_i_mem;
        self.i_ahp = next_i_ahp;
        self.refractory_time = next_refractory;
        i32::from(spiked)
    }

    pub fn reset(&mut self) {
        self.i_mem = self.i_reset;
        self.i_ahp = self.i_0;
        self.refractory_time = 0.0;
    }
}
impl Default for DPINeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpi_fires() {
        let mut n = DPINeuron::new();
        let t: i32 = (0..20_000).map(|_| n.step(5.0)).sum();
        assert_eq!(t, 44);
    }
    #[test]
    fn dpi_silent() {
        let mut n = DPINeuron::new();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn dpi_reset() {
        let mut n = DPINeuron::new();
        for _ in 0..500 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.i_mem, n.i_reset);
        assert_eq!(n.i_ahp, n.i_0);
        assert_eq!(n.refractory_time, 0.0);
    }
    #[test]
    fn dpi_nan_no_panic() {
        let mut n = DPINeuron::new();
        let old = (n.i_mem, n.i_ahp, n.refractory_time);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.i_mem, n.i_ahp, n.refractory_time), old);
    }
    #[test]
    fn dpi_coupled_euler_anchor() {
        let mut n = DPINeuron::new();
        assert_eq!(n.step(5.0), 0);
        assert!((n.i_mem - 0.010201975272610835).abs() < 1.0e-17);
        assert!((n.i_ahp - 0.00999).abs() < 1.0e-17);
    }
    #[test]
    fn dpi_refractory_pulse_drives_adaptation() {
        let mut n = DPINeuron {
            refractory_time: 2.0,
            ..DPINeuron::new()
        };
        assert_eq!(n.step(0.0), 0);
        assert_eq!(n.i_mem, n.i_reset);
        assert!(n.i_ahp > 0.01);
        assert_eq!(n.refractory_time, 1.9);
    }
}
