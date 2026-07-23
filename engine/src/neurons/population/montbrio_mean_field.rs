// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Montbrió-Pazó-Roxin Mean-Field Model

/// Montbrio-Pazo-Roxin 2015 — exact mean-field of QIF neuron population.
///
/// Reduces an infinite population of quadratic integrate-and-fire (QIF)
/// neurons to 2 ODEs for the population firing rate r and mean membrane
/// potential v. This is mathematically exact (not an approximation) for
/// Lorentzian-distributed heterogeneity.
///
/// dr/dt = (delta / (pi * tau^2)) + (2 * r * v / tau)
/// dv/dt = (v^2 + eta + I - (pi * tau * r)^2) / tau
///
/// where delta = heterogeneity width, eta = mean excitability,
/// tau = membrane time constant, I = external input.
///
/// Spike emitted when r exceeds a threshold (proxy for population burst).
///
/// Montbrio, Pazo & Roxin, Phys Rev X 5:021028, 2015.
#[derive(Clone, Debug)]
pub struct MontbrioMeanField {
    pub r: f64,     // Population firing rate (Hz)
    pub v: f64,     // Mean membrane potential
    pub delta: f64, // Heterogeneity width (Lorentzian)
    pub eta: f64,   // Mean excitability
    pub tau: f64,   // Membrane time constant (ms)
    pub j: f64,     // Synaptic coupling strength
    pub dt: f64,
    pub r_threshold: f64,
    pub gain: f64,
}

impl Default for MontbrioMeanField {
    fn default() -> Self {
        Self::new()
    }
}

impl MontbrioMeanField {
    pub fn new() -> Self {
        Self {
            r: 0.01,
            v: -2.0,
            delta: 1.0,
            eta: -5.0, // Below threshold for spontaneous activity
            tau: 1.0,
            j: 15.0,  // Excitatory coupling
            dt: 0.01, // Small dt for stability
            r_threshold: 0.5,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let r_prev = self.r;

        let pi = std::f64::consts::PI;
        let tau = self.tau;

        // MPR equations with synaptic coupling (j * r adds recurrence)
        let dr = (self.delta / (pi * tau * tau)) + (2.0 * self.r * self.v / tau);
        let dv = (self.v * self.v + self.eta + input + self.j * tau * self.r
            - (pi * tau * self.r).powi(2))
            / tau;

        self.r += self.dt * dr;
        self.v += self.dt * dv;

        // Safety bounds
        self.r = self.r.clamp(0.0, 100.0);
        self.v = self.v.clamp(-50.0, 50.0);
        if !self.r.is_finite() {
            self.r = 0.01;
        }
        if !self.v.is_finite() {
            self.v = -2.0;
        }

        // "Spike" = population burst: r crosses threshold
        if self.r >= self.r_threshold && r_prev < self.r_threshold {
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
    fn mpr_fires_with_input() {
        let mut n = MontbrioMeanField::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(10.0);
        }
        assert!(
            spikes > 0,
            "MPR must produce bursts with strong input, got {spikes}"
        );
    }

    #[test]
    fn mpr_silent_without_input() {
        // eta = -5 (below threshold), no input → quiescent
        let mut n = MontbrioMeanField::new();
        let mut spikes = 0;
        for _ in 0..50_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "MPR must be quiescent without input (eta<0), got {spikes}"
        );
    }

    #[test]
    fn mpr_rate_increases_with_input() {
        let mut low = MontbrioMeanField::new();
        let mut high = MontbrioMeanField::new();
        for _ in 0..10_000 {
            low.step(3.0);
            high.step(15.0);
        }
        assert!(
            high.r > low.r,
            "Higher input → higher rate: high={:.3} vs low={:.3}",
            high.r,
            low.r
        );
    }

    #[test]
    fn mpr_two_ode_dynamics() {
        // Verify both r and v evolve from initial conditions
        let mut n = MontbrioMeanField::new();
        let r0 = n.r;
        let v0 = n.v;
        for _ in 0..1000 {
            n.step(5.0);
        }
        assert!(
            n.r != r0 || n.v != v0,
            "State must evolve from initial conditions"
        );
    }

    #[test]
    fn mpr_rate_non_negative() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..50_000 {
            n.step(-10.0);
        }
        assert!(n.r >= 0.0, "Rate must be non-negative, r={}", n.r);
    }

    #[test]
    fn mpr_negative_input_no_crash() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..50_000 {
            n.step(-100.0);
        }
        assert!(n.r.is_finite());
        assert!(n.v.is_finite());
    }

    #[test]
    fn mpr_nan_input_stays_finite() {
        let mut n = MontbrioMeanField::new();
        n.step(f64::NAN);
        assert!(n.r.is_finite());
        assert!(n.v.is_finite());
    }

    #[test]
    fn mpr_extreme_input_bounded() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..10_000 {
            n.step(1e6);
        }
        assert!(n.r.is_finite() && n.r <= 100.0);
        assert!(n.v.is_finite() && n.v <= 50.0);
    }

    #[test]
    fn mpr_reset_clears_state() {
        let mut n = MontbrioMeanField::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.r, 0.01);
        assert_eq!(n.v, -2.0);
    }

    #[test]
    fn mpr_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = MontbrioMeanField::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }
}
