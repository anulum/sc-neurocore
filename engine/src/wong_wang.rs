// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wong and Wang 2006 reduced decision circuit

//! Explicit-Euler NMDA dynamics with the published AMPA Ornstein-Uhlenbeck
//! current noise.  The deterministic sample-taking boundary is shared by the
//! scalar engine class and the batch accelerator.

use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const A: f64 = 270.0;
const B: f64 = 108.0;
const D: f64 = 0.154;

/// Four dynamic states of one reduced Wong-Wang decision circuit.
#[derive(Clone, Debug)]
pub struct WongWangUnit {
    pub s1: f64,
    pub s2: f64,
    pub noise1: f64,
    pub noise2: f64,
    pub tau_s: f64,
    pub tau_ampa: f64,
    pub gamma: f64,
    pub j_n: f64,
    pub j_cross: f64,
    pub i_0: f64,
    pub sigma: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl WongWangUnit {
    /// Construct the published parameter set with a reproducible RNG stream.
    pub fn new(seed: u64) -> Self {
        Self {
            s1: 0.1,
            s2: 0.1,
            noise1: 0.0,
            noise2: 0.0,
            tau_s: 0.1,
            tau_ampa: 0.002,
            gamma: 0.641,
            j_n: 0.2609,
            j_cross: 0.0497,
            i_0: 0.3255,
            sigma: 0.02,
            dt: 0.0001,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Construct one configured state while retaining the supplied RNG seed.
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        s1: f64,
        s2: f64,
        noise1: f64,
        noise2: f64,
        tau_s: f64,
        tau_ampa: f64,
        gamma: f64,
        j_n: f64,
        j_cross: f64,
        i_0: f64,
        sigma: f64,
        dt: f64,
        seed: u64,
    ) -> Result<Self, String> {
        let unit = Self {
            s1,
            s2,
            noise1,
            noise2,
            tau_s,
            tau_ampa,
            gamma,
            j_n,
            j_cross,
            i_0,
            sigma,
            dt,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        };
        unit.validate()?;
        Ok(unit)
    }

    fn validate(&self) -> Result<(), String> {
        let finite = [
            self.s1,
            self.s2,
            self.noise1,
            self.noise2,
            self.tau_s,
            self.tau_ampa,
            self.gamma,
            self.j_n,
            self.j_cross,
            self.i_0,
            self.sigma,
            self.dt,
        ];
        if !finite.iter().all(|value| value.is_finite()) {
            return Err("Wong-Wang state and parameters must be finite".into());
        }
        if !(0.0..=1.0).contains(&self.s1) || !(0.0..=1.0).contains(&self.s2) {
            return Err("Wong-Wang gating state must remain in [0, 1]".into());
        }
        if self.tau_s <= 0.0 || self.tau_ampa <= 0.0 || self.gamma <= 0.0 || self.dt <= 0.0 {
            return Err("Wong-Wang time constants, gamma, and dt must be positive".into());
        }
        if self.j_n < 0.0 || self.j_cross < 0.0 || self.sigma < 0.0 {
            return Err("Wong-Wang couplings and sigma must be non-negative".into());
        }
        Ok(())
    }

    #[inline]
    fn phi(i_syn: f64) -> Result<f64, String> {
        if !i_syn.is_finite() {
            return Err("Wong-Wang synaptic current must be finite".into());
        }
        let x = A * i_syn - B;
        let scaled = -D * x;
        let response = if scaled > 700.0 {
            0.0
        } else if x.abs() < 1.0e-7 {
            1.0 / D
        } else {
            x / -scaled.exp_m1()
        };
        if !response.is_finite() {
            return Err("Wong-Wang transfer response must be finite".into());
        }
        Ok(response.max(0.0))
    }

    /// Advance one Euler/OU update from externally supplied normal samples.
    pub fn step_with_gaussian_samples(
        &mut self,
        stim1: f64,
        stim2: f64,
        xi1: f64,
        xi2: f64,
    ) -> Result<(f64, f64), String> {
        self.validate()?;
        if ![stim1, stim2, xi1, xi2]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("Wong-Wang stimuli and Gaussian samples must be finite".into());
        }
        let current1 = self.j_n * self.s1 - self.j_cross * self.s2 + self.i_0 + stim1 + self.noise1;
        let current2 = self.j_n * self.s2 - self.j_cross * self.s1 + self.i_0 + stim2 + self.noise2;
        let rate1 = Self::phi(current1)?;
        let rate2 = Self::phi(current2)?;
        let ds1 = -self.s1 / self.tau_s + (1.0 - self.s1) * self.gamma * rate1;
        let ds2 = -self.s2 / self.tau_s + (1.0 - self.s2) * self.gamma * rate2;
        let noise_scale = (self.dt / self.tau_ampa).sqrt() * self.sigma;
        let next_s1 = self.s1 + self.dt * ds1;
        let next_s2 = self.s2 + self.dt * ds2;
        let next_noise1 = self.noise1 - (self.dt / self.tau_ampa) * self.noise1 + noise_scale * xi1;
        let next_noise2 = self.noise2 - (self.dt / self.tau_ampa) * self.noise2 + noise_scale * xi2;
        if ![next_s1, next_s2, next_noise1, next_noise2]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("Wong-Wang candidate state must remain finite".into());
        }
        if !(0.0..=1.0).contains(&next_s1) || !(0.0..=1.0).contains(&next_s2) {
            return Err("Wong-Wang candidate gating state left [0, 1]".into());
        }
        self.s1 = next_s1;
        self.s2 = next_s2;
        self.noise1 = next_noise1;
        self.noise2 = next_noise2;
        Ok((rate1, rate2))
    }

    /// Advance one stochastic update using the internal seeded RNG.
    pub fn step(&mut self, stim1: f64, stim2: f64) -> Result<(f64, f64), String> {
        let xi1 = self.randn();
        let xi2 = self.randn();
        self.step_with_gaussian_samples(stim1, stim2, xi1, xi2)
    }

    fn randn(&mut self) -> f64 {
        let u1 = self.rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = self.rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Restore only dynamic state variables.
    pub fn reset(&mut self) {
        self.s1 = 0.1;
        self.s2 = 0.1;
        self.noise1 = 0.0;
        self.noise2 = 0.0;
    }
}

/// Per-step state and rate traces returned by the batch implementation.
pub struct WongWangTrace {
    pub s1: Vec<f64>,
    pub s2: Vec<f64>,
    pub noise1: Vec<f64>,
    pub noise2: Vec<f64>,
    pub rate1: Vec<f64>,
    pub rate2: Vec<f64>,
    pub final_s1: f64,
    pub final_s2: f64,
    pub final_noise1: f64,
    pub final_noise2: f64,
}

/// Simulate a complete deterministic-sample batch.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    s1: f64,
    s2: f64,
    noise1: f64,
    noise2: f64,
    tau_s: f64,
    tau_ampa: f64,
    gamma: f64,
    j_n: f64,
    j_cross: f64,
    i_0: f64,
    sigma: f64,
    dt: f64,
    stim1: &[f64],
    stim2: &[f64],
    xi: &[f64],
) -> Result<WongWangTrace, String> {
    let n_steps = stim1.len();
    if stim2.len() != n_steps {
        return Err(format!(
            "stim1 and stim2 length mismatch: {n_steps} vs {}",
            stim2.len()
        ));
    }
    if xi.len() != 2 * n_steps {
        return Err(format!(
            "xi length must be 2 * n_steps ({}): got {}",
            2 * n_steps,
            xi.len()
        ));
    }
    let mut unit = WongWangUnit::with_parameters(
        s1, s2, noise1, noise2, tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt, 0,
    )?;
    let mut trace = WongWangTrace {
        s1: Vec::with_capacity(n_steps),
        s2: Vec::with_capacity(n_steps),
        noise1: Vec::with_capacity(n_steps),
        noise2: Vec::with_capacity(n_steps),
        rate1: Vec::with_capacity(n_steps),
        rate2: Vec::with_capacity(n_steps),
        final_s1: s1,
        final_s2: s2,
        final_noise1: noise1,
        final_noise2: noise2,
    };
    for step in 0..n_steps {
        let (rate1, rate2) = unit.step_with_gaussian_samples(
            stim1[step],
            stim2[step],
            xi[2 * step],
            xi[2 * step + 1],
        )?;
        trace.s1.push(unit.s1);
        trace.s2.push(unit.s2);
        trace.noise1.push(unit.noise1);
        trace.noise2.push(unit.noise2);
        trace.rate1.push(rate1);
        trace.rate2.push(rate2);
    }
    trace.final_s1 = unit.s1;
    trace.final_s2 = unit.s2;
    trace.final_noise1 = unit.noise1;
    trace.final_noise2 = unit.noise2;
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_matches_appendix_euler_and_ou_equations() {
        let mut unit = WongWangUnit::with_parameters(
            0.1, 0.2, 0.01, -0.02, 0.1, 0.002, 0.641, 0.2609, 0.0497, 0.3255, 0.02, 0.0001, 3,
        )
        .unwrap();
        let old = (unit.s1, unit.s2, unit.noise1, unit.noise2);
        let (rate1, rate2) = unit
            .step_with_gaussian_samples(0.03, -0.01, 0.5, -1.0)
            .unwrap();
        let expected_s1 = old.0 + 0.0001 * (-old.0 / 0.1 + (1.0 - old.0) * 0.641 * rate1);
        let expected_s2 = old.1 + 0.0001 * (-old.1 / 0.1 + (1.0 - old.1) * 0.641 * rate2);
        let scale = (0.0001_f64 / 0.002).sqrt() * 0.02;
        assert_eq!(unit.s1, expected_s1);
        assert_eq!(unit.s2, expected_s2);
        assert_eq!(unit.noise1, old.2 - 0.05 * old.2 + scale * 0.5);
        assert_eq!(unit.noise2, old.3 - 0.05 * old.3 - scale);
    }

    #[test]
    fn batch_matches_scalar_and_preserves_empty_initial_state() {
        let empty = simulate(
            0.2,
            0.3,
            0.01,
            -0.02,
            0.1,
            0.002,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.0001,
            &[],
            &[],
            &[],
        )
        .unwrap();
        assert!(empty.s1.is_empty());
        assert_eq!(empty.final_s1, 0.2);
        assert_eq!(empty.final_noise2, -0.02);

        let stim1 = [0.02, 0.01, -0.01];
        let stim2 = [-0.01, 0.0, 0.03];
        let xi = [0.5, -1.0, 0.25, 0.75, -0.5, 0.0];
        let batch = simulate(
            0.2, 0.3, 0.01, -0.02, 0.12, 0.003, 0.7, 0.28, 0.06, 0.31, 0.015, 0.0002, &stim1,
            &stim2, &xi,
        )
        .unwrap();
        let mut scalar = WongWangUnit::with_parameters(
            0.2, 0.3, 0.01, -0.02, 0.12, 0.003, 0.7, 0.28, 0.06, 0.31, 0.015, 0.0002, 0,
        )
        .unwrap();
        for step in 0..stim1.len() {
            let rates = scalar
                .step_with_gaussian_samples(
                    stim1[step],
                    stim2[step],
                    xi[2 * step],
                    xi[2 * step + 1],
                )
                .unwrap();
            assert_eq!(batch.s1[step], scalar.s1);
            assert_eq!(batch.s2[step], scalar.s2);
            assert_eq!(batch.noise1[step], scalar.noise1);
            assert_eq!(batch.noise2[step], scalar.noise2);
            assert_eq!((batch.rate1[step], batch.rate2[step]), rates);
        }
        assert_eq!(batch.final_s1, scalar.s1);
        assert_eq!(batch.final_s2, scalar.s2);
        assert_eq!(batch.final_noise1, scalar.noise1);
        assert_eq!(batch.final_noise2, scalar.noise2);
    }

    #[test]
    fn invalid_input_is_rejected_before_state_commit() {
        let mut unit = WongWangUnit::new(5);
        let before = (unit.s1, unit.s2, unit.noise1, unit.noise2);
        assert!(unit
            .step_with_gaussian_samples(f64::NAN, 0.0, 0.0, 0.0)
            .is_err());
        assert_eq!((unit.s1, unit.s2, unit.noise1, unit.noise2), before);
        assert!(simulate(
            0.1,
            0.1,
            0.0,
            0.0,
            0.1,
            0.002,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.0001,
            &[0.0],
            &[0.0],
            &[],
        )
        .is_err());
    }

    #[test]
    fn reset_preserves_configured_parameters() {
        let mut unit = WongWangUnit::with_parameters(
            0.2, 0.3, 0.01, -0.02, 0.12, 0.003, 0.7, 0.3, 0.04, 0.31, 0.03, 0.0002, 7,
        )
        .unwrap();
        unit.reset();
        assert_eq!(
            (unit.s1, unit.s2, unit.noise1, unit.noise2),
            (0.1, 0.1, 0.0, 0.0)
        );
        assert_eq!((unit.tau_s, unit.tau_ampa, unit.dt), (0.12, 0.003, 0.0002));
    }
}
