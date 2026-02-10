//! # Kuramoto Solver
//!
//! High-performance Kuramoto oscillator integration for SCPN Layer-4 and SSGF loops.
//! The solver keeps all scratch buffers preallocated to avoid per-step allocations.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const TWO_PI: f64 = std::f64::consts::TAU;

/// High-performance Kuramoto oscillator solver.
///
/// Baseline equation:
/// `dθ_n/dt = ω_n + Σ_m K_nm sin(θ_m - θ_n) + noise`
///
/// SSGF extension (`step_ssgf`) adds geometry/PGBO/field terms:
/// `+ σ_g Σ_m W_nm sin(θ_m - θ_n) + pgbo_w Σ_m h_nm sin(θ_m - θ_n) + F cos(θ_n)`.
pub struct KuramotoSolver {
    /// Number of oscillators.
    pub n: usize,
    /// Natural frequencies `ω_n`, shape `(n,)`.
    pub omega: Vec<f64>,
    /// Baseline coupling matrix `K_nm`, row-major shape `(n*n)`.
    pub coupling: Vec<f64>,
    /// Current phase vector `θ_n`, shape `(n,)`.
    pub phases: Vec<f64>,
    /// Gaussian noise amplitude.
    pub noise_amp: f64,
    /// Field pressure strength `F` used by `step_ssgf`.
    pub field_pressure: f64,
    /// Scratch: per-oscillator phase derivative.
    dtheta: Vec<f64>,
    /// Scratch: `sin(θ_m - θ_n)` matrix, row-major `(n*n)`.
    sin_diff: Vec<f64>,
    /// Scratch: per-step standard normal draws.
    noise: Vec<f64>,
    /// Scratch: `cos(θ_n)` vector for field-pressure term.
    cos_theta: Vec<f64>,
    /// Scratch: geometry coupling contribution per oscillator.
    geo_coupling: Vec<f64>,
    /// Scratch: PGBO coupling contribution per oscillator.
    pgbo_coupling: Vec<f64>,
}

impl KuramotoSolver {
    /// Create a new solver with preallocated scratch buffers.
    pub fn new(
        omega: Vec<f64>,
        coupling_flat: Vec<f64>,
        initial_phases: Vec<f64>,
        noise_amp: f64,
    ) -> Self {
        let n = omega.len();
        assert!(n > 0, "omega must not be empty");
        assert_eq!(
            initial_phases.len(),
            n,
            "initial_phases length mismatch: got {}, expected {}",
            initial_phases.len(),
            n
        );
        assert_eq!(
            coupling_flat.len(),
            n * n,
            "coupling length mismatch: got {}, expected {}",
            coupling_flat.len(),
            n * n
        );

        Self {
            n,
            omega,
            coupling: coupling_flat,
            phases: initial_phases,
            noise_amp,
            field_pressure: 0.0,
            dtheta: vec![0.0; n],
            sin_diff: vec![0.0; n * n],
            noise: vec![0.0; n],
            cos_theta: vec![0.0; n],
            geo_coupling: vec![0.0; n],
            pgbo_coupling: vec![0.0; n],
        }
    }

    /// Set external field pressure `F` for SSGF mode.
    pub fn set_field_pressure(&mut self, f: f64) {
        self.field_pressure = f;
    }

    /// Advance one baseline Euler step.
    ///
    /// Returns Kuramoto order parameter `R ∈ [0, 1]`.
    pub fn step(&mut self, dt: f64, seed: u64) -> f64 {
        let n = self.n;
        let phases = &self.phases;

        self.sin_diff
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let theta_n = phases[row_idx];
                for (col_idx, value) in row.iter_mut().enumerate() {
                    *value = (phases[col_idx] - theta_n).sin();
                }
            });

        if seed == 0 || self.noise_amp == 0.0 {
            self.noise.fill(0.0);
        } else {
            fill_standard_normals(&mut self.noise, seed);
        }

        self.dtheta
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, dtheta_n)| {
                let coupling_row = &self.coupling[row_idx * n..(row_idx + 1) * n];
                let sin_row = &self.sin_diff[row_idx * n..(row_idx + 1) * n];
                let coupling_sum = coupling_row
                    .iter()
                    .zip(sin_row.iter())
                    .map(|(k_nm, sin_diff)| k_nm * sin_diff)
                    .sum::<f64>();
                *dtheta_n =
                    self.omega[row_idx] + coupling_sum + self.noise_amp * self.noise[row_idx];
            });

        for (phase, dtheta) in self.phases.iter_mut().zip(self.dtheta.iter()) {
            *phase = (*phase + dtheta * dt).rem_euclid(TWO_PI);
        }

        self.order_parameter()
    }

    /// Advance N baseline steps and return `R` after each step.
    pub fn run(&mut self, n_steps: usize, dt: f64, seed: u64) -> Vec<f64> {
        let mut order_values = Vec::with_capacity(n_steps);
        for step_idx in 0..n_steps {
            let step_seed = if seed == 0 {
                0
            } else {
                seed.wrapping_add(step_idx as u64)
            };
            order_values.push(self.step(dt, step_seed));
        }
        order_values
    }

    /// SSGF-compatible step with geometry and PGBO coupling.
    ///
    /// `w_flat`: row-major geometry matrix `W` (`n*n`). Empty slice disables geometry term.
    /// `sigma_g`: geometry coupling gain.
    /// `h_flat`: row-major PGBO tensor `h` (`n*n`). Empty slice disables PGBO term.
    /// `pgbo_weight`: PGBO coupling gain.
    ///
    /// Returns Kuramoto order parameter `R ∈ [0, 1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_ssgf(
        &mut self,
        dt: f64,
        seed: u64,
        w_flat: &[f64],
        sigma_g: f64,
        h_flat: &[f64],
        pgbo_weight: f64,
    ) -> f64 {
        let n = self.n;
        let phases = &self.phases;

        // 1) Shared sin-difference matrix.
        self.sin_diff
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let theta_n = phases[row_idx];
                for (col_idx, value) in row.iter_mut().enumerate() {
                    *value = (phases[col_idx] - theta_n).sin();
                }
            });

        // 2) Noise vector.
        if seed == 0 || self.noise_amp == 0.0 {
            self.noise.fill(0.0);
        } else {
            fill_standard_normals(&mut self.noise, seed);
        }

        // 3) Geometry term: sigma_g * Σ_m W_nm sin(diff).
        let has_geo = !w_flat.is_empty() && sigma_g != 0.0;
        if has_geo {
            assert_eq!(
                w_flat.len(),
                n * n,
                "w_flat length mismatch: got {}, expected {}",
                w_flat.len(),
                n * n
            );
            self.geo_coupling
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, geo_n)| {
                    let w_row = &w_flat[row_idx * n..(row_idx + 1) * n];
                    let sin_row = &self.sin_diff[row_idx * n..(row_idx + 1) * n];
                    *geo_n = sigma_g
                        * w_row
                            .iter()
                            .zip(sin_row.iter())
                            .map(|(w, s)| w * s)
                            .sum::<f64>();
                });
        } else {
            self.geo_coupling.fill(0.0);
        }

        // 4) PGBO term: pgbo_weight * Σ_m h_nm sin(diff).
        let has_pgbo = !h_flat.is_empty() && pgbo_weight != 0.0;
        if has_pgbo {
            assert_eq!(
                h_flat.len(),
                n * n,
                "h_flat length mismatch: got {}, expected {}",
                h_flat.len(),
                n * n
            );
            self.pgbo_coupling
                .par_iter_mut()
                .enumerate()
                .for_each(|(row_idx, pgbo_n)| {
                    let h_row = &h_flat[row_idx * n..(row_idx + 1) * n];
                    let sin_row = &self.sin_diff[row_idx * n..(row_idx + 1) * n];
                    *pgbo_n = pgbo_weight
                        * h_row
                            .iter()
                            .zip(sin_row.iter())
                            .map(|(h, s)| h * s)
                            .sum::<f64>();
                });
        } else {
            self.pgbo_coupling.fill(0.0);
        }

        // 5) Field pressure term input.
        if self.field_pressure != 0.0 {
            for (c, &theta) in self.cos_theta.iter_mut().zip(phases.iter()) {
                *c = theta.cos();
            }
        } else {
            self.cos_theta.fill(0.0);
        }

        // 6) Assemble dtheta from all enabled terms.
        self.dtheta
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dtheta_n)| {
                let coupling_row = &self.coupling[i * n..(i + 1) * n];
                let sin_row = &self.sin_diff[i * n..(i + 1) * n];
                let coupling_sum = coupling_row
                    .iter()
                    .zip(sin_row.iter())
                    .map(|(k, s)| k * s)
                    .sum::<f64>();

                *dtheta_n = self.omega[i]
                    + coupling_sum
                    + self.geo_coupling[i]
                    + self.pgbo_coupling[i]
                    + self.field_pressure * self.cos_theta[i]
                    + self.noise_amp * self.noise[i];
            });

        for (phase, dtheta) in self.phases.iter_mut().zip(self.dtheta.iter()) {
            *phase = (*phase + dtheta * dt).rem_euclid(TWO_PI);
        }

        self.order_parameter()
    }

    /// Run N SSGF-compatible steps and return `R` after each step.
    #[allow(clippy::too_many_arguments)]
    pub fn run_ssgf(
        &mut self,
        n_steps: usize,
        dt: f64,
        seed: u64,
        w_flat: &[f64],
        sigma_g: f64,
        h_flat: &[f64],
        pgbo_weight: f64,
    ) -> Vec<f64> {
        let mut order_values = Vec::with_capacity(n_steps);
        for step_idx in 0..n_steps {
            let step_seed = if seed == 0 {
                0
            } else {
                seed.wrapping_add(step_idx as u64)
            };
            order_values.push(self.step_ssgf(dt, step_seed, w_flat, sigma_g, h_flat, pgbo_weight));
        }
        order_values
    }

    /// Compute Kuramoto order parameter
    /// `R = sqrt(mean(cos θ)^2 + mean(sin θ)^2)`.
    pub fn order_parameter(&self) -> f64 {
        if self.phases.is_empty() {
            return 0.0;
        }

        let n_inv = 1.0 / self.phases.len() as f64;
        let mean_cos = self.phases.iter().map(|theta| theta.cos()).sum::<f64>() * n_inv;
        let mean_sin = self.phases.iter().map(|theta| theta.sin()).sum::<f64>() * n_inv;
        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }

    /// Borrow current phase vector.
    pub fn get_phases(&self) -> &[f64] {
        &self.phases
    }

    /// Replace phase vector.
    pub fn set_phases(&mut self, phases: Vec<f64>) {
        assert_eq!(
            phases.len(),
            self.n,
            "phases length mismatch: got {}, expected {}",
            phases.len(),
            self.n
        );
        self.phases = phases;
    }

    /// Replace baseline coupling matrix.
    pub fn set_coupling(&mut self, coupling_flat: Vec<f64>) {
        assert_eq!(
            coupling_flat.len(),
            self.n * self.n,
            "coupling length mismatch: got {}, expected {}",
            coupling_flat.len(),
            self.n * self.n
        );
        self.coupling = coupling_flat;
    }
}

fn fill_standard_normals(out: &mut [f64], seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut i = 0usize;

    while i + 1 < out.len() {
        let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TWO_PI * u2;
        out[i] = r * theta.cos();
        out[i + 1] = r * theta.sin();
        i += 2;
    }

    if i < out.len() {
        let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.gen::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        out[i] = r * (TWO_PI * u2).cos();
    }
}
