use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const TWO_PI: f64 = std::f64::consts::TAU;

/// High-performance Kuramoto oscillator solver.
///
/// Implements: dθ_n/dt = ω_n + Σ_m K_nm sin(θ_m - θ_n) + noise
pub struct KuramotoSolver {
    pub n: usize,
    /// Natural frequencies ω_n: (n,)
    pub omega: Vec<f64>,
    /// Coupling matrix K_nm: flat row-major (n * n)
    pub coupling: Vec<f64>,
    /// Current phases θ_n: (n,)
    pub phases: Vec<f64>,
    /// Noise amplitude
    pub noise_amp: f64,
    /// Scratch arrays (pre-allocated for performance)
    dtheta: Vec<f64>,
    sin_diff: Vec<f64>,
    noise: Vec<f64>,
}

impl KuramotoSolver {
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
            dtheta: vec![0.0; n],
            sin_diff: vec![0.0; n * n],
            noise: vec![0.0; n],
        }
    }

    /// Advance one Euler step of size dt.
    /// Updates self.phases in-place.
    /// Returns the Kuramoto order parameter R ∈ [0, 1].
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

    /// Advance N steps, returning R after each step.
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

    /// Compute the Kuramoto order parameter:
    /// R = |1/N Σ_n exp(i θ_n)|
    pub fn order_parameter(&self) -> f64 {
        if self.phases.is_empty() {
            return 0.0;
        }

        let n_inv = 1.0 / self.phases.len() as f64;
        let mean_cos = self.phases.iter().map(|theta| theta.cos()).sum::<f64>() * n_inv;
        let mean_sin = self.phases.iter().map(|theta| theta.sin()).sum::<f64>() * n_inv;
        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }

    /// Get current phases.
    pub fn get_phases(&self) -> &[f64] {
        &self.phases
    }

    /// Set phases (for synchronization with Python layers).
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

    /// Set coupling matrix (for dynamic coupling updates).
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
