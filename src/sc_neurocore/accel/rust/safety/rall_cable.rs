// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for rall_cable

#[derive(Debug, Clone)]
pub struct RallCableNeuron {
    pub n_comp: usize,
    pub tau_m: f64,
    pub v_rest: f64,
    pub g_ratio: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
    pub v: Vec<f64>,
}

impl RallCableNeuron {
    pub fn new() -> Self {
        Self::with_compartments(5)
    }

    pub fn with_compartments(n_comp: usize) -> Self {
        let count = n_comp.max(1);
        Self {
            n_comp: count,
            tau_m: 20.0_f64,
            v_rest: -65.0_f64,
            g_ratio: 0.5_f64,
            v_threshold: -50.0_f64,
            v_reset: -65.0_f64,
            dt: 0.1_f64,
            v: vec![-65.0_f64; count],
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        let Some(mut candidate) = self.candidate(i_ext) else {
            return -1;
        };
        let previous_soma = self.v[0];
        if candidate[0] >= self.v_threshold && previous_soma < self.v_threshold {
            candidate[0] = self.v_reset;
            self.v = candidate;
            return 1;
        }
        self.v = candidate;
        0
    }

    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
    }

    fn candidate(&self, i_ext: f64) -> Option<Vec<f64>> {
        if !validate_rall_cable(self) || !i_ext.is_finite() {
            return None;
        }
        let alpha = self.dt / self.tau_m;
        let offdiag = -alpha * self.g_ratio;
        let mut diagonal = vec![1.0 + alpha + 2.0 * alpha * self.g_ratio; self.n_comp];
        if self.n_comp == 1 {
            diagonal[0] = 1.0 + alpha;
        } else {
            diagonal[0] = 1.0 + alpha + alpha * self.g_ratio;
            diagonal[self.n_comp - 1] = 1.0 + alpha + alpha * self.g_ratio;
        }
        let lower = vec![offdiag; self.n_comp.saturating_sub(1)];
        let upper = vec![offdiag; self.n_comp.saturating_sub(1)];
        let mut rhs: Vec<f64> = self.v.iter().map(|value| value - self.v_rest).collect();
        rhs[self.n_comp - 1] += alpha * i_ext;
        let mut solved = solve_tridiagonal(&lower, &diagonal, &upper, &rhs)?;
        for value in &mut solved {
            *value += self.v_rest;
        }
        Some(solved)
    }
}

pub fn validate_rall_cable(state: &RallCableNeuron) -> bool {
    state.n_comp >= 1
        && state.v.len() == state.n_comp
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.v_rest.is_finite()
        && state.g_ratio.is_finite()
        && state.g_ratio >= 0.0
        && state.v_threshold.is_finite()
        && state.v_reset.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v.iter().all(|value| value.is_finite())
}

fn solve_tridiagonal(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = diagonal.len();
    if n == 0
        || rhs.len() != n
        || lower.len() != n.saturating_sub(1)
        || upper.len() != n.saturating_sub(1)
    {
        return None;
    }
    let mut c_prime = vec![0.0; n.saturating_sub(1)];
    let mut d_prime = vec![0.0; n];
    let mut pivot = diagonal[0];
    if !pivot.is_finite() || pivot == 0.0 {
        return None;
    }
    if n > 1 {
        c_prime[0] = upper[0] / pivot;
    }
    d_prime[0] = rhs[0] / pivot;
    for i in 1..n {
        pivot = diagonal[i] - lower[i - 1] * c_prime[i - 1];
        if !pivot.is_finite() || pivot == 0.0 {
            return None;
        }
        if i < n - 1 {
            c_prime[i] = upper[i] / pivot;
        }
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot;
    }
    let mut solution = vec![0.0; n];
    solution[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rall_cable_new() {
        let state = RallCableNeuron::new();
        assert_eq!(state.v.len(), 5);
        assert!(validate_rall_cable(&state));
    }

    #[test]
    fn test_rall_cable_step_matches_implicit_solve() {
        let mut state = RallCableNeuron::with_compartments(3);
        let spike = state.step(100.0);
        assert_eq!(spike, 0);
        assert!((state.v[0] - -64.99999695179709).abs() < 1e-12);
        assert!((state.v[1] - -64.99877157422763).abs() < 1e-12);
        assert!((state.v[2] - -64.50371903616434).abs() < 1e-12);
    }

    #[test]
    fn test_rall_cable_invalid_current_preserves_state() {
        let mut state = RallCableNeuron::with_compartments(3);
        let before = state.v.clone();
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.v, before);
    }

    #[test]
    fn test_rall_cable_invalid_state_preserves_state() {
        let mut state = RallCableNeuron::with_compartments(3);
        state.v[1] = f64::NAN;
        assert_eq!(state.step(1.0), -1);
        assert!(state.v[1].is_nan());
    }
}
