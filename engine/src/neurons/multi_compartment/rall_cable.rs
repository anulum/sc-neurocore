// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rall cable neuron model

//! Rall cable neuron model.

/// Rall cable — N-compartment passive dendrite model. Rall 1964.
#[derive(Clone, Debug)]
pub struct RallCableNeuron {
    pub v: Vec<f64>,
    pub n_comp: usize,
    pub tau_m: f64,
    pub v_rest: f64,
    pub g_ratio: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl RallCableNeuron {
    pub fn new(n_comp: usize) -> Self {
        let count = n_comp.max(1);
        Self {
            v: vec![-65.0; count],
            n_comp: count,
            tau_m: 20.0,
            v_rest: -65.0,
            g_ratio: 0.5,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let Some(mut candidate) = self.candidate(current) else {
            return -1;
        };
        let previous_soma = self.v[0];
        if candidate[0] >= self.v_threshold && previous_soma < self.v_threshold {
            candidate[0] = self.v_reset;
            self.v = candidate;
            1
        } else {
            self.v = candidate;
            0
        }
    }
    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
    }

    fn valid(&self) -> bool {
        self.n_comp >= 1
            && self.v.len() == self.n_comp
            && self.tau_m.is_finite()
            && self.tau_m > 0.0
            && self.v_rest.is_finite()
            && self.g_ratio.is_finite()
            && self.g_ratio >= 0.0
            && self.v_threshold.is_finite()
            && self.v_reset.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v.iter().all(|value| value.is_finite())
    }

    fn candidate(&self, current: f64) -> Option<Vec<f64>> {
        if !self.valid() || !current.is_finite() {
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
        rhs[self.n_comp - 1] += alpha * current;
        let mut solved = solve_rall_tridiagonal(&lower, &diagonal, &upper, &rhs)?;
        for value in &mut solved {
            *value += self.v_rest;
        }
        Some(solved)
    }
}

fn solve_rall_tridiagonal(
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
    fn rall_fires() {
        let mut n = RallCableNeuron::new(2);
        n.g_ratio = 5.0;
        let t: i32 = (0..5000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn rall_reset() {
        let mut n = RallCableNeuron::new(5);
        for _ in 0..100 {
            n.step(50.0);
        }
        n.reset();
        assert!(n.v.iter().all(|&x| (x - n.v_rest).abs() < 1e-10));
    }

    #[test]
    fn rall_bounded() {
        let mut n = RallCableNeuron::new(5);
        for _ in 0..1000 {
            n.step(500.0);
        }
        assert!(n.v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn rall_implicit_step_reference() {
        let mut n = RallCableNeuron::new(3);
        assert_eq!(n.step(100.0), 0);
        assert!((n.v[0] - -64.99999695179709).abs() < 1e-12);
        assert!((n.v[1] - -64.99877157422763).abs() < 1e-12);
        assert!((n.v[2] - -64.50371903616434).abs() < 1e-12);
    }

    #[test]
    fn rall_nan_no_panic() {
        let mut n = RallCableNeuron::new(5);
        let before = n.v.clone();
        assert_eq!(n.step(f64::NAN), -1);
        assert_eq!(n.v, before);
    }
}
