// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rall branching dendrite model in Rust

//! Compartmental dendritic tree with Rall's 3/2 power rule.
//! Distal → proximal propagation with inter-compartment coupling.

/// Rall branching dendrite with configurable topology.
pub struct RallDendriteRust {
    pub n_branches: usize,
    pub branch_length: usize,
    tau: f64,
    coupling: f64,
    dt: f64,
    decay: f64,
    dt_over_tau: f64,
    attenuation: Vec<f64>,
    /// Compartment voltages: [n_branches][branch_length]
    v: Vec<Vec<f64>>,
    pub soma_v: f64,
}

impl RallDendriteRust {
    pub fn new(n_branches: usize, branch_length: usize, tau: f64, coupling: f64, dt: f64) -> Self {
        let decay = (-dt / tau).exp();
        let dt_over_tau = dt / tau;
        // Rall 3/2: d_parent^1.5 = n * d_daughter^1.5, d_daughter = 1
        let parent_d = (n_branches as f64).powf(2.0 / 3.0);
        let attenuation: Vec<f64> = (0..n_branches)
            .map(|_| (1.0 / parent_d).powf(1.5))
            .collect();

        Self {
            n_branches,
            branch_length,
            tau,
            coupling,
            dt,
            decay,
            dt_over_tau,
            attenuation,
            v: vec![vec![0.0; branch_length]; n_branches],
            soma_v: 0.0,
        }
    }

    /// Advance one timestep. branch_inputs: [n_branches] injected at distal tip.
    pub fn step(&mut self, branch_inputs: &[f64]) -> f64 {
        let nb = self.n_branches;
        let bl = self.branch_length;

        // Decay all compartments
        for b in 0..nb {
            for k in 0..bl {
                self.v[b][k] *= self.decay;
            }
        }

        // Inject at distal tip (last compartment)
        for b in 0..nb.min(branch_inputs.len()) {
            self.v[b][bl - 1] += branch_inputs[b] * self.dt_over_tau;
        }

        // Propagate distal → proximal
        for k in (1..bl).rev() {
            for b in 0..nb {
                let flow = self.coupling * (self.v[b][k] - self.v[b][k - 1]);
                self.v[b][k] -= flow;
                self.v[b][k - 1] += flow;
            }
        }

        // Sum proximal with Rall attenuation
        let mut soma_input = 0.0;
        for b in 0..nb {
            soma_input += self.v[b][0] * self.attenuation[b];
        }
        self.soma_v = self.decay * self.soma_v + soma_input * self.dt_over_tau;
        self.soma_v
    }

    pub fn reset(&mut self) {
        for b in &mut self.v {
            b.fill(0.0);
        }
        self.soma_v = 0.0;
    }

    pub fn branch_voltages(&self) -> Vec<Vec<f64>> {
        self.v.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_zero() {
        let d = RallDendriteRust::new(4, 3, 10.0, 0.5, 1.0);
        assert_eq!(d.soma_v, 0.0);
        assert!(d.v.iter().all(|b| b.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_input_reaches_soma() {
        let mut d = RallDendriteRust::new(2, 3, 10.0, 0.5, 1.0);
        for _ in 0..20 {
            d.step(&[1.0, 0.0]);
        }
        assert!(d.soma_v > 0.0);
    }

    #[test]
    fn test_more_branches_more_input() {
        let mut d1 = RallDendriteRust::new(4, 2, 10.0, 0.5, 1.0);
        let mut d2 = RallDendriteRust::new(4, 2, 10.0, 0.5, 1.0);
        for _ in 0..20 {
            d1.step(&[1.0, 0.0, 0.0, 0.0]);
            d2.step(&[1.0, 1.0, 1.0, 1.0]);
        }
        assert!(d2.soma_v > d1.soma_v);
    }

    #[test]
    fn test_reset() {
        let mut d = RallDendriteRust::new(2, 2, 10.0, 0.5, 1.0);
        d.step(&[5.0, 5.0]);
        d.reset();
        assert_eq!(d.soma_v, 0.0);
        assert!(d.v.iter().all(|b| b.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_distal_higher_than_proximal() {
        let mut d = RallDendriteRust::new(1, 5, 20.0, 0.3, 1.0);
        for _ in 0..10 {
            d.step(&[2.0]);
        }
        let bv = &d.v[0];
        assert!(bv[4] > bv[0], "Distal should be higher than proximal");
    }
}
