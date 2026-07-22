// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-compartment MCN neuron model

//! Multi-compartment MCN neuron model.

/// Multi-compartment neuron (MCN) matching the Spiking-WM architecture.
///
/// Dual-dendrite model with basal and apical compartments. The apical dendrite
/// gates how strongly basal information influences the soma, enabling
/// nonlinear integration for long-term temporal memory in RL tasks. The engine
/// uses candidate-first RK4 over `(u, v_basal, v_apical)` so all compartments
/// are advanced from one consistent state before the reset is committed.
///
/// Exact equations from arXiv:2503.00713 (Spiking-WM, PNAS 2025):
///
///   τ_b dV_b/dt = -V_b + x_b                                  (basal)
///   τ_a dV_a/dt = -V_a + x_a                                  (apical)
///   τ   dU/dt   = -U + σ(V_a)·[g_B/g_L·(V_b - U) + W_s·I]   (soma)
///   S[t] = Θ(U[t] - V_th)                                     (spike)
///   U[t] ← U[t]·(1 - S[t])                                    (soft reset)
///
/// Default parameters from Table II: τ = τ_a = τ_b = 2.0, g_B/g_L = 1.0,
/// β = 1.0 (sigmoid steepness), V_th = 1.0.
///
/// Reference: Brain-Cog-Lab, arXiv:2503.00713, PNAS 2025.
#[derive(Clone, Debug)]
pub struct MulticompartmentMCNNeuron {
    /// Somatic membrane potential.
    pub u: f64,
    /// Basal dendrite potential.
    pub v_basal: f64,
    /// Apical dendrite potential.
    pub v_apical: f64,
    /// Soma time constant.
    pub tau: f64,
    /// Basal dendrite time constant.
    pub tau_b: f64,
    /// Apical dendrite time constant.
    pub tau_a: f64,
    /// Basal-to-soma conductance ratio (g_B/g_L).
    pub g_ratio: f64,
    /// Sigmoid steepness for apical gating.
    pub beta: f64,
    /// Spike threshold.
    pub v_th: f64,
    /// Time step.
    pub dt: f64,
}

impl MulticompartmentMCNNeuron {
    pub fn new() -> Self {
        Self {
            u: 0.0,
            v_basal: 0.0,
            v_apical: 0.0,
            tau: 2.0,
            tau_b: 2.0,
            tau_a: 2.0,
            g_ratio: 1.0,
            beta: 1.0,
            v_th: 1.0,
            dt: 1.0,
        }
    }

    /// Sigmoid gating function σ(x) = 1/(1 + exp(-βx)).
    fn sigma(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-self.beta * x).exp())
    }

    fn valid(&self) -> bool {
        self.tau.is_finite()
            && self.tau > 0.0
            && self.tau_b.is_finite()
            && self.tau_b > 0.0
            && self.tau_a.is_finite()
            && self.tau_a > 0.0
            && self.g_ratio.is_finite()
            && self.g_ratio >= 0.0
            && self.beta.is_finite()
            && self.beta > 0.0
            && self.v_th.is_finite()
            && self.v_th > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.u.is_finite()
            && self.v_basal.is_finite()
            && self.v_apical.is_finite()
    }

    fn derivatives(
        &self,
        u: f64,
        v_basal: f64,
        v_apical: f64,
        x_basal: f64,
        x_apical: f64,
        i_soma: f64,
    ) -> [f64; 3] {
        let gate = self.sigma(v_apical);
        let du = (-u + gate * (self.g_ratio * (v_basal - u) + i_soma)) / self.tau;
        let dv_basal = (-v_basal + x_basal) / self.tau_b;
        let dv_apical = (-v_apical + x_apical) / self.tau_a;
        [du, dv_basal, dv_apical]
    }

    fn rk4_substep(&self, state: [f64; 3], x_basal: f64, x_apical: f64, i_soma: f64) -> [f64; 3] {
        let dt = self.dt;
        let k1 = self.derivatives(state[0], state[1], state[2], x_basal, x_apical, i_soma);
        let k2 = self.derivatives(
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
            x_basal,
            x_apical,
            i_soma,
        );
        let k3 = self.derivatives(
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            x_basal,
            x_apical,
            i_soma,
        );
        let k4 = self.derivatives(
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            x_basal,
            x_apical,
            i_soma,
        );
        [
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        ]
    }

    fn threshold_reached(&self, candidate_u: f64) -> bool {
        let margin = 16.0 * f64::EPSILON * self.v_th.abs().max(1.0);
        candidate_u >= self.v_th || (candidate_u - self.v_th).abs() <= margin
    }

    /// Step with basal input (x_b), apical input (x_a), and direct somatic input.
    pub fn step_compartments(&mut self, x_basal: f64, x_apical: f64, i_soma: f64) -> i32 {
        if !x_basal.is_finite() || !x_apical.is_finite() || !i_soma.is_finite() || !self.valid() {
            return 0;
        }
        let next = self.rk4_substep(
            [self.u, self.v_basal, self.v_apical],
            x_basal,
            x_apical,
            i_soma,
        );
        if !next.iter().all(|value| value.is_finite()) {
            return 0;
        }
        let spike = self.threshold_reached(next[0]);
        self.u = if spike { 0.0 } else { next[0] };
        self.v_basal = next[1];
        self.v_apical = next[2];
        i32::from(spike)
    }

    /// Simple step: input goes to basal dendrite only.
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_compartments(current, 0.0, 0.0)
    }

    pub fn reset(&mut self) {
        self.u = 0.0;
        self.v_basal = 0.0;
        self.v_apical = 0.0;
    }
}

impl Default for MulticompartmentMCNNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcn_apical_gating() {
        // Without apical input, gate = σ(0) = 0.5, moderate drive.
        let mut n_no_apical = MulticompartmentMCNNeuron::new();
        let mut spikes_no = 0;
        for _ in 0..1000 {
            spikes_no += n_no_apical.step_compartments(2.5, 0.0, 0.0);
        }
        // With strong apical input, gate ≈ 1.0, full basal→soma coupling.
        let mut n_apical = MulticompartmentMCNNeuron::new();
        let mut spikes_yes = 0;
        for _ in 0..1000 {
            spikes_yes += n_apical.step_compartments(2.5, 5.0, 0.0);
        }
        assert!(
            spikes_yes >= spikes_no && spikes_yes > 0,
            "Apical gating should boost firing: apical={spikes_yes} >= none={spikes_no}"
        );
    }

    #[test]
    fn mcn_rk4_cross_backend_anchor() {
        let mut n = MulticompartmentMCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..200_000 {
            spikes += n.step(3.2);
        }
        assert_eq!(spikes, 49_999);
    }

    #[test]
    fn mcn_threshold_boundary_accepts_one_ulp_roundoff() {
        let n = MulticompartmentMCNNeuron::new();
        let one_ulp_below = f64::from_bits(n.v_th.to_bits() - 1);
        assert!(n.threshold_reached(one_ulp_below));
        assert!(!n.threshold_reached(n.v_th - 1.0e-9));
    }

    #[test]
    fn mcn_invalid_input_preserves_state() {
        let mut n = MulticompartmentMCNNeuron::new();
        for _ in 0..5 {
            let _ = n.step(3.2);
        }
        let old = (n.u, n.v_basal, n.v_apical);
        assert_eq!(n.step(f64::INFINITY), 0);
        assert_eq!((n.u, n.v_basal, n.v_apical), old);
    }

    #[test]
    fn mcn_basal_dendrite_memory() {
        // τ_b = 2.0, dt = 1.0: V_b decays by factor (1 - dt/τ) = 0.5 per step.
        let mut n = MulticompartmentMCNNeuron::new();
        n.step_compartments(5.0, 0.0, 0.0);
        let v_after = n.v_basal;
        n.step_compartments(0.0, 0.0, 0.0);
        let v_decay = n.v_basal;
        assert!(
            v_decay.abs() > 0.1 * v_after.abs(),
            "Basal dendrite retains memory: {v_decay:.3} vs {v_after:.3}"
        );
    }

    #[test]
    fn mcn_reset_clears_all() {
        let mut n = MulticompartmentMCNNeuron::new();
        for _ in 0..50 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.u, 0.0);
        assert_eq!(n.v_basal, 0.0);
        assert_eq!(n.v_apical, 0.0);
    }
}
