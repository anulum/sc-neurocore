// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hybrid linear-attention neuron model

/// Hybrid linear attention neuron for spiking environments.
///
/// Combines local windowed attention with linear (kernel-based) global attention,
/// achieving near-linear training complexity O(L) instead of O(L²).
/// Inspired by SpikingBrain's hybrid attention architecture.
///
/// The neuron accumulates spike-weighted keys and values via a recurrent
/// state S, avoiding the quadratic attention matrix:
///
///   S(t+1) = λ S(t) + φ(k_t) ⊗ v_t
///   output = φ(q_t)ᵀ S(t)
///
/// where φ is an elu+1 feature map.
#[derive(Clone, Debug)]
pub struct HybridLinearAttentionNeuron {
    pub v: f64,
    state_kv: Vec<f64>,
    pub dim: usize,
    pub lambda: f64,
    pub window_size: usize,
    window_buf: Vec<f64>,
    window_idx: usize,
    pub dt: f64,
}

impl HybridLinearAttentionNeuron {
    pub fn new(dim: usize) -> Self {
        Self {
            v: 0.0,
            state_kv: vec![0.0; dim],
            dim,
            lambda: 0.95,
            window_size: 16,
            window_buf: vec![0.0; 16],
            window_idx: 0,
            dt: 1.0,
        }
    }

    /// Step with query, key, value (each scalar projections).
    pub fn step_qkv(&mut self, query: f64, key: f64, value: f64) -> f64 {
        // Feature map: elu(x) + 1.
        let phi_q = if query > 0.0 {
            query + 1.0
        } else {
            query.exp()
        };
        let phi_k = if key > 0.0 { key + 1.0 } else { key.exp() };

        // Update recurrent KV state (linear attention).
        for s in &mut self.state_kv {
            *s *= self.lambda;
        }
        let idx = (phi_k.abs() * self.dim as f64) as usize % self.dim;
        self.state_kv[idx] += phi_k * value;

        // Global attention output.
        let global = phi_q * self.state_kv[idx];

        // Local windowed attention (sliding window buffer).
        self.window_buf[self.window_idx % self.window_size] = value;
        self.window_idx += 1;
        let local: f64 = self.window_buf.iter().sum::<f64>() / self.window_size as f64;

        // Combine global + local.
        self.v = 0.5 * global + 0.5 * local;
        self.v
    }

    /// Simple step (input treated as combined qkv).
    pub fn step(&mut self, current: f64) -> i32 {
        let out = self.step_qkv(current, current, current);
        if out > 1.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.state_kv.fill(0.0);
        self.window_buf.fill(0.0);
        self.window_idx = 0;
    }
}

impl Default for HybridLinearAttentionNeuron {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_linear_attention_step() {
        let mut n = HybridLinearAttentionNeuron::new(8);
        let mut nonzero = false;
        for i in 0..100 {
            let out = n.step_qkv(i as f64 * 0.1, 0.5, 1.0);
            if out.abs() > 1e-10 {
                nonzero = true;
            }
        }
        assert!(nonzero, "Should produce non-zero output");
    }

    #[test]
    fn hybrid_linear_attention_deterministic() {
        let mut n1 = HybridLinearAttentionNeuron::new(8);
        let mut n2 = HybridLinearAttentionNeuron::new(8);
        for i in 0..50 {
            let a = n1.step_qkv(i as f64 * 0.1, 0.3, 0.7);
            let b = n2.step_qkv(i as f64 * 0.1, 0.3, 0.7);
            assert_eq!(a, b, "Must be deterministic");
        }
    }

    #[test]
    fn hybrid_linear_attention_reset() {
        let mut n = HybridLinearAttentionNeuron::new(8);
        for _ in 0..50 {
            n.step_qkv(1.0, 1.0, 1.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert!(n.state_kv.iter().all(|&x| x == 0.0));
    }
}
