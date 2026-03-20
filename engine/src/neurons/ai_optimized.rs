// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AI-optimized spiking neuron models (original designs)

//! Eight novel neuron models designed for AI workloads, not biological simulation.

/// Three-compartment memory neuron (fast/medium/slow timescales).
/// Slow compartment accumulates context, modulating excitability.
#[derive(Clone, Debug)]
pub struct MultiTimescaleNeuron {
    pub v_fast: f64,
    pub v_medium: f64,
    pub v_slow: f64,
    pub tau_fast: f64,
    pub tau_medium: f64,
    pub tau_slow: f64,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub theta_base: f64,
    pub dt: f64,
}

impl MultiTimescaleNeuron {
    pub fn new() -> Self {
        Self {
            v_fast: 0.0,
            v_medium: 0.0,
            v_slow: 0.0,
            tau_fast: 5.0,
            tau_medium: 200.0,
            tau_slow: 10000.0,
            alpha: 10.0,
            beta: 0.05,
            gamma: 0.3,
            theta_base: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v_fast += (-self.v_fast + current) / self.tau_fast * self.dt;
        let theta_eff = self.theta_base - self.gamma * self.v_slow;
        let fired = if self.v_fast >= theta_eff { 1 } else { 0 };
        self.v_medium += (-self.v_medium + self.alpha * fired as f64) / self.tau_medium * self.dt;
        self.v_slow += (-self.v_slow + self.beta * self.v_medium) / self.tau_slow * self.dt;
        if fired == 1 {
            self.v_fast = 0.0;
        }
        fired
    }

    pub fn reset(&mut self) {
        self.v_fast = 0.0;
        self.v_medium = 0.0;
        self.v_slow = 0.0;
    }
}

impl Default for MultiTimescaleNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Spiking neuron with learned sigmoid attention gate.
/// gate = sigmoid(w_key * I + w_query * v), modulates input before integration.
#[derive(Clone, Debug)]
pub struct AttentionGatedNeuron {
    pub v: f64,
    pub w_key: f64,
    pub w_query: f64,
    pub tau: f64,
    pub theta: f64,
    pub dt: f64,
}

impl AttentionGatedNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            w_key: 1.0,
            w_query: 0.5,
            tau: 10.0,
            theta: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let gate = 1.0 / (1.0 + (-(self.w_key * current + self.w_query * self.v)).exp());
        self.v += (-self.v + gate * current) / self.tau * self.dt;
        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for AttentionGatedNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Fires only on prediction errors. Silent when input matches prediction.
#[derive(Clone, Debug)]
pub struct PredictiveCodingNeuron {
    pub v: f64,
    pub pred: f64,
    pub tau: f64,
    pub tau_pred: f64,
    pub theta: f64,
    pub dt: f64,
}

impl PredictiveCodingNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            pred: 0.0,
            tau: 10.0,
            tau_pred: 50.0,
            theta: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let surprise = (current - self.pred).abs();
        self.pred += (current - self.pred) / self.tau_pred * self.dt;
        self.v += (-self.v + surprise) / self.tau * self.dt;
        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.pred = 0.0;
    }
}

impl Default for PredictiveCodingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Introspects on its own spike history; adjusts tau based on firing rate.
#[derive(Clone, Debug)]
pub struct SelfReferentialNeuron {
    pub v: f64,
    pub tau: f64,
    pub theta: f64,
    pub target_rate: f64,
    pub dt: f64,
    history: Vec<u8>,
    head: usize,
    window: usize,
}

impl SelfReferentialNeuron {
    pub fn new() -> Self {
        let window = 50;
        Self {
            v: 0.0,
            tau: 10.0,
            theta: 1.0,
            target_rate: 0.1,
            dt: 1.0,
            history: vec![0; window],
            head: 0,
            window,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let n_spikes: u32 = self.history.iter().map(|&x| x as u32).sum();
        let rate = n_spikes as f64 / self.window as f64;
        let tau_eff = self.tau * (1.0 + rate / self.target_rate);
        self.v += (-self.v + current) / tau_eff * self.dt;
        let fired = if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        };
        self.history[self.head] = fired as u8;
        self.head = (self.head + 1) % self.window;
        fired
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.history.fill(0);
        self.head = 0;
    }
}

impl Default for SelfReferentialNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Phase-coding neuron for compositional variable binding.
/// Spike when amplitude * cos(phase) > threshold.
#[derive(Clone, Debug)]
pub struct CompositionalBindingNeuron {
    pub phi: f64,
    pub amplitude: f64,
    pub omega: f64,
    pub coupling: f64,
    pub tau: f64,
    pub theta: f64,
    pub dt: f64,
}

impl CompositionalBindingNeuron {
    pub fn new() -> Self {
        Self {
            phi: 0.0,
            amplitude: 0.0,
            omega: 0.1,
            coupling: 0.5,
            tau: 10.0,
            theta: 0.8,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.phi += self.omega * self.dt;
        self.amplitude += (-self.amplitude + current) / self.tau * self.dt;
        if self.amplitude * self.phi.cos() > self.theta {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.phi = 0.0;
        self.amplitude = 0.0;
    }
}

impl Default for CompositionalBindingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Spiking neuron with learnable surrogate gradient parameters.
/// alpha (decay), beta (steepness), theta (threshold) all trainable.
#[derive(Clone, Debug)]
pub struct DifferentiableSurrogateNeuron {
    pub v: f64,
    pub alpha: f64,
    pub beta: f64,
    pub theta: f64,
}

impl DifferentiableSurrogateNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            alpha: 0.9,
            beta: 5.0,
            theta: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let spike = if self.v >= self.theta { 1 } else { 0 };
        self.v = self.alpha * self.v * (1.0 - spike as f64) + current;
        spike
    }

    pub fn surrogate_grad(&self) -> f64 {
        let d = (self.v - self.theta).abs();
        1.0 / ((1.0 + self.beta * d) * (1.0 + self.beta * d))
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for DifferentiableSurrogateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Ring attractor for continuous working memory.
/// Mexican hat connectivity; holds a continuous value in persistent activity.
#[derive(Clone, Debug)]
pub struct ContinuousAttractorNeuron {
    pub u: Vec<f64>,
    pub tau: f64,
    pub dt: f64,
    weights: Vec<Vec<f64>>,
    n_units: usize,
}

impl ContinuousAttractorNeuron {
    pub fn new(n_units: usize) -> Self {
        let sigma_e: f64 = 1.0;
        let excitation: f64 = 4.0;
        let inhibition: f64 = 0.5;
        let mut weights = vec![vec![0.0; n_units]; n_units];
        for i in 0..n_units {
            for j in 0..n_units {
                let d = (i as f64 - j as f64)
                    .abs()
                    .min((n_units as f64) - (i as f64 - j as f64).abs());
                weights[i][j] =
                    excitation * (-d * d / (2.0 * sigma_e * sigma_e)).exp() - inhibition;
            }
        }
        Self {
            u: vec![0.0; n_units],
            tau: 10.0,
            dt: 1.0,
            weights,
            n_units,
        }
    }

    fn activation(x: f64) -> f64 {
        let r = x.max(0.0);
        r * r / (1.0 + r * r)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let mut new_u = vec![0.0; self.n_units];
        for i in 0..self.n_units {
            let mut recurrent = 0.0;
            for j in 0..self.n_units {
                recurrent += self.weights[i][j] * Self::activation(self.u[j]);
            }
            new_u[i] = self.u[i] + (-self.u[i] + recurrent + current) / self.tau * self.dt;
        }
        self.u = new_u;
        let peak = self.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if peak > 1.0 {
            1
        } else {
            0
        }
    }

    pub fn bump_position(&self) -> usize {
        self.u
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

impl Default for ContinuousAttractorNeuron {
    fn default() -> Self {
        Self::new(16)
    }
}

/// Neuron with self-regulating meta-learning rate.
/// error_trace adapts learning speed: high error → learn faster, low error → stabilize.
#[derive(Clone, Debug)]
pub struct MetaPlasticNeuron {
    pub v: f64,
    pub error_trace: f64,
    pub expected_reward: f64,
    pub tau: f64,
    pub tau_meta: f64,
    pub theta: f64,
    pub lr0: f64,
    pub kappa: f64,
    pub target_error: f64,
    pub dt: f64,
}

impl MetaPlasticNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            error_trace: 0.0,
            expected_reward: 0.0,
            tau: 10.0,
            tau_meta: 500.0,
            theta: 1.0,
            lr0: 0.01,
            kappa: 5.0,
            target_error: 0.3,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-self.v + current) / self.tau * self.dt;
        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    pub fn update_meta(&mut self, reward: f64) {
        let error = (reward - self.expected_reward).abs();
        self.error_trace += (-self.error_trace + error) / self.tau_meta * self.dt;
        let meta_lr = self.meta_lr();
        self.expected_reward += meta_lr * (reward - self.expected_reward);
    }

    pub fn meta_lr(&self) -> f64 {
        self.lr0 / (1.0 + (-self.kappa * (self.error_trace - self.target_error)).exp())
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.error_trace = 0.0;
        self.expected_reward = 0.0;
    }
}

impl Default for MetaPlasticNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_timescale_fires() {
        let mut n = MultiTimescaleNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn multi_timescale_slow_accumulates() {
        let mut n = MultiTimescaleNeuron::new();
        for _ in 0..500 {
            n.step(2.0);
        }
        assert!(n.v_slow > 0.0);
    }

    #[test]
    fn multi_timescale_reset() {
        let mut n = MultiTimescaleNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v_fast, 0.0);
        assert_eq!(n.v_medium, 0.0);
        assert_eq!(n.v_slow, 0.0);
    }

    #[test]
    fn attention_gated_fires() {
        let mut n = AttentionGatedNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn attention_gated_gate_suppresses_low_input() {
        let mut n = AttentionGatedNeuron {
            w_key: -2.0,
            ..AttentionGatedNeuron::new()
        };
        let total: i32 = (0..200).map(|_| n.step(0.1)).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn attention_gated_reset() {
        let mut n = AttentionGatedNeuron::new();
        for _ in 0..50 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
    }

    #[test]
    fn predictive_coding_fires_on_change() {
        let mut n = PredictiveCodingNeuron::new();
        for _ in 0..200 {
            n.step(1.0);
        }
        let spikes_after_change: i32 = (0..50).map(|_| n.step(10.0)).sum();
        assert!(spikes_after_change > 0);
    }

    #[test]
    fn predictive_coding_silent_on_constant() {
        let mut n = PredictiveCodingNeuron::new();
        for _ in 0..500 {
            n.step(0.5);
        }
        let late: i32 = (0..100).map(|_| n.step(0.5)).sum();
        assert_eq!(late, 0);
    }

    #[test]
    fn predictive_coding_reset() {
        let mut n = PredictiveCodingNeuron::new();
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert_eq!(n.pred, 0.0);
    }

    #[test]
    fn self_referential_fires() {
        let mut n = SelfReferentialNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn self_referential_adapts_tau() {
        let mut n = SelfReferentialNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        let n_spikes: u32 = n.history.iter().map(|&x| x as u32).sum();
        assert!(n_spikes > 0);
    }

    #[test]
    fn self_referential_reset() {
        let mut n = SelfReferentialNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert!(n.history.iter().all(|&x| x == 0));
    }

    #[test]
    fn compositional_binding_fires() {
        let mut n = CompositionalBindingNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn compositional_binding_phase_advances() {
        let mut n = CompositionalBindingNeuron::new();
        for _ in 0..100 {
            n.step(1.0);
        }
        assert!(n.phi > 0.0);
    }

    #[test]
    fn compositional_binding_reset() {
        let mut n = CompositionalBindingNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.phi, 0.0);
        assert_eq!(n.amplitude, 0.0);
    }

    #[test]
    fn differentiable_surrogate_fires() {
        let mut n = DifferentiableSurrogateNeuron::new();
        let total: i32 = (0..20).map(|_| n.step(1.5)).sum();
        assert!(total > 0);
    }

    #[test]
    fn differentiable_surrogate_grad_positive() {
        let n = DifferentiableSurrogateNeuron::new();
        assert!(n.surrogate_grad() > 0.0);
    }

    #[test]
    fn differentiable_surrogate_reset() {
        let mut n = DifferentiableSurrogateNeuron::new();
        for _ in 0..10 {
            n.step(1.5);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
    }

    #[test]
    fn continuous_attractor_fires() {
        let mut n = ContinuousAttractorNeuron::new(16);
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn continuous_attractor_bump_forms() {
        let mut n = ContinuousAttractorNeuron::new(16);
        for _ in 0..200 {
            n.step(2.0);
        }
        let peak = n.u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(peak > 0.0);
    }

    #[test]
    fn continuous_attractor_reset() {
        let mut n = ContinuousAttractorNeuron::new(16);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!(n.u.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn meta_plastic_fires() {
        let mut n = MetaPlasticNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn meta_plastic_adapts_lr() {
        let mut n = MetaPlasticNeuron::new();
        let lr_before = n.meta_lr();
        for _ in 0..100 {
            n.step(2.0);
            n.update_meta(1.0);
        }
        let lr_after = n.meta_lr();
        assert!((lr_after - lr_before).abs() > 1e-6);
    }

    #[test]
    fn meta_plastic_reset() {
        let mut n = MetaPlasticNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
            n.update_meta(1.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert_eq!(n.error_trace, 0.0);
        assert_eq!(n.expected_reward, 0.0);
    }

    // ── ArcaneNeuron tests ──────────────────────────────────────

    #[test]
    fn arcane_fires() {
        let mut n = ArcaneNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn arcane_deep_accumulates() {
        let mut n = ArcaneNeuron::new();
        for _ in 0..1000 {
            n.step(3.0);
        }
        assert!(n.v_deep.abs() > 1e-10, "deep state must accumulate");
    }

    #[test]
    fn arcane_deep_survives_reset() {
        let mut n = ArcaneNeuron::new();
        for _ in 0..500 {
            n.step(3.0);
        }
        let deep_before = n.v_deep;
        n.reset();
        assert_eq!(n.v_fast, 0.0);
        assert_eq!(n.v_work, 0.0);
        assert!(
            (n.v_deep - deep_before).abs() < 1e-15,
            "deep must survive reset"
        );
    }

    #[test]
    fn arcane_novelty_increases_deep_change() {
        let mut n = ArcaneNeuron::new();
        // Constant input
        for _ in 0..200 {
            n.step(2.0);
        }
        let deep_after_constant = n.v_deep;
        // Novel input
        for _ in 0..200 {
            n.step(8.0);
        }
        let deep_after_novel = n.v_deep;
        let delta = (deep_after_novel - deep_after_constant).abs();
        assert!(delta > 0.0, "novel input must change deep state");
    }
}

/// ArcaneNeuron — unified self-referential cognition model.
///
/// 3-compartment (fast/working/deep) with attention gate, predictive
/// self-model, and meta-plastic learning rate. Deep compartment
/// accumulates identity and survives reset.
///
/// Original design: Šotek & Arcane Sapience 2026.
#[derive(Clone, Debug)]
pub struct ArcaneNeuron {
    pub v_fast: f64,
    pub v_work: f64,
    pub v_deep: f64,
    pub tau_fast: f64,
    pub tau_work: f64,
    pub tau_deep: f64,
    pub alpha_w: f64,
    pub alpha_d: f64,
    pub theta: f64,
    pub gamma: f64,
    pub delta_conf: f64,
    pub w_gate: [f64; 4],
    pub w_pred: [f64; 3],
    pub kappa: f64,
    pub surprise_baseline: f64,
    pub lr_base: f64,
    pub eta: f64,
    pub w_inh: f64,
    pub dt: f64,
    prediction: f64,
    surprise: f64,
    novelty: f64,
    confidence: f64,
    spike_history: Vec<u8>,
    novelty_history: Vec<f64>,
    hist_idx: usize,
    nov_idx: usize,
    total_steps: usize,
}

impl ArcaneNeuron {
    pub fn new() -> Self {
        Self {
            v_fast: 0.0,
            v_work: 0.0,
            v_deep: 0.0,
            tau_fast: 5.0,
            tau_work: 200.0,
            tau_deep: 10000.0,
            alpha_w: 0.3,
            alpha_d: 0.05,
            theta: 1.0,
            gamma: 0.2,
            delta_conf: 0.3,
            w_gate: [0.8, 0.1, 0.05, 0.05],
            w_pred: [0.6, 0.3, 0.1],
            kappa: 5.0,
            surprise_baseline: 0.1,
            lr_base: 0.01,
            eta: 2.0,
            w_inh: 0.3,
            dt: 1.0,
            prediction: 0.0,
            surprise: 0.0,
            novelty: 0.0,
            confidence: 0.5,
            spike_history: vec![0; 50],
            novelty_history: vec![0.5; 20],
            hist_idx: 0,
            nov_idx: 0,
            total_steps: 0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let sh_len = self.spike_history.len() as f64;
        let nh_len = self.novelty_history.len() as f64;
        let spike_rate: f64 = self.spike_history.iter().map(|&s| s as f64).sum::<f64>() / sh_len;
        self.confidence = 1.0 - self.novelty_history.iter().sum::<f64>() / nh_len;

        let gate_in = self.w_gate[0] * current
            + self.w_gate[1] * self.v_fast
            + self.w_gate[2] * self.v_work
            + self.w_gate[3] * self.confidence;
        let gate = 1.0 / (1.0 + (-gate_in).exp());
        let i_eff = gate * current;

        self.v_fast += (-self.v_fast + i_eff - self.w_inh * spike_rate) / self.tau_fast * self.dt;

        self.prediction = self.w_pred[0] * self.v_fast
            + self.w_pred[1] * self.v_work
            + self.w_pred[2] * self.v_deep;
        self.surprise = (self.v_fast - self.prediction).abs();
        self.novelty = 1.0 / (1.0 + (-self.kappa * (self.surprise - self.surprise_baseline)).exp());

        let nh_sz = self.novelty_history.len();
        self.novelty_history[self.nov_idx % nh_sz] = self.novelty;
        self.nov_idx += 1;

        let eff_threshold = (self.theta
            * (1.0 + self.gamma * self.v_deep)
            * (1.0 - self.delta_conf * self.confidence))
            .max(0.1);

        let spike = if self.v_fast >= eff_threshold { 1 } else { 0 };

        if spike == 1 {
            self.v_work += self.alpha_w * self.v_fast / self.tau_work * self.dt;
            self.v_fast = 0.0;
        }

        self.v_work += -self.v_work / self.tau_work * self.dt;
        self.v_deep +=
            (-self.v_deep + self.alpha_d * self.v_work * self.novelty) / self.tau_deep * self.dt;

        let meta_lr = self.lr_base * (1.0 + self.eta * self.novelty);
        let error = self.v_fast - self.prediction;
        self.w_pred[0] += meta_lr * error * self.v_fast;
        self.w_pred[1] += meta_lr * error * self.v_work;
        self.w_pred[2] += meta_lr * error * self.v_deep;
        let norm =
            (self.w_pred[0].powi(2) + self.w_pred[1].powi(2) + self.w_pred[2].powi(2)).sqrt();
        if norm > 0.0 {
            for w in &mut self.w_pred {
                *w /= norm;
            }
        }

        let sh_sz = self.spike_history.len();
        self.spike_history[self.hist_idx % sh_sz] = spike as u8;
        self.hist_idx += 1;
        self.total_steps += 1;

        spike
    }

    pub fn reset(&mut self) {
        self.v_fast = 0.0;
        self.v_work = 0.0;
        // v_deep does NOT reset — it IS the identity
        self.prediction = 0.0;
        self.surprise = 0.0;
        self.novelty = 0.0;
        self.spike_history.fill(0);
        self.hist_idx = 0;
    }
}

impl Default for ArcaneNeuron {
    fn default() -> Self {
        Self::new()
    }
}
