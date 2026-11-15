// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Arcane neuron model

#![deny(missing_docs)]

/// ArcaneNeuron — unified self-referential cognition model.
///
/// 3-compartment (fast/working/deep) with attention gate, predictive
/// self-model, and meta-plastic learning rate. Deep compartment
/// accumulates identity and survives reset.
///
/// Original design: Šotek & Arcane Sapience 2026.
#[derive(Clone, Debug)]
pub struct ArcaneNeuron {
    /// Fast (perceptual) compartment; a spike sets it to 0.
    pub v_fast: f64,
    /// Working-memory compartment; each spike adds `alpha_w * v_fast * dt / tau_work`, and
    /// it decays with `tau_work`.
    pub v_work: f64,
    /// Deep (identity) compartment, driven by `alpha_d * v_work * novelty` with `tau_deep`;
    /// `reset` keeps it.
    pub v_deep: f64,
    /// Fast (perceptual) compartment time constant (ms).
    pub tau_fast: f64,
    /// Working-memory compartment time constant (ms).
    pub tau_work: f64,
    /// Deep (identity) compartment time constant (ms).
    pub tau_deep: f64,
    /// Fast-to-working-memory coupling weight (dimensionless).
    pub alpha_w: f64,
    /// Working-to-deep coupling weight (dimensionless).
    pub alpha_d: f64,
    /// Base firing threshold, before the deep-state and confidence factors (dimensionless).
    pub theta: f64,
    /// Deep-state gain on the adaptive threshold (dimensionless).
    pub gamma: f64,
    /// Confidence-driven reduction of the firing threshold (dimensionless).
    pub delta_conf: f64,
    /// Attention-gate weights on the input, `v_fast`, `v_work` and the confidence.
    pub w_gate: [f64; 4],
    /// Self-model weights predicting `v_fast` from `v_fast`, `v_work` and `v_deep`; learned
    /// online and kept at unit norm.
    pub w_pred: [f64; 3],
    /// Steepness of the novelty (surprise) sigmoid (dimensionless).
    pub kappa: f64,
    /// Baseline surprise subtracted before the novelty sigmoid (dimensionless).
    pub surprise_baseline: f64,
    /// Baseline meta-learning rate (dimensionless).
    pub lr_base: f64,
    /// Novelty modulation strength of the learning rate (dimensionless).
    pub eta: f64,
    /// Recurrent inhibition weight on the fast compartment (dimensionless).
    pub w_inh: f64,
    /// Step (ms).
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
    /// Create a neuron with every compartment at 0, the default parameters, `dt` = 1 ms, a
    /// 50-step spike history and a 20-step novelty history at 0.5 (confidence 0.5).
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

    /// Advance one step with `current` and return 1 on a spike, else 0.
    ///
    /// In order:
    /// 1. The confidence is 1 minus the mean of the novelty history.
    /// 2. The attention gate is the logistic of `w_gate` applied to the input, `v_fast`,
    ///    `v_work` and the confidence, and it scales the input.
    /// 3. `v_fast` takes an Euler step with the gated input and the inhibition
    ///    `w_inh * spike_rate`, where the spike rate is taken over the history window.
    /// 4. The prediction, surprise and novelty follow from the new `v_fast`.
    /// 5. The threshold is `max(theta * (1 + gamma * v_deep) * (1 - delta_conf * confidence), 0.1)`.
    /// 6. A spike feeds `v_work` and sets `v_fast` to 0.
    /// 7. `v_work` and `v_deep` decay and integrate.
    /// 8. `w_pred` takes a novelty-modulated learning step on the prediction error and is
    ///    renormalised.
    ///
    /// The input is not validated.
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

    /// Clear the fast and working compartments, the prediction, the surprise, the novelty
    /// and the spike history. `v_deep`, the learned `w_pred`, the novelty history (and so
    /// the confidence) and every parameter are kept.
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

#[cfg(test)]
mod tests {
    use super::*;

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
