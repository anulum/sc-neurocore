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
    /// The step is the Python model's, in its order and with its arithmetic, so the engine
    /// follows it bit for bit:
    /// 1. The spike rate is the spike count of the history over its length, and the
    ///    confidence is 1 minus the left-to-right mean of the novelty history.
    /// 2. The attention gate is the logistic of `w_gate` applied to the input, `v_fast`,
    ///    `v_work` and the confidence, and it scales the input.
    /// 3. `v_fast` relaxes exactly over `dt` towards the gated input minus
    ///    `w_inh * spike_rate`, with `tau_fast`.
    /// 4. The prediction, surprise and novelty follow from that candidate `v_fast` and the
    ///    previous `v_work` and `v_deep`.
    /// 5. The threshold is `max(theta * (1 + gamma * v_deep) * (1 - delta_conf * confidence), 0.1)`
    ///    with the previous `v_deep`; a candidate at or above it spikes and `v_fast` becomes 0.
    /// 6. `v_work` relaxes exactly towards `alpha_w * v_fast` on a spike and towards 0
    ///    otherwise; `v_deep` relaxes exactly towards `alpha_d * v_work * novelty`.
    /// 7. `w_pred` takes a novelty-modulated learning step on the error of the accepted
    ///    `v_fast` and is renormalised by the square root of its plain sum of squares.
    ///
    /// The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let old_v_fast = self.v_fast;
        let old_v_work = self.v_work;
        let old_v_deep = self.v_deep;
        let spike_count: u32 = self.spike_history.iter().map(|&s| s as u32).sum();
        let spike_rate = spike_count as f64 / self.spike_history.len() as f64;
        let mut novelty_total = 0.0;
        for &value in &self.novelty_history {
            novelty_total += value;
        }
        let confidence = 1.0 - novelty_total / self.novelty_history.len() as f64;

        let gate_input = self.w_gate[0] * current
            + self.w_gate[1] * old_v_fast
            + self.w_gate[2] * old_v_work
            + self.w_gate[3] * confidence;
        let gate = Self::sigmoid(gate_input);
        let i_eff = gate * current;
        let fast_drive = i_eff - self.w_inh * spike_rate;
        let next_v_fast = Self::exact_relaxation(old_v_fast, fast_drive, self.dt, self.tau_fast);

        let prediction = self.w_pred[0] * next_v_fast
            + self.w_pred[1] * old_v_work
            + self.w_pred[2] * old_v_deep;
        let surprise = (next_v_fast - prediction).abs();
        let novelty = Self::sigmoid(self.kappa * (surprise - self.surprise_baseline));

        let eff_threshold =
            (self.theta * (1.0 + self.gamma * old_v_deep) * (1.0 - self.delta_conf * confidence))
                .max(0.1);
        let spike = if next_v_fast >= eff_threshold { 1 } else { 0 };
        let accepted_v_fast = if spike == 1 { 0.0 } else { next_v_fast };

        let work_drive = if spike == 1 {
            self.alpha_w * next_v_fast
        } else {
            0.0
        };
        let next_v_work = Self::exact_relaxation(old_v_work, work_drive, self.dt, self.tau_work);
        let deep_drive = self.alpha_d * next_v_work * novelty;
        let next_v_deep = Self::exact_relaxation(old_v_deep, deep_drive, self.dt, self.tau_deep);

        let meta_lr = self.lr_base * (1.0 + self.eta * novelty);
        let error = accepted_v_fast - prediction;
        let mut w_pred = self.w_pred;
        w_pred[0] += meta_lr * error * accepted_v_fast;
        w_pred[1] += meta_lr * error * next_v_work;
        w_pred[2] += meta_lr * error * next_v_deep;
        let norm = (w_pred[0] * w_pred[0] + w_pred[1] * w_pred[1] + w_pred[2] * w_pred[2]).sqrt();
        if norm > 0.0 {
            for weight in &mut w_pred {
                *weight /= norm;
            }
        }

        let novelty_len = self.novelty_history.len();
        self.novelty_history[self.nov_idx % novelty_len] = novelty;
        let spike_len = self.spike_history.len();
        self.spike_history[self.hist_idx % spike_len] = spike as u8;

        self.v_fast = accepted_v_fast;
        self.v_work = next_v_work;
        self.v_deep = next_v_deep;
        self.prediction = prediction;
        self.surprise = surprise;
        self.novelty = novelty;
        self.confidence = confidence;
        self.w_pred = w_pred;
        self.nov_idx += 1;
        self.hist_idx += 1;
        self.total_steps += 1;
        spike
    }

    fn sigmoid(x: f64) -> f64 {
        if !x.is_finite() {
            return if x > 0.0 { 1.0 } else { 0.0 };
        }
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }

    fn exact_relaxation(state: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        let decay = (-dt / tau).exp();
        decay * state + (1.0 - decay) * steady_state
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
    fn arcane_matches_the_python_model_bit_for_bit() {
        // Python's ArcaneNeuron after 300 steps of 2.0 from its defaults: 75 spikes, then
        // v_fast = 0, v_work = 0.052144899505804664, v_deep = 2.041571933657978e-05 and
        // w_pred = [0.9161249065184631, 0.37682124738564304, 0.13682435154337574].
        let mut n = ArcaneNeuron::new();
        let spikes: i32 = (0..300).map(|_| n.step(2.0)).sum();
        assert_eq!(spikes, 75);
        assert_eq!(n.v_fast, 0.0);
        assert_eq!(n.v_work, 0.052144899505804664);
        assert_eq!(n.v_deep, 2.041571933657978e-05);
        assert_eq!(
            n.w_pred,
            [0.9161249065184631, 0.37682124738564304, 0.13682435154337574]
        );
    }

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
