// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for arcane_neuron

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ArcaneNeuron {
    pub v_fast: f64,
    pub tau_fast: f64,
    pub v_work: f64,
    pub tau_work: f64,
    pub alpha_w: f64,
    pub v_deep: f64,
    pub tau_deep: f64,
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
    pub prediction: f64,
    pub surprise: f64,
    pub novelty: f64,
    pub confidence: f64,
    pub spike_history: [f64; 50],
    pub novelty_history: [f64; 20],
    pub hist_idx: usize,
    pub nov_idx: usize,
    pub total_steps: usize,
    pub identity_drift: f64,
    pub w_inh: f64,
    pub dt: f64,
}

impl ArcaneNeuron {
    pub fn new() -> Self {
        Self {
            v_fast: 0.0,
            tau_fast: 5.0,
            v_work: 0.0,
            tau_work: 200.0,
            alpha_w: 0.3,
            v_deep: 0.0,
            tau_deep: 10000.0,
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
            prediction: 0.0,
            surprise: 0.0,
            novelty: 0.0,
            confidence: 0.5,
            spike_history: [0.0; 50],
            novelty_history: [0.5; 20],
            hist_idx: 0,
            nov_idx: 0,
            total_steps: 0,
            identity_drift: 0.0,
            w_inh: 0.3,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !validate_arcane_neuron(self) {
            return Err("ArcaneNeuron state/current must be finite and physically valid");
        }

        let spike_rate = mean50(&self.spike_history);
        let confidence = 1.0 - mean20(&self.novelty_history);
        let gate_input = self.w_gate[0] * i_ext
            + self.w_gate[1] * self.v_fast
            + self.w_gate[2] * self.v_work
            + self.w_gate[3] * confidence;
        let gate = stable_sigmoid(gate_input);
        let i_eff = gate * i_ext;
        let fast_drive = i_eff - self.w_inh * spike_rate;
        let next_v_fast_continuous =
            exact_relaxation(self.v_fast, fast_drive, self.dt, self.tau_fast);
        if !next_v_fast_continuous.is_finite() {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }

        let prediction = self.w_pred[0] * next_v_fast_continuous
            + self.w_pred[1] * self.v_work
            + self.w_pred[2] * self.v_deep;
        if !prediction.is_finite() {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }
        let surprise = (next_v_fast_continuous - prediction).abs();
        let novelty = stable_sigmoid(self.kappa * (surprise - self.surprise_baseline));

        let mut eff_threshold =
            self.theta * (1.0 + self.gamma * self.v_deep) * (1.0 - self.delta_conf * confidence);
        if !eff_threshold.is_finite() {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }
        if eff_threshold < 0.1 {
            eff_threshold = 0.1;
        }

        let mut spike = 0;
        let mut accepted_v_fast = next_v_fast_continuous;
        if next_v_fast_continuous >= eff_threshold {
            spike = 1;
            accepted_v_fast = 0.0;
        }

        let work_drive = if spike == 1 {
            self.alpha_w * next_v_fast_continuous
        } else {
            0.0
        };
        let next_v_work = exact_relaxation(self.v_work, work_drive, self.dt, self.tau_work);
        if !next_v_work.is_finite() {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }

        let deep_drive = self.alpha_d * next_v_work * novelty;
        let next_v_deep = exact_relaxation(self.v_deep, deep_drive, self.dt, self.tau_deep);
        if !next_v_deep.is_finite() {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }

        let meta_lr = self.lr_base * (1.0 + self.eta * novelty);
        let error = accepted_v_fast - prediction;
        let mut next_w_pred = self.w_pred;
        next_w_pred[0] += meta_lr * error * accepted_v_fast;
        next_w_pred[1] += meta_lr * error * next_v_work;
        next_w_pred[2] += meta_lr * error * next_v_deep;
        let norm = (next_w_pred[0] * next_w_pred[0]
            + next_w_pred[1] * next_w_pred[1]
            + next_w_pred[2] * next_w_pred[2])
            .sqrt();
        if !norm.is_finite() {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }
        if norm > 0.0 {
            next_w_pred[0] /= norm;
            next_w_pred[1] /= norm;
            next_w_pred[2] /= norm;
        }
        if !next_w_pred.iter().all(|value| value.is_finite()) {
            return Err("ArcaneNeuron exact relaxation update became non-finite");
        }

        let mut next_novelty_history = self.novelty_history;
        next_novelty_history[self.nov_idx % next_novelty_history.len()] = novelty;
        let mut next_spike_history = self.spike_history;
        next_spike_history[self.hist_idx % next_spike_history.len()] = spike as f64;

        let old_v_deep = self.v_deep;
        self.v_fast = accepted_v_fast;
        self.v_work = next_v_work;
        self.v_deep = next_v_deep;
        self.prediction = prediction;
        self.surprise = surprise;
        self.novelty = novelty;
        self.confidence = confidence;
        self.novelty_history = next_novelty_history;
        self.nov_idx += 1;
        self.identity_drift += (next_v_deep - old_v_deep).abs();
        self.w_pred = next_w_pred;
        self.spike_history = next_spike_history;
        self.hist_idx += 1;
        self.total_steps += 1;
        Ok(spike)
    }

    pub fn reset(&mut self) {
        self.v_fast = 0.0;
        self.v_work = 0.0;
        self.prediction = 0.0;
        self.surprise = 0.0;
        self.novelty = 0.0;
        self.spike_history = [0.0; 50];
        self.hist_idx = 0;
        self.identity_drift = 0.0;
    }

    pub fn identity_state(&self) -> f64 {
        self.v_deep
    }

    pub fn meta_learning_rate(&self) -> f64 {
        self.lr_base * (1.0 + self.eta * self.novelty)
    }
}

fn exact_relaxation(state: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
    let decay = (-dt / tau).exp();
    decay * state + (1.0 - decay) * steady_state
}

fn stable_sigmoid(x: f64) -> f64 {
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x >= 0.0 {
        let z = (-x).exp();
        return 1.0 / (1.0 + z);
    }
    let z = x.exp();
    z / (1.0 + z)
}

fn mean50(values: &[f64; 50]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean20(values: &[f64; 20]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn validate_arcane_neuron(state: &ArcaneNeuron) -> bool {
    state.v_fast.is_finite()
        && state.v_work.is_finite()
        && state.v_deep.is_finite()
        && state.prediction.is_finite()
        && state.surprise.is_finite()
        && state.novelty.is_finite()
        && state.confidence.is_finite()
        && state.identity_drift.is_finite()
        && state.tau_fast.is_finite()
        && state.tau_fast > 0.0
        && state.tau_work.is_finite()
        && state.tau_work > 0.0
        && state.tau_deep.is_finite()
        && state.tau_deep > 0.0
        && state.alpha_w.is_finite()
        && state.alpha_w >= 0.0
        && state.alpha_d.is_finite()
        && state.alpha_d >= 0.0
        && state.theta.is_finite()
        && state.theta > 0.0
        && state.gamma.is_finite()
        && state.delta_conf.is_finite()
        && state.kappa.is_finite()
        && state.surprise_baseline.is_finite()
        && state.lr_base.is_finite()
        && state.lr_base >= 0.0
        && state.eta.is_finite()
        && state.w_inh.is_finite()
        && state.w_inh >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.w_gate.iter().all(|value| value.is_finite())
        && state.w_pred.iter().all(|value| value.is_finite())
        && state
            .spike_history
            .iter()
            .all(|value| *value == 0.0 || *value == 1.0)
        && state.novelty_history.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arcane_neuron_new() {
        let state = ArcaneNeuron::new();
        assert!(validate_arcane_neuron(&state));
    }

    #[test]
    fn test_arcane_neuron_step() {
        let mut state = ArcaneNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    fn exact_reference(state: f64, steady_state: f64, dt: f64, tau: f64) -> f64 {
        let decay = (-dt / tau).exp();
        decay * state + (1.0 - decay) * steady_state
    }

    fn stable_sigmoid_reference(x: f64) -> f64 {
        if x >= 0.0 {
            let z = (-x).exp();
            return 1.0 / (1.0 + z);
        }
        let z = x.exp();
        z / (1.0 + z)
    }

    #[test]
    fn test_arcane_neuron_exact_relaxation_no_spike() {
        let mut state = ArcaneNeuron::new();
        state.v_fast = 0.4;
        state.v_work = 0.2;
        state.v_deep = 0.01;
        state.theta = 100.0;
        state.dt = 25.0;
        state.novelty_history = [0.2; 20];

        let current = 1.5;
        let confidence = 0.8;
        let gate_input = state.w_gate[0] * current
            + state.w_gate[1] * state.v_fast
            + state.w_gate[2] * state.v_work
            + state.w_gate[3] * confidence;
        let gate = stable_sigmoid_reference(gate_input);
        let expected_fast = exact_reference(state.v_fast, gate * current, state.dt, state.tau_fast);
        let expected_work = exact_reference(state.v_work, 0.0, state.dt, state.tau_work);

        let spike = state.step(current).unwrap();
        assert_eq!(spike, 0);
        assert!((state.v_fast - expected_fast).abs() < 1.0e-12);
        assert!((state.v_work - expected_work).abs() < 1.0e-12);
    }

    #[test]
    fn test_arcane_neuron_invalid_state_preserves_state() {
        let mut state = ArcaneNeuron::new();
        state.v_fast = 0.25;
        state.v_work = 0.1;
        state.v_deep = 0.01;
        let before = (state.v_fast, state.v_work, state.v_deep);
        state.tau_fast = 0.0;
        assert!(state.step(0.5).is_err());
        assert_eq!((state.v_fast, state.v_work, state.v_deep), before);
    }
}
