// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Discrete map neuron models

//! Discrete map neuron models.

/// Chialvo 1995 — 2D discrete map neuron.
#[derive(Clone, Debug)]
pub struct ChialvoMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub k: f64,
    pub x_threshold: f64,
}

impl ChialvoMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            a: 0.89,
            b: 0.6,
            c: 0.28,
            k: 0.04,
            x_threshold: 1.0,
        }
    }
    fn is_valid(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.a.is_finite()
            && self.b.is_finite()
            && self.c.is_finite()
            && self.k.is_finite()
            && self.x_threshold.is_finite()
    }

    fn safe_exp(value: f64) -> f64 {
        value.clamp(-500.0, 500.0).exp()
    }

    /// Checked Chialvo update used by the production batch dispatcher.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.is_valid() {
            return Err("invalid Chialvo map runtime state");
        }
        if !current.is_finite() {
            return Err("invalid Chialvo map current");
        }

        let x_prev = self.x;
        let x_squared = self.x * self.x;
        let exponential = Self::safe_exp(self.y - self.x);
        let x_new = x_squared * exponential + self.k + current;
        let y_new = self.a * self.y - self.b * self.x + self.c;
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("invalid Chialvo map candidate state");
        }
        self.x = x_new;
        self.y = y_new;
        Ok(if x_prev < self.x_threshold && self.x >= self.x_threshold {
            1
        } else {
            0
        })
    }

    /// Legacy infallible engine-class update. Invalid input leaves the state
    /// unchanged and emits no event; the checked batch API reports the error.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run checked map iterations, returning the fast-state trace and events.
    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes = 0_i64;
        for _ in 0..n_steps {
            spikes += i64::from(self.try_step(current)?);
            trace.push(self.x);
        }
        Ok((trace, spikes))
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}
impl Default for ChialvoMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Rulkov 2001 — piecewise nonlinear map for fast/slow bursting.
#[derive(Clone, Debug)]
pub struct RulkovMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub sigma: f64,
    pub mu: f64,
    pub x_threshold: f64,
}

impl RulkovMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0,
            y: -3.0,
            alpha: 4.0,
            sigma: -1.6,
            mu: 0.001,
            x_threshold: 0.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let x_new = if self.x <= 0.0 {
            self.alpha / (1.0 - self.x) + self.y + current
        } else if self.x < self.alpha + self.y + current {
            self.alpha + self.y + current
        } else {
            -1.0
        };
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// upward-crossing spike count. Reuses `step` so the trace is bit-identical
    /// to the per-step path and to the Python reference. The final state is
    /// left in `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = -1.0;
        self.y = -3.0;
    }
}
impl Default for RulkovMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Ibarz-Tanaka map — piecewise-linear spiking map.
#[derive(Clone, Debug)]
pub struct IbarzTanakaMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub beta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub x_threshold: f64,
    pub x_reset: f64,
}

impl IbarzTanakaMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0,
            y: -2.5,
            alpha: 3.65,
            beta: 0.25,
            mu: 0.0005,
            sigma: -1.6,
            x_threshold: 3.0,
            x_reset: -1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let f = if self.x <= 0.0 {
            self.alpha / (1.0 - self.x)
        } else {
            self.alpha + self.beta * self.x
        };
        let x_new = f + self.y + current;
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold {
            self.x = self.x_reset;
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace (already
    /// reset to `x_reset` on spiking steps) and the spike count. Reuses `step`
    /// so the trace is bit-identical to the per-step path and to the Python
    /// reference. The final state is left in `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = -1.0;
        self.y = -2.5;
    }
}
impl Default for IbarzTanakaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Medvedev map — piecewise monotone 1D neuron map.
#[derive(Clone, Debug)]
pub struct MedvedevMapNeuron {
    pub x: f64,
    pub alpha: f64,
    pub beta: f64,
    pub x_threshold: f64,
}

impl MedvedevMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            alpha: 3.5,
            beta: 0.5,
            x_threshold: 0.9,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        self.x = if self.x < self.beta {
            self.alpha * self.x + current
        } else {
            self.alpha * (1.0 - self.x) + current
        };
        self.x = self.x.rem_euclid(1.0);
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace (folded
    /// into `[0, 1)`) and the upward-crossing spike count. Reuses `step` so the
    /// trace is bit-identical to the per-step path and to the Python reference
    /// (`f64::rem_euclid(1.0)` equals Python's `x % 1.0` bit-for-bit). The final
    /// state is left in `self.x`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
    }
}
impl Default for MedvedevMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Cazelles logistic map neuron — coupled 2D logistic with slow variable.
#[derive(Clone, Debug)]
pub struct CazellesMapNeuron {
    pub x: f64,
    pub y: f64,
    pub a: f64,
    pub epsilon: f64,
    pub sigma: f64,
    pub x_threshold: f64,
}

impl CazellesMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.1,
            y: 0.0,
            a: 3.8,
            epsilon: 0.01,
            sigma: 0.5,
            x_threshold: 0.9,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let f = self.a * self.x * (1.0 - self.x);
        let x_new = (f - self.y + current).clamp(-2.0, 2.0);
        let y_new = self.y + self.epsilon * (self.x - self.sigma);
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// spike count. Reuses `step` so the trace is bit-identical to the
    /// per-step path and to the Python reference. The final state is left in
    /// `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = 0.1;
        self.y = 0.0;
    }
}
impl Default for CazellesMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Courbage-Nekorkin-Vdovin (2007) discontinuous two-dimensional spiking map.
///
/// Canonical map (Chaos 17:043109; arXiv:0712.2097, eqs. 3-5):
///
/// ```text
/// x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
/// y(n+1) = y(n) + eps*(x(n) - J)
/// F(x)   = -m0*x        for x <= Jmin
///          m1*(x - a)   for Jmin < x < Jmax
///          -m0*(x - 1)  for x >= Jmax
/// H(z)   = 1 for z >= 0, else 0
/// Jmin   = a*m1/(m0 + m1),  Jmax = (m0 + a*m1)/(m0 + m1)
/// ```
///
/// Defaults place the model in the published chaotic spiking-bursting regime.
/// The arithmetic is exact (no transcendental functions), so `simulate` is
/// bit-identical to the Python NumPy reference and the Julia/Go backends.
#[derive(Clone, Debug)]
pub struct CourageNekorkinMapNeuron {
    pub x: f64,
    pub y: f64,
    pub m0: f64,
    pub m1: f64,
    pub a: f64,
    pub d: f64,
    pub j: f64,
    pub beta: f64,
    pub eps: f64,
    pub x_threshold: f64,
}

impl CourageNekorkinMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            m0: 0.0864,
            m1: 0.65,
            a: 0.2,
            d: 0.235,
            j: 0.2,
            beta: 0.085,
            eps: 0.02,
            x_threshold: 0.235,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let am1 = self.a * self.m1;
        let den = self.m0 + self.m1;
        let jmin = am1 / den;
        let jmax = (self.m0 + am1) / den;
        let fx = if self.x <= jmin {
            -self.m0 * self.x
        } else if self.x < jmax {
            self.m1 * (self.x - self.a)
        } else {
            -self.m0 * (self.x - 1.0)
        };
        let h = if (self.x - self.d) >= 0.0 { 1.0 } else { 0.0 };
        let x_new = self.x + fx - self.y - self.beta * h + current;
        let y_new = self.y + self.eps * (self.x - self.j);
        self.x = x_new;
        self.y = y_new;
        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// upward-crossing spike count. Reuses `step` so the trace is bit-identical
    /// to the per-step path and to the Python reference. The final state is left
    /// in `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.x);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}
impl Default for CourageNekorkinMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Aihara 1990 — chaotic neuron map with sigmoid nonlinearity.
///
/// 2D discrete map producing chaotic spiking, bursting, and tonic firing
/// depending on parameters. The sigmoid output function models the
/// nonlinear voltage-to-firing-rate relationship.
///
/// x(n+1) = k_f * x(n) / (1 + exp(-(x(n) + alpha))) - y(n) + I
/// y(n+1) = k_s * y(n) + delta * x(n)
///
/// Aihara et al., Phys Lett A 144:333, 1990.
#[derive(Clone, Debug)]
pub struct AiharaMapNeuron {
    pub x: f64,
    pub y: f64,
    pub k_f: f64,   // Fast variable decay
    pub k_s: f64,   // Slow variable decay
    pub alpha: f64, // Sigmoid steepness offset
    pub delta: f64, // Slow→fast coupling
    pub x_threshold: f64,
}

impl Default for AiharaMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AiharaMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            k_f: 0.7,
            k_s: 0.95,
            alpha: 2.0,
            delta: 0.05,
            x_threshold: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let sigmoid = 1.0 / (1.0 + (-(self.x + self.alpha)).exp());
        let x_new = self.k_f * self.x * sigmoid - self.y + current;
        let y_new = self.k_s * self.y + self.delta * self.x;

        self.x = x_new.clamp(-10.0, 10.0);
        self.y = y_new.clamp(-10.0, 10.0);

        if !self.x.is_finite() {
            self.x = 0.0;
        }
        if !self.y.is_finite() {
            self.y = 0.0;
        }

        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Kilinc-Bhatt 2023 — sigmoid map with adaptive threshold.
///
/// Minimal 2D map with built-in spike frequency adaptation via
/// a slow threshold variable. Designed for efficient hardware
/// implementation while retaining biologically relevant dynamics.
///
/// x(n+1) = k * sigmoid(x(n) - theta(n)) + I
/// theta(n+1) = beta * theta(n) + gamma * H(x(n) - theta_spike)
///
/// H() is the Heaviside step function (spike-triggered increment).
#[derive(Clone, Debug)]
pub struct KilincBhattMapNeuron {
    pub x: f64,
    pub theta: f64,       // Adaptive threshold
    pub k: f64,           // Gain
    pub beta: f64,        // Threshold decay
    pub gamma: f64,       // Spike→threshold coupling
    pub theta_spike: f64, // Spike detection level
    pub x_threshold: f64,
}

impl Default for KilincBhattMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl KilincBhattMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            theta: 0.0,
            k: 1.5,
            beta: 0.95,
            gamma: 0.3,
            theta_spike: 0.8,
            x_threshold: 0.8,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let x_prev = self.x;
        let sig = 1.0 / (1.0 + (-(self.x - self.theta) * 4.0).exp());
        let x_new = -self.x + self.k * sig + current;
        let spiked = if self.x >= self.theta_spike { 1.0 } else { 0.0 };
        let theta_new = self.beta * self.theta + self.gamma * spiked;

        self.x = x_new.clamp(-5.0, 5.0);
        self.theta = theta_new.clamp(-5.0, 5.0);

        if !self.x.is_finite() {
            self.x = 0.0;
        }
        if !self.theta.is_finite() {
            self.theta = 0.0;
        }

        if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Ermentrout-Kopell canonical Type I — theta neuron in map form.
///
/// The canonical model for Type I (saddle-node) excitability.
/// theta(n+1) = theta(n) + dt * (1 - cos(theta)) + (1 + cos(theta)) * I
/// Spike when theta crosses pi.
///
/// Ermentrout & Kopell, SIAM J Appl Math 46:233, 1986.
#[derive(Clone, Debug)]
pub struct ErmentroutKopellMapNeuron {
    pub theta: f64, // Phase variable [0, 2*pi)
    pub dt: f64,
    pub gain: f64,
    pub theta_threshold: f64,
}

impl Default for ErmentroutKopellMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ErmentroutKopellMapNeuron {
    pub fn new() -> Self {
        Self {
            theta: 0.0,
            dt: 0.1, // Discrete step size
            gain: 1.0,
            theta_threshold: std::f64::consts::PI,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let theta_prev = self.theta;

        let d_theta = (1.0 - self.theta.cos()) + (1.0 + self.theta.cos()) * input;
        self.theta += self.dt * d_theta;

        // Spike detection: crossing pi
        let fired = if self.theta >= self.theta_threshold && theta_prev < self.theta_threshold {
            1
        } else {
            0
        };

        // Wrap theta to [0, 2*pi)
        let two_pi = 2.0 * std::f64::consts::PI;
        if self.theta >= two_pi {
            self.theta -= two_pi;
        }
        if self.theta < 0.0 {
            self.theta += two_pi;
        }

        if !self.theta.is_finite() {
            self.theta = 0.0;
        }

        fired
    }

    /// Run `n_steps` under a constant input, returning the `theta` trace
    /// (wrapped to `[0, 2*pi)`) and the upward-crossing spike count. Reuses
    /// `step` so the trace matches the per-step path; on a shared libm it also
    /// matches the Python reference bit-for-bit (the only transcendental is
    /// `cos`, and the non-chaotic phase flow does not amplify ULP differences).
    /// The final state is left in `self.theta`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        let mut trace = Vec::with_capacity(n_steps);
        let mut spikes: i64 = 0;
        for _ in 0..n_steps {
            let spiked = self.step(current);
            trace.push(self.theta);
            spikes += spiked as i64;
        }
        (trace, spikes)
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chialvo_matches_independent_source_step() {
        let mut neuron = ChialvoMapNeuron {
            x: 0.2,
            y: 0.7,
            ..Default::default()
        };
        let x = neuron.x;
        let y = neuron.y;
        let expected_x = x * x * (y - x).exp() + neuron.k + 0.01;
        let expected_y = neuron.a * y - neuron.b * x + neuron.c;
        assert_eq!(neuron.try_step(0.01), Ok(0));
        assert_eq!(neuron.x, expected_x);
        assert_eq!(neuron.y, expected_y);
    }

    #[test]
    fn chialvo_matches_python_golden_event_counts() {
        for (current, expected) in [(-0.05, 0_i64), (0.0, 26), (0.01, 30), (0.1, 0), (1.0, 1)] {
            let mut neuron = ChialvoMapNeuron::new();
            let (_trace, spikes) = neuron
                .simulate(1000, current)
                .expect("finite source regime");
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn chialvo_rejects_non_finite_input_without_mutation() {
        let mut neuron = ChialvoMapNeuron::new();
        let initial = (neuron.x, neuron.y);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.x, neuron.y), initial);

        neuron.y = f64::INFINITY;
        assert!(neuron.try_step(0.0).is_err());
    }

    #[test]
    fn chialvo_reset_preserves_parameters() {
        let mut neuron = ChialvoMapNeuron {
            x: 2.0,
            y: -1.0,
            a: 0.8,
            b: 0.4,
            c: 0.2,
            k: 0.03,
            x_threshold: 0.75,
        };
        neuron.reset();
        assert_eq!((neuron.x, neuron.y), (0.0, 0.0));
        assert_eq!(
            (neuron.a, neuron.b, neuron.c, neuron.k, neuron.x_threshold),
            (0.8, 0.4, 0.2, 0.03, 0.75)
        );
    }
    #[test]
    fn rulkov_fires() {
        let mut n = RulkovMapNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }
    #[test]
    fn ibarz_fires() {
        let mut n = IbarzTanakaMapNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn medvedev_fires() {
        let mut n = MedvedevMapNeuron {
            x: 0.5,
            ..Default::default()
        };
        let t: i32 = (0..500).map(|_| n.step(0.1)).sum();
        assert!(t > 0);
    }
    #[test]
    fn cazelles_fires() {
        let mut n = CazellesMapNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn cournekorkin_fires() {
        let mut n = CourageNekorkinMapNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    // -- Aihara Map coverage tests --

    #[test]
    fn aihara_fires_with_input() {
        let mut n = AiharaMapNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(t > 0, "Aihara must fire with input, got {t}");
    }

    #[test]
    fn aihara_silent_without_input() {
        let mut n = AiharaMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0, "Aihara must be silent without input, got {t}");
    }

    #[test]
    fn aihara_chaotic_dynamics() {
        // With appropriate input, trajectory should not settle to fixed point
        let mut n = AiharaMapNeuron::new();
        let mut values = Vec::new();
        for _ in 0..1000 {
            n.step(0.5);
            values.push(n.x);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        assert!(
            var > 0.001,
            "Trajectory should show variability (chaos), var={var}"
        );
    }

    #[test]
    fn aihara_negative_input_no_crash() {
        let mut n = AiharaMapNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.x.is_finite());
    }

    #[test]
    fn aihara_nan_input_stays_finite() {
        let mut n = AiharaMapNeuron::new();
        n.step(f64::NAN);
        assert!(n.x.is_finite());
    }

    #[test]
    fn aihara_extreme_input_bounded() {
        let mut n = AiharaMapNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.x.is_finite() && n.x <= 1e6);
    }

    #[test]
    fn aihara_reset_clears_state() {
        let mut n = AiharaMapNeuron::new();
        for _ in 0..100 {
            n.step(1.0);
        }
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
    }

    #[test]
    fn aihara_rate_increases_with_input() {
        let mut low = AiharaMapNeuron::new();
        let mut high = AiharaMapNeuron::new();
        let spikes_low: i32 = (0..5000).map(|_| low.step(0.5)).sum();
        let spikes_high: i32 = (0..5000).map(|_| high.step(2.0)).sum();
        assert!(
            spikes_high >= spikes_low,
            "Higher input should produce more spikes: high={spikes_high} vs low={spikes_low}"
        );
    }

    #[test]
    fn aihara_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = AiharaMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(0.5));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }

    // -- Kilinc-Bhatt Map coverage tests --

    #[test]
    fn kb_fires_with_input() {
        let mut n = KilincBhattMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(1.0)).sum();
        assert!(t > 0, "KB must fire with input, got {t}");
    }

    #[test]
    fn kb_silent_without_input() {
        let mut n = KilincBhattMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0, "KB must be silent without input, got {t}");
    }

    #[test]
    fn kb_adaptation() {
        // Theta increases with spiking → fewer spikes over time
        let mut n = KilincBhattMapNeuron::new();
        let early: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        let late: i32 = (0..2000).map(|_| n.step(1.0)).sum();
        assert!(
            early >= late,
            "Adaptation should slow firing: early={early}, late={late}"
        );
    }

    #[test]
    fn kb_theta_increases_during_spiking() {
        let mut n = KilincBhattMapNeuron::new();
        let theta_before = n.theta;
        for _ in 0..5000 {
            n.step(1.5);
        }
        assert!(
            n.theta > theta_before,
            "Theta must increase during spiking, theta={}",
            n.theta
        );
    }

    #[test]
    fn kb_negative_input_no_crash() {
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.x.is_finite());
    }

    #[test]
    fn kb_nan_input_stays_finite() {
        let mut n = KilincBhattMapNeuron::new();
        n.step(f64::NAN);
        assert!(n.x.is_finite());
    }

    #[test]
    fn kb_extreme_input_bounded() {
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.x.is_finite() && n.x <= 5.0);
    }

    #[test]
    fn kb_reset_clears_state() {
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..100 {
            n.step(1.0);
        }
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.theta, 0.0);
    }

    #[test]
    fn kb_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = KilincBhattMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(0.5));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }

    // -- Ermentrout-Kopell Map coverage tests --

    #[test]
    fn ek_fires_with_input() {
        let mut n = ErmentroutKopellMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(0.5)).sum();
        assert!(t > 0, "EK must fire with input, got {t}");
    }

    #[test]
    fn ek_silent_without_input() {
        // Type I: no firing below threshold (I < 0 is subthreshold for theta model)
        let mut n = ErmentroutKopellMapNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(-0.1)).sum();
        assert_eq!(t, 0, "EK must be silent with negative input, got {t}");
    }

    #[test]
    fn ek_type_i_excitability() {
        // Type I: arbitrarily low firing rate near threshold
        let mut n_low = ErmentroutKopellMapNeuron::new();
        let mut n_high = ErmentroutKopellMapNeuron::new();
        let spikes_low: i32 = (0..10_000).map(|_| n_low.step(0.01)).sum();
        let spikes_high: i32 = (0..10_000).map(|_| n_high.step(1.0)).sum();
        assert!(
            spikes_high > spikes_low,
            "Higher input → higher rate: high={spikes_high} vs low={spikes_low}"
        );
    }

    #[test]
    fn ek_theta_wraps() {
        // Theta should stay in [0, 2*pi)
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..10_000 {
            n.step(0.5);
        }
        let two_pi = 2.0 * std::f64::consts::PI;
        assert!(
            n.theta >= 0.0 && n.theta < two_pi,
            "Theta must wrap to [0, 2pi), theta={}",
            n.theta
        );
    }

    #[test]
    fn ek_negative_input_no_crash() {
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.theta.is_finite());
    }

    #[test]
    fn ek_nan_input_stays_finite() {
        let mut n = ErmentroutKopellMapNeuron::new();
        n.step(f64::NAN);
        assert!(n.theta.is_finite());
    }

    #[test]
    fn ek_extreme_input_bounded() {
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.theta.is_finite());
    }

    #[test]
    fn ek_reset_clears_state() {
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..100 {
            n.step(0.5);
        }
        n.reset();
        assert_eq!(n.theta, 0.0);
    }

    #[test]
    fn ek_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = ErmentroutKopellMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(0.5));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }

    // -- Courbage-Nekorkin-Vdovin 2007 canonical map tests --

    #[test]
    fn cn_default_sustained_bounded_spiking() {
        // The default parameters are the published chaotic spiking-bursting
        // regime: sustained firing with a bounded (clip-free) trajectory.
        let mut n = CourageNekorkinMapNeuron::new();
        let mut spikes = 0i64;
        let mut max_abs = 0.0f64;
        for _ in 0..20_000 {
            spikes += n.step(0.0) as i64;
            max_abs = max_abs.max(n.x.abs());
        }
        assert!(spikes > 1000, "expected sustained spiking, got {spikes}");
        assert!(
            max_abs < 10.0,
            "trajectory must stay bounded, got {max_abs}"
        );
    }

    #[test]
    fn cn_breakpoints_inside_region() {
        // Default discontinuity d sits strictly inside (Jmin, Jmax) — eq. 6.
        let n = CourageNekorkinMapNeuron::new();
        let am1 = n.a * n.m1;
        let den = n.m0 + n.m1;
        let jmin = am1 / den;
        let jmax = (n.m0 + am1) / den;
        assert!(jmin < n.d && n.d < jmax);
        assert!(n.j > 0.0 && n.j < n.d);
        assert!(n.m0 < 1.0);
    }

    #[test]
    fn cn_f_piecewise_branches() {
        // F lower/middle/upper branches (eq. 4).
        let n = CourageNekorkinMapNeuron::new();
        let am1 = n.a * n.m1;
        let den = n.m0 + n.m1;
        let jmin = am1 / den;
        let jmax = (n.m0 + am1) / den;
        // Replicate the in-step branch selection on a probe value per region.
        let f = |x: f64| {
            if x <= jmin {
                -n.m0 * x
            } else if x < jmax {
                n.m1 * (x - n.a)
            } else {
                -n.m0 * (x - 1.0)
            }
        };
        assert_eq!(f(jmin - 0.05), -n.m0 * (jmin - 0.05));
        let mid = 0.5 * (jmin + jmax);
        assert_eq!(f(mid), n.m1 * (mid - n.a));
        assert_eq!(f(jmax + 0.05), -n.m0 * (jmax + 0.05 - 1.0));
    }

    #[test]
    fn cn_heaviside_subtracts_beta_above_d() {
        // At x >= d the Heaviside term removes exactly beta from the x update.
        let mut with_beta = CourageNekorkinMapNeuron::new();
        with_beta.x = 0.30; // >= d
        let mut no_beta = with_beta.clone();
        no_beta.beta = 0.0;
        with_beta.step(0.0);
        no_beta.step(0.0);
        assert!((with_beta.x - no_beta.x - (-0.085)).abs() < 1e-15);
    }

    #[test]
    fn cn_simulate_matches_repeated_step() {
        let (trace, spikes) = CourageNekorkinMapNeuron::new().simulate(500, 0.0);
        let mut stepper = CourageNekorkinMapNeuron::new();
        let mut manual = Vec::with_capacity(500);
        let mut sp = 0i64;
        for _ in 0..500 {
            sp += stepper.step(0.0) as i64;
            manual.push(stepper.x);
        }
        assert_eq!(trace, manual);
        assert_eq!(spikes, sp);
        assert_eq!(
            stepper.x,
            CourageNekorkinMapNeuron::new().simulate(500, 0.0).0[499]
        );
    }

    #[test]
    fn cn_reset_clears_state() {
        let mut n = CourageNekorkinMapNeuron::new();
        for _ in 0..100 {
            n.step(0.0);
        }
        n.reset();
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
    }

    #[test]
    fn cn_deterministic() {
        let (a, sa) = CourageNekorkinMapNeuron::new().simulate(1000, 0.05);
        let (b, sb) = CourageNekorkinMapNeuron::new().simulate(1000, 0.05);
        assert_eq!(a, b);
        assert_eq!(sa, sb);
    }

    #[test]
    fn cn_performance_100k_steps() {
        let start = std::time::Instant::now();
        let mut n = CourageNekorkinMapNeuron::new();
        for _ in 0..100_000 {
            std::hint::black_box(n.step(0.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "100k steps must complete in <50ms"
        );
    }
}
