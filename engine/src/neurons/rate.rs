// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rate-based models, synaptic plasticity neurons, and other special types.

/// McCulloch-Pitts 1943 — binary threshold unit.
#[derive(Clone, Debug)]
pub struct McCullochPittsNeuron {
    pub theta: f64,
}

impl McCullochPittsNeuron {
    pub fn new(theta: f64) -> Self {
        Self { theta }
    }
    pub fn step(&self, weighted_input: f64) -> i32 {
        if weighted_input >= self.theta {
            1
        } else {
            0
        }
    }
}
impl Default for McCullochPittsNeuron {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Sigmoid rate neuron — Wilson-Cowan-style single unit.
#[derive(Clone, Debug)]
pub struct SigmoidRateNeuron {
    pub r: f64,
    pub tau: f64,
    pub beta: f64,
    pub theta: f64,
    pub dt: f64,
}

impl SigmoidRateNeuron {
    pub fn new() -> Self {
        Self {
            r: 0.0,
            tau: 10.0,
            beta: 1.0,
            theta: 0.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> f64 {
        let sigma = 1.0 / (1.0 + (-self.beta * (current - self.theta)).exp());
        self.r += (-self.r + sigma) / self.tau * self.dt;
        self.r
    }
    pub fn reset(&mut self) {
        self.r = 0.0;
    }
}
impl Default for SigmoidRateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Threshold-linear rate neuron — ReLU-like firing curve.
#[derive(Clone, Debug)]
pub struct ThresholdLinearRateNeuron {
    pub r: f64,
    pub theta: f64,
    pub gain: f64,
}

impl ThresholdLinearRateNeuron {
    pub fn new() -> Self {
        Self {
            r: 0.0,
            theta: 0.0,
            gain: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> f64 {
        self.r = self.gain * (current - self.theta).max(0.0);
        self.r
    }
    pub fn reset(&mut self) {
        self.r = 0.0;
    }
}
impl Default for ThresholdLinearRateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Li-Rinzel IP3R astrocyte model — Ca²⁺ dynamics.
#[derive(Clone, Debug)]
pub struct AstrocyteModel {
    pub ca: f64,
    pub h: f64,
    pub ip3: f64,
    pub v_er: f64,
    pub k_er: f64,
    pub v_serca: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d5: f64,
    pub c0: f64,
    pub c1: f64,
    pub dt: f64,
}

impl AstrocyteModel {
    pub fn new() -> Self {
        Self {
            ca: 0.05,
            h: 0.8,
            ip3: 0.5,
            v_er: 0.9,
            k_er: 0.15,
            v_serca: 0.4,
            d1: 0.13,
            d2: 1.049,
            d3: 0.9434,
            d5: 0.08234,
            c0: 2.0,
            c1: 0.185,
            dt: 0.01,
        }
    }
    pub fn step(&mut self, current: f64) -> f64 {
        let ca_er = (self.c0 - self.ca) / self.c1;
        let m_inf = self.ip3 / (self.ip3 + self.d1);
        let n_inf = self.ca / (self.ca + self.d5);
        let j_chan = self.v_er * (m_inf * n_inf * self.h).powi(3) * (ca_er - self.ca);
        let j_leak = self.k_er * (ca_er - self.ca);
        let j_pump = self.v_serca * self.ca.powi(2) / (self.ca.powi(2) + 0.1_f64.powi(2));
        let q2 = self.d2 * (self.ip3 + self.d1) / (self.ip3 + self.d3);
        let h_inf = q2 / (q2 + self.ca);
        let tau_h = 1.0 / (0.2 * (q2 + self.ca));
        self.ca += (j_chan + j_leak - j_pump + current) * self.dt;
        self.ca = self.ca.max(0.0);
        self.h += (h_inf - self.h) / tau_h * self.dt;
        self.ca
    }
    pub fn reset(&mut self) {
        self.ca = 0.05;
        self.h = 0.8;
        self.ip3 = 0.5;
    }
}
impl Default for AstrocyteModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Tsodyks-Markram 1997 — LIF with short-term synaptic plasticity.
#[derive(Clone, Debug)]
pub struct TsodyksMarkramNeuron {
    pub v: f64,
    pub x: f64,
    pub u: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_d: f64,
    pub tau_f: f64,
    pub u_se: f64,
    pub a_se: f64,
    pub r_m: f64,
    pub dt: f64,
}

impl TsodyksMarkramNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            x: 1.0,
            u: 0.2,
            v_rest: -65.0,
            v_reset: -65.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            tau_d: 200.0,
            tau_f: 600.0,
            u_se: 0.2,
            a_se: 50.0,
            r_m: 1.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64, presynaptic_spike: bool) -> i32 {
        self.x += (1.0 - self.x) / self.tau_d * self.dt;
        self.u += (self.u_se - self.u) / self.tau_f * self.dt;
        let mut i_syn = 0.0;
        if presynaptic_spike {
            self.u += self.u_se * (1.0 - self.u);
            i_syn = self.a_se * self.u * self.x;
            self.x -= self.u * self.x;
        }
        self.v += (-(self.v - self.v_rest) + self.r_m * (i_syn + current)) / self.tau_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.x = 1.0;
        self.u = self.u_se;
    }
}
impl Default for TsodyksMarkramNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Liquid Time-Constant neuron — input-dependent time constant. Hasani et al. 2021.
#[derive(Clone, Debug)]
pub struct LiquidTimeConstantNeuron {
    pub x: f64,
    pub tau_base: f64,
    pub w_tau: f64,
    pub w_x: f64,
    pub w_in: f64,
    pub bias: f64,
    pub v_threshold: f64,
    pub dt: f64,
}

impl LiquidTimeConstantNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            tau_base: 10.0,
            w_tau: -0.5,
            w_x: 0.8,
            w_in: 1.0,
            bias: 0.0,
            v_threshold: 1.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let sigma_tau = 1.0 / (1.0 + (-(self.w_tau * current + self.bias)).exp());
        let tau = (self.tau_base * sigma_tau).max(0.1);
        let f_target = (self.w_x * self.x + self.w_in * current).tanh();
        let decay = (-self.dt / tau).exp();
        self.x = self.x * decay + f_target * (1.0 - decay);
        if self.x >= self.v_threshold {
            self.x -= self.v_threshold;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.x = 0.0;
    }
}
impl Default for LiquidTimeConstantNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Compte WM — NMDA-based working-memory neuron. Compte et al. 2000.
#[derive(Clone, Debug)]
pub struct CompteWMNeuron {
    pub v: f64,
    pub s_ampa: f64,
    pub s_nmda: f64,
    pub x_nmda: f64,
    pub s_gaba: f64,
    pub g_l: f64,
    pub g_ampa: f64,
    pub g_nmda: f64,
    pub g_gaba: f64,
    pub e_l: f64,
    pub e_ampa: f64,
    pub e_nmda: f64,
    pub e_gaba: f64,
    pub c_m: f64,
    pub mg: f64,
    pub tau_ampa: f64,
    pub tau_nmda_rise: f64,
    pub tau_nmda_decay: f64,
    pub tau_gaba: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl CompteWMNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            s_ampa: 0.0,
            s_nmda: 0.0,
            x_nmda: 0.0,
            s_gaba: 0.0,
            g_l: 0.025,
            g_ampa: 0.005,
            g_nmda: 0.165,
            g_gaba: 0.013,
            e_l: -70.0,
            e_ampa: 0.0,
            e_nmda: 0.0,
            e_gaba: -70.0,
            c_m: 0.5,
            mg: 1.0,
            tau_ampa: 2.0,
            tau_nmda_rise: 2.0,
            tau_nmda_decay: 100.0,
            tau_gaba: 5.0,
            v_threshold: -50.0,
            v_reset: -55.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64, spike_in: bool) -> i32 {
        let mg_block = 1.0 / (1.0 + self.mg * (-0.062 * self.v).exp() / 3.57);
        let i_l = self.g_l * (self.v - self.e_l);
        let i_ampa = self.g_ampa * self.s_ampa * (self.v - self.e_ampa);
        let i_nmda = self.g_nmda * self.s_nmda * mg_block * (self.v - self.e_nmda);
        let i_gaba = self.g_gaba * self.s_gaba * (self.v - self.e_gaba);
        self.v += (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m * self.dt;
        let spike_f = if spike_in { 1.0 } else { 0.0 };
        self.s_ampa += (-self.s_ampa / self.tau_ampa + spike_f) * self.dt;
        self.x_nmda += (-self.x_nmda / self.tau_nmda_rise + spike_f) * self.dt;
        self.s_nmda += (-self.s_nmda / self.tau_nmda_decay
            + 0.5 * self.x_nmda * (1.0 - self.s_nmda))
            * self.dt;
        self.s_gaba += (-self.s_gaba / self.tau_gaba + spike_f * 0.5) * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.s_ampa = 0.0;
        self.s_nmda = 0.0;
        self.x_nmda = 0.0;
        self.s_gaba = 0.0;
    }
}
impl Default for CompteWMNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel Spiking Neuron — convolution-based filter. Fang et al. 2023.
#[derive(Clone, Debug)]
pub struct ParallelSpikingNeuron {
    pub kernel: Vec<f64>,
    pub buffer: Vec<f64>,
    pub v_threshold: f64,
    ptr: usize,
}

impl ParallelSpikingNeuron {
    pub fn new(kernel_size: usize, v_threshold: f64) -> Self {
        let k = 1.0 / kernel_size as f64;
        Self {
            kernel: vec![k; kernel_size],
            buffer: vec![0.0; kernel_size],
            v_threshold,
            ptr: 0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.buffer[self.ptr] = current;
        self.ptr = (self.ptr + 1) % self.buffer.len();
        let v: f64 = self
            .kernel
            .iter()
            .enumerate()
            .map(|(i, &w)| w * self.buffer[(self.ptr + i) % self.buffer.len()])
            .sum();
        if v >= self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.ptr = 0;
    }
}

/// Fractional-order LIF — Grünwald-Letnikov approximation. Teka et al. 2014.
#[derive(Clone, Debug)]
pub struct FractionalLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub alpha: f64,
    pub resistance: f64,
    pub dt: f64,
    history: Vec<f64>,
    gl_coeffs: Vec<f64>,
    max_hist: usize,
}

impl FractionalLIFNeuron {
    pub fn new(alpha: f64, max_hist: usize) -> Self {
        let mut coeffs = vec![0.0; max_hist + 1];
        coeffs[0] = 1.0;
        for j in 1..=max_hist {
            coeffs[j] = coeffs[j - 1] * (1.0 - (alpha + 1.0) / j as f64);
        }
        Self {
            v: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            v_threshold: 1.0,
            alpha,
            resistance: 1.0,
            dt: 1.0,
            history: vec![0.0; max_hist],
            gl_coeffs: coeffs,
            max_hist,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        // Grünwald-Letnikov: D^α v ≈ (1/dt^α) Σ_j c_j v(t-j·dt)
        let mut gl_sum = 0.0;
        let n = self.history.len().min(self.gl_coeffs.len() - 1);
        for j in 0..n {
            gl_sum += self.gl_coeffs[j + 1] * self.history[n - 1 - j];
        }
        let rhs = -(self.v - self.v_rest) + self.resistance * current;
        self.v = rhs * self.dt.powf(self.alpha) - gl_sum;
        // Shift history
        let len = self.history.len();
        if len > 0 {
            for i in 0..len - 1 {
                self.history[i] = self.history[i + 1];
            }
            self.history[len - 1] = self.v;
        }
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.history.fill(0.0);
    }
}

/// Siegert transfer function — analytical stationary firing rate of a LIF neuron.
#[derive(Clone, Debug)]
pub struct SiegertTransferFunction {
    pub tau_m: f64,
    pub tau_rp: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub v_rest: f64,
}

impl SiegertTransferFunction {
    pub fn new() -> Self {
        Self {
            tau_m: 20.0,
            tau_rp: 2.0,
            v_threshold: -50.0,
            v_reset: -70.0,
            v_rest: -65.0,
        }
    }
    pub fn step(&self, current: f64) -> f64 {
        let mu = self.v_rest + current;
        let sigma: f64 = 3.0; // noise amplitude (fixed default)
        if sigma.abs() < 1e-10 {
            return if mu > self.v_threshold {
                1000.0 / self.tau_rp
            } else {
                0.0
            };
        }
        let upper = (self.v_threshold - mu) / sigma;
        let lower = (self.v_reset - mu) / sigma;
        // Gauss-Legendre 10-point quadrature for ∫ exp(u²)(1+erf(u)) du
        let nodes = [
            -0.973906528517172,
            -0.865063366688985,
            -0.679409568299024,
            -0.433395394129247,
            -0.148874338981631,
            0.148874338981631,
            0.433395394129247,
            0.679409568299024,
            0.865063366688985,
            0.973906528517172,
        ];
        let weights = [
            0.066671344308688,
            0.149451349150581,
            0.219086362515982,
            0.269266719309996,
            0.295524224714753,
            0.295524224714753,
            0.269266719309996,
            0.219086362515982,
            0.149451349150581,
            0.066671344308688,
        ];
        let half = (upper - lower) / 2.0;
        let mid = (upper + lower) / 2.0;
        let mut integral = 0.0;
        for (&node, &w) in nodes.iter().zip(weights.iter()) {
            let u = mid + half * node;
            let eu2 = (u * u).min(500.0).exp();
            let erf_u = Self::erf_approx(u);
            integral += w * eu2 * (1.0 + erf_u);
        }
        integral *= half;
        let rate = 1.0 / (self.tau_rp + self.tau_m * std::f64::consts::PI.sqrt() * integral);
        rate.max(0.0) * 1000.0
    }
    fn erf_approx(x: f64) -> f64 {
        // Abramowitz-Stegun approximation
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let result = 1.0 - poly * (-x * x).exp();
        if x >= 0.0 {
            result
        } else {
            -result
        }
    }
}
impl Default for SiegertTransferFunction {
    fn default() -> Self {
        Self::new()
    }
}

/// Amari neural field — continuous attractor with Mexican hat kernel.
#[derive(Clone, Debug)]
pub struct AmariNeuralField {
    pub u: Vec<f64>,
    pub n: usize,
    pub tau: f64,
    pub a_exc: f64,
    pub a_width: f64,
    pub b_inh: f64,
    pub b_width: f64,
    pub dx: f64,
    pub dt: f64,
    w: Vec<f64>,
}

impl AmariNeuralField {
    pub fn new(n: usize) -> Self {
        let dx = 0.5;
        let mut w = vec![0.0; n];
        let a_exc = 1.5;
        let a_width = 1.0;
        let b_inh = 0.75;
        let b_width = 2.0;
        for i in 0..n {
            let d = (i as f64 - n as f64 / 2.0) * dx;
            w[i] = a_exc * (-d * d / (2.0 * a_width * a_width)).exp()
                - b_inh * (-d * d / (2.0 * b_width * b_width)).exp();
        }
        Self {
            u: vec![0.0; n],
            n,
            tau: 10.0,
            a_exc,
            a_width,
            b_inh,
            b_width,
            dx,
            dt: 0.5,
            w,
        }
    }
    pub fn step(&mut self, input: &[f64]) -> f64 {
        let n = self.n;
        let mut du = vec![0.0; n];
        for i in 0..n {
            let s_i = if self.u[i] > 0.0 { self.u[i] } else { 0.0 };
            let mut conv = 0.0;
            for j in 0..n {
                let s_j = if self.u[j] > 0.0 { self.u[j] } else { 0.0 };
                let idx = ((i as i64 - j as i64).unsigned_abs() as usize).min(n - 1);
                conv += self.w[idx] * s_j * self.dx;
            }
            let inp = if i < input.len() { input[i] } else { 0.0 };
            du[i] = (-self.u[i] + conv + inp) / self.tau * self.dt;
            let _ = s_i; // used for convolution input
        }
        for i in 0..n {
            self.u[i] += du[i];
        }
        self.u.iter().sum::<f64>() / n as f64
    }
    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_threshold() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.step(2.0), 1);
        assert_eq!(n.step(0.5), 0);
    }
    #[test]
    fn sigmoid_rate() {
        let mut n = SigmoidRateNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        assert!(n.r > 0.5);
    }
    #[test]
    fn tl_rate() {
        let mut n = ThresholdLinearRateNeuron::new();
        assert!(n.step(5.0) > 0.0);
        assert!(n.step(-1.0) == 0.0);
    }
    #[test]
    fn astrocyte_ca() {
        let mut n = AstrocyteModel::new();
        let mut max_ca = 0.0_f64;
        for _ in 0..5000 {
            let c = n.step(0.1);
            max_ca = max_ca.max(c);
        }
        assert!(max_ca > 0.05);
    }
    #[test]
    fn tm_fires() {
        let mut n = TsodyksMarkramNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(50.0, false)).sum();
        assert!(t > 0);
    }
    #[test]
    fn ltc_fires() {
        let mut n = LiquidTimeConstantNeuron {
            v_threshold: 0.9,
            ..LiquidTimeConstantNeuron::new()
        };
        let t: i32 = (0..100).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn compte_fires() {
        let mut n = CompteWMNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0, false)).sum();
        assert!(t > 0);
    }
    #[test]
    fn psn_fires() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        let t: i32 = (0..20).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn frac_lif_fires() {
        let mut n = FractionalLIFNeuron::new(0.8, 50);
        let t: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn siegert_rate() {
        let n = SiegertTransferFunction::new();
        let r = n.step(20.0);
        assert!(r > 0.0);
    }
    #[test]
    fn amari_activates() {
        let mut n = AmariNeuralField::new(32);
        let inp = vec![0.5; 32];
        for _ in 0..100 {
            n.step(&inp);
        }
        assert!(n.u.iter().any(|&x| x.abs() > 0.01));
    }
}
