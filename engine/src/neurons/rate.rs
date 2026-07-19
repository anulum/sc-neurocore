// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rate-based models, synaptic plasticity neurons, and

//! Rate-based models, synaptic plasticity neurons, and other special types.

/// McCulloch-Pitts 1943 — excitatory-count threshold with absolute inhibition.
#[derive(Clone, Debug)]
pub struct McCullochPittsNeuron {
    pub theta: i32,
}

impl McCullochPittsNeuron {
    /// Construct a source-faithful neuron with a positive afferent-count threshold.
    pub fn new(theta: i32) -> Result<Self, String> {
        if theta <= 0 {
            return Err("theta must be a positive signed 32-bit integer".into());
        }
        Ok(Self { theta })
    }

    /// Revalidate the public fixed threshold before any execution boundary.
    pub fn validate(&self) -> Result<(), String> {
        if self.theta <= 0 {
            return Err("theta must be a positive signed 32-bit integer".into());
        }
        Ok(())
    }

    /// Evaluate one preceding-instant afferent pattern without cell state.
    pub fn try_step(&self, excitatory_count: i32, inhibitory_active: bool) -> Result<i32, String> {
        self.validate()?;
        if excitatory_count < 0 {
            return Err("excitatory_count must be a non-negative signed 32-bit integer".into());
        }
        Ok(i32::from(
            !inhibitory_active && excitatory_count >= self.theta,
        ))
    }
}
impl Default for McCullochPittsNeuron {
    fn default() -> Self {
        Self::new(1).expect("the default McCulloch-Pitts threshold is valid")
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
    /// Construct the maintained factory-default rate unit.
    pub fn new() -> Self {
        Self::with_parameters(0.0, 10.0, 1.0, 0.0, 0.1)
            .expect("the factory-default sigmoid-rate contract is valid")
    }

    /// Construct a fully configurable, validated sigmoid-rate unit.
    pub fn with_parameters(
        r: f64,
        tau: f64,
        beta: f64,
        theta: f64,
        dt: f64,
    ) -> Result<Self, String> {
        let neuron = Self {
            r,
            tau,
            beta,
            theta,
            dt,
        };
        neuron.validate()?;
        Ok(neuron)
    }

    /// Validate the complete mutable numeric contract.
    pub fn validate(&self) -> Result<(), String> {
        if !self.r.is_finite()
            || !(0.0..=1.0).contains(&self.r)
            || !self.tau.is_finite()
            || self.tau <= 0.0
            || !self.beta.is_finite()
            || !self.theta.is_finite()
            || !self.dt.is_finite()
            || self.dt <= 0.0
        {
            return Err(
                "sigmoid-rate state and parameters must be finite, with r in [0,1] and positive tau/dt"
                    .into(),
            );
        }
        Ok(())
    }

    /// Advance one step, preserving the previous state when validation fails.
    pub fn try_step(&mut self, current: f64) -> Result<f64, String> {
        self.validate()?;
        if !current.is_finite() {
            return Err("sigmoid-rate current must be finite".into());
        }
        let target = stable_sigmoid(self.beta, current, self.theta)?;
        let decay = (-self.dt / self.tau).exp();
        let candidate = decay * self.r + (1.0 - decay) * target;
        if !candidate.is_finite() || !(0.0..=1.0).contains(&candidate) {
            return Err("sigmoid-rate exact relaxation left the finite unit interval".into());
        }
        self.r = candidate;
        Ok(candidate)
    }

    /// Advance one step through the legacy non-throwing engine boundary.
    pub fn step(&mut self, current: f64) -> f64 {
        self.try_step(current).unwrap_or(self.r)
    }

    /// Restore the dynamic rate state without changing configured parameters.
    pub fn reset(&mut self) {
        self.r = 0.0;
    }
}

fn stable_sigmoid(beta: f64, current: f64, theta: f64) -> Result<f64, String> {
    let argument = beta * (current - theta);
    if argument.is_infinite() {
        return Ok(if argument.is_sign_positive() {
            1.0
        } else {
            0.0
        });
    }
    if !argument.is_finite() {
        return Err("sigmoid-rate transfer argument must be finite or saturating".into());
    }
    if argument >= 0.0 {
        Ok(1.0 / (1.0 + (-argument).exp()))
    } else {
        let exponential = argument.exp();
        Ok(exponential / (1.0 + exponential))
    }
}
impl Default for SigmoidRateNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Threshold-linear continuous-rate transfer with cached output.
#[derive(Clone, Debug)]
pub struct ThresholdLinearRateNeuron {
    pub r: f64,
    pub theta: f64,
    pub gain: f64,
}

impl ThresholdLinearRateNeuron {
    /// Construct the maintained factory-default transfer.
    pub fn new() -> Self {
        Self::with_parameters(0.0, 0.0, 1.0)
            .expect("the factory-default threshold-linear contract is valid")
    }

    /// Construct a fully configurable, validated transfer.
    pub fn with_parameters(r: f64, theta: f64, gain: f64) -> Result<Self, String> {
        let neuron = Self { r, theta, gain };
        neuron.validate()?;
        Ok(neuron)
    }

    /// Validate the complete mutable numeric contract.
    pub fn validate(&self) -> Result<(), String> {
        if !self.r.is_finite()
            || self.r < 0.0
            || !self.theta.is_finite()
            || !self.gain.is_finite()
            || self.gain < 0.0
        {
            return Err(
                "threshold-linear rate state and parameters must be finite, with non-negative r/gain"
                    .into(),
            );
        }
        Ok(())
    }

    /// Evaluate one input, preserving the cached output on any failure.
    pub fn try_step(&mut self, current: f64) -> Result<f64, String> {
        self.validate()?;
        if !current.is_finite() {
            return Err("threshold-linear rate current must be finite".into());
        }
        let candidate = self.gain * (current - self.theta).max(0.0);
        if !candidate.is_finite() || candidate < 0.0 {
            return Err("threshold-linear rate output must remain finite and non-negative".into());
        }
        self.r = candidate;
        Ok(candidate)
    }

    /// Evaluate one input through the legacy non-throwing engine boundary.
    pub fn step(&mut self, current: f64) -> f64 {
        self.try_step(current).unwrap_or(self.r)
    }

    /// Clear the cached output without changing threshold or gain.
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
        let j_pump = self.v_serca * self.ca.powi(2) / (self.ca.powi(2) + self.k_er.powi(2));
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
        self.x += self.dt / tau * (-self.x + f_target);
        if self.x >= self.v_threshold {
            self.x = 0.0;
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
    pub e_exc: f64,
    pub e_inh: f64,
    pub c_m: f64,
    pub mg: f64,
    pub tau_ampa: f64,
    pub tau_nmda: f64,
    pub tau_x: f64,
    pub alpha_nmda: f64,
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
            e_exc: 0.0,
            e_inh: -70.0,
            c_m: 0.5,
            mg: 1.0,
            tau_ampa: 2.0,
            tau_nmda: 100.0,
            tau_x: 2.0,
            alpha_nmda: 0.5,
            v_threshold: -50.0,
            v_reset: -55.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64, spike_in: bool) -> i32 {
        if spike_in {
            self.s_ampa += 1.0;
            self.x_nmda += 1.0;
        }
        self.s_ampa *= (-self.dt / self.tau_ampa).exp();
        self.s_nmda += (-self.s_nmda / self.tau_nmda
            + self.alpha_nmda * self.x_nmda * (1.0 - self.s_nmda))
            * self.dt;
        self.x_nmda *= (-self.dt / self.tau_x).exp();
        self.s_gaba *= (-self.dt / 5.0).exp();
        let mg_block = 1.0 / (1.0 + self.mg / 3.57 * (-0.062 * self.v).exp());
        let i_l = self.g_l * (self.v - self.e_l);
        let i_ampa = self.g_ampa * self.s_ampa * (self.v - self.e_exc);
        let i_nmda = self.g_nmda * mg_block * self.s_nmda * (self.v - self.e_exc);
        let i_gaba = self.g_gaba * self.s_gaba * (self.v - self.e_inh);
        self.v += (-i_l - i_ampa - i_nmda - i_gaba + current) / self.c_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.s_gaba += 1.0;
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
        let ks = self.buffer.len();
        self.buffer[self.ptr % ks] = current;
        self.ptr += 1;
        let n = self.ptr.min(ks);
        let score: f64 = self.kernel[..n]
            .iter()
            .zip(self.buffer[..n].iter())
            .map(|(&w, &b)| w * b)
            .sum();
        if score >= self.v_threshold {
            self.buffer.fill(0.0);
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
    _max_hist: usize,
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
            _max_hist: max_hist,
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
        let sigma = current.abs().max(1e-6) * 0.1;
        let upper = (self.v_threshold - mu) / sigma;
        let lower = (self.v_reset - mu) / sigma;
        // Gauss-Legendre 20-point quadrature for ∫ exp(u²)(1+erf(u)) du
        let nodes = [
            -0.993128599185095,
            -0.963971927277914,
            -0.912234428251326,
            -0.839116971822219,
            -0.746331906460151,
            -0.636053680726515,
            -0.510867001950827,
            -0.373706088715420,
            -0.227785851141645,
            -0.076526521133497,
            0.076526521133497,
            0.227785851141645,
            0.373706088715420,
            0.510867001950827,
            0.636053680726515,
            0.746331906460151,
            0.839116971822219,
            0.912234428251326,
            0.963971927277914,
            0.993128599185095,
        ];
        let weights = [
            0.017614007139152,
            0.040601429800387,
            0.062672048334109,
            0.083276741576704,
            0.101930119817240,
            0.118194531961518,
            0.131688638449177,
            0.142096109318382,
            0.149172986472604,
            0.152753387130726,
            0.152753387130726,
            0.149172986472604,
            0.142096109318382,
            0.131688638449177,
            0.118194531961518,
            0.101930119817240,
            0.083276741576704,
            0.062672048334109,
            0.040601429800387,
            0.017614007139152,
        ];
        let half = (upper - lower) / 2.0;
        let mid = (upper + lower) / 2.0;
        let mut integral = 0.0;
        for (&node, &w) in nodes.iter().zip(weights.iter()) {
            let u = mid + half * node;
            let eu2 = (u * u).min(50.0).exp();
            let erf_u = Self::erf_approx(u);
            integral += w * eu2 * (1.0 + erf_u);
        }
        integral *= half;
        let t_isi = self.tau_rp + self.tau_m * std::f64::consts::PI.sqrt() * integral;
        1000.0 / t_isi.max(0.01)
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
        // Exponential Mexican hat kernel, rolled to match Python's np.roll
        for i in 0..n {
            let x = ((i as isize - n as isize / 2).unsigned_abs() as f64) * dx;
            w[i] = a_exc * (-a_width * x).exp() - b_inh * (-b_width * x).exp();
        }
        // Roll by -n/2 to match Python's np.roll(k, -n//2)
        w.rotate_left(n / 2);
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
        // f(u) = max(0, u)
        let f_u: Vec<f64> = self.u.iter().map(|&v| v.max(0.0)).collect();
        // Circular convolution (matching Python FFT-based conv)
        let mut conv = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                let idx = (i + n - j) % n;
                s += self.w[idx] * f_u[j];
            }
            conv[i] = s * self.dx;
        }
        for i in 0..n {
            let inp = if i < input.len() { input[i] } else { 0.0 };
            self.u[i] += (-self.u[i] + conv[i] + inp) / self.tau * self.dt;
        }
        // Return mean of ReLU activations
        self.u.iter().map(|&v| v.max(0.0)).sum::<f64>() / n as f64
    }
    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

/// Leaky Compete-and-Fire — winner-take-all with lateral inhibition. Oster et al. 2009.
#[derive(Clone, Debug)]
pub struct LeakyCompeteFireNeuron {
    pub v: Vec<f64>,
    pub n_units: usize,
    pub tau: f64,
    pub v_threshold: f64,
    pub w_inh: f64,
    pub dt: f64,
}

impl LeakyCompeteFireNeuron {
    pub fn new(n_units: usize) -> Self {
        Self {
            v: vec![0.0; n_units],
            n_units,
            tau: 10.0,
            v_threshold: 1.0,
            w_inh: 0.5,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, currents: &[f64]) -> Vec<i32> {
        let n = self.n_units;
        for i in 0..n {
            let c = if i < currents.len() { currents[i] } else { 0.0 };
            self.v[i] += (-self.v[i] + c) / self.tau * self.dt;
        }
        let mut spikes = vec![0i32; n];
        for i in 0..n {
            if self.v[i] >= self.v_threshold {
                spikes[i] = 1;
                self.v[i] = 0.0;
                for j in 0..n {
                    if j != i {
                        self.v[j] = (self.v[j] - self.w_inh).max(0.0);
                    }
                }
            }
        }
        spikes
    }

    pub fn reset(&mut self) {
        self.v.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_threshold() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.try_step(2, false), Ok(1));
        assert_eq!(n.try_step(0, false), Ok(0));
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

    // ── Multi-angle tests for rate models ──

    // -- McCullochPitts --
    #[test]
    fn mcp_below_threshold() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.try_step(0, false), Ok(0));
    }
    #[test]
    fn mcp_absolute_inhibition_vetoes_maximum_excitation() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.try_step(i32::MAX, true), Ok(0));
    }
    #[test]
    fn mcp_theta_two_is_and() {
        let n = McCullochPittsNeuron::new(2).unwrap();
        assert_eq!(n.try_step(0, false), Ok(0));
        assert_eq!(n.try_step(1, false), Ok(0));
        assert_eq!(n.try_step(2, false), Ok(1));
    }
    #[test]
    fn mcp_rejects_non_positive_thresholds() {
        assert!(McCullochPittsNeuron::new(0).is_err());
        assert!(McCullochPittsNeuron::new(-1).is_err());
    }
    #[test]
    fn mcp_rejects_negative_excitation() {
        let n = McCullochPittsNeuron::default();
        assert!(n.try_step(-1, false).is_err());
    }
    #[test]
    fn mcp_revalidates_public_threshold_mutation() {
        let n = McCullochPittsNeuron { theta: 0 };
        assert!(n.try_step(1, false).is_err());
    }
    #[test]
    fn mcp_is_stateless_across_history() {
        let n = McCullochPittsNeuron::new(2).unwrap();
        let outputs: Vec<i32> = [2, 0, 2, 0]
            .into_iter()
            .map(|count| n.try_step(count, false).unwrap())
            .collect();
        assert_eq!(outputs, vec![1, 0, 1, 0]);
    }

    // -- SigmoidRate --
    #[test]
    fn sigmoid_rate_matches_python_exact_relaxation_golden() {
        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5).unwrap();
        let expected = [
            0.2857007338135623,
            0.3196603222932904,
            0.3519636820991432,
            0.38269158845670403,
            0.41192087713731845,
            0.43972463658754457,
        ];
        for target in expected {
            let rate = neuron.try_step(3.0).unwrap();
            assert!((rate - target).abs() <= 2.0e-15, "{rate} != {target}");
        }
    }

    #[test]
    fn sigmoid_rate_exact_relaxation_is_bounded_for_large_timestep() {
        let mut neuron = SigmoidRateNeuron::with_parameters(1.0, 0.1, 1.0, 0.0, 5.0).unwrap();
        let rate = neuron.try_step(-100.0).unwrap();
        assert!((rate - 1.9287498479639178e-22).abs() <= 1.0e-36);
        assert!((0.0..=1.0).contains(&rate));
    }

    #[test]
    fn sigmoid_rate_rejects_invalid_contract_without_mutation() {
        let invalid_contracts = [
            (-0.1, 10.0, 1.0, 0.0, 0.1),
            (1.1, 10.0, 1.0, 0.0, 0.1),
            (0.0, 0.0, 1.0, 0.0, 0.1),
            (0.0, 10.0, f64::NAN, 0.0, 0.1),
            (0.0, 10.0, 1.0, f64::INFINITY, 0.1),
            (0.0, 10.0, 1.0, 0.0, -0.1),
        ];
        for (r, tau, beta, theta, dt) in invalid_contracts {
            assert!(SigmoidRateNeuron::with_parameters(r, tau, beta, theta, dt).is_err());
        }

        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5).unwrap();
        let before = neuron.r;
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!(neuron.r, before);
        neuron.tau = 0.0;
        assert!(neuron.try_step(3.0).is_err());
        assert_eq!(neuron.r, before);
    }

    #[test]
    fn sigmoid_rate_saturates_extreme_finite_drive() {
        let mut high = SigmoidRateNeuron::with_parameters(0.0, 10.0, 1.0e308, 0.0, 0.1).unwrap();
        let mut low = high.clone();
        assert!(high.try_step(1.0e308).unwrap() > 0.0);
        assert_eq!(low.try_step(-1.0e308).unwrap(), 0.0);
    }

    #[test]
    fn sigmoid_rate_reset_preserves_configuration() {
        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 7.0, 2.5, -0.4, 0.2).unwrap();
        neuron.try_step(3.0).unwrap();
        neuron.reset();
        assert_eq!(neuron.r, 0.0);
        assert_eq!(
            (neuron.tau, neuron.beta, neuron.theta, neuron.dt),
            (7.0, 2.5, -0.4, 0.2)
        );
    }

    #[test]
    fn sigmoid_rate_legacy_step_fails_closed() {
        let mut neuron = SigmoidRateNeuron::with_parameters(0.25, 10.0, 2.0, 1.0, 0.5).unwrap();
        assert_eq!(neuron.step(f64::NAN), 0.25);
        assert_eq!(neuron.r, 0.25);
    }

    // -- ThresholdLinear --
    #[test]
    fn tl_rate_reset() {
        let mut n = ThresholdLinearRateNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.r - 0.0).abs() < 1e-10);
    }
    #[test]
    fn tl_rate_nan_no_panic() {
        let mut neuron = ThresholdLinearRateNeuron::with_parameters(0.25, 0.5, 2.0).unwrap();
        assert_eq!(neuron.step(f64::NAN), 0.25);
        assert_eq!(neuron.r, 0.25);
    }
    #[test]
    fn tl_rate_below_threshold() {
        let mut n = ThresholdLinearRateNeuron::new();
        assert!(n.step(-5.0) == 0.0, "below threshold → zero rate");
    }

    #[test]
    fn tl_rate_configured_transfer_and_reset() {
        let mut neuron = ThresholdLinearRateNeuron::with_parameters(0.25, 1.5, 2.0).unwrap();
        assert_eq!(neuron.try_step(3.0), Ok(3.0));
        neuron.reset();
        assert_eq!((neuron.r, neuron.theta, neuron.gain), (0.0, 1.5, 2.0));
    }

    #[test]
    fn tl_rate_rejects_invalid_contract_without_mutation() {
        assert!(ThresholdLinearRateNeuron::with_parameters(-1.0, 0.0, 1.0).is_err());
        assert!(ThresholdLinearRateNeuron::with_parameters(0.0, f64::NAN, 1.0).is_err());
        assert!(ThresholdLinearRateNeuron::with_parameters(0.0, 0.0, -1.0).is_err());
        let mut neuron = ThresholdLinearRateNeuron::with_parameters(0.25, 0.0, 1.0e308).unwrap();
        assert!(neuron.try_step(1.0e308).is_err());
        assert_eq!(neuron.r, 0.25);
    }

    // -- Astrocyte --
    #[test]
    fn astrocyte_reset() {
        let mut n = AstrocyteModel::new();
        for _ in 0..1000 {
            n.step(0.1);
        }
        n.reset();
        assert!((n.ca - 0.05).abs() < 1e-10);
    }
    #[test]
    fn astrocyte_nan_no_panic() {
        AstrocyteModel::new().step(f64::NAN);
    }
    #[test]
    fn astrocyte_ca_nonneg() {
        let mut n = AstrocyteModel::new();
        for _ in 0..5000 {
            n.step(0.1);
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative");
    }

    // -- TsodyksMarkram --
    #[test]
    fn tm_reset() {
        let mut n = TsodyksMarkramNeuron::new();
        for _ in 0..100 {
            n.step(50.0, false);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.x - 1.0).abs() < 1e-10);
    }
    #[test]
    fn tm_bounded() {
        let mut n = TsodyksMarkramNeuron::new();
        for _ in 0..1000 {
            n.step(1e4, false);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn tm_nan_no_panic() {
        TsodyksMarkramNeuron::new().step(f64::NAN, false);
    }
    #[test]
    fn tm_stp_depression() {
        let mut n = TsodyksMarkramNeuron::new();
        for _ in 0..500 {
            n.step(50.0, true);
        }
        // With repeated presynaptic spikes, x (available fraction) should decrease
        assert!(
            n.x < 1.0,
            "STP depression: x should be < 1.0 after spikes: {}",
            n.x
        );
    }

    // -- LiquidTimeConstant --
    #[test]
    fn ltc_reset() {
        let mut n = LiquidTimeConstantNeuron {
            v_threshold: 0.9,
            ..LiquidTimeConstantNeuron::new()
        };
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.x - 0.0).abs() < 1e-10);
    }
    #[test]
    fn ltc_nan_no_panic() {
        LiquidTimeConstantNeuron::new().step(f64::NAN);
    }

    // -- CompteWM --
    #[test]
    fn compte_reset() {
        let mut n = CompteWMNeuron::new();
        for _ in 0..100 {
            n.step(5.0, false);
        }
        n.reset();
        assert!((n.v - n.e_l).abs() < 1e-10);
    }
    #[test]
    fn compte_nan_no_panic() {
        CompteWMNeuron::new().step(f64::NAN, false);
    }

    // -- ParallelSpiking --
    #[test]
    fn psn_reset() {
        let mut n = ParallelSpikingNeuron::new(4, 0.5);
        for _ in 0..20 {
            n.step(1.0);
        }
        n.reset();
    }
    #[test]
    fn psn_nan_no_panic() {
        ParallelSpikingNeuron::new(4, 0.5).step(f64::NAN);
    }

    // -- FractionalLIF --
    #[test]
    fn frac_lif_reset() {
        let mut n = FractionalLIFNeuron::new(0.8, 50);
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn frac_lif_nan_no_panic() {
        FractionalLIFNeuron::new(0.8, 50).step(f64::NAN);
    }

    // -- Siegert --
    #[test]
    fn siegert_zero() {
        let n = SiegertTransferFunction::new();
        let r = n.step(0.0);
        assert!(r >= 0.0);
    }
    #[test]
    fn siegert_nan_no_panic() {
        SiegertTransferFunction::new().step(f64::NAN);
    }

    // -- Amari --
    #[test]
    fn amari_reset() {
        let mut n = AmariNeuralField::new(32);
        let inp = vec![0.5; 32];
        for _ in 0..100 {
            n.step(&inp);
        }
        n.reset();
        assert!(n.u.iter().all(|&x| x == 0.0));
    }
    #[test]
    fn amari_bounded() {
        let mut n = AmariNeuralField::new(16);
        let inp = vec![1e3; 16];
        for _ in 0..1000 {
            n.step(&inp);
        }
        assert!(n.u.iter().all(|x| x.is_finite()));
    }

    // -- LeakyCompeteFireNeuron tests --

    #[test]
    fn lcf_fires_with_strong_input() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        let inp = vec![5.0, 0.0, 0.0];
        let mut any_spike = false;
        for _ in 0..200 {
            let spikes = n.step(&inp);
            if spikes.contains(&1) {
                any_spike = true;
            }
        }
        assert!(any_spike, "LeakyCompeteFire should fire with strong input");
    }

    #[test]
    fn lcf_silent_without_input() {
        let mut n = LeakyCompeteFireNeuron::new(4);
        let inp = vec![0.0; 4];
        for _ in 0..200 {
            let spikes = n.step(&inp);
            assert!(
                spikes.iter().all(|&s| s == 0),
                "should be silent at zero input"
            );
        }
    }

    #[test]
    fn lcf_winner_take_all() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        // Unit 0 receives strong input, others receive moderate
        let inp = vec![5.0, 2.0, 2.0];
        let mut spike_counts = [0i32; 3];
        for _ in 0..1000 {
            let spikes = n.step(&inp);
            for (i, &s) in spikes.iter().enumerate() {
                spike_counts[i] += s;
            }
        }
        // Winner (unit 0) should spike more than losers due to lateral inhibition
        assert!(
            spike_counts[0] > spike_counts[1],
            "unit 0 ({}) should spike more than unit 1 ({}) — winner-take-all",
            spike_counts[0],
            spike_counts[1]
        );
    }

    #[test]
    fn lcf_lateral_inhibition_suppresses() {
        let mut n = LeakyCompeteFireNeuron::new(2);
        n.w_inh = 2.0; // Strong inhibition
        let inp = vec![3.0, 3.0];
        let mut spike_counts = [0i32; 2];
        for _ in 0..500 {
            let spikes = n.step(&inp);
            for (i, &s) in spikes.iter().enumerate() {
                spike_counts[i] += s;
            }
        }
        // With equal input + strong inhibition, total spikes should be less
        // than with no inhibition (competitive suppression)
        let mut n_no_inh = LeakyCompeteFireNeuron::new(2);
        n_no_inh.w_inh = 0.0;
        let mut total_no_inh = 0i32;
        for _ in 0..500 {
            let spikes = n_no_inh.step(&inp);
            total_no_inh += spikes.iter().sum::<i32>();
        }
        let total_inh: i32 = spike_counts.iter().sum();
        assert!(
            total_inh <= total_no_inh,
            "inhibition ({}) should reduce total spikes vs no inhibition ({})",
            total_inh,
            total_no_inh
        );
    }

    #[test]
    fn lcf_reset_clears_state() {
        let mut n = LeakyCompeteFireNeuron::new(4);
        let inp = vec![3.0; 4];
        for _ in 0..100 {
            n.step(&inp);
        }
        n.reset();
        assert!(
            n.v.iter().all(|&x| x == 0.0),
            "reset must zero all voltages"
        );
    }

    #[test]
    fn lcf_voltages_bounded() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        let inp = vec![1e6, 1e6, 1e6];
        for _ in 0..1000 {
            n.step(&inp);
        }
        assert!(
            n.v.iter().all(|x| x.is_finite()),
            "voltages must stay finite under extreme input"
        );
    }

    #[test]
    fn lcf_negative_input_no_crash() {
        let mut n = LeakyCompeteFireNeuron::new(3);
        let inp = vec![-10.0, -5.0, -1.0];
        for _ in 0..500 {
            n.step(&inp);
        }
        assert!(
            n.v.iter().all(|x| x.is_finite()),
            "must handle negative input"
        );
    }

    #[test]
    fn lcf_output_length_matches_units() {
        let n_units = 7;
        let mut n = LeakyCompeteFireNeuron::new(n_units);
        let inp = vec![1.0; n_units];
        let spikes = n.step(&inp);
        assert_eq!(spikes.len(), n_units, "output length must match n_units");
    }
}
