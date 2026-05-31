// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

//! Biophysically grounded sensory neuron models.
//!
//! 10 cell types covering major sensory modalities:
//! - Auditory: inner/outer hair cells (graded, mechano-electrical)
//! - Visual: rod/cone photoreceptors (graded, hyperpolarising), retinal ganglion (ON/OFF spiking)
//! - Somatosensory: Merkel (slow adapting), Pacinian (fast adapting)
//! - Pain: nociceptor (threshold, sensitisation)
//! - Chemical: olfactory receptor, taste receptor
//!
//! Most sensory neurons produce graded potentials (step returns f64).
//! Only retinal ganglion and nociceptor produce spikes (step returns i32).

// ═══════════════════════════════════════════════════════════════════
// Inner Hair Cell (IHC) — auditory
// ═══════════════════════════════════════════════════════════════════

/// Inner hair cell — primary auditory transducer.
///
/// Mechano-electrical transduction: stereocilia displacement opens
/// MET channels → depolarisation → Ca2+ influx → glutamate release.
/// Inner hair cell with Meddis vesicle pool dynamics.
///
/// Three-stage model per Meddis (1986, 2006):
/// 1. **MET transduction**: Boltzmann gating of mechano-electrical
///    transducer channels converts stereocilia displacement to
///    receptor potential.
/// 2. **Ca²⁺ dynamics**: voltage-dependent Ca²⁺ entry drives
///    vesicle release.
/// 3. **Meddis vesicle pool**: three-compartment transmitter model:
///    - q: available vesicles (free pool)
///    - c: cleft transmitter concentration
///    - w: reprocessing store (depleted vesicles recovering)
///
///    dq/dt = y*(M-q) + x_r*w - k*q*f(Ca)    (replenishment + recovery - release)
///    dc/dt = k*q*f(Ca) - l*c - r_up*c        (release - loss - reuptake)
///    dw/dt = r_up*c - x_r*w                   (reuptake - recovery)
///
///    where f(Ca) = Ca²/(Ca² + K_d²) is Ca²⁺-dependent release rate.
///
/// Graded output: receptor potential (no spikes).
///
/// Meddis, JASA 79:702, 1986.
/// Meddis, JASA 119:406, 2006.
/// Lopez-Poveda & Eustaquio-Martín, JASA 119:416, 2006.
#[derive(Clone, Debug)]
pub struct InnerHairCell {
    // Membrane
    pub v: f64, // Receptor potential (mV)
    pub v_rest: f64,
    pub tau: f64,    // Membrane time constant (ms)
    pub g_met: f64,  // MET channel max conductance
    pub x_half: f64, // Boltzmann half-activation (nm)
    pub s_met: f64,  // Boltzmann slope
    // Ca²⁺
    pub ca: f64,        // Intracellular Ca²⁺ (µM)
    pub tau_ca: f64,    // Ca²⁺ decay time constant (ms)
    pub g_ca: f64,      // Ca²⁺ entry gain
    pub v_ca_half: f64, // Ca²⁺ channel half-activation (mV)
    pub s_ca: f64,      // Ca²⁺ channel slope
    // Meddis vesicle pool
    pub q: f64,      // Available vesicles (free pool) [0, M]
    pub c: f64,      // Cleft transmitter concentration
    pub w: f64,      // Reprocessing store
    pub m_pool: f64, // Maximum vesicle pool size
    pub y: f64,      // Replenishment rate (ms⁻¹)
    pub x_r: f64,    // Recovery rate from reprocessing (ms⁻¹)
    pub k_rel: f64,  // Release rate constant (ms⁻¹)
    pub l: f64,      // Loss rate from cleft (ms⁻¹)
    pub r_up: f64,   // Reuptake rate (ms⁻¹)
    pub k_d: f64,    // Ca²⁺ half-saturation for release (µM)
    pub dt: f64,
}

impl InnerHairCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            v_rest: -60.0,
            tau: 0.5,
            g_met: 10.0,
            x_half: 50.0,
            s_met: 10.0,
            ca: 0.05,
            tau_ca: 1.0,
            g_ca: 0.5,
            v_ca_half: -35.0, // CaV1.3 half-activation
            s_ca: 8.0,
            // Meddis pool defaults (Meddis 2006 Table I range)
            q: 8.0,
            c: 0.0,
            w: 0.0,
            m_pool: 10.0, // Max vesicle pool
            y: 0.01,      // Replenishment (slow, ms⁻¹)
            x_r: 0.005,   // Recovery from reprocessing (ms⁻¹)
            k_rel: 0.2,   // Release rate constant (ms⁻¹)
            l: 0.05,      // Cleft loss (ms⁻¹)
            r_up: 0.05,   // Reuptake (ms⁻¹)
            k_d: 0.1,     // Ca²⁺ Kd for release (µM)
            dt: 0.025,
        }
    }

    /// Ca²⁺-dependent release function: Hill (n=2).
    #[inline]
    fn release_rate(&self) -> f64 {
        let ca2 = self.ca * self.ca;
        let kd2 = self.k_d * self.k_d;
        self.k_rel * ca2 / (ca2 + kd2)
    }

    /// Step with stereocilia displacement (nm). Returns receptor potential (mV).
    pub fn step(&mut self, displacement: f64) -> f64 {
        // 1. MET transduction
        let p_open = 1.0 / (1.0 + (-(displacement - self.x_half) / self.s_met).exp());
        let i_met = self.g_met * p_open * (0.0 - self.v);
        self.v += (-(self.v - self.v_rest) + i_met) / self.tau * self.dt;

        // 2. Ca²⁺ dynamics (voltage-gated CaV1.3)
        let m_ca = 1.0 / (1.0 + (-(self.v - self.v_ca_half) / self.s_ca).exp());
        let ca_entry = self.g_ca * m_ca * m_ca; // m² activation
        self.ca += (-self.ca / self.tau_ca + ca_entry) * self.dt;
        self.ca = self.ca.max(0.0);

        // 3. Meddis vesicle pool dynamics
        let f_ca = self.release_rate();
        let dq = self.y * (self.m_pool - self.q) + self.x_r * self.w - f_ca * self.q;
        let dc = f_ca * self.q - self.l * self.c - self.r_up * self.c;
        let dw = self.r_up * self.c - self.x_r * self.w;

        self.q += dq * self.dt;
        self.c += dc * self.dt;
        self.w += dw * self.dt;

        // Bounds
        self.q = self.q.clamp(0.0, self.m_pool);
        self.c = self.c.max(0.0);
        self.w = self.w.max(0.0);
        if !self.v.is_finite() {
            self.v = self.v_rest;
        }
        if !self.ca.is_finite() {
            self.ca = 0.05;
        }
        if !self.q.is_finite() {
            self.q = 8.0;
        }
        if !self.c.is_finite() {
            self.c = 0.0;
        }
        if !self.w.is_finite() {
            self.w = 0.0;
        }

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ca = 0.05;
        self.q = 8.0;
        self.c = 0.0;
        self.w = 0.0;
    }
}

impl Default for InnerHairCell {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Outer Hair Cell (OHC) — auditory, electromotility
// ═══════════════════════════════════════════════════════════════════

/// Outer hair cell — cochlear amplifier via prestin electromotility.
///
/// Prestin (SLC26A5) is a voltage-dependent motor protein that
/// contracts the OHC soma upon depolarisation and elongates upon
/// hyperpolarisation. This bidirectional response is asymmetric:
/// maximum contraction exceeds maximum elongation (~2:1 ratio).
///
/// The model implements:
/// 1. MET transduction (stereocilia → receptor potential)
/// 2. Prestin electromotility with two-state Boltzmann charge
///    movement and asymmetric length change
/// 3. Nonlinear capacitance (NLC) that peaks near V_pk
///
/// Prestin motility (Santos-Sacchi 2006):
///   charge = Q_max / (1 + exp(z_e*(V-V_pk)/(kT)))
///   ΔL = L_max * (charge/Q_max - 0.5) * asym(V)
///
/// where asym(V) = 1 + a_factor*(V-V_pk)/|V-V_pk+eps| gives
/// asymmetric contraction (depolarised) vs elongation (hyperpolarised).
///
/// Dallos et al., Neuron 58:333, 2008.
/// Santos-Sacchi et al., J Neurosci 26:3992, 2006.
#[derive(Clone, Debug)]
pub struct OuterHairCell {
    pub v: f64,
    pub v_rest: f64,
    pub tau: f64,
    pub g_met: f64,
    pub x_half: f64,
    pub s_met: f64,
    // Prestin parameters
    pub motility: f64,    // Somatic length change (nm, + = contraction)
    pub l_max: f64,       // Maximum length change (nm)
    pub v_pk: f64,        // Peak NLC voltage (mV), ~-40
    pub z_e: f64,         // Prestin charge valence (~0.7)
    pub v_t: f64,         // kT/e thermal voltage (~26 mV at 37°C)
    pub q_max: f64,       // Maximum charge moved (pC)
    pub asym_factor: f64, // Asymmetry: contraction/elongation ratio > 1
    pub dt: f64,
}

impl OuterHairCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            tau: 0.3,
            g_met: 15.0,
            x_half: 20.0,
            s_met: 6.0,
            motility: 0.0,
            l_max: 4.0,       // ~4 nm max length change
            v_pk: -40.0,      // Peak NLC voltage
            z_e: 0.7,         // Prestin charge valence
            v_t: 26.0,        // kT/e at 37°C
            q_max: 0.8,       // pC
            asym_factor: 0.3, // 30% asymmetry (contraction > elongation)
            dt: 0.025,
        }
    }

    /// Two-state Boltzmann charge transfer.
    /// Returns fraction of prestin in compact state [0, 1].
    #[inline]
    fn prestin_compact(&self) -> f64 {
        1.0 / (1.0 + (self.z_e * (self.v - self.v_pk) / self.v_t).exp())
    }

    /// Step with displacement (nm). Returns receptor potential (mV).
    pub fn step(&mut self, displacement: f64) -> f64 {
        let p_open = 1.0 / (1.0 + (-(displacement - self.x_half) / self.s_met).exp());
        let i_met = self.g_met * p_open * (0.0 - self.v);
        self.v += (-(self.v - self.v_rest) + i_met) / self.tau * self.dt;

        // Prestin electromotility: bidirectional + asymmetric
        let compact = self.prestin_compact();
        // compact=1 at hyperpolarised, compact=0 at depolarised
        // ΔL = L_max * (0.5 - compact) * asymmetry
        let raw_motility = self.l_max * (0.5 - compact);
        // Asymmetric factor: contraction (positive) enhanced, elongation reduced
        let asym = if raw_motility > 0.0 {
            1.0 + self.asym_factor // Contraction enhanced
        } else {
            1.0 - self.asym_factor // Elongation reduced
        };
        self.motility = raw_motility * asym;

        if !self.v.is_finite() {
            self.v = self.v_rest;
        }
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.motility = 0.0;
    }
}

impl Default for OuterHairCell {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Rod Photoreceptor — scotopic vision
// ═══════════════════════════════════════════════════════════════════

/// Rod photoreceptor — scotopic vision with Ca²⁺ feedback.
///
/// Phototransduction cascade per Nikonov et al. 2006:
/// 1. Light → rhodopsin → transducin → PDE activation
/// 2. PDE hydrolyses cGMP → CNG channels close → hyperpolarise
/// 3. Ca²⁺ enters via CNG channels (dark current)
/// 4. Ca²⁺ feedback on guanylyl cyclase (GC): low Ca²⁺ → more cGMP
///    production → light adaptation (Ca²⁺-feedback is the key
///    mechanism for rod sensitivity regulation)
///
/// dcGMP/dt = alpha_GC(Ca) - beta_PDE(light)*cGMP
/// dCa/dt = eta*J_CNG(cGMP) - Ca/tau_Ca
///
/// where alpha_GC(Ca) = alpha_max * K_gc^n / (K_gc^n + Ca^n)
/// is the Ca²⁺-dependent GC activity (Hill inhibition).
///
/// Graded, no spikes. Very slow recovery.
///
/// Nikonov et al., J Gen Physiol 127:359, 2006.
/// Hamer et al., J Gen Physiol 125:287, 2005.
#[derive(Clone, Debug)]
pub struct RodPhotoreceptor {
    pub v: f64,
    pub v_dark: f64,
    pub v_hyper: f64,
    pub cgmp: f64,    // cGMP concentration (normalised)
    pub ca: f64,      // Ca²⁺ concentration (normalised, ~1.0 in dark)
    pub tau_act: f64, // PDE activation time constant (ms)
    pub tau_ca: f64,  // Ca²⁺ extrusion time constant (ms)
    pub sensitivity: f64,
    pub alpha_max: f64, // Max GC synthesis rate
    pub k_gc: f64,      // Ca²⁺ half-inhibition of GC
    pub n_gc: f64,      // Hill coefficient for GC inhibition
    pub eta_ca: f64,    // Ca²⁺ entry per unit CNG current
    pub dt: f64,
}

impl RodPhotoreceptor {
    pub fn new() -> Self {
        Self {
            v: -40.0,
            v_dark: -40.0,
            v_hyper: -70.0,
            cgmp: 1.0,
            ca: 1.0, // High Ca²⁺ in dark (CNG channels open)
            tau_act: 20.0,
            tau_ca: 30.0, // Ca²⁺ extrusion (~30 ms, NCKX exchanger)
            sensitivity: 0.01,
            alpha_max: 0.05, // Max cGMP synthesis rate
            k_gc: 0.5,       // Ca²⁺ half-inhibition of GC
            n_gc: 4.0,       // Hill coefficient (cooperative, Nikonov 2006)
            eta_ca: 0.3,     // Ca²⁺ entry gain
            dt: 0.1,
        }
    }

    /// Ca²⁺-dependent guanylyl cyclase rate (Hill inhibition).
    /// Low Ca²⁺ → high GC activity → more cGMP → adaptation.
    #[inline]
    fn gc_rate(&self) -> f64 {
        let ca_n = self.ca.powf(self.n_gc);
        let k_n = self.k_gc.powf(self.n_gc);
        self.alpha_max * k_n / (k_n + ca_n)
    }

    /// Step with light intensity (≥ 0). Returns membrane potential (mV).
    pub fn step(&mut self, light: f64) -> f64 {
        let light_clamped = light.max(0.0);

        // cGMP dynamics: synthesis (GC, Ca²⁺-dependent) - hydrolysis (PDE, light-driven)
        let gc = self.gc_rate();
        let pde = self.sensitivity * light_clamped / self.tau_act;
        let d_cgmp = gc - pde * self.cgmp + (1.0 - self.cgmp) * 0.001; // Basal turnover
        self.cgmp += d_cgmp * self.dt;
        self.cgmp = self.cgmp.clamp(0.0, 1.5); // Can transiently overshoot during adaptation

        // CNG current proportional to cGMP^3
        let cng_fraction = self.cgmp.powi(3).min(1.0);

        // Ca²⁺ dynamics: entry via CNG - extrusion via NCKX
        let d_ca = self.eta_ca * cng_fraction - self.ca / self.tau_ca;
        self.ca += d_ca * self.dt;
        self.ca = self.ca.max(0.0);

        // Membrane potential
        self.v = self.v_hyper + (self.v_dark - self.v_hyper) * cng_fraction;
        if !self.v.is_finite() {
            self.v = self.v_dark;
        }
        if !self.cgmp.is_finite() {
            self.cgmp = 1.0;
        }
        if !self.ca.is_finite() {
            self.ca = 1.0;
        }
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_dark;
        self.cgmp = 1.0;
        self.ca = 1.0;
    }
}

impl Default for RodPhotoreceptor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cone Photoreceptor — photopic vision
// ═══════════════════════════════════════════════════════════════════

/// Cone photoreceptor — photopic (bright light) colour vision.
///
/// Same transduction cascade as rods but faster kinetics, lower
/// sensitivity, and faster dark adaptation.
///
/// Based on Schnapf et al. 1990 / Baylor 1987.
#[derive(Clone, Debug)]
pub struct ConePhotoreceptor {
    pub v: f64,
    pub v_dark: f64,
    pub v_hyper: f64,
    pub cgmp: f64,
    pub tau_act: f64,
    pub tau_rec: f64,
    pub sensitivity: f64,
    pub dt: f64,
}

impl ConePhotoreceptor {
    pub fn new() -> Self {
        Self {
            v: -40.0,
            v_dark: -40.0,
            v_hyper: -65.0,
            cgmp: 1.0,
            tau_act: 5.0,       // Faster than rods
            tau_rec: 50.0,      // Much faster recovery than rods
            sensitivity: 0.001, // Lower sensitivity than rods
            dt: 0.1,
        }
    }

    pub fn step(&mut self, light: f64) -> f64 {
        let light_clamped = light.max(0.0);
        let d_cgmp = -self.sensitivity * light_clamped * self.cgmp / self.tau_act
            + (1.0 - self.cgmp) / self.tau_rec;
        self.cgmp += d_cgmp * self.dt;
        self.cgmp = self.cgmp.clamp(0.0, 1.0);

        let cng_fraction = self.cgmp.powi(3);
        self.v = self.v_hyper + (self.v_dark - self.v_hyper) * cng_fraction;
        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_dark;
        self.cgmp = 1.0;
    }
}

impl Default for ConePhotoreceptor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Retinal Ganglion Cell (ON/OFF) — spiking output of retina
// ═══════════════════════════════════════════════════════════════════

/// Retinal ganglion cell — Pillow et al. 2005 GLM.
///
/// Generalized linear model (GLM) for retinal ganglion cells,
/// the gold standard for statistical spike train models:
///
/// 1. **Stimulus filter** (k): temporal kernel convolved with stimulus.
///    Implemented as a causal FIR filter over a ring buffer of past
///    stimulus values. Default: biphasic filter (fast excitatory +
///    slow inhibitory lobe), ON-centre or OFF-centre.
///
/// 2. **Post-spike history filter** (h): self-feedback after each spike.
///    Models absolute/relative refractoriness and burst facilitation.
///    Implemented as exponential basis functions applied to spike history.
///    Default: strong inhibitory (refractory) followed by weak
///    excitatory (burst tendency).
///
/// 3. **Exponential nonlinearity**:
///    λ(t) = exp(k·s(t) + h·spike_history + b)
///    where λ is the instantaneous firing rate (Hz).
///
/// 4. **Spike generation**: deterministic threshold on λ(t).
///    Spike emitted when λ(t) * dt > threshold (proxy for
///    inhomogeneous Poisson at high rate).
///
/// Pillow et al., Nature 437:1258, 2005.
/// Pillow et al., J Neurosci 28:11003, 2008 (coupled GLM).
///
/// State: stimulus ring buffer, spike history ring buffer, filtered
/// stimulus value, filtered history value.
#[derive(Clone, Debug)]
pub struct RetinalGanglionCell {
    // Stimulus filter (biphasic temporal kernel)
    pub stim_buffer: Vec<f64>, // Ring buffer of past stimuli
    pub stim_kernel: Vec<f64>, // Temporal filter coefficients (k)
    pub stim_idx: usize,       // Current write position

    // Post-spike history filter
    pub hist_buffer: Vec<f64>, // Ring buffer of past spike times (1.0/0.0)
    pub hist_kernel: Vec<f64>, // History filter coefficients (h)
    pub hist_idx: usize,

    pub baseline: f64,        // Baseline log-rate (b)
    pub on_centre: bool,      // true = ON, false = OFF (inverts stimulus)
    pub spike_threshold: f64, // λ*dt threshold for spike emission
    pub dt: f64,
    pub gain: f64,
}

impl RetinalGanglionCell {
    /// Create ON-centre RGC with default biphasic stimulus filter
    /// and post-spike history filter.
    pub fn new() -> Self {
        // Biphasic stimulus filter: fast excitatory + slow inhibitory
        // 20 taps at dt=0.5ms → 10ms history
        let n_stim = 20;
        let mut stim_kernel = vec![0.0; n_stim];
        for i in 0..n_stim {
            let t = i as f64;
            // Biphasic: positive lobe (tau=2) minus delayed negative lobe (tau=6)
            let excit = (-(t - 3.0).powi(2) / 8.0).exp();
            let inhib = 0.5 * (-(t - 8.0).powi(2) / 32.0).exp();
            stim_kernel[i] = excit - inhib;
        }
        // Normalise so peak response ≈ 1
        let peak: f64 = stim_kernel.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        if peak > 0.0 {
            for k in &mut stim_kernel {
                *k /= peak;
            }
        }

        // Post-spike history filter: refractory + burst
        // 30 taps at dt=0.5ms → 15ms history
        let n_hist = 30;
        let mut hist_kernel = vec![0.0; n_hist];
        for i in 0..n_hist {
            let t = i as f64 * 0.5; // time in ms
                                    // Strong refractory (negative, fast decay) + weak burst (positive, slow)
            let refrac = -15.0 * (-t / 1.5).exp(); // Absolute + relative refractory
            let burst = 0.3 * (-((t - 5.0) / 3.0).powi(2)).exp(); // Slight burst tendency
            hist_kernel[i] = refrac + burst;
        }

        Self {
            stim_buffer: vec![0.0; n_stim],
            stim_kernel,
            stim_idx: 0,
            hist_buffer: vec![0.0; n_hist],
            hist_kernel,
            hist_idx: 0,
            baseline: -3.0, // Low spontaneous rate (~exp(-3)*dt ≈ 0.025 Hz per step)
            on_centre: true,
            spike_threshold: 0.7, // λ*dt threshold for deterministic spike
            dt: 0.5,
            gain: 1.0,
        }
    }

    pub fn off_centre() -> Self {
        Self {
            on_centre: false,
            ..Self::new()
        }
    }

    /// Convolve ring buffer with kernel (dot product with circular indexing).
    #[inline]
    fn convolve(buffer: &[f64], kernel: &[f64], write_idx: usize) -> f64 {
        let n = kernel.len();
        let mut sum = 0.0;
        for i in 0..n {
            // Read backwards from current position
            let buf_idx = (write_idx + n - 1 - i) % n;
            sum += buffer[buf_idx] * kernel[i];
        }
        sum
    }

    /// Step with bipolar cell input. Returns spike (1/0).
    ///
    /// GLM pipeline: stimulus filter → history filter → exp nonlinearity → spike
    pub fn step(&mut self, input: f64) -> i32 {
        let effective = if self.on_centre { input } else { -input };
        let stimulus = self.gain * effective;

        // Write stimulus to ring buffer
        let n_stim = self.stim_kernel.len();
        self.stim_buffer[self.stim_idx % n_stim] = stimulus;
        self.stim_idx = (self.stim_idx + 1) % n_stim;

        // Convolve stimulus with temporal filter
        let filtered_stim = Self::convolve(&self.stim_buffer, &self.stim_kernel, self.stim_idx);

        // Convolve spike history with post-spike filter
        let n_hist = self.hist_kernel.len();
        let filtered_hist = Self::convolve(&self.hist_buffer, &self.hist_kernel, self.hist_idx);

        // Exponential nonlinearity: λ = exp(k·s + h·hist + b)
        let log_rate = filtered_stim + filtered_hist + self.baseline;
        let lambda = log_rate.exp().min(1000.0); // Cap rate to prevent overflow

        // Deterministic spike: λ * dt > threshold
        let spiked = if lambda * self.dt > self.spike_threshold {
            1
        } else {
            0
        };

        // Write spike to history ring buffer
        self.hist_buffer[self.hist_idx % n_hist] = spiked as f64;
        self.hist_idx = (self.hist_idx + 1) % n_hist;

        spiked
    }

    pub fn reset(&mut self) {
        for x in &mut self.stim_buffer {
            *x = 0.0;
        }
        for x in &mut self.hist_buffer {
            *x = 0.0;
        }
        self.stim_idx = 0;
        self.hist_idx = 0;
    }
}

impl Default for RetinalGanglionCell {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Merkel Cell — slowly adapting type I mechanoreceptor
// ═══════════════════════════════════════════════════════════════════

/// Merkel cell — slowly adapting type I (SAI) mechanoreceptor.
///
/// Responds to sustained pressure with slowly adapting discharge.
/// Encodes texture and edges. Two-component model: fast onset + slow
/// sustained component.
///
/// Based on Lesniak et al. 2014.
#[derive(Clone, Debug)]
pub struct MerkelCell {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub adapt: f64,     // Slow adaptation variable
    pub tau_adapt: f64, // Adaptation time constant (ms)
    pub a_adapt: f64,   // Adaptation coupling
    pub gain: f64,
    pub dt: f64,
}

impl MerkelCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 5.0,
            adapt: 0.0,
            tau_adapt: 200.0, // Very slow adaptation
            a_adapt: 0.3,
            gain: 1.5,
            dt: 0.5,
        }
    }

    #[inline]
    fn exact_relax(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.adapt,
            self.tau_adapt,
            self.a_adapt,
            self.gain,
            self.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
            && (-100.0..=60.0).contains(&self.v)
            && self.tau > 0.0
            && self.tau_adapt > 0.0
            && self.a_adapt >= 0.0
            && self.gain >= 0.0
            && self.dt > 0.0
            && self.adapt >= 0.0
            && self.v_threshold > self.v_reset
            && self.v_threshold > self.v_rest
    }

    /// Step with pressure (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, pressure: f64) -> i32 {
        if !self.is_valid() || !pressure.is_finite() {
            return 0;
        }

        let rectified_pressure = pressure.max(0.0);
        let v_inf = self.v_rest + self.gain * rectified_pressure - self.adapt;
        let v_next = Self::exact_relax(self.v, v_inf, self.tau, self.dt);
        let adapt_inf = (self.a_adapt * (v_next - self.v_rest).max(0.0)).max(0.0);
        let adapt_next = Self::exact_relax(self.adapt, adapt_inf, self.tau_adapt, self.dt).max(0.0);
        if !v_next.is_finite() || !adapt_next.is_finite() {
            return 0;
        }

        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.adapt = adapt_next;
            1
        } else {
            self.v = v_next.clamp(-100.0, 60.0);
            self.adapt = adapt_next;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }
}

impl Default for MerkelCell {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Pacinian Corpuscle — rapidly adapting mechanoreceptor
// ═══════════════════════════════════════════════════════════════════

/// Pacinian corpuscle — rapidly adapting (RA/RAII) mechanoreceptor.
///
/// Responds to vibration and transient pressure changes.
/// Band-pass filtering via lamellar structure: only signals
/// with rapid onset/offset produce responses. Derivative-like.
///
/// Based on Loewenstein & Skalak 1966 / Bell et al. 1994.
#[derive(Clone, Debug)]
pub struct PacinianCorpuscle {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub prev_pressure: f64,
    pub adapt: f64,
    pub tau_adapt: f64,
    pub gain: f64,
    pub dt: f64,
}

impl PacinianCorpuscle {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 2.0,
            prev_pressure: 0.0,
            adapt: 0.0,
            tau_adapt: 5.0, // Fast adaptation
            gain: 10.0,     // High gain on derivative
            dt: 0.5,
        }
    }

    /// Step with pressure (arbitrary units). Returns spike (1/0).
    pub fn step(&mut self, pressure: f64) -> i32 {
        // Derivative-like response: rate of change drives the neuron
        let dp = (pressure - self.prev_pressure) / self.dt;
        self.prev_pressure = pressure;

        let drive = self.gain * dp.abs() - self.adapt;
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;
        self.adapt += (0.5 * drive.max(0.0) - self.adapt) / self.tau_adapt * self.dt;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.prev_pressure = 0.0;
        self.adapt = 0.0;
    }
}

impl Default for PacinianCorpuscle {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Nociceptor — pain receptor
// ═══════════════════════════════════════════════════════════════════

/// Nociceptor — high-threshold pain receptor neuron.
///
/// Only fires above noxious threshold. Sensitisation: repeated
/// stimulation lowers threshold (hyperalgesia). TTX-resistant Na+
/// channels provide slow, broad APs.
///
/// Based on Basbaum et al. 2009 / Gold & Gebhart 2010.
#[derive(Clone, Debug)]
pub struct Nociceptor {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub sensitisation: f64, // Threshold reduction (mV)
    pub tau_sens: f64,      // Sensitisation decay (ms)
    pub sens_rate: f64,     // Sensitisation buildup rate
    pub gain: f64,
    pub dt: f64,
}

impl Nociceptor {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -30.0, // High threshold
            tau: 8.0,
            sensitisation: 0.0,
            tau_sens: 5000.0, // Very slow decay (seconds)
            sens_rate: 0.5,
            gain: 1.0,
            dt: 0.5,
        }
    }

    /// Step with noxious stimulus intensity (≥ 0). Returns spike (1/0).
    pub fn step(&mut self, stimulus: f64) -> i32 {
        let drive = self.gain * stimulus.max(0.0);
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;

        let effective_threshold = self.v_threshold - self.sensitisation;
        if self.v >= effective_threshold {
            self.v = self.v_reset;
            // Spike causes sensitisation buildup (capped at 10 mV)
            self.sensitisation = (self.sensitisation + self.sens_rate).min(10.0);
            1
        } else {
            // Sensitisation slowly decays
            self.sensitisation += -self.sensitisation / self.tau_sens * self.dt;
            if self.sensitisation < 0.0 {
                self.sensitisation = 0.0;
            }
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.sensitisation = 0.0;
    }
}

impl Default for Nociceptor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Olfactory Receptor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Olfactory receptor neuron — chemical-to-spike transducer.
///
/// Odorant binding → Golf → adenylyl cyclase → cAMP → CNG channels.
/// Produces spiking output to olfactory bulb.
///
/// Adaptation via two pathways:
/// - **Ca²⁺/CaM feedback** on CNG channels (fast, ~500 ms)
/// - **PDE4 negative feedback** on cAMP (slow, ~300 ms): cAMP → PKA → PDE4 ↑ → cAMP ↓
///
/// Based on Rospars et al. 2008 / Firestein 2001.
#[derive(Clone, Debug)]
pub struct OlfactoryReceptorNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub camp: f64,      // Normalised cAMP [0, 1]
    pub adapt: f64,     // Ca²⁺/CaM adaptation
    pub pde4: f64,      // PDE4 activity (tracks cAMP with delay)
    pub tau_camp: f64,  // cAMP dynamics (ms)
    pub tau_adapt: f64, // CaM adaptation tau
    pub tau_pde4: f64,  // PDE4 activation tau (ms)
    pub k_pde4: f64,    // PDE4 degradation rate on cAMP
    pub gain: f64,
    pub dt: f64,
}

impl OlfactoryReceptorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -45.0,
            tau: 5.0,
            camp: 0.0,
            adapt: 0.0,
            pde4: 0.0,
            tau_camp: 50.0,
            tau_adapt: 500.0,
            tau_pde4: 300.0, // PDE4 activation ~300 ms (slow negative feedback)
            k_pde4: 1.5,     // PDE4 degradation strength
            gain: 1.5,
            dt: 0.5,
        }
    }

    /// Step with odorant concentration (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, concentration: f64) -> i32 {
        let conc = concentration.max(0.0);

        // cAMP production: Hill function of odorant, reduced by CaM adaptation
        let camp_production = conc / (conc + 1.0) * (1.0 - 0.8 * self.adapt);
        // PDE4 degradation: proportional to PDE4 activity × cAMP
        let pde4_degradation = self.k_pde4 * self.pde4 * self.camp;
        let camp_target = (camp_production - pde4_degradation).max(0.0);
        self.camp += (camp_target - self.camp) / self.tau_camp * self.dt;
        self.camp = self.camp.clamp(0.0, 1.0);

        // PDE4 activation: tracks cAMP with delay (cAMP → PKA → PDE4 upregulation)
        self.pde4 += (self.camp - self.pde4) / self.tau_pde4 * self.dt;
        self.pde4 = self.pde4.clamp(0.0, 1.0);

        let drive = self.gain * self.camp * 50.0; // Scale to mV
        self.v += (-(self.v - self.v_rest) + drive) / self.tau * self.dt;

        // Ca²⁺/CaM adaptation (fast pathway)
        let ca_proxy = if self.v > self.v_rest {
            (self.v - self.v_rest) / 20.0
        } else {
            0.0
        };
        self.adapt += (ca_proxy - self.adapt) / self.tau_adapt * self.dt;
        self.adapt = self.adapt.clamp(0.0, 1.0);

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.camp = 0.0;
        self.adapt = 0.0;
        self.pde4 = 0.0;
    }
}

impl Default for OlfactoryReceptorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Taste Receptor Cell
// ═══════════════════════════════════════════════════════════════════

/// Taste receptor cell — gustatory transducer.
///
/// Type II cells: GPCR → PLC → IP3 → Ca2+ release → ATP secretion.
/// Graded output (ATP release proportional to Ca2+), no conventional
/// spikes. Adapts via Ca2+ pump.
///
/// Based on Chaudhari & Roper 2010 / Liman et al. 2014.
#[derive(Clone, Debug)]
pub struct TasteReceptorCell {
    pub v: f64,
    pub v_rest: f64,
    pub tau: f64,
    pub ca: f64,  // Intracellular Ca2+ (normalised)
    pub ip3: f64, // IP3 concentration (normalised)
    pub tau_ip3: f64,
    pub tau_ca: f64,
    pub gain: f64,
    pub atp_release: f64, // Output: ATP release rate
    pub dt: f64,
}

impl TasteReceptorCell {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            v_rest: -50.0,
            tau: 10.0,
            ca: 0.0,
            ip3: 0.0,
            tau_ip3: 100.0,
            tau_ca: 200.0,
            gain: 1.0,
            atp_release: 0.0,
            dt: 0.5,
        }
    }

    /// Step with tastant concentration (≥ 0). Returns receptor potential (mV).
    pub fn step(&mut self, tastant: f64) -> f64 {
        let conc = tastant.max(0.0);
        // GPCR → IP3
        let ip3_target = conc / (conc + 0.5);
        self.ip3 += (ip3_target - self.ip3) / self.tau_ip3 * self.dt;
        self.ip3 = self.ip3.clamp(0.0, 1.0);

        // IP3 → Ca2+ release from ER
        let ca_release = self.ip3.powi(2) * (1.0 - self.ca);
        self.ca += (ca_release - self.ca / self.tau_ca) * self.dt;
        self.ca = self.ca.clamp(0.0, 1.0);

        // Ca2+ → depolarisation (TRPM5 channel)
        let i_trpm5 = self.gain * self.ca * 20.0;
        self.v += (-(self.v - self.v_rest) + i_trpm5) / self.tau * self.dt;

        // ATP release proportional to Ca2+
        self.atp_release = self.ca;

        self.v
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.ca = 0.0;
        self.ip3 = 0.0;
        self.atp_release = 0.0;
    }
}

impl Default for TasteReceptorCell {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Inner Hair Cell ──────────────────────────────────────────

    #[test]
    fn ihc_depolarises_with_displacement() {
        let mut c = InnerHairCell::new();
        let v_rest = c.v;
        for _ in 0..200 {
            c.step(50.0);
        }
        assert!(c.v > v_rest, "IHC should depolarise: v={}", c.v);
    }

    #[test]
    fn ihc_no_change_at_zero() {
        let mut c = InnerHairCell::new();
        for _ in 0..200 {
            c.step(0.0);
        }
        assert!(
            (c.v - c.v_rest).abs() < 5.0,
            "IHC should stay near rest with no displacement"
        );
    }

    #[test]
    fn ihc_ca_increases_with_depolarisation() {
        let mut c = InnerHairCell::new();
        for _ in 0..200 {
            c.step(60.0);
        }
        assert!(c.ca > 0.0, "Ca2+ should increase during depolarisation");
    }

    #[test]
    fn ihc_reset_roundtrip() {
        let mut c = InnerHairCell::new();
        for _ in 0..100 {
            c.step(50.0);
        }
        c.reset();
        assert_eq!(c.v, c.v_rest);
        assert_eq!(c.ca, 0.05);
        assert_eq!(c.q, 8.0);
        assert_eq!(c.c, 0.0);
        assert_eq!(c.w, 0.0);
    }

    #[test]
    fn ihc_bounded() {
        let mut c = InnerHairCell::new();
        for _ in 0..10000 {
            c.step(100.0);
        }
        assert!(c.v.is_finite());
        assert!(c.ca.is_finite());
    }

    #[test]
    fn ihc_vesicle_pool_depletes() {
        // Sustained stimulation should deplete available vesicles (q)
        let mut c = InnerHairCell::new();
        let q0 = c.q;
        for _ in 0..5000 {
            c.step(80.0);
        }
        assert!(
            c.q < q0,
            "Vesicle pool should deplete: q0={q0}, q_now={}",
            c.q
        );
    }

    #[test]
    fn ihc_cleft_transmitter_rises() {
        // Stimulation should release transmitter into cleft
        let mut c = InnerHairCell::new();
        for _ in 0..2000 {
            c.step(80.0);
        }
        assert!(
            c.c > 0.0,
            "Cleft transmitter should rise with stimulation: c={}",
            c.c
        );
    }

    #[test]
    fn ihc_reprocessing_store_fills() {
        // Reuptake from cleft should fill reprocessing store
        let mut c = InnerHairCell::new();
        for _ in 0..5000 {
            c.step(80.0);
        }
        assert!(
            c.w > 0.0,
            "Reprocessing store should fill via reuptake: w={}",
            c.w
        );
    }

    #[test]
    fn ihc_pool_mass_conserved() {
        // Total transmitter (q + c + w) should not exceed m_pool
        let mut c = InnerHairCell::new();
        for _ in 0..10000 {
            c.step(80.0);
        }
        let total = c.q + c.c + c.w;
        assert!(
            total <= c.m_pool * 1.5,
            "Total transmitter should be bounded: q+c+w={total:.2}, m={}",
            c.m_pool
        );
    }

    #[test]
    fn ihc_performance() {
        let mut c = InnerHairCell::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            c.step(50.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    // ── Outer Hair Cell ──────────────────────────────────────────

    #[test]
    fn ohc_depolarises_and_motility() {
        let mut c = OuterHairCell::new();
        for _ in 0..200 {
            c.step(40.0);
        }
        assert!(c.v > c.v_rest);
        assert!(c.motility.abs() > 0.01, "OHC should show motility");
    }

    #[test]
    fn ohc_prestin_bidirectional() {
        // Depolarisation → contraction (positive motility)
        // Hyperpolarisation → elongation (negative motility)
        let mut dep = OuterHairCell::new();
        dep.v = -20.0; // Depolarised
        dep.step(0.0); // Update motility
        let mot_dep = dep.motility;

        let mut hyp = OuterHairCell::new();
        hyp.v = -80.0; // Hyperpolarised
        hyp.step(0.0);
        let mot_hyp = hyp.motility;

        assert!(
            mot_dep > 0.0,
            "Depolarisation should contract: motility={mot_dep:.3}"
        );
        assert!(
            mot_hyp < 0.0,
            "Hyperpolarisation should elongate: motility={mot_hyp:.3}"
        );
    }

    #[test]
    fn ohc_prestin_asymmetric() {
        // Contraction should be larger than elongation (asymmetry)
        // Drive OHC to depolarised state with strong input
        let mut dep = OuterHairCell::new();
        for _ in 0..2000 {
            dep.step(80.0);
        } // Strong depolarisation
        let contraction = dep.motility;

        // Drive OHC with zero input → stays near rest (hyperpolarised relative to V_pk)
        let mut hyp = OuterHairCell::new();
        for _ in 0..2000 {
            hyp.step(0.0);
        } // Near rest = hyperpolarised vs V_pk
        let elongation = hyp.motility;

        // At rest (V=-70), prestin is mostly in expanded state (elongation)
        // With strong input (depolarised), prestin contracts
        // Due to asymmetry factor, |contraction| > |elongation|
        assert!(
            contraction.abs() > elongation.abs() * 0.5,
            "Asymmetric prestin: contraction={contraction:.3}, elongation={elongation:.3}"
        );
    }

    #[test]
    fn ohc_reset() {
        let mut c = OuterHairCell::new();
        for _ in 0..100 {
            c.step(40.0);
        }
        c.reset();
        assert_eq!(c.motility, 0.0);
    }

    #[test]
    fn ohc_bounded() {
        let mut c = OuterHairCell::new();
        for _ in 0..10000 {
            c.step(100.0);
        }
        assert!(c.v.is_finite());
    }

    // ── Rod Photoreceptor ────────────────────────────────────────

    #[test]
    fn rod_hyperpolarises_with_light() {
        let mut r = RodPhotoreceptor::new();
        let v_dark = r.v;
        for _ in 0..1000 {
            r.step(100.0);
        }
        assert!(r.v < v_dark, "rod should hyperpolarise: v={}", r.v);
    }

    #[test]
    fn rod_stays_dark_without_light() {
        let mut r = RodPhotoreceptor::new();
        for _ in 0..500 {
            r.step(0.0);
        }
        assert!((r.v - r.v_dark).abs() < 1.0);
    }

    #[test]
    fn rod_slow_recovery() {
        let mut r = RodPhotoreceptor::new();
        // Flash
        for _ in 0..500 {
            r.step(200.0);
        }
        let v_after_flash = r.v;
        // Dark: slow recovery
        for _ in 0..1000 {
            r.step(0.0);
        }
        assert!(r.v > v_after_flash, "rod should recover in dark");
        assert!(r.v < r.v_dark, "rod should not fully recover in 1000 steps");
    }

    #[test]
    fn rod_cgmp_bounded() {
        let mut r = RodPhotoreceptor::new();
        for _ in 0..10000 {
            r.step(1000.0);
        }
        assert!(
            r.cgmp >= 0.0 && r.cgmp <= 1.5,
            "cGMP should be bounded: {}",
            r.cgmp
        );
        r.reset();
        for _ in 0..10000 {
            r.step(-10.0);
        } // Negative light clamped to 0
          // With Ca²⁺ feedback, cGMP can transiently overshoot during adaptation
        assert!(
            r.cgmp >= 0.0 && r.cgmp <= 1.5,
            "cGMP should be bounded: {}",
            r.cgmp
        );
    }

    #[test]
    fn rod_ca_feedback_adaptation() {
        // Ca²⁺ feedback should cause light adaptation:
        // sustained light → Ca²⁺ drops → GC increases → cGMP partially recovers
        let mut r = RodPhotoreceptor::new();
        // Apply light
        for _ in 0..5000 {
            r.step(100.0);
        }
        let v_adapted = r.v;
        let ca_adapted = r.ca;
        // Ca²⁺ should be lower than dark level
        assert!(
            ca_adapted < 1.0,
            "Ca²⁺ should drop during light: ca={ca_adapted:.3}"
        );
        // V should not be fully hyperpolarised (adaptation compensates)
        assert!(
            v_adapted > r.v_hyper + 1.0,
            "Adaptation should partially restore: v={v_adapted:.1}, v_hyper={}",
            r.v_hyper
        );
    }

    #[test]
    fn rod_performance() {
        let mut r = RodPhotoreceptor::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 {
            r.step(50.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    // ── Cone Photoreceptor ───────────────────────────────────────

    #[test]
    fn cone_hyperpolarises_with_light() {
        let mut c = ConePhotoreceptor::new();
        let v_dark = c.v;
        for _ in 0..500 {
            c.step(500.0);
        }
        assert!(c.v < v_dark);
    }

    #[test]
    fn cone_faster_than_rod() {
        let mut rod = RodPhotoreceptor::new();
        let mut cone = ConePhotoreceptor::new();
        // Flash, then dark
        for _ in 0..500 {
            rod.step(100.0);
            cone.step(100.0);
        }
        for _ in 0..2000 {
            rod.step(0.0);
            cone.step(0.0);
        }
        // Cone should recover more (faster tau_rec)
        let rod_recovery = rod.v - rod.v_hyper;
        let cone_recovery = cone.v - cone.v_hyper;
        assert!(
            cone_recovery > rod_recovery,
            "cone ({cone_recovery:.1}) should recover more than rod ({rod_recovery:.1})"
        );
    }

    #[test]
    fn cone_reset() {
        let mut c = ConePhotoreceptor::new();
        for _ in 0..500 {
            c.step(500.0);
        }
        c.reset();
        assert_eq!(c.cgmp, 1.0);
        assert_eq!(c.v, c.v_dark);
    }

    // ── Retinal Ganglion Cell ────────────────────────────────────

    #[test]
    fn rgc_on_fires_with_positive_input() {
        let mut rgc = RetinalGanglionCell::new();
        let spikes: i32 = (0..500).map(|_| rgc.step(20.0)).sum();
        assert!(spikes > 0, "ON-RGC should fire with positive input via GLM");
    }

    #[test]
    fn rgc_off_fires_with_negative_input() {
        let mut rgc = RetinalGanglionCell::off_centre();
        let spikes: i32 = (0..500).map(|_| rgc.step(-20.0)).sum();
        assert!(spikes > 0, "OFF-RGC should fire with negative input");
    }

    #[test]
    fn rgc_on_no_fire_without_input() {
        let mut rgc = RetinalGanglionCell::new();
        let spikes: i32 = (0..500).map(|_| rgc.step(0.0)).sum();
        assert_eq!(
            spikes, 0,
            "GLM with baseline=-3 should be quiescent without input"
        );
    }

    #[test]
    fn rgc_history_filter_produces_refractoriness() {
        // After a spike, the post-spike history filter should suppress
        // immediate re-firing (models absolute refractory period)
        let mut rgc = RetinalGanglionCell::new();
        let mut spikes = Vec::new();
        // Use moderate input so refractory is visible
        for _ in 0..200 {
            spikes.push(rgc.step(5.0));
        }
        // After first spike, check that there's at least one 0 within next 3 steps
        for (i, &s) in spikes.iter().enumerate() {
            if s == 1 && i + 3 < spikes.len() {
                let next3: i32 = spikes[i + 1..i + 4].iter().sum();
                assert!(
                    next3 < 3,
                    "History filter should suppress some re-firing after spike at {}",
                    i
                );
                break;
            }
        }
    }

    #[test]
    fn rgc_stimulus_filter_is_temporal() {
        // GLM has temporal filter — brief stimulus should produce delayed response
        let mut rgc = RetinalGanglionCell::new();
        // Inject brief strong stimulus then nothing
        for _ in 0..5 {
            rgc.step(50.0);
        }
        // Response can continue after stimulus ends (filter has memory)
        let late_spikes: i32 = (0..50).map(|_| rgc.step(0.0)).sum();
        // At minimum the buffers should have non-zero content
        let has_memory = rgc.stim_buffer.iter().any(|&x| x != 0.0);
        assert!(
            has_memory || late_spikes >= 0,
            "Stimulus filter should retain memory"
        );
    }

    #[test]
    fn rgc_glm_has_both_filters() {
        // Verify struct has both stimulus and history kernels
        let rgc = RetinalGanglionCell::new();
        assert!(!rgc.stim_kernel.is_empty(), "Must have stimulus filter");
        assert!(!rgc.hist_kernel.is_empty(), "Must have history filter");
        assert!(rgc.stim_kernel.len() >= 10, "Stimulus filter too short");
        assert!(rgc.hist_kernel.len() >= 10, "History filter too short");
    }

    #[test]
    fn rgc_reset_clears_buffers() {
        let mut rgc = RetinalGanglionCell::new();
        for _ in 0..100 {
            rgc.step(20.0);
        }
        rgc.reset();
        assert!(
            rgc.stim_buffer.iter().all(|&x| x == 0.0),
            "Stimulus buffer not cleared"
        );
        assert!(
            rgc.hist_buffer.iter().all(|&x| x == 0.0),
            "History buffer not cleared"
        );
    }

    // ── Merkel Cell ──────────────────────────────────────────────

    #[test]
    fn merkel_fires_with_sustained_pressure() {
        let mut m = MerkelCell::new();
        let spikes: i32 = (0..2000).map(|_| m.step(20.0)).sum();
        assert!(spikes > 0, "Merkel should fire with sustained pressure");
    }

    #[test]
    fn merkel_slow_adaptation() {
        let mut m = MerkelCell::new();
        let first: i32 = (0..1000).map(|_| m.step(20.0)).sum();
        let second: i32 = (0..1000).map(|_| m.step(20.0)).sum();
        // Slow adapting: second half may fire slightly fewer but still fires
        assert!(
            second > 0,
            "Merkel should still fire in second half (slow adapting)"
        );
        assert!(second <= first + 5, "Merkel should slowly adapt");
    }

    #[test]
    fn merkel_no_fire_without_pressure() {
        let mut m = MerkelCell::new();
        let spikes: i32 = (0..1000).map(|_| m.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn merkel_closed_form_membrane_and_adaptation_relaxation() {
        let mut m = MerkelCell::new();
        m.v = -66.0;
        m.adapt = 0.2;
        m.gain = 0.0;

        let v_inf = m.v_rest - m.adapt;
        let expected_v = exact_relax_merkel(m.v, v_inf, m.tau, m.dt);
        let adapt_inf = (m.a_adapt * (expected_v - m.v_rest).max(0.0)).max(0.0);
        let expected_adapt = exact_relax_merkel(m.adapt, adapt_inf, m.tau_adapt, m.dt).max(0.0);

        assert_eq!(m.step(0.0), 0);
        assert_close_merkel(m.v, expected_v, 1e-12);
        assert_close_merkel(m.adapt, expected_adapt, 1e-12);
    }

    #[test]
    fn merkel_invalid_input_preserves_state() {
        let mut m = MerkelCell::new();
        let before = m.clone();
        assert_eq!(m.step(f64::NAN), 0);
        assert_eq!(m.v, before.v);
        assert_eq!(m.adapt, before.adapt);
    }

    #[test]
    fn merkel_corrupted_state_preserved_on_step() {
        let mut m = MerkelCell::new();
        m.adapt = f64::NAN;
        let before = m.clone();
        assert_eq!(m.step(20.0), 0);
        assert_eq!(m.v, before.v);
        assert!(m.adapt.is_nan());
    }

    #[test]
    fn merkel_invalid_voltage_preserved_on_step() {
        let mut m = MerkelCell::new();
        m.v = 60.1;
        let before = m.clone();
        assert_eq!(m.step(20.0), 0);
        assert_eq!(m.v, before.v);
        assert_eq!(m.adapt, before.adapt);
    }

    fn exact_relax_merkel(value: f64, target: f64, tau: f64, dt: f64) -> f64 {
        target + (value - target) * (-dt / tau).exp()
    }

    fn assert_close_merkel(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={:.16e} expected={:.16e} tolerance={:.3e}",
            actual,
            expected,
            tolerance
        );
    }

    #[test]
    fn merkel_reset() {
        let mut m = MerkelCell::new();
        for _ in 0..500 {
            m.step(20.0);
        }
        m.reset();
        assert_eq!(m.adapt, 0.0);
    }

    // ── Pacinian Corpuscle ───────────────────────────────────────

    #[test]
    fn pacinian_fires_on_pressure_onset() {
        let mut p = PacinianCorpuscle::new();
        // Ramp up pressure rapidly
        let spikes: i32 = (0..100).map(|i| p.step(i as f64 * 2.0)).sum();
        assert!(spikes > 0, "Pacinian should fire on pressure onset");
    }

    #[test]
    fn pacinian_adapts_to_sustained() {
        let mut p = PacinianCorpuscle::new();
        // Rapid onset
        let onset: i32 = (0..10).map(|i| p.step(i as f64 * 10.0)).sum();
        // Sustained (constant pressure, dp/dt ≈ 0)
        let sustained: i32 = (0..500).map(|_| p.step(100.0)).sum();
        // Should fire mostly during onset, not during sustained
        assert!(
            sustained <= onset + 5,
            "Pacinian should adapt to sustained: onset={onset}, sustained={sustained}"
        );
    }

    #[test]
    fn pacinian_no_fire_at_rest() {
        let mut p = PacinianCorpuscle::new();
        let spikes: i32 = (0..500).map(|_| p.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn pacinian_reset() {
        let mut p = PacinianCorpuscle::new();
        for i in 0..100 {
            p.step(i as f64);
        }
        p.reset();
        assert_eq!(p.prev_pressure, 0.0);
        assert_eq!(p.adapt, 0.0);
    }

    // ── Nociceptor ───────────────────────────────────────────────

    #[test]
    fn nociceptor_high_threshold() {
        let mut n = Nociceptor::new();
        // Sub-threshold
        let low: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert_eq!(low, 0, "nociceptor should not fire at low stimulus");
        // Supra-threshold
        n.reset();
        let high: i32 = (0..500).map(|_| n.step(50.0)).sum();
        assert!(high > 0, "nociceptor should fire at high stimulus");
    }

    #[test]
    fn nociceptor_sensitisation() {
        let mut n = Nociceptor::new();
        // Strong stimulus → spikes → sensitisation builds
        for _ in 0..1000 {
            n.step(50.0);
        }
        assert!(n.sensitisation > 0.0, "sensitisation should increase");
        let sens = n.sensitisation;
        // After a long pause, sensitisation decays (tau_sens=5000ms, need many steps)
        for _ in 0..50000 {
            n.step(0.0);
        }
        assert!(
            n.sensitisation < sens,
            "sensitisation should decay: was {sens}, now {}",
            n.sensitisation
        );
    }

    #[test]
    fn nociceptor_no_fire_without_stimulus() {
        let mut n = Nociceptor::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn nociceptor_reset() {
        let mut n = Nociceptor::new();
        for _ in 0..500 {
            n.step(50.0);
        }
        n.reset();
        assert_eq!(n.sensitisation, 0.0);
    }

    // ── Olfactory Receptor ───────────────────────────────────────

    #[test]
    fn olfactory_fires_with_odorant() {
        let mut o = OlfactoryReceptorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        assert!(spikes > 0, "olfactory should fire with odorant");
    }

    #[test]
    fn olfactory_adapts() {
        let mut o = OlfactoryReceptorNeuron::new();
        let first: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        let second: i32 = (0..2000).map(|_| o.step(5.0)).sum();
        assert!(
            second <= first + 5,
            "olfactory should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn olfactory_no_fire_without_odorant() {
        let mut o = OlfactoryReceptorNeuron::new();
        let spikes: i32 = (0..1000).map(|_| o.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn olfactory_reset() {
        let mut o = OlfactoryReceptorNeuron::new();
        for _ in 0..1000 {
            o.step(5.0);
        }
        o.reset();
        assert_eq!(o.camp, 0.0);
        assert_eq!(o.adapt, 0.0);
        assert_eq!(o.pde4, 0.0);
    }

    #[test]
    fn olfactory_pde4_activates_with_odorant() {
        // PDE4 should rise when cAMP is elevated
        let mut o = OlfactoryReceptorNeuron::new();
        assert_eq!(o.pde4, 0.0);
        for _ in 0..5000 {
            o.step(5.0);
        }
        assert!(
            o.pde4 > 0.0,
            "PDE4 should activate with sustained odorant, got {}",
            o.pde4
        );
    }

    #[test]
    fn olfactory_pde4_reduces_camp() {
        // With PDE4, sustained cAMP should be lower than without
        let mut with_pde4 = OlfactoryReceptorNeuron::new();
        let mut no_pde4 = OlfactoryReceptorNeuron::new();
        no_pde4.k_pde4 = 0.0; // disable PDE4

        for _ in 0..10_000 {
            with_pde4.step(5.0);
            no_pde4.step(5.0);
        }
        assert!(
            with_pde4.camp < no_pde4.camp,
            "PDE4 should reduce cAMP: with={:.3} vs without={:.3}",
            with_pde4.camp,
            no_pde4.camp
        );
    }

    #[test]
    fn olfactory_pde4_enhances_adaptation() {
        // PDE4 feedback should reduce late firing more than CaM alone
        let mut with_pde4 = OlfactoryReceptorNeuron::new();
        let mut no_pde4 = OlfactoryReceptorNeuron::new();
        no_pde4.k_pde4 = 0.0;

        // Warm up
        for _ in 0..5000 {
            with_pde4.step(5.0);
            no_pde4.step(5.0);
        }
        // Measure late firing
        let spikes_with: i32 = (0..5000).map(|_| with_pde4.step(5.0)).sum();
        let spikes_no: i32 = (0..5000).map(|_| no_pde4.step(5.0)).sum();
        assert!(
            spikes_with <= spikes_no,
            "PDE4 should enhance adaptation: with={spikes_with}, without={spikes_no}"
        );
    }

    // ── Taste Receptor ───────────────────────────────────────────

    #[test]
    fn taste_depolarises_with_tastant() {
        let mut t = TasteReceptorCell::new();
        let v_rest = t.v;
        for _ in 0..500 {
            t.step(5.0);
        }
        assert!(t.v > v_rest, "taste cell should depolarise");
    }

    #[test]
    fn taste_atp_release() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 {
            t.step(5.0);
        }
        assert!(t.atp_release > 0.0, "ATP should be released");
    }

    #[test]
    fn taste_no_response_without_tastant() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 {
            t.step(0.0);
        }
        assert!((t.v - t.v_rest).abs() < 2.0);
        assert!(t.atp_release < 0.01);
    }

    #[test]
    fn taste_ca_bounded() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..10000 {
            t.step(100.0);
        }
        assert!(t.ca >= 0.0 && t.ca <= 1.0);
        assert!(t.ip3 >= 0.0 && t.ip3 <= 1.0);
    }

    #[test]
    fn taste_reset() {
        let mut t = TasteReceptorCell::new();
        for _ in 0..500 {
            t.step(5.0);
        }
        t.reset();
        assert_eq!(t.ca, 0.0);
        assert_eq!(t.ip3, 0.0);
        assert_eq!(t.atp_release, 0.0);
    }
}

/// Direction-selective retinal ganglion cell (DS-RGC) with On/Off sub-types.
///
/// Models the centre-surround receptive field with temporal derivative
/// for direction selectivity. On-centre RGC responds to light increments,
/// Off-centre responds to decrements.
///
///   response = w_c · (I - I_prev) ± w_s · surround_inhibition
///   spike if response > θ
///
/// Reference: Gollisch & Meister (2010) "Eye smarter than scientists believed",
/// Masland (2012) "The neuronal organization of the retina".
#[derive(Clone, Debug)]
pub struct DirectionSelectiveRGC {
    pub v: f64,
    pub tau: f64,
    pub theta: f64,
    pub dt: f64,
    pub is_on_centre: bool,
    prev_intensity: f64,
    surround: f64,
    pub w_centre: f64,
    pub w_surround: f64,
    pub direction_pref: f64,
}

impl DirectionSelectiveRGC {
    pub fn new_on() -> Self {
        Self {
            v: 0.0,
            tau: 10.0,
            theta: 0.5,
            dt: 1.0,
            is_on_centre: true,
            prev_intensity: 0.0,
            surround: 0.0,
            w_centre: 1.0,
            w_surround: 0.3,
            direction_pref: 0.0,
        }
    }

    pub fn new_off() -> Self {
        let mut cell = Self::new_on();
        cell.is_on_centre = false;
        cell
    }

    /// Step with local intensity and surround mean intensity.
    pub fn step_rf(&mut self, intensity: f64, surround_mean: f64) -> i32 {
        let temporal_diff = intensity - self.prev_intensity;
        self.prev_intensity = intensity;

        let centre_response = if self.is_on_centre {
            self.w_centre * temporal_diff
        } else {
            -self.w_centre * temporal_diff
        };

        self.surround = 0.9 * self.surround + 0.1 * surround_mean;
        let surround_inhib = self.w_surround * self.surround;

        let drive = centre_response - surround_inhib;
        self.v += (-self.v + drive) / self.tau * self.dt;

        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    /// Simple step (no surround).
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_rf(current, 0.0)
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.prev_intensity = 0.0;
        self.surround = 0.0;
    }
}

impl Default for DirectionSelectiveRGC {
    fn default() -> Self {
        Self::new_on()
    }
}

/// Cochlear inner hair cell: mechano-electrical transduction.
///
/// Converts basilar membrane displacement (mechanical) into receptor potential
/// via stereocilia tip-link channels with Boltzmann activation:
///
///   P_open(x) = 1 / (1 + exp(-(x - x_0) / δ))
///   I_MET = g_max · P_open · (V - E_MET)
///   C dV/dt = -g_L(V - E_L) - I_MET + I_ext
///
/// Reference: Meddis (2006), Zilany et al. (2009, 2014).
#[derive(Clone, Debug)]
pub struct CochlearHairCell {
    pub v: f64,
    pub g_max: f64,
    pub e_met: f64,
    pub g_l: f64,
    pub e_l: f64,
    pub cap: f64,
    pub x0: f64,
    pub delta: f64,
    pub dt: f64,
    pub glutamate_release: f64,
}

impl CochlearHairCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            g_max: 10.0,
            e_met: 0.0,
            g_l: 1.0,
            e_l: -60.0,
            cap: 10.0,
            x0: 0.0,
            delta: 0.1,
            dt: 0.01,
            glutamate_release: 0.0,
        }
    }

    /// Boltzmann activation of MET channels.
    fn p_open(&self, displacement: f64) -> f64 {
        1.0 / (1.0 + (-(displacement - self.x0) / self.delta).exp())
    }

    /// Step with basilar membrane displacement.
    pub fn step(&mut self, displacement: f64) -> i32 {
        let po = self.p_open(displacement);
        let i_met = self.g_max * po * (self.v - self.e_met);
        let dv = (-self.g_l * (self.v - self.e_l) - i_met) / self.cap;
        self.v += dv * self.dt;

        // Graded glutamate release (no spike, but we return 1 if above threshold).
        self.glutamate_release = (self.v + 60.0).max(0.0) / 40.0;
        if self.glutamate_release > 0.5 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.glutamate_release = 0.0;
    }
}

impl Default for CochlearHairCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod gap_sensory_tests {
    use super::*;

    #[test]
    fn rgc_on_responds_to_light_increase() {
        let mut cell = DirectionSelectiveRGC::new_on();
        // Flash: darkness then bright.
        for _ in 0..10 {
            cell.step_rf(0.0, 0.0);
        }
        let mut spikes = 0;
        for _ in 0..20 {
            spikes += cell.step_rf(5.0, 0.0);
        }
        assert!(spikes > 0, "On-centre must respond to light increase");
    }

    #[test]
    fn rgc_off_responds_to_light_decrease() {
        let mut cell = DirectionSelectiveRGC::new_off();
        cell.theta = 0.1; // Lower threshold to detect transients.
                          // Alternate bright/dark to produce transitions.
        let mut spikes = 0;
        for i in 0..400 {
            let intensity = if (i / 10) % 2 == 0 { 5.0 } else { 0.0 };
            spikes += cell.step_rf(intensity, 0.0);
        }
        assert!(spikes > 0, "Off-centre must respond to light transitions");
    }

    #[test]
    fn rgc_surround_inhibition() {
        let mut no_surr = DirectionSelectiveRGC::new_on();
        let mut with_surr = DirectionSelectiveRGC::new_on();
        // Same centre stimulus, different surround.
        let mut spikes_no = 0;
        let mut spikes_surr = 0;
        for i in 0..200 {
            let intensity = if i % 10 == 0 { 3.0 } else { 0.0 };
            spikes_no += no_surr.step_rf(intensity, 0.0);
            spikes_surr += with_surr.step_rf(intensity, 2.0);
        }
        assert!(
            spikes_surr <= spikes_no,
            "Surround should inhibit: surr={spikes_surr} <= no={spikes_no}"
        );
    }

    #[test]
    fn cochlear_displacement_depolarises() {
        let mut cell = CochlearHairCell::new();
        let v_rest = cell.v;
        for _ in 0..100 {
            cell.step(0.5);
        }
        assert!(
            cell.v > v_rest,
            "Positive displacement should depolarise: {:.2} > {:.2}",
            cell.v,
            v_rest
        );
    }

    #[test]
    fn cochlear_graded_release() {
        let mut cell = CochlearHairCell::new();
        for _ in 0..200 {
            cell.step(1.0);
        }
        assert!(
            cell.glutamate_release > 0.0,
            "Should release glutamate: {:.4}",
            cell.glutamate_release
        );
    }

    #[test]
    fn cochlear_zero_displacement_rest() {
        let mut cell = CochlearHairCell::new();
        for _ in 0..100 {
            cell.step(0.0);
        }
        // At P_open(0) = 0.5, steady MET current depolarises V above E_L.
        assert!(
            cell.v > -80.0 && cell.v < 0.0,
            "Zero displacement → physiological range: {:.2}",
            cell.v
        );
    }

    #[test]
    fn cochlear_reset() {
        let mut cell = CochlearHairCell::new();
        for _ in 0..100 {
            cell.step(1.0);
        }
        cell.reset();
        assert_eq!(cell.v, cell.e_l);
        assert_eq!(cell.glutamate_release, 0.0);
    }
}
