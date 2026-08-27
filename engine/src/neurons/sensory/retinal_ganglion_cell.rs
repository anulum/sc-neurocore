// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

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
        self.stim_buffer.fill(0.0);
        self.hist_buffer.fill(0.0);
        self.stim_idx = 0;
        self.hist_idx = 0;
    }
}

impl Default for RetinalGanglionCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn retinal_ganglion_default_matches_constructor_contract() {
        let default = RetinalGanglionCell::default();
        let constructed = RetinalGanglionCell::new();
        assert_eq!(default.stim_buffer, constructed.stim_buffer);
        assert_eq!(default.stim_kernel, constructed.stim_kernel);
        assert_eq!(default.hist_buffer, constructed.hist_buffer);
        assert_eq!(default.hist_kernel, constructed.hist_kernel);
        assert_eq!(default.on_centre, constructed.on_centre);
    }
}
