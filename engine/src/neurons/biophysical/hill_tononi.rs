// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hill and Tononi 2005 hybrid thalamocortical neuron

//! Source cortical-excitatory waking recurrence with optional Ih and IT.

#[derive(Clone, Debug)]
/// Hill-Tononi cortical-waking state and complete scalar-cell configuration.
pub struct HillTononiNeuron {
    pub v: f64,
    pub theta: f64,
    pub d_k: f64,
    pub m_h: f64,
    pub m_t: f64,
    pub h_t: f64,
    pub spike_timer: f64,
    pub g_na_l: f64,
    pub g_k_l: f64,
    pub g_na_p: f64,
    pub g_dk: f64,
    pub g_h: f64,
    pub g_t: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_na_p: f64,
    pub e_dk: f64,
    pub e_h: f64,
    pub e_t: f64,
    pub n_na_p: f64,
    pub n_t: f64,
    pub tau_m: f64,
    pub theta_eq: f64,
    pub tau_theta: f64,
    pub g_spike: f64,
    pub t_spike: f64,
    pub tau_spike: f64,
    pub tau_d: f64,
    pub d_influx_peak: f64,
    pub d_threshold: f64,
    pub d_slope: f64,
    pub d_eq: f64,
    pub d_half: f64,
    pub dt: f64,
}

impl HillTononiNeuron {
    /// Return the publication's cortical-excitatory waking profile.
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta: -51.0,
            d_k: 0.001,
            m_h: 0.2871859013825026,
            m_t: 0.1450215950687922,
            h_t: 0.03732688734412946,
            spike_timer: 0.0,
            g_na_l: 0.2,
            g_k_l: 1.0,
            g_na_p: 0.5,
            g_dk: 0.5,
            g_h: 0.0,
            g_t: 0.0,
            e_na: 30.0,
            e_k: -90.0,
            e_na_p: 30.0,
            e_dk: -90.0,
            e_h: -40.0,
            e_t: 0.0,
            n_na_p: 3.0,
            n_t: 2.0,
            tau_m: 16.0,
            theta_eq: -51.0,
            tau_theta: 2.0,
            g_spike: 1.0,
            t_spike: 2.0,
            tau_spike: 1.75,
            tau_d: 1250.0,
            d_influx_peak: 0.025,
            d_threshold: -10.0,
            d_slope: 5.0,
            d_eq: 0.001,
            d_half: 0.25,
            dt: 0.25,
        }
    }

    fn m_h_inf(v: f64) -> f64 {
        1.0 / (1.0 + ((v + 75.0) / 5.5).exp())
    }

    fn tau_m_h(v: f64) -> f64 {
        1.0 / ((-14.59 - 0.086 * v).exp() + (-1.87 + 0.0701 * v).exp())
    }

    fn m_t_inf(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 59.0) / 6.2).exp())
    }

    fn tau_m_t(v: f64) -> f64 {
        0.22 / ((-(v + 132.0) / 16.7).exp() + ((v + 16.8) / 18.2).exp()) + 0.13
    }

    fn h_t_inf(v: f64) -> f64 {
        1.0 / (1.0 + ((v + 83.0) / 4.0).exp())
    }

    fn tau_h_t(v: f64) -> f64 {
        8.2 + (56.6 + 0.27 * ((v + 115.2) / 5.0).exp()) / (1.0 + ((v + 86.0) / 3.2).exp())
    }

    fn d_k_inf(&self, v: f64) -> f64 {
        let influx = self.d_influx_peak / (1.0 + (-(v - self.d_threshold) / self.d_slope).exp());
        self.tau_d * influx + self.d_eq
    }

    fn derivatives(&self, state: [f64; 6], current: f64, spike_active: bool) -> [f64; 6] {
        let [v, theta, d_k, m_h, m_t, h_t] = state;
        let m_na_p = 1.0 / (1.0 + (-(v + 55.7) / 7.7).exp());
        let d_activation = 1.0 / (1.0 + (self.d_half / d_k.max(1e-15)).powf(3.5));
        let i_na_l = -self.g_na_l * (v - self.e_na);
        let i_k_l = -self.g_k_l * (v - self.e_k);
        let i_na_p = -self.g_na_p * m_na_p.powf(self.n_na_p) * (v - self.e_na_p);
        let i_dk = -self.g_dk * d_activation * (v - self.e_dk);
        let i_h = -self.g_h * m_h * (v - self.e_h);
        let i_t = -self.g_t * m_t.powf(self.n_t) * h_t * (v - self.e_t);
        let i_spike = if spike_active {
            -self.g_spike * (v - self.e_k) / self.tau_spike
        } else {
            0.0
        };
        [
            (i_na_l + i_k_l + i_na_p + i_dk + i_h + i_t + current) / self.tau_m + i_spike,
            -(theta - self.theta_eq) / self.tau_theta,
            (self.d_k_inf(v) - d_k) / self.tau_d,
            (Self::m_h_inf(v) - m_h) / Self::tau_m_h(v),
            (Self::m_t_inf(v) - m_t) / Self::tau_m_t(v),
            (Self::h_t_inf(v) - h_t) / Self::tau_h_t(v),
        ]
    }

    fn shifted(state: [f64; 6], slope: [f64; 6], scale: f64) -> [f64; 6] {
        std::array::from_fn(|index| state[index] + scale * slope[index])
    }

    fn candidate(&self, state: [f64; 6], current: f64, spike_active: bool) -> [f64; 6] {
        let k1 = self.derivatives(state, current, spike_active);
        let k2 = self.derivatives(
            Self::shifted(state, k1, 0.5 * self.dt),
            current,
            spike_active,
        );
        let k3 = self.derivatives(
            Self::shifted(state, k2, 0.5 * self.dt),
            current,
            spike_active,
        );
        let k4 = self.derivatives(Self::shifted(state, k3, self.dt), current, spike_active);
        std::array::from_fn(|index| {
            state[index]
                + self.dt * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0
        })
    }

    fn configuration_is_valid(&self) -> bool {
        let values = [
            self.v,
            self.theta,
            self.d_k,
            self.m_h,
            self.m_t,
            self.h_t,
            self.spike_timer,
            self.g_na_l,
            self.g_k_l,
            self.g_na_p,
            self.g_dk,
            self.g_h,
            self.g_t,
            self.e_na,
            self.e_k,
            self.e_na_p,
            self.e_dk,
            self.e_h,
            self.e_t,
            self.n_na_p,
            self.n_t,
            self.tau_m,
            self.theta_eq,
            self.tau_theta,
            self.g_spike,
            self.t_spike,
            self.tau_spike,
            self.tau_d,
            self.d_influx_peak,
            self.d_threshold,
            self.d_slope,
            self.d_eq,
            self.d_half,
            self.dt,
        ];
        values.iter().all(|value| value.is_finite())
            && self.d_k >= 0.0
            && self.spike_timer >= 0.0
            && [
                self.g_na_l,
                self.g_k_l,
                self.g_na_p,
                self.g_dk,
                self.g_h,
                self.g_t,
                self.g_spike,
                self.d_influx_peak,
                self.d_eq,
            ]
            .iter()
            .all(|value| *value >= 0.0)
            && [
                self.n_na_p,
                self.n_t,
                self.tau_m,
                self.tau_theta,
                self.t_spike,
                self.tau_spike,
                self.tau_d,
                self.d_slope,
                self.d_half,
                self.dt,
            ]
            .iter()
            .all(|value| *value > 0.0)
    }

    /// Advance one source RK4 step, committing state only after validation.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !self.configuration_is_valid() {
            return Err("Hill-Tononi configuration and current must be finite and physical");
        }
        let refractory = self.spike_timer > 0.0;
        let state = [self.v, self.theta, self.d_k, self.m_h, self.m_t, self.h_t];
        let mut next = self.candidate(state, current, refractory);
        if !next.iter().all(|value| value.is_finite()) || next[2] < 0.0 {
            return Err("Hill-Tononi candidate must be finite and physical");
        }
        let mut timer = (self.spike_timer - self.dt).max(0.0);
        let spike = !refractory && next[0] >= next[1];
        if spike {
            next[0] = self.e_na;
            next[1] = self.e_na;
            timer = self.t_spike;
        }
        [self.v, self.theta, self.d_k, self.m_h, self.m_t, self.h_t] = next;
        self.spike_timer = timer;
        Ok(i32::from(spike))
    }

    /// Advance one step through the historical integer-event API.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Restore the source cortical-excitatory waking initial state.
    pub fn reset(&mut self) {
        self.v = -70.0;
        self.theta = -51.0;
        self.d_k = 0.001;
        self.m_h = Self::m_h_inf(self.v);
        self.m_t = Self::m_t_inf(self.v);
        self.h_t = Self::h_t_inf(self.v);
        self.spike_timer = 0.0;
    }
}

impl Default for HillTononiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_defaults_and_python_anchor() {
        let mut neuron = HillTononiNeuron::new();
        assert_eq!(
            [neuron.v, neuron.theta, neuron.d_k, neuron.dt],
            [-70.0, -51.0, 0.001, 0.25]
        );
        assert_eq!(neuron.step(12.0), 0);
        let expected = [
            -69.81228106951788,
            -51.0,
            0.0010000391293823398,
            0.2871847356365222,
            0.1451785200593081,
            0.037318086618308974,
        ];
        let observed = [
            neuron.v,
            neuron.theta,
            neuron.d_k,
            neuron.m_h,
            neuron.m_t,
            neuron.h_t,
        ];
        for (actual, target) in observed.into_iter().zip(expected) {
            assert!((actual - target).abs() < 2e-12, "{actual} != {target}");
        }
    }

    #[test]
    fn spike_sets_dynamic_threshold_and_pulse() {
        let mut neuron = HillTononiNeuron::new();
        neuron.v = -50.0;
        neuron.theta = -51.0;
        assert_eq!(neuron.try_step(0.0), Ok(1));
        assert_eq!(
            [neuron.v, neuron.theta, neuron.spike_timer],
            [30.0, 30.0, 2.0]
        );
        assert_eq!(neuron.try_step(0.0), Ok(0));
        assert_eq!(neuron.spike_timer, 1.75);
        assert!(neuron.v < 30.0);
    }

    #[test]
    fn invalid_step_is_atomic() {
        let mut neuron = HillTononiNeuron::new();
        let before = [
            neuron.v,
            neuron.theta,
            neuron.d_k,
            neuron.m_h,
            neuron.m_t,
            neuron.h_t,
        ];
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!(
            [
                neuron.v,
                neuron.theta,
                neuron.d_k,
                neuron.m_h,
                neuron.m_t,
                neuron.h_t
            ],
            before
        );
    }
}
