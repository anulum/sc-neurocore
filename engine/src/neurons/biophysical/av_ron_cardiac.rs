// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Av-Ron Cardiac Ganglion Neuron Model

//! Av-Ron cardiac-ganglion Type III burster dynamics.

/// Av-Ron cardiac ganglion — Type III burster. Av-Ron et al. 1991.
#[derive(Clone, Debug)]
pub struct AvRonCardiacNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub s: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_s: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_s: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl AvRonCardiacNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            h: 0.6,
            n: 0.3,
            s: 0.5,
            g_na: 80.0,
            g_k: 40.0,
            g_s: 20.0,
            g_l: 0.1,
            e_na: 40.0,
            e_k: -80.0,
            e_s: -25.0,
            e_l: -60.0,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }

    fn finite_values(values: &[f64]) -> bool {
        values.iter().all(|value| value.is_finite())
    }

    fn gate_in_range(value: f64) -> bool {
        (0.0..=1.0).contains(&value)
    }

    fn bounded_exp(value: f64) -> f64 {
        value.clamp(-745.0, 709.0).exp()
    }

    fn sigmoid_pos(value: f64) -> f64 {
        1.0 / (1.0 + Self::bounded_exp(-value))
    }

    fn sigmoid_neg(value: f64) -> f64 {
        1.0 / (1.0 + Self::bounded_exp(value))
    }

    fn valid_runtime(&self) -> bool {
        Self::finite_values(&[
            self.v,
            self.h,
            self.n,
            self.s,
            self.g_na,
            self.g_k,
            self.g_s,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_s,
            self.e_l,
            self.dt,
            self.v_threshold,
        ]) && self.dt > 0.0
            && self.g_na >= 0.0
            && self.g_k >= 0.0
            && self.g_s >= 0.0
            && self.g_l >= 0.0
            && Self::gate_in_range(self.h)
            && Self::gate_in_range(self.n)
            && Self::gate_in_range(self.s)
    }

    fn rates(&self, voltage: f64) -> [f64; 7] {
        [
            Self::sigmoid_pos((voltage + 40.0) / 7.0),
            Self::sigmoid_neg((voltage + 45.0) / 5.0),
            Self::sigmoid_pos((voltage + 40.0) / 15.0),
            Self::sigmoid_neg((voltage + 35.0) / 3.0),
            1.0 + 12.0 * Self::sigmoid_neg((voltage + 50.0) / 8.0),
            1.0 + 8.0 * Self::sigmoid_neg((voltage + 35.0) / 8.0),
            200.0 + 1000.0 * Self::sigmoid_neg((voltage + 30.0) / 5.0),
        ]
    }

    fn derivatives(&self, state: [f64; 4], current: f64) -> [f64; 4] {
        let [voltage, h_gate, n_gate, s_gate] = state;
        if !Self::finite_values(&state)
            || !Self::gate_in_range(h_gate)
            || !Self::gate_in_range(n_gate)
            || !Self::gate_in_range(s_gate)
        {
            return [f64::NAN; 4];
        }
        let rates = self.rates(voltage);
        let i_na = self.g_na * rates[0].powi(3) * h_gate * (voltage - self.e_na);
        let i_k = self.g_k * n_gate.powi(4) * (voltage - self.e_k);
        let i_s = self.g_s * s_gate * (voltage - self.e_s);
        let i_l = self.g_l * (voltage - self.e_l);
        [
            -i_na - i_k - i_s - i_l + current,
            (rates[1] - h_gate) / rates[4],
            (rates[2] - n_gate) / rates[5],
            (rates[3] - s_gate) / rates[6],
        ]
    }

    fn add_scaled(state: [f64; 4], slope: [f64; 4], scale: f64) -> [f64; 4] {
        [
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        ]
    }

    fn rk4_candidate(&self, current: f64) -> Option<[f64; 4]> {
        let state = [self.v, self.h, self.n, self.s];
        let half_dt = 0.5 * self.dt;
        let k1 = self.derivatives(state, current);
        let k2 = self.derivatives(Self::add_scaled(state, k1, half_dt), current);
        let k3 = self.derivatives(Self::add_scaled(state, k2, half_dt), current);
        let k4 = self.derivatives(Self::add_scaled(state, k3, self.dt), current);
        let candidate = [
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        ];
        if Self::finite_values(&candidate)
            && Self::gate_in_range(candidate[1])
            && Self::gate_in_range(candidate[2])
            && Self::gate_in_range(candidate[3])
        {
            Some(candidate)
        } else {
            None
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid_runtime() {
            return 0;
        }
        let v_prev = self.v;
        let Some(candidate) = self.rk4_candidate(current) else {
            return 0;
        };
        self.v = candidate[0];
        self.h = candidate[1];
        self.n = candidate[2];
        self.s = candidate[3];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -60.0;
        self.h = 0.6;
        self.n = 0.3;
        self.s = 0.5;
    }
}
impl Default for AvRonCardiacNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = AvRonCardiacNeuron::default();
        let constructed = AvRonCardiacNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn invalid_derivative_state_is_rejected() {
        let n = AvRonCardiacNeuron::new();
        assert!(n
            .derivatives([n.v, f64::NAN, n.n, n.s], 0.0)
            .iter()
            .all(|value| value.is_nan()));
    }

    #[test]
    fn nonfinite_rk4_candidate_preserves_state() {
        let mut n = AvRonCardiacNeuron::new();
        let before = (n.v, n.h, n.n, n.s);
        assert_eq!(n.step(f64::MAX), 0);
        assert_eq!((n.v, n.h, n.n, n.s), before);
    }

    #[test]
    fn avron_fires() {
        let mut n = AvRonCardiacNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- AvRonCardiac --
    #[test]
    fn avron_rk4_reference_point() {
        let mut n = AvRonCardiacNeuron::new();
        n.v = -55.0;
        n.h = 0.55;
        n.n = 0.35;
        n.s = 0.45;
        assert_eq!(n.step(2.0), 0);
        assert!((n.v - (-50.0840498399381)).abs() < 1e-12);
        assert!((n.h - 0.5506609782132562).abs() < 1e-15);
        assert!((n.n - 0.34988677751350306).abs() < 1e-15);
        assert!((n.s - 0.4500091998827305).abs() < 1e-15);
    }
    #[test]
    fn avron_invalid_input_preserves_state() {
        let mut n = AvRonCardiacNeuron::new();
        n.v = -52.0;
        n.h = 0.4;
        n.n = 0.2;
        n.s = 0.8;
        let before = (n.v, n.h, n.n, n.s);
        assert_eq!(n.step(f64::NAN), 0);
        n.dt = 0.0;
        assert_eq!(n.step(1.0), 0);
        assert_eq!((n.v, n.h, n.n, n.s), before);
    }
    #[test]
    fn avron_rejects_corrupted_gate_state() {
        let mut n = AvRonCardiacNeuron::new();
        n.h = 1.2;
        let before = (n.v, n.h, n.n, n.s);
        assert_eq!(n.step(1.0), 0);
        assert_eq!((n.v, n.h, n.n, n.s), before);
    }
    #[test]
    fn avron_slow_current_remains_physical() {
        let mut n = AvRonCardiacNeuron::new();
        for _ in 0..1000 {
            n.step(5.0);
            assert!((0.0..=1.0).contains(&n.h));
            assert!((0.0..=1.0).contains(&n.n));
            assert!((0.0..=1.0).contains(&n.s));
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn avron_reset_clears_state() {
        let mut n = AvRonCardiacNeuron::new();
        n.v = -40.0;
        n.h = 0.1;
        n.n = 0.9;
        n.s = 0.2;
        n.reset();
        assert_eq!((n.v, n.h, n.n, n.s), (-60.0, 0.6, 0.3, 0.5));
    }
}
