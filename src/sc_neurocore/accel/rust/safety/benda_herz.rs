// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for benda_herz

#[derive(Debug, Clone)]
pub struct BendaHerzNeuron {
    pub a: f64,
    pub f_max: f64,
    pub beta: f64,
    pub i_half: f64,
    pub tau_a: f64,
    pub delta_a: f64,
    pub dt: f64,
    pub _rng: f64,
}

impl BendaHerzNeuron {
    pub fn new() -> Self {
        Self {
            a: 0.0_f64,
            f_max: 200.0_f64,
            beta: 0.1_f64,
            i_half: 5.0_f64,
            tau_a: 100.0_f64,
            delta_a: 0.5_f64,
            dt: 1.0_f64,
            _rng: 0.0_f64,
        }
    }

    pub fn _f_onset(&self, x: f64) -> f64 {
        let z = self.beta * (x - self.i_half);
        if z == f64::INFINITY {
            return self.f_max;
        }
        if z == f64::NEG_INFINITY {
            return 0.0;
        }
        if !z.is_finite() {
            return f64::NAN;
        }
        if z >= 0.0 {
            self.f_max / (1.0 + (-z).exp())
        } else {
            let exp_z = z.exp();
            self.f_max * exp_z / (1.0 + exp_z)
        }
    }

    fn adaptation_rhs(&self, a: f64, i_ext: f64) -> Option<(f64, f64)> {
        if !a.is_finite() || a < 0.0 {
            return None;
        }
        let rate = self._f_onset(i_ext - a);
        if !rate.is_finite() || rate < 0.0 || rate > self.f_max {
            return None;
        }
        Some((-a / self.tau_a + self.delta_a * rate, rate))
    }

    pub fn rk4_candidate(&self, i_ext: f64) -> Option<(f64, f64)> {
        let (k1, r1) = self.adaptation_rhs(self.a, i_ext)?;
        let (k2, r2) = self.adaptation_rhs(self.a + 0.5 * self.dt * k1, i_ext)?;
        let (k3, r3) = self.adaptation_rhs(self.a + 0.5 * self.dt * k2, i_ext)?;
        let (k4, r4) = self.adaptation_rhs(self.a + self.dt * k3, i_ext)?;
        let next_a = self.a + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        let average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0;
        let hazard = average_rate * self.dt / 1000.0;
        if !next_a.is_finite() || next_a < 0.0 || !hazard.is_finite() || hazard < 0.0 {
            return None;
        }
        let probability = -(-hazard).exp_m1();
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return None;
        }
        Some((next_a, probability))
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !validate_benda_herz(self) {
            return 0;
        }

        let Some((next_a, p)) = self.rk4_candidate(i_ext) else {
            return 0;
        };

        self.a = next_a;
        if self._rng < p {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.a = 0.0_f64;
    }
}

#[cfg(test)]
fn rk4_reference(n: &BendaHerzNeuron, i_ext: f64) -> (f64, f64) {
    let rhs = |a: f64| {
        let rate = n._f_onset(i_ext - a);
        (-a / n.tau_a + n.delta_a * rate, rate)
    };
    let (k1, r1) = rhs(n.a);
    let (k2, r2) = rhs(n.a + 0.5 * n.dt * k1);
    let (k3, r3) = rhs(n.a + 0.5 * n.dt * k2);
    let (k4, r4) = rhs(n.a + n.dt * k3);
    let next_a = n.a + (n.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    let average_rate = (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0;
    let probability = -(-(average_rate * n.dt / 1000.0)).exp_m1();
    (next_a, probability)
}

pub fn validate_benda_herz(state: &BendaHerzNeuron) -> bool {
    state.a.is_finite()
        && state.a >= 0.0
        && state.f_max.is_finite()
        && state.f_max > 0.0
        && state.beta.is_finite()
        && state.beta > 0.0
        && state.i_half.is_finite()
        && state.tau_a.is_finite()
        && state.tau_a > 0.0
        && state.delta_a.is_finite()
        && state.delta_a >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state._rng.is_finite()
        && state._rng >= 0.0
        && state._rng < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk4_candidate_matches_reference() {
        let state = BendaHerzNeuron {
            a: 0.35,
            dt: 0.25,
            _rng: 0.5,
            ..BendaHerzNeuron::new()
        };
        let expected = rk4_reference(&state, 12.5);
        let candidate = state.rk4_candidate(12.5).expect("valid RK4 candidate");

        assert!((candidate.0 - expected.0).abs() < 1.0e-12);
        assert!((candidate.1 - expected.1).abs() < 1.0e-15);
    }

    #[test]
    fn step_commits_rk4_candidate() {
        let mut state = BendaHerzNeuron {
            a: 0.25,
            dt: 0.5,
            _rng: 0.5,
            ..BendaHerzNeuron::new()
        };
        let expected = rk4_reference(&state, 15.0);

        state.step(15.0);

        assert!((state.a - expected.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_benda_herz_new() {
        let state = BendaHerzNeuron::new();
        assert!(validate_benda_herz(&state));
    }

    #[test]
    fn test_benda_herz_step() {
        let mut state = BendaHerzNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn onset_rate_is_sigmoid_bounded_and_monotonic() {
        let state = BendaHerzNeuron::new();
        let low = state._f_onset(0.0);
        let mid = state._f_onset(state.i_half);
        let high = state._f_onset(50.0);

        assert!(low >= 0.0);
        assert!(high <= state.f_max);
        assert!(low < mid && mid < high);
        assert!((mid - state.f_max / 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn adaptation_increases_under_subunit_probability_drive() {
        let mut state = BendaHerzNeuron::new();
        let before = state.a;

        for _ in 0..100 {
            state.step(10.0);
        }

        assert!(state.a > before);
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = BendaHerzNeuron::new();
        state.a = 0.5;

        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.a, 0.5);
    }

    #[test]
    fn supraunit_probability_does_not_mutate_state() {
        let mut state = BendaHerzNeuron::new();
        state.f_max = 2_000.0;
        state.a = 0.5;

        assert_eq!(state.step(100.0), 0);
        assert_eq!(state.a, 0.5);
    }

    #[test]
    fn non_finite_adaptation_update_does_not_mutate_state() {
        let mut state = BendaHerzNeuron::new();
        state.f_max = 1.0e-306;
        state.delta_a = 1.0e308;
        state.dt = 1.0e308;
        state.a = 0.5;

        assert_eq!(state.step(100.0), 0);
        assert_eq!(state.a, 0.5);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = BendaHerzNeuron {
            a: 42.0,
            f_max: 150.0,
            beta: 0.2,
            i_half: 6.0,
            tau_a: 120.0,
            delta_a: 0.7,
            dt: 0.5,
            _rng: 0.25,
        };

        state.reset();

        assert_eq!(state.a, 0.0);
        assert_eq!(state.f_max, 150.0);
        assert_eq!(state.beta, 0.2);
        assert_eq!(state.i_half, 6.0);
        assert_eq!(state.tau_a, 120.0);
        assert_eq!(state.delta_a, 0.7);
        assert_eq!(state.dt, 0.5);
        assert_eq!(state._rng, 0.25);
    }
}
