// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for resonate_and_fire

#[derive(Debug, Clone)]
pub struct ResonateAndFireNeuron {
    pub x: f64,
    pub y: f64,
    pub b: f64,
    pub omega: f64,
    pub threshold: f64,
    pub dt: f64,
}

impl ResonateAndFireNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            y: 0.0_f64,
            b: -0.1_f64,
            omega: 1.0_f64,
            threshold: 1.0_f64,
            dt: 0.05_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !i_ext.is_finite() || !validate_resonate_and_fire(self) {
            return Err("resonate-and-fire state/current must be finite and well-formed");
        }

        let (next_x, next_y) = resonate_exact_flow(self.x, self.y, i_ext, self.b, self.omega, self.dt)?;
        let radius = next_x.hypot(next_y);
        if !next_x.is_finite() || !next_y.is_finite() || !radius.is_finite() {
            return Err("resonate-and-fire exact-flow update became non-finite");
        }

        if radius >= self.threshold {
            self.x = 0.0_f64;
            self.y = 0.0_f64;
            return Ok(1);
        }
        self.x = next_x;
        self.y = next_y;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.x = 0.0_f64;
        self.y = 0.0_f64;
    }
}

pub fn validate_resonate_and_fire(state: &ResonateAndFireNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.b.is_finite()
        && state.omega.is_finite()
        && state.omega > 0.0_f64
        && state.threshold.is_finite()
        && state.threshold > 0.0_f64
        && state.dt.is_finite()
        && state.dt > 0.0_f64
}

fn resonate_exact_flow(
    x: f64,
    y: f64,
    current: f64,
    b: f64,
    omega: f64,
    dt: f64,
) -> Result<(f64, f64), &'static str> {
    let denominator = b * b + omega * omega;
    let damping_argument = b * dt;
    let angle = omega * dt;
    if !denominator.is_finite()
        || denominator <= 0.0_f64
        || !damping_argument.is_finite()
        || !angle.is_finite()
    {
        return Err("resonate-and-fire exact-flow update became non-finite");
    }

    let x_ss = -b * current / denominator;
    let y_ss = omega * current / denominator;
    let decay = damping_argument.exp();
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    if !x_ss.is_finite()
        || !y_ss.is_finite()
        || !decay.is_finite()
        || !cos_angle.is_finite()
        || !sin_angle.is_finite()
    {
        return Err("resonate-and-fire exact-flow update became non-finite");
    }

    let dx = x - x_ss;
    let dy = y - y_ss;
    let next_x = x_ss + decay * (dx * cos_angle - dy * sin_angle);
    let next_y = y_ss + decay * (dx * sin_angle + dy * cos_angle);
    if !next_x.is_finite() || !next_y.is_finite() {
        return Err("resonate-and-fire exact-flow update became non-finite");
    }
    Ok((next_x, next_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonate_and_fire_new() {
        let state = ResonateAndFireNeuron::new();
        assert!(validate_resonate_and_fire(&state));
    }

    #[test]
    fn test_resonate_and_fire_step() {
        let mut state = ResonateAndFireNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    fn exact_flow(
        x: f64,
        y: f64,
        current: f64,
        b: f64,
        omega: f64,
        dt: f64,
    ) -> (f64, f64) {
        let denominator = b * b + omega * omega;
        let x_ss = -b * current / denominator;
        let y_ss = omega * current / denominator;
        let dx = x - x_ss;
        let dy = y - y_ss;
        let decay = (b * dt).exp();
        let angle = omega * dt;
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        (
            x_ss + decay * (dx * cos_angle - dy * sin_angle),
            y_ss + decay * (dx * sin_angle + dy * cos_angle),
        )
    }

    #[test]
    fn exact_flow_without_spike() {
        let mut state = ResonateAndFireNeuron {
            x: 0.3,
            y: -0.2,
            b: -0.2,
            omega: 1.7,
            threshold: 100.0,
            dt: 1.25,
        };
        let (expected_x, expected_y) =
            exact_flow(state.x, state.y, 0.8, state.b, state.omega, state.dt);

        let spike = state.step(0.8).unwrap();

        assert_eq!(spike, 0);
        assert!((state.x - expected_x).abs() < 1.0e-12);
        assert!((state.y - expected_y).abs() < 1.0e-12);
    }

    #[test]
    fn large_timestep_damped_flow_stays_bounded() {
        let mut state = ResonateAndFireNeuron {
            x: 1.0,
            y: 0.0,
            b: -0.5,
            omega: 4.0,
            threshold: 100.0,
            dt: 10.0,
        };

        state.step(0.0).unwrap();

        assert!((state.x.hypot(state.y) - (-5.0_f64).exp()).abs() < 1.0e-12);
    }

    #[test]
    fn positive_current_spikes_and_resets() {
        let mut state = ResonateAndFireNeuron::new();
        for _ in 0..50_000 {
            if state.step(2.0).unwrap() == 1 {
                assert_eq!(state.x, 0.0);
                assert_eq!(state.y, 0.0);
                return;
            }
        }
        panic!("positive current should produce at least one spike");
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = ResonateAndFireNeuron::new();
        state.x = 0.25;
        state.y = -0.5;

        assert!(state.step(f64::NAN).is_err());
        assert_eq!(state.x, 0.25);
        assert_eq!(state.y, -0.5);
    }

    #[test]
    fn invalid_exact_flow_update_does_not_mutate_state() {
        let mut state = ResonateAndFireNeuron::new();
        state.x = 0.25;
        state.y = -0.5;
        state.threshold = 1.0e308;
        state.b = 1.0e308;
        state.dt = 1.0e308;

        assert!(state.step(1.0e308).is_err());
        assert_eq!(state.x, 0.25);
        assert_eq!(state.y, -0.5);
    }

    #[test]
    fn reset_preserves_parameters() {
        let mut state = ResonateAndFireNeuron {
            x: 0.5,
            y: -0.25,
            b: -0.5,
            omega: 2.0,
            threshold: 3.0,
            dt: 0.02,
        };

        state.reset();

        assert_eq!(state.x, 0.0);
        assert_eq!(state.y, 0.0);
        assert_eq!(state.b, -0.5);
        assert_eq!(state.omega, 2.0);
        assert_eq!(state.threshold, 3.0);
        assert_eq!(state.dt, 0.02);
    }
}
