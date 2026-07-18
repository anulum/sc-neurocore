// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for resonate-and-fire

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
            x: 0.0,
            y: 0.0,
            b: -1.0,
            omega: 10.0,
            threshold: 1.0,
            dt: 0.01,
        }
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !current.is_finite() || !validate_resonate_and_fire(self) {
            return Err("resonate-and-fire state/current must be finite and well-formed");
        }
        let old_y = self.y;
        let (next_x, next_y) =
            resonate_exact_flow(self.x, self.y, current, self.b, self.omega, self.dt)?;
        if old_y < self.threshold && next_y >= self.threshold {
            self.x = 0.0;
            self.y = self.threshold;
            return Ok(1);
        }
        self.x = next_x;
        self.y = next_y;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

impl Default for ResonateAndFireNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_resonate_and_fire(state: &ResonateAndFireNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.b.is_finite()
        && state.omega.is_finite()
        && state.omega > 0.0
        && state.threshold.is_finite()
        && state.threshold > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
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
    let x_ss = -b * current / denominator;
    let y_ss = omega * current / denominator;
    let decay = damping_argument.exp();
    let cos_angle = angle.cos();
    let sin_angle = angle.sin();
    if ![
        denominator,
        damping_argument,
        angle,
        x_ss,
        y_ss,
        decay,
        cos_angle,
        sin_angle,
    ]
    .into_iter()
    .all(f64::is_finite)
        || denominator <= 0.0
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
    fn source_defaults_are_valid() {
        let state = ResonateAndFireNeuron::new();
        assert!(validate_resonate_and_fire(&state));
        assert_eq!((state.b, state.omega, state.threshold, state.dt), (-1.0, 10.0, 1.0, 0.01));
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
        let expected =
            resonate_exact_flow(state.x, state.y, 0.8, state.b, state.omega, state.dt).unwrap();
        assert_eq!(state.step(0.8).unwrap(), 0);
        assert!((state.x - expected.0).abs() < 1.0e-12);
        assert!((state.y - expected.1).abs() < 1.0e-12);
    }

    #[test]
    fn voltage_crossing_installs_source_reset() {
        let mut state = ResonateAndFireNeuron {
            x: 0.0,
            y: 0.99,
            b: 0.0,
            omega: 1.0,
            threshold: 1.0,
            dt: 0.1,
        };
        assert_eq!(state.step(10.0).unwrap(), 1);
        assert_eq!((state.x, state.y), (0.0, 1.0));
        assert_eq!(state.step(0.0).unwrap(), 0);
    }

    #[test]
    fn radius_is_not_the_event_surface() {
        let mut state = ResonateAndFireNeuron {
            x: 2.0,
            y: 0.0,
            b: 0.0,
            omega: 1.0,
            threshold: 1.0,
            dt: 0.01,
        };
        assert_eq!(state.step(0.0).unwrap(), 0);
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = ResonateAndFireNeuron::new();
        state.x = 0.25;
        state.y = -0.5;
        let before = (state.x, state.y);
        assert!(state.step(f64::NAN).is_err());
        assert_eq!((state.x, state.y), before);
    }

    #[test]
    fn invalid_exact_flow_update_does_not_mutate_state() {
        let mut state = ResonateAndFireNeuron::new();
        state.x = 0.25;
        state.y = -0.5;
        state.b = 1.0e308;
        state.dt = 1.0e308;
        let before = (state.x, state.y);
        assert!(state.step(1.0e308).is_err());
        assert_eq!((state.x, state.y), before);
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
        assert_eq!((state.x, state.y), (0.0, 0.0));
        assert_eq!((state.b, state.omega, state.threshold, state.dt), (-0.5, 2.0, 3.0, 0.02));
    }
}
