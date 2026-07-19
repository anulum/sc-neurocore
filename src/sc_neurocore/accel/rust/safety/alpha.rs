// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Standalone Rust safety mirror for dual alpha-synapse LIF

#[derive(Debug, Clone)]
pub struct AlphaNeuron {
    pub v: f64,
    pub a_exc: f64,
    pub i_exc: f64,
    pub a_inh: f64,
    pub i_inh: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub tau_v: f64,
    pub tau_exc: f64,
    pub tau_inh: f64,
    pub dt: f64,
}

impl AlphaNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            a_exc: 0.0,
            i_exc: 0.0,
            a_inh: 0.0,
            i_inh: 0.0,
            v_rest: 0.0,
            v_threshold: 1.0,
            tau_v: 20.0,
            tau_exc: 5.0,
            tau_inh: 10.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, exc_current: f64, inh_current: f64) -> Result<i32, &'static str> {
        if !exc_current.is_finite() || !inh_current.is_finite() || !validate_alpha(self) {
            return Err("alpha state/current must be finite and well-formed");
        }
        let (a_exc_next, i_exc_next) =
            filter_candidates(self.a_exc, self.i_exc, exc_current, self.tau_exc, self.dt)?;
        let (a_inh_next, i_inh_next) =
            filter_candidates(self.a_inh, self.i_inh, inh_current, self.tau_inh, self.dt)?;
        let exc_steady = self.tau_exc * exc_current;
        let inh_steady = self.tau_inh * inh_current;
        let v_steady = self.v_rest + exc_steady - inh_steady;
        let decay_v = (-self.dt / self.tau_v).exp();
        let v_next = v_steady
            + (self.v - v_steady) * decay_v
            + drive_contribution(
                self.i_exc - exc_steady,
                self.a_exc - exc_steady,
                self.tau_exc,
                self.tau_v,
                self.dt,
            )?
            - drive_contribution(
                self.i_inh - inh_steady,
                self.a_inh - inh_steady,
                self.tau_inh,
                self.tau_v,
                self.dt,
            )?;
        if !v_next.is_finite() {
            return Err("alpha exact-flow update became non-finite");
        }
        self.a_exc = a_exc_next;
        self.i_exc = i_exc_next;
        self.a_inh = a_inh_next;
        self.i_inh = i_inh_next;
        if v_next >= self.v_threshold {
            self.v = self.v_rest;
            return Ok(1);
        }
        self.v = v_next;
        Ok(0)
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.a_exc = 0.0;
        self.i_exc = 0.0;
        self.a_inh = 0.0;
        self.i_inh = 0.0;
    }
}

impl Default for AlphaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_alpha(state: &AlphaNeuron) -> bool {
    state.v.is_finite()
        && state.a_exc.is_finite()
        && state.i_exc.is_finite()
        && state.a_inh.is_finite()
        && state.i_inh.is_finite()
        && state.v_rest.is_finite()
        && state.v_threshold.is_finite()
        && state.v_threshold > state.v_rest
        && state.tau_v.is_finite()
        && state.tau_v > 0.0
        && state.tau_exc.is_finite()
        && state.tau_exc > 0.0
        && state.tau_inh.is_finite()
        && state.tau_inh > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

fn filter_candidates(
    rise_state: f64,
    current_state: f64,
    drive: f64,
    tau: f64,
    dt: f64,
) -> Result<(f64, f64), &'static str> {
    let steady_state = tau * drive;
    let rise_delta = rise_state - steady_state;
    let current_delta = current_state - steady_state;
    let decay = (-dt / tau).exp();
    let rise_next = steady_state + rise_delta * decay;
    let current_next = steady_state + decay * (current_delta + rise_delta * dt / tau);
    if !rise_next.is_finite() || !current_next.is_finite() {
        return Err("alpha exact-flow update became non-finite");
    }
    Ok((rise_next, current_next))
}

fn drive_contribution(
    current_delta: f64,
    rise_delta: f64,
    tau_drive: f64,
    tau_v: f64,
    dt: f64,
) -> Result<f64, &'static str> {
    let rate_v = 1.0 / tau_v;
    let rate_drive = 1.0 / tau_drive;
    let decay_v = (-dt / tau_v).exp();
    let decay_drive = (-dt / tau_drive).exp();
    let contribution = if (rate_v - rate_drive).abs() <= 1.0e-14 {
        rate_v * decay_v * (current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive))
    } else {
        let rate_delta = rate_v - rate_drive;
        let first_order = current_delta * (decay_drive - decay_v) / rate_delta;
        let second_order = rise_delta / tau_drive
            * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
            / (rate_delta * rate_delta);
        rate_v * (first_order + second_order)
    };
    if !contribution.is_finite() {
        return Err("alpha exact-flow update became non-finite");
    }
    Ok(contribution)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalogue_defaults_are_valid() {
        let state = AlphaNeuron::new();
        assert!(validate_alpha(&state));
        assert_eq!(
            (
                state.v,
                state.v_threshold,
                state.tau_v,
                state.tau_exc,
                state.tau_inh,
                state.dt
            ),
            (0.0, 1.0, 20.0, 5.0, 10.0, 1.0)
        );
    }

    #[test]
    fn filter_matches_exact_alpha_cascade() {
        let (rise_next, current_next) = filter_candidates(0.25, 0.1, 2.0, 5.0, 0.5).unwrap();
        let steady = 5.0 * 2.0;
        let decay = (-0.5_f64 / 5.0).exp();
        let expected_rise = steady + (0.25 - steady) * decay;
        let expected_current = steady + decay * ((0.1 - steady) + (0.25 - steady) * 0.5 / 5.0);
        assert!((rise_next - expected_rise).abs() < 1.0e-12);
        assert!((current_next - expected_current).abs() < 1.0e-12);
    }

    #[test]
    fn drive_contribution_handles_equal_time_constants() {
        let exact = drive_contribution(0.3, 0.2, 20.0, 20.0, 0.5).unwrap();
        let rate = 1.0 / 20.0;
        let decay = (-0.5_f64 / 20.0).exp();
        let expected = rate * decay * (0.3 * 0.5 + 0.2 * 0.5 * 0.5 / (2.0 * 20.0));
        assert!((exact - expected).abs() < 1.0e-12);
    }

    #[test]
    fn spike_resets_only_the_membrane() {
        let mut state = AlphaNeuron {
            v: 0.9,
            a_exc: 0.4,
            i_exc: 0.6,
            a_inh: 0.2,
            i_inh: 0.1,
            v_threshold: 0.5,
            ..AlphaNeuron::new()
        };
        let before = (state.a_exc, state.i_exc, state.a_inh, state.i_inh);
        assert_eq!(state.step(0.0, 0.0).unwrap(), 1);
        assert_eq!(state.v, 0.0);
        let decay_exc = (-1.0_f64 / 5.0).exp();
        let decay_inh = (-1.0_f64 / 10.0).exp();
        assert!((state.a_exc - before.0 * decay_exc).abs() < 1.0e-12);
        assert!((state.i_exc - decay_exc * (before.1 + before.0 * 1.0 / 5.0)).abs() < 1.0e-12);
        assert!((state.a_inh - before.2 * decay_inh).abs() < 1.0e-12);
        assert!((state.i_inh - decay_inh * (before.3 + before.2 * 1.0 / 10.0)).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_current_does_not_mutate_state() {
        let mut state = AlphaNeuron::new();
        state.v = 0.25;
        state.a_exc = 0.5;
        let before = (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh);
        assert!(state.step(f64::NAN, 0.0).is_err());
        assert_eq!(
            (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh),
            before
        );
    }

    #[test]
    fn invalid_configuration_does_not_mutate_state() {
        let mut state = AlphaNeuron::new();
        state.v = 0.25;
        state.a_exc = 0.5;
        state.tau_exc = -1.0;
        let before = (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh);
        assert!(state.step(0.5, 0.0).is_err());
        assert_eq!(
            (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh),
            before
        );
    }

    #[test]
    fn invalid_update_does_not_mutate_state() {
        let mut state = AlphaNeuron::new();
        state.v = -f64::MAX;
        let before = (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh);
        assert!(state.step(f64::MAX, 0.0).is_err());
        assert_eq!(
            (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh),
            before
        );
    }

    #[test]
    fn reset_preserves_configuration() {
        let mut state = AlphaNeuron {
            v: 0.4,
            a_exc: 0.3,
            i_exc: 0.2,
            a_inh: 0.1,
            i_inh: 0.05,
            v_rest: -0.5,
            v_threshold: 1.5,
            tau_v: 8.0,
            tau_exc: 3.0,
            tau_inh: 6.0,
            dt: 0.25,
        };
        state.reset();
        assert_eq!(
            (state.v, state.a_exc, state.i_exc, state.a_inh, state.i_inh),
            (-0.5, 0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            (
                state.v_rest,
                state.v_threshold,
                state.tau_v,
                state.tau_exc,
                state.tau_inh,
                state.dt
            ),
            (-0.5, 1.5, 8.0, 3.0, 6.0, 0.25)
        );
    }
}
