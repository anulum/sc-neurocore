// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for motor_unit

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct MotorUnit {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub adapt: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64,
    pub gain: f64,
    pub force: f64,
    pub twitch_amp: f64,
    pub tau_twitch: f64,
    pub force_decay: f64,
    pub dt: f64,
}

impl MotorUnit {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            v_rest: -65.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 10.0_f64,
            adapt: 0.0_f64,
            tau_adapt: 100.0_f64,
            a_adapt: 0.2_f64,
            gain: 1.0_f64,
            force: 0.0_f64,
            twitch_amp: 0.05_f64,
            tau_twitch: 90.0_f64,
            force_decay: 0.0_f64,
            dt: 0.5_f64,
        }
    }

    pub fn slow() -> Self {
        Self::new()
    }

    pub fn fast() -> Self {
        Self {
            tau_m: 6.0_f64,
            tau_adapt: 50.0_f64,
            a_adapt: 0.1_f64,
            twitch_amp: 0.3_f64,
            tau_twitch: 30.0_f64,
            ..Self::new()
        }
    }

    fn voltage_valid(value: f64) -> bool {
        value.is_finite() && (-150.0..=100.0).contains(&value)
    }

    fn force_valid(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    fn exact_relax(previous: f64, steady: f64, tau: f64, dt: f64) -> Option<f64> {
        if !previous.is_finite()
            || !steady.is_finite()
            || !tau.is_finite()
            || !dt.is_finite()
            || tau <= 0.0
            || dt <= 0.0
        {
            return None;
        }
        Some(steady + (previous - steady) * (-dt / tau).exp())
    }

    fn valid_state(&self) -> bool {
        Self::voltage_valid(self.v)
            && Self::voltage_valid(self.v_rest)
            && Self::voltage_valid(self.v_reset)
            && Self::voltage_valid(self.v_threshold)
            && Self::force_valid(self.force)
            && self.tau_m.is_finite()
            && self.adapt.is_finite()
            && self.tau_adapt.is_finite()
            && self.a_adapt.is_finite()
            && self.gain.is_finite()
            && self.twitch_amp.is_finite()
            && self.tau_twitch.is_finite()
            && self.force_decay.is_finite()
            && self.dt.is_finite()
            && self.tau_m > 0.0
            && self.tau_adapt > 0.0
            && self.tau_twitch > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
            && self.twitch_amp >= 0.0
            && self.v_reset < self.v_threshold
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_state() {
            return 0;
        }

        let mut force = self.force * (-self.dt / self.tau_twitch).exp();
        let input_drive = self.gain * i_ext.max(0.0) - self.adapt;
        let v_target = self.v_rest + input_drive;
        let Some(mut v_candidate) = Self::exact_relax(self.v, v_target, self.tau_m, self.dt) else {
            return 0;
        };
        if !Self::voltage_valid(v_candidate) {
            return 0;
        }
        let adapt_target = self.a_adapt * (v_candidate - self.v_rest);
        let Some(adapt_candidate) =
            Self::exact_relax(self.adapt, adapt_target, self.tau_adapt, self.dt)
        else {
            return 0;
        };
        if !adapt_candidate.is_finite() {
            return 0;
        }

        let mut spike = 0;
        if v_candidate >= self.v_threshold {
            v_candidate = self.v_reset;
            force = (force + self.twitch_amp).min(1.0);
            spike = 1;
        }
        if !Self::voltage_valid(v_candidate) || !Self::force_valid(force) {
            return 0;
        }

        self.v = v_candidate;
        self.adapt = adapt_candidate;
        self.force = force;
        spike
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

pub fn validate_motor_unit(state: &MotorUnit) -> bool {
    state.valid_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_unit_new() {
        let state = MotorUnit::new();
        assert!(state.v.is_finite());
        assert!(validate_motor_unit(&state));
    }

    #[test]
    fn test_motor_unit_step() {
        let mut state = MotorUnit::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    fn relax(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
        steady + (previous - steady) * (-dt / tau).exp()
    }

    fn reference_step(mut unit: MotorUnit, drive: f64) -> MotorUnit {
        let mut force = unit.force * (-unit.dt / unit.tau_twitch).exp();
        let input_drive = unit.gain * drive.max(0.0) - unit.adapt;
        let v_target = unit.v_rest + input_drive;
        let mut v_candidate = relax(unit.v, v_target, unit.tau_m, unit.dt);
        let adapt_target = unit.a_adapt * (v_candidate - unit.v_rest);
        let adapt = relax(unit.adapt, adapt_target, unit.tau_adapt, unit.dt);
        if v_candidate >= unit.v_threshold {
            v_candidate = unit.v_reset;
            force = (force + unit.twitch_amp).min(1.0);
        }
        unit.v = v_candidate;
        unit.adapt = adapt;
        unit.force = force;
        unit
    }

    fn snapshot(unit: &MotorUnit) -> (f64, f64, f64) {
        (unit.v, unit.adapt, unit.force)
    }

    #[test]
    fn test_motor_unit_exact_lif_adaptation_and_force_decay_step() {
        let mut state = MotorUnit::new();
        let expected = reference_step(MotorUnit::new(), 20.0);

        assert_eq!(state.step(20.0), 0);

        assert!((state.v - expected.v).abs() <= 1e-12);
        assert!((state.adapt - expected.adapt).abs() <= 1e-12);
        assert!((state.force - expected.force).abs() <= 1e-12);
    }

    #[test]
    fn test_motor_unit_invalid_drive_preserves_state() {
        let mut state = MotorUnit::new();
        for _ in 0..20 {
            state.step(20.0);
        }
        let before = snapshot(&state);

        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(snapshot(&state), before);
        assert_eq!(state.step(f64::INFINITY), 0);
        assert_eq!(snapshot(&state), before);
    }

    #[test]
    fn test_motor_unit_excess_drive_preserves_state() {
        let mut state = MotorUnit::new();
        let before = snapshot(&state);

        assert_eq!(state.step(1.0e8), 0);

        assert_eq!(snapshot(&state), before);
    }

    #[test]
    fn test_motor_unit_spike_adds_twitch_and_force_stays_bounded() {
        let mut state = MotorUnit::fast();
        let spikes: i32 = (0..1000).map(|_| state.step(50.0)).sum();

        assert!(spikes > 0);
        assert!((0.0..=1.0).contains(&state.force));
        let force_after_drive = state.force;
        for _ in 0..200 {
            state.step(0.0);
        }
        assert!((0.0..=force_after_drive).contains(&state.force));
    }
}
