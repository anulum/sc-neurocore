// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sensory Neuron Models

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

    fn valid_runtime(&self) -> bool {
        [
            self.v,
            self.tau,
            self.theta,
            self.dt,
            self.prev_intensity,
            self.surround,
            self.w_centre,
            self.w_surround,
            self.direction_pref,
        ]
        .iter()
        .all(|x| x.is_finite())
            && self.tau > 0.0
            && self.theta > 0.0
            && self.dt > 0.0
            && self.prev_intensity >= 0.0
            && self.surround >= 0.0
            && self.w_centre >= 0.0
            && self.w_surround >= 0.0
    }

    /// Step with local intensity and surround mean intensity.
    pub fn step_rf(&mut self, intensity: f64, surround_mean: f64) -> i32 {
        if !intensity.is_finite()
            || !surround_mean.is_finite()
            || intensity < 0.0
            || surround_mean < 0.0
            || !self.valid_runtime()
        {
            return 0;
        }
        let temporal_diff = intensity - self.prev_intensity;
        let centre_response = if self.is_on_centre {
            self.w_centre * temporal_diff
        } else {
            -self.w_centre * temporal_diff
        };

        let next_surround = 0.9 * self.surround + 0.1 * surround_mean;
        let surround_inhib = self.w_surround * next_surround;
        let drive = centre_response - surround_inhib;
        let decay = (-self.dt / self.tau).exp();
        let next_v = drive + (self.v - drive) * decay;
        if !next_surround.is_finite()
            || !drive.is_finite()
            || !decay.is_finite()
            || !next_v.is_finite()
            || next_surround < 0.0
        {
            return 0;
        }

        self.prev_intensity = intensity;
        self.surround = next_surround;
        if next_v >= self.theta {
            self.v = 0.0;
            1
        } else {
            self.v = next_v;
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

#[cfg(test)]
mod tests {
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
            spikes += cell.step_rf(6.0, 0.0);
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
    fn rgc_exact_membrane_relaxation() {
        let mut cell = DirectionSelectiveRGC::new_on();
        cell.tau = 7.0;
        cell.theta = 100.0;
        cell.dt = 1.25;
        cell.w_centre = 1.4;
        cell.w_surround = 0.2;
        cell.v = 0.35;
        let expected_surround = 0.9 * cell.surround + 0.1 * 0.5;
        let expected_drive =
            cell.w_centre * (2.0 - cell.prev_intensity) - cell.w_surround * expected_surround;
        let expected_v = expected_drive + (cell.v - expected_drive) * (-cell.dt / cell.tau).exp();
        assert_eq!(cell.step_rf(2.0, 0.5), 0);
        assert!((cell.v - expected_v).abs() < 1e-12);
        assert!((cell.surround - expected_surround).abs() < 1e-12);
    }

    #[test]
    fn rgc_invalid_drive_preserves_state() {
        let mut cell = DirectionSelectiveRGC::new_on();
        let before = (cell.v, cell.prev_intensity, cell.surround);
        assert_eq!(cell.step_rf(f64::NAN, 0.0), 0);
        assert_eq!((cell.v, cell.prev_intensity, cell.surround), before);
    }

    #[test]
    fn rgc_corrupt_runtime_state_preserves_state() {
        let mut cell = DirectionSelectiveRGC::new_on();
        cell.surround = f64::INFINITY;
        let before = (cell.v, cell.prev_intensity, cell.surround);
        assert_eq!(cell.step_rf(1.0, 0.0), 0);
        assert_eq!((cell.v, cell.prev_intensity, cell.surround), before);
    }

    #[test]
    fn direction_selective_rgc_default_matches_on_constructor_contract() {
        let default = DirectionSelectiveRGC::default();
        let constructed = DirectionSelectiveRGC::new_on();
        assert_eq!(default.v, constructed.v);
        assert_eq!(default.is_on_centre, constructed.is_on_centre);
        assert_eq!(default.w_centre, constructed.w_centre);
        assert_eq!(default.dt, constructed.dt);
    }

    #[test]
    fn rgc_simple_step_and_reset_contract() {
        let mut cell = DirectionSelectiveRGC::new_on();
        assert!(matches!(cell.step(2.0), 0 | 1));
        cell.reset();
        assert_eq!(cell.v, 0.0);
        assert_eq!(cell.prev_intensity, 0.0);
        assert_eq!(cell.surround, 0.0);
    }

    #[test]
    fn rgc_nonfinite_candidate_preserves_state() {
        let mut cell = DirectionSelectiveRGC::new_on();
        cell.w_centre = f64::MAX;
        let before = (cell.v, cell.prev_intensity, cell.surround);
        assert_eq!(cell.step_rf(2.0, 0.0), 0);
        assert_eq!((cell.v, cell.prev_intensity, cell.surround), before);
    }
}
