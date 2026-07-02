// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for dvs_input

#![allow(non_snake_case)]

const DEFAULT_RNG_SEED: u64 = 0x4456_535f_494e_5054;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DVSEvent {
    pub x: isize,
    pub y: isize,
    pub timestamp_ms: f64,
    pub polarity: i8,
}

impl DVSEvent {
    pub fn new(x: isize, y: isize, timestamp_ms: f64, polarity: i8) -> Self {
        Self {
            x,
            y,
            timestamp_ms,
            polarity,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DVSInputLayer {
    pub height: usize,
    pub width: usize,
    pub decay_tau: f64,
    pub surface: Vec<Vec<f64>>,
    pub last_update_time: f64,
    pub rng_state: u64,
}

impl DVSInputLayer {
    pub fn new() -> Self {
        Self::try_new(1, 1, 100.0, None).expect("default DVS parameters are valid")
    }

    pub fn try_new(
        height: usize,
        width: usize,
        decay_tau: f64,
        rng_seed: Option<u64>,
    ) -> Result<Self, String> {
        validate_params(height, width, decay_tau)?;
        Ok(Self {
            height,
            width,
            decay_tau,
            surface: vec![vec![0.0; width]; height],
            last_update_time: 0.0,
            rng_state: rng_seed.unwrap_or(DEFAULT_RNG_SEED),
        })
    }

    pub fn process_events(&mut self, events: &[DVSEvent]) -> Result<Vec<Vec<f64>>, String> {
        if events.is_empty() {
            return Ok(self.output_probabilities());
        }
        self.validate_events(events)?;

        let current_time = events[events.len() - 1].timestamp_ms;
        let dt = current_time - self.last_update_time;
        let decay_factor = (-dt / self.decay_tau).exp();
        if !decay_factor.is_finite() {
            return Err("event decay factor must remain finite".to_string());
        }

        for row in &mut self.surface {
            for value in row {
                *value *= decay_factor;
            }
        }

        for event in events {
            if self.contains(event.x, event.y) {
                self.surface[event.y as usize][event.x as usize] += 1.0;
            }
        }

        self.last_update_time = current_time;
        Ok(self.output_probabilities())
    }

    pub fn generate_bitstream_frame(&mut self, length: usize) -> Result<Vec<Vec<Vec<u8>>>, String> {
        if length == 0 {
            return Err("bitstream frame length must be positive".to_string());
        }

        let probabilities = self.output_probabilities();
        let mut bits = vec![vec![vec![0u8; length]; self.width]; self.height];
        for y in 0..self.height {
            for x in 0..self.width {
                let probability = probabilities[y][x];
                for idx in 0..length {
                    bits[y][x][idx] = u8::from(self.next_unit_interval() < probability);
                }
            }
        }
        Ok(bits)
    }

    fn output_probabilities(&self) -> Vec<Vec<f64>> {
        self.surface
            .iter()
            .map(|row| row.iter().map(|value| value.tanh()).collect())
            .collect()
    }

    fn contains(&self, x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    fn validate_events(&self, events: &[DVSEvent]) -> Result<(), String> {
        let mut previous_time: Option<f64> = None;
        for event in events {
            if !event.timestamp_ms.is_finite() {
                return Err("event timestamp must be finite".to_string());
            }
            if previous_time.is_some_and(|previous| event.timestamp_ms < previous) {
                return Err("event timestamps must be monotonically non-decreasing".to_string());
            }
            if event.timestamp_ms < self.last_update_time {
                return Err("event timestamp cannot be earlier than last update time".to_string());
            }
            if !matches!(event.polarity, -1 | 0 | 1) {
                return Err("event polarity must be -1, 0, or 1".to_string());
            }
            previous_time = Some(event.timestamp_ms);
        }
        Ok(())
    }

    fn next_unit_interval(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn validate_params(height: usize, width: usize, decay_tau: f64) -> Result<(), String> {
    if height == 0 {
        return Err("height must be positive".to_string());
    }
    if width == 0 {
        return Err("width must be positive".to_string());
    }
    if !decay_tau.is_finite() || decay_tau <= 0.0 {
        return Err("decay_tau must be finite and positive".to_string());
    }
    Ok(())
}

pub fn validate_dvs_input(state: &DVSInputLayer) -> bool {
    if validate_params(state.height, state.width, state.decay_tau).is_err() {
        return false;
    }
    if !state.last_update_time.is_finite() || state.last_update_time < 0.0 {
        return false;
    }
    if state.surface.len() != state.height
        || state.surface.iter().any(|row| row.len() != state.width)
    {
        return false;
    }
    state
        .surface
        .iter()
        .flatten()
        .all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dvs_input_new() {
        let state = DVSInputLayer::new();
        assert!(validate_dvs_input(&state));
        assert_eq!(state.height, 1);
        assert_eq!(state.width, 1);
    }

    #[test]
    fn test_dvs_process_events_updates_surface_with_decay() {
        let mut layer = DVSInputLayer::try_new(1, 1, 10.0, Some(17)).unwrap();
        let first = layer
            .process_events(&[DVSEvent::new(0, 0, 0.0, 1)])
            .unwrap();
        assert!((first[0][0] - 1.0_f64.tanh()).abs() < 1.0e-12);

        let second = layer
            .process_events(&[DVSEvent::new(0, 0, 10.0, -1)])
            .unwrap();
        assert!(second[0][0] < (2.0_f64).tanh());
        assert!(second[0][0] > 1.0_f64.tanh());
        assert!((layer.last_update_time - 10.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_dvs_empty_batch_returns_probabilities_without_mutating_surface() {
        let mut layer = DVSInputLayer::try_new(1, 1, 100.0, Some(18)).unwrap();
        let _ = layer
            .process_events(&[DVSEvent::new(0, 0, 0.0, 1), DVSEvent::new(0, 0, 0.0, -1)])
            .unwrap();
        let surface_before = layer.surface.clone();
        let last_update_before = layer.last_update_time;

        let output = layer.process_events(&[]).unwrap();

        assert!((output[0][0] - surface_before[0][0].tanh()).abs() < 1.0e-12);
        assert!(output[0][0] < surface_before[0][0]);
        assert_eq!(layer.surface, surface_before);
        assert_eq!(layer.last_update_time, last_update_before);
    }

    #[test]
    fn test_dvs_ignores_out_of_bounds_events_and_preserves_shape() {
        let mut layer = DVSInputLayer::try_new(2, 3, 100.0, Some(19)).unwrap();
        let output = layer
            .process_events(&[DVSEvent::new(4, 1, 1.0, 1), DVSEvent::new(-1, -1, 2.0, 1)])
            .unwrap();

        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 3);
        assert!(output.iter().flatten().all(|value| *value == 0.0));
    }

    #[test]
    fn test_dvs_rejects_invalid_parameters_and_events() {
        assert!(DVSInputLayer::try_new(0, 1, 100.0, None).is_err());
        assert!(DVSInputLayer::try_new(1, 0, 100.0, None).is_err());
        assert!(DVSInputLayer::try_new(1, 1, 0.0, None).is_err());

        let mut layer = DVSInputLayer::try_new(2, 2, 100.0, None).unwrap();
        assert!(layer
            .process_events(&[DVSEvent::new(0, 0, f64::NAN, 1)])
            .is_err());
        assert!(layer
            .process_events(&[DVSEvent::new(0, 0, 1.0, 7)])
            .is_err());
        assert!(layer
            .process_events(&[DVSEvent::new(0, 0, 2.0, 1), DVSEvent::new(1, 1, 1.0, -1),])
            .is_err());
    }

    #[test]
    fn test_dvs_rejects_cross_batch_timestamp_rewind_without_mutation() {
        let mut layer = DVSInputLayer::try_new(2, 2, 100.0, None).unwrap();
        let _ = layer
            .process_events(&[DVSEvent::new(0, 0, 5.0, 1)])
            .unwrap();
        let surface_before = layer.surface.clone();
        let last_update_before = layer.last_update_time;

        let result = layer.process_events(&[DVSEvent::new(1, 1, 4.0, 1)]);

        assert!(result.is_err());
        assert_eq!(layer.surface, surface_before);
        assert_eq!(layer.last_update_time, last_update_before);
    }

    #[test]
    fn test_dvs_generate_bitstream_frame_shape_and_binary_values() {
        let mut layer = DVSInputLayer::try_new(2, 3, 100.0, Some(23)).unwrap();
        let _ = layer
            .process_events(&[DVSEvent::new(1, 1, 1.0, 1)])
            .unwrap();
        let bits = layer.generate_bitstream_frame(8).unwrap();

        assert_eq!(bits.len(), 2);
        assert_eq!(bits[0].len(), 3);
        assert_eq!(bits[0][0].len(), 8);
        assert!(bits
            .iter()
            .flatten()
            .flatten()
            .all(|bit| *bit == 0 || *bit == 1));
    }
}
