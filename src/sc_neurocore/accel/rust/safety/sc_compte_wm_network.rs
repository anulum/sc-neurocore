// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — dependency-free safety oracle for SC-COMPTE-WM-NETWORK

//! Independent state, input, and dense-connectivity safety oracle.
//!
//! This file deliberately has no crate dependencies and is compiled directly
//! with `rustc --test`. It does not share the production FFT implementation.

/// Fixed pyramidal population size.
pub const N_EXCITATORY: usize = 2048;
/// Fixed interneuron population size.
pub const N_INHIBITORY: usize = 512;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const STEP_MIX: u64 = 0xD1B5_4A32_D192_ED03;
const STREAM_MIX: u64 = 0x94D0_49BB_1331_11EB;

/// Complete independent state view required by the safety validator.
pub struct SafetyState<'a> {
    /// Excitatory voltages in mV.
    pub v_exc_mv: &'a [f64],
    /// Inhibitory voltages in mV.
    pub v_inh_mv: &'a [f64],
    /// Excitatory refractory durations in ms.
    pub refractory_exc_ms: &'a [f64],
    /// Inhibitory refractory durations in ms.
    pub refractory_inh_ms: &'a [f64],
    /// Excitatory external AMPA gates.
    pub external_ampa_exc: &'a [f64],
    /// Inhibitory external AMPA gates.
    pub external_ampa_inh: &'a [f64],
    /// Excitatory recurrent NMDA gates.
    pub recurrent_nmda: &'a [f64],
    /// Excitatory recurrent NMDA rise precursors.
    pub recurrent_nmda_rise: &'a [f64],
    /// Inhibitory recurrent GABAA gates.
    pub recurrent_gabaa: &'a [f64],
}

/// Return whether every full-network state array obeys the SC safety envelope.
#[must_use]
pub fn validate_state(state: &SafetyState<'_>) -> bool {
    state.v_exc_mv.len() == N_EXCITATORY
        && state.v_inh_mv.len() == N_INHIBITORY
        && state.refractory_exc_ms.len() == N_EXCITATORY
        && state.refractory_inh_ms.len() == N_INHIBITORY
        && state.external_ampa_exc.len() == N_EXCITATORY
        && state.external_ampa_inh.len() == N_INHIBITORY
        && state.recurrent_nmda.len() == N_EXCITATORY
        && state.recurrent_nmda_rise.len() == N_EXCITATORY
        && state.recurrent_gabaa.len() == N_INHIBITORY
        && state
            .v_exc_mv
            .iter()
            .chain(state.v_inh_mv)
            .all(|value| value.is_finite() && (-200.0..=100.0).contains(value))
        && state
            .refractory_exc_ms
            .iter()
            .chain(state.refractory_inh_ms)
            .chain(state.external_ampa_exc)
            .chain(state.external_ampa_inh)
            .chain(state.recurrent_nmda_rise)
            .chain(state.recurrent_gabaa)
            .all(|value| value.is_finite() && (0.0..=1.0e6).contains(value))
        && state
            .recurrent_nmda
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

/// Produce the independent portable Poisson fixture for one population.
pub fn counter_poisson_counts(
    population_size: usize,
    rate_hz: f64,
    dt_ms: f64,
    seed: u64,
    stream: u64,
    step_index: u64,
) -> Result<Vec<u64>, &'static str> {
    if population_size == 0
        || !rate_hz.is_finite()
        || rate_hz < 0.0
        || !dt_ms.is_finite()
        || dt_ms <= 0.0
    {
        return Err("invalid input configuration");
    }
    let mean = rate_hz * dt_ms / 1000.0;
    if mean > 32.0 {
        return Err("input mean exceeds safety envelope");
    }
    let mut probability = (-mean).exp();
    let mut cumulative = probability;
    let mut cdf = vec![cumulative];
    let mut event = 0_u64;
    while cumulative < 1.0 - 1.0e-15 {
        event += 1;
        if event > 255 {
            return Err("input inverse CDF exceeded event range");
        }
        probability *= mean / event as f64;
        cumulative += probability;
        cdf.push(cumulative.min(1.0));
    }
    let last = cdf.last_mut().ok_or("input CDF is empty")?;
    *last = 1.0;
    Ok((0..population_size)
        .map(|cell| {
            let counter = seed
                .wrapping_add(step_index.wrapping_mul(STEP_MIX))
                .wrapping_add(stream.wrapping_mul(STREAM_MIX))
                .wrapping_add((cell as u64).wrapping_mul(GOLDEN));
            let bits = splitmix64(counter);
            let uniform = ((bits >> 11) as f64 + 0.5) * 2.0_f64.powi(-53);
            cdf.partition_point(|threshold| *threshold < uniform) as u64
        })
        .collect())
}

/// Compute one independent dense E→E aggregate with no recurrent autapse.
pub fn dense_ee_aggregate(nmda: &[f64], target: usize) -> Result<f64, &'static str> {
    if nmda.len() != N_EXCITATORY
        || target >= N_EXCITATORY
        || !nmda
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        return Err("invalid dense aggregate input");
    }
    let gaussian_mean = (0..N_EXCITATORY)
        .map(|index| gaussian(index, 18.0))
        .sum::<f64>()
        / N_EXCITATORY as f64;
    let j_minus = (1.0 - 1.62 * gaussian_mean) / (1.0 - gaussian_mean);
    let raw: Vec<f64> = (0..N_EXCITATORY)
        .map(|index| j_minus + (1.62 - j_minus) * gaussian(index, 18.0))
        .collect();
    let mean = raw.iter().sum::<f64>() / N_EXCITATORY as f64;
    Ok(nmda
        .iter()
        .enumerate()
        .filter(|(source, _)| *source != target)
        .map(|(source, gate)| {
            let offset = (target + N_EXCITATORY - source) % N_EXCITATORY;
            gate * raw[offset] / mean
        })
        .sum())
}

fn gaussian(index: usize, sigma_deg: f64) -> f64 {
    let angle = index as f64 * 360.0 / N_EXCITATORY as f64;
    let distance = (angle + 180.0).rem_euclid(360.0) - 180.0;
    (-0.5 * (distance / sigma_deg).powi(2)).exp()
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_fixture_matches_production_and_python() {
        let counts = counter_poisson_counts(64, 1800.0, 0.02, 42, 0, 0).unwrap();
        let active: Vec<usize> = counts
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (*value > 0).then_some(index))
            .collect();
        assert_eq!(active, [49, 61]);
    }

    #[test]
    fn complete_zero_state_is_valid_and_wrong_shape_is_not() {
        let exc = vec![0.0; N_EXCITATORY];
        let inh = vec![0.0; N_INHIBITORY];
        let valid = SafetyState {
            v_exc_mv: &vec![-70.0; N_EXCITATORY],
            v_inh_mv: &vec![-70.0; N_INHIBITORY],
            refractory_exc_ms: &exc,
            refractory_inh_ms: &inh,
            external_ampa_exc: &exc,
            external_ampa_inh: &inh,
            recurrent_nmda: &exc,
            recurrent_nmda_rise: &exc,
            recurrent_gabaa: &inh,
        };
        assert!(validate_state(&valid));
        let invalid = SafetyState {
            v_exc_mv: &exc[..8],
            ..valid
        };
        assert!(!validate_state(&invalid));
    }

    #[test]
    fn dense_oracle_removes_self_connection() {
        let mut nmda = vec![0.0; N_EXCITATORY];
        nmda[17] = 0.5;
        assert_eq!(dense_ee_aggregate(&nmda, 17), Ok(0.0));
        assert!(dense_ee_aggregate(&nmda, 18).unwrap() > 0.0);
    }
}
