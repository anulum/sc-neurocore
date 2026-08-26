// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for retained unit-capacitance respiratory identity

mod source {
    include!("butera_respiratory.rs");
}

/// Count-neutral safety wrapper preserving the former SC recurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct SCUnitCapacitanceRespiratoryNeuron {
    pub inner: source::ButeraRespiratoryNeuron,
}

impl SCUnitCapacitanceRespiratoryNeuron {
    pub fn new() -> Self {
        let mut inner = source::ButeraRespiratoryNeuron::new();
        inner.capacitance = 1.0;
        inner.e_syn = -10.0;
        Self { inner }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

impl Default for SCUnitCapacitanceRespiratoryNeuron {
    fn default() -> Self {
        Self::new()
    }
}

pub fn validate_sc_unit_capacitance_respiratory(
    state: &SCUnitCapacitanceRespiratoryNeuron,
) -> bool {
    source::validate_butera_respiratory(&state.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_profile_is_valid_and_event_stable() {
        let mut state = SCUnitCapacitanceRespiratoryNeuron::new();
        assert!(validate_sc_unit_capacitance_respiratory(&state));
        assert_eq!(state.inner.capacitance, 1.0);
        assert_eq!(state.inner.e_syn, -10.0);
        assert_eq!((0..20_000).map(|_| state.step(20.0)).sum::<i32>(), 5);
    }

    #[test]
    fn invalid_input_preserves_state() {
        let mut state = SCUnitCapacitanceRespiratoryNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state, before);
    }
}
