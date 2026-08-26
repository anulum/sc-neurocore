// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained unit-capacitance respiratory recurrence

use super::butera_respiratory::ButeraRespiratoryNeuron;

/// Count-neutral wrapper retaining the historical SC timing and events.
#[derive(Clone, Debug)]
pub struct SCUnitCapacitanceRespiratoryNeuron {
    pub inner: ButeraRespiratoryNeuron,
}

impl SCUnitCapacitanceRespiratoryNeuron {
    pub fn new() -> Self {
        let mut inner = ButeraRespiratoryNeuron::new();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_unit_capacitance_profile() {
        let mut neuron = SCUnitCapacitanceRespiratoryNeuron::new();
        assert_eq!(neuron.inner.capacitance, 1.0);
        assert_eq!(neuron.inner.e_syn, -10.0);
        assert_eq!((0..20_000).map(|_| neuron.step(20.0)).sum::<i32>(), 5);
    }
}
