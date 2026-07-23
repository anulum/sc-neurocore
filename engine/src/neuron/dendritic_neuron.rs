// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — XOR-nonlinearity dendritic neuron

/// XOR-nonlinearity dendritic neuron.
///
/// Koch, Biophysics of Computation, 1999, Ch. 12.
/// Output = 1 if `(d1 + d2 - 2*d1*d2) > threshold`.
#[derive(Clone, Debug)]
pub struct DendriticNeuron {
    pub threshold: f64,
    last_current: f64,
}

impl DendriticNeuron {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            last_current: 0.0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.5)
    }

    pub fn step(&mut self, input_a: f64, input_b: f64) -> i32 {
        self.last_current = input_a + input_b - 2.0 * input_a * input_b;
        if self.last_current > self.threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.last_current = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::DendriticNeuron;

    #[test]
    fn xor_nonlinearity_matches_truth_table() {
        let mut neuron = DendriticNeuron::new(0.5);
        assert_eq!(neuron.step(0.0, 0.0), 0);
        assert_eq!(neuron.step(1.0, 0.0), 1);
        assert_eq!(neuron.step(0.0, 1.0), 1);
        assert_eq!(neuron.step(1.0, 1.0), 0);
    }

    #[test]
    fn subthreshold_current_does_not_fire() {
        let mut neuron = DendriticNeuron::new(0.5);
        assert_eq!(neuron.step(0.2, 0.1), 0);
    }

    #[test]
    fn reset_clears_last_current() {
        let mut neuron = DendriticNeuron::with_defaults();
        neuron.step(1.0, 0.0);
        neuron.reset();
        assert!(neuron.last_current.abs() < 1e-12);
    }
}
