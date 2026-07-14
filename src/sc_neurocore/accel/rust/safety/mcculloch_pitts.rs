// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source-faithful McCulloch-Pitts safety kernel

/// McCulloch and Pitts' all-or-none logical neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McCullochPittsNeuron {
    pub theta: i32,
}

impl McCullochPittsNeuron {
    /// Construct a neuron with a positive excitatory-afferent threshold.
    pub fn new(theta: i32) -> Result<Self, &'static str> {
        if theta <= 0 {
            return Err("theta must be a positive signed 32-bit integer");
        }
        Ok(Self { theta })
    }

    /// Revalidate the publicly visible fixed parameter.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.theta <= 0 {
            return Err("theta must be a positive signed 32-bit integer");
        }
        Ok(())
    }

    /// Evaluate one preceding-instant afferent pattern.
    pub fn step(
        &self,
        excitatory_count: i32,
        inhibitory_active: bool,
    ) -> Result<i32, &'static str> {
        self.validate()?;
        if excitatory_count < 0 {
            return Err("excitatory_count must be a non-negative signed 32-bit integer");
        }
        Ok(i32::from(
            !inhibitory_active && excitatory_count >= self.theta,
        ))
    }

    /// Evaluate a complete varying-input batch after validating every row.
    pub fn evaluate_batch(
        &self,
        excitatory_counts: &[i64],
        inhibitory_flags: &[u8],
    ) -> Result<(Vec<u8>, usize), &'static str> {
        self.validate()?;
        if excitatory_counts.len() != inhibitory_flags.len() {
            return Err("inhibitory_flags must match excitatory_counts length");
        }

        let mut validated = Vec::with_capacity(excitatory_counts.len());
        for (&count, &flag) in excitatory_counts.iter().zip(inhibitory_flags) {
            let count = i32::try_from(count).map_err(|_| {
                "excitatory counts must be non-negative signed 32-bit integers"
            })?;
            if count < 0 {
                return Err("excitatory counts must be non-negative signed 32-bit integers");
            }
            if flag > 1 {
                return Err("inhibitory flags must contain only zero or one");
            }
            validated.push((count, flag != 0));
        }

        let events: Vec<u8> = validated
            .into_iter()
            .map(|(count, inhibited)| {
                self.step(count, inhibited)
                    .expect("all rows and the fixed threshold were validated")
                    as u8
            })
            .collect();
        let event_count = events.iter().map(|&event| usize::from(event)).sum();
        Ok((events, event_count))
    }

    /// The formal neuron carries no cell state.
    pub fn reset(&self) -> Result<(), &'static str> {
        self.validate()
    }
}

impl Default for McCullochPittsNeuron {
    fn default() -> Self {
        Self::new(1).expect("the default McCulloch-Pitts threshold is valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_positive_and_stateless() {
        let neuron = McCullochPittsNeuron::default();
        assert_eq!(neuron.theta, 1);
        assert_eq!(neuron.reset(), Ok(()));
    }

    #[test]
    fn theta_one_is_or() {
        let neuron = McCullochPittsNeuron::default();
        assert_eq!(neuron.step(0, false), Ok(0));
        assert_eq!(neuron.step(1, false), Ok(1));
        assert_eq!(neuron.step(2, false), Ok(1));
    }

    #[test]
    fn theta_two_is_and() {
        let neuron = McCullochPittsNeuron::new(2).unwrap();
        assert_eq!(neuron.step(0, false), Ok(0));
        assert_eq!(neuron.step(1, false), Ok(0));
        assert_eq!(neuron.step(2, false), Ok(1));
    }

    #[test]
    fn inhibition_is_an_absolute_veto() {
        let neuron = McCullochPittsNeuron::default();
        assert_eq!(neuron.step(i32::MAX, true), Ok(0));
    }

    #[test]
    fn invalid_threshold_and_count_fail_closed() {
        assert!(McCullochPittsNeuron::new(0).is_err());
        assert!(McCullochPittsNeuron::new(-1).is_err());
        assert!(McCullochPittsNeuron::default().step(-1, false).is_err());
    }

    #[test]
    fn public_threshold_mutation_is_revalidated() {
        let neuron = McCullochPittsNeuron { theta: 0 };
        assert!(neuron.step(1, false).is_err());
        assert!(neuron.reset().is_err());
    }

    #[test]
    fn batch_is_exact_for_varying_excitation_and_inhibition() {
        let neuron = McCullochPittsNeuron::new(2).unwrap();
        let result = neuron.evaluate_batch(&[0, 1, 2, i32::MAX.into()], &[0, 0, 0, 1]);
        assert_eq!(result, Ok((vec![0, 0, 1, 0], 1)));
    }

    #[test]
    fn empty_batch_is_valid() {
        assert_eq!(
            McCullochPittsNeuron::default().evaluate_batch(&[], &[]),
            Ok((Vec::new(), 0))
        );
    }

    #[test]
    fn malformed_batches_fail_before_returning_output() {
        let neuron = McCullochPittsNeuron::default();
        assert!(neuron.evaluate_batch(&[1], &[]).is_err());
        assert!(neuron.evaluate_batch(&[-1], &[0]).is_err());
        assert!(neuron
            .evaluate_batch(&[i64::from(i32::MAX) + 1], &[0])
            .is_err());
        assert!(neuron.evaluate_batch(&[1], &[2]).is_err());
    }
}
