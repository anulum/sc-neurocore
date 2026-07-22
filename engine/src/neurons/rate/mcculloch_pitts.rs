// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — McCulloch-Pitts neuron model

/// McCulloch-Pitts 1943 — excitatory-count threshold with absolute inhibition.
#[derive(Clone, Debug)]
pub struct McCullochPittsNeuron {
    pub theta: i32,
}

impl McCullochPittsNeuron {
    /// Construct a source-faithful neuron with a positive afferent-count threshold.
    pub fn new(theta: i32) -> Result<Self, String> {
        if theta <= 0 {
            return Err("theta must be a positive signed 32-bit integer".into());
        }
        Ok(Self { theta })
    }

    /// Revalidate the public fixed threshold before any execution boundary.
    pub fn validate(&self) -> Result<(), String> {
        if self.theta <= 0 {
            return Err("theta must be a positive signed 32-bit integer".into());
        }
        Ok(())
    }

    /// Evaluate one preceding-instant afferent pattern without cell state.
    pub fn try_step(&self, excitatory_count: i32, inhibitory_active: bool) -> Result<i32, String> {
        self.validate()?;
        if excitatory_count < 0 {
            return Err("excitatory_count must be a non-negative signed 32-bit integer".into());
        }
        Ok(i32::from(
            !inhibitory_active && excitatory_count >= self.theta,
        ))
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
    fn mcp_threshold() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.try_step(2, false), Ok(1));
        assert_eq!(n.try_step(0, false), Ok(0));
    }

    #[test]
    fn mcp_below_threshold() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.try_step(0, false), Ok(0));
    }

    #[test]
    fn mcp_absolute_inhibition_vetoes_maximum_excitation() {
        let n = McCullochPittsNeuron::default();
        assert_eq!(n.try_step(i32::MAX, true), Ok(0));
    }

    #[test]
    fn mcp_theta_two_is_and() {
        let n = McCullochPittsNeuron::new(2).unwrap();
        assert_eq!(n.try_step(0, false), Ok(0));
        assert_eq!(n.try_step(1, false), Ok(0));
        assert_eq!(n.try_step(2, false), Ok(1));
    }

    #[test]
    fn mcp_rejects_non_positive_thresholds() {
        assert!(McCullochPittsNeuron::new(0).is_err());
        assert!(McCullochPittsNeuron::new(-1).is_err());
    }

    #[test]
    fn mcp_rejects_negative_excitation() {
        let n = McCullochPittsNeuron::default();
        assert!(n.try_step(-1, false).is_err());
    }

    #[test]
    fn mcp_revalidates_public_threshold_mutation() {
        let n = McCullochPittsNeuron { theta: 0 };
        assert!(n.try_step(1, false).is_err());
    }

    #[test]
    fn mcp_is_stateless_across_history() {
        let n = McCullochPittsNeuron::new(2).unwrap();
        let outputs: Vec<i32> = [2, 0, 2, 0]
            .into_iter()
            .map(|count| n.try_step(count, false).unwrap())
            .collect();
        assert_eq!(outputs, vec![1, 0, 1, 0]);
    }
}
