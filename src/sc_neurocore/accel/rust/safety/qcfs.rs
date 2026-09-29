// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust QCFS quantisation and surrogate derivatives

//! Shifted, clipped QCFS quantisation matching the Python activation.

/// Positive finite threshold and positive integer quantisation grid.
#[derive(Debug, Clone, Copy)]
pub struct QCFSActivation {
    /// Number of simulation steps and quantisation intervals.
    pub steps: u32,
    /// Upper activation bound and neuron threshold.
    pub theta: f64,
}

/// Invalid quantisation step count or threshold scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidQcfs;

impl QCFSActivation {
    /// Construct the public default eight-step, unit-threshold activation.
    pub fn new() -> Self {
        Self {
            steps: 8,
            theta: 1.0,
        }
    }

    /// Refuse grids and threshold scales that cannot represent firing rates.
    pub fn with_parameters(steps: u32, theta: f64) -> Result<Self, InvalidQcfs> {
        let state = Self { steps, theta };
        if validate_qcfs(&state) {
            Ok(state)
        } else {
            Err(InvalidQcfs)
        }
    }

    /// Quantise a scalar, saturating infinities and propagating NaN inputs.
    pub fn forward(&self, x: f64) -> Result<f64, InvalidQcfs> {
        if !validate_qcfs(self) {
            return Err(InvalidQcfs);
        }
        let steps = f64::from(self.steps);
        let shifted = x * steps / self.theta + 0.5;
        Ok(shifted.clamp(0.0, steps).floor() * self.theta / steps)
    }

    /// Straight-through derivatives of one element (Bu et al., 2022, Eq. 17).
    ///
    /// Returns the input derivative, `upstream` on the open interior
    /// `0 < s < T` and zero elsewhere, and the threshold derivative a
    /// one-element batch receives, `upstream * floor(c) / T` plus, on the
    /// interior, `-upstream * x / theta`, in the Python autograd operation order.
    pub fn backward(&self, x: f64, upstream: f64) -> Result<(f64, f64), InvalidQcfs> {
        if !validate_qcfs(self) {
            return Err(InvalidQcfs);
        }
        let steps = f64::from(self.steps);
        let theta = self.theta;
        let shifted = x * steps / theta + 0.5;
        let interior = shifted > 0.0 && shifted < steps;
        let lattice = shifted.clamp(0.0, steps).floor();
        let output_gradient = upstream / steps;
        let carried = if interior {
            output_gradient * theta
        } else {
            0.0
        };
        let scaled_input = if interior { x } else { 0.0 } * steps;
        let input_gradient = if interior {
            carried / theta * steps
        } else {
            0.0
        };
        let threshold_gradient =
            (0.0 + output_gradient * lattice) + (0.0 + (-carried) * (scaled_input / theta / theta));
        Ok((input_gradient, threshold_gradient))
    }

    /// Describe the grid and current threshold using Python-compatible precision.
    pub fn extra_repr(&self) -> String {
        format!("T={}, theta={:.2}", self.steps, self.theta)
    }
}

impl Default for QCFSActivation {
    /// Use the same default quantisation grid as the Python API.
    fn default() -> Self {
        Self::new()
    }
}

/// Test mutable public state before every forward operation.
pub fn validate_qcfs(state: &QCFSActivation) -> bool {
    state.steps > 0 && state.theta.is_finite() && state.theta > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grid_and_saturation() {
        let state = QCFSActivation::new();
        assert!(validate_qcfs(&state));
        assert_eq!(state.forward(0.5), Ok(0.5));
        assert_eq!(state.forward(-f64::INFINITY), Ok(0.0));
        assert_eq!(state.forward(f64::INFINITY), Ok(1.0));
        assert!(state.forward(f64::NAN).unwrap().is_nan());
        assert_eq!(state.extra_repr(), "T=8, theta=1.00");
    }

    #[test]
    fn surrogate_derivatives_and_saturation() {
        let state = QCFSActivation::with_parameters(4, 2.0).unwrap();
        assert_eq!(state.backward(-0.125, 1.0), Ok((1.0, 0.0625)));
        assert_eq!(state.backward(0.625, 1.0), Ok((1.0, -0.0625)));
        assert_eq!(state.backward(f64::INFINITY, 1.0), Ok((0.0, 1.0)));
        assert_eq!(state.backward(-f64::INFINITY, 1.0), Ok((0.0, 0.0)));
        let (input, threshold) = state.backward(-1.0, -1.0).unwrap();
        assert!(input == 0.0 && threshold.to_bits() == 0.0f64.to_bits());
        let (input, threshold) = state.backward(f64::NAN, 1.0).unwrap();
        assert!(input == 0.0 && threshold.is_nan());
    }

    #[test]
    fn parameters_and_mutated_state_refuse() {
        assert!(QCFSActivation::with_parameters(0, 1.0).is_err());
        for theta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(QCFSActivation::with_parameters(4, theta).is_err());
            let state = QCFSActivation {
                theta,
                ..QCFSActivation::default()
            };
            assert_eq!(state.forward(0.5), Err(InvalidQcfs));
            assert_eq!(state.backward(0.5, 1.0), Err(InvalidQcfs));
        }
    }
}
