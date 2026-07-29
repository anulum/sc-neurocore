// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Safe Amari 1977 periodic neural field

// Fail-closed vector mirror of the maintained Amari field contract.

/// Validated periodic field state and numerical configuration.
#[derive(Debug, Clone)]
pub struct AmariNeuralField {
    /// Dynamic field potential at every site.
    pub u: Vec<f64>,
    /// Field time constant.
    pub tau: f64,
    /// Excitation amplitude.
    pub a_exc: f64,
    /// Excitation inverse width.
    pub a_width: f64,
    /// Inhibition amplitude.
    pub b_inh: f64,
    /// Inhibition inverse width.
    pub b_width: f64,
    /// Spatial interval.
    pub dx: f64,
    /// Euler interval.
    pub dt: f64,
    kernel: Vec<f64>,
}

impl AmariNeuralField {
    /// Construct the 64-site maintained default field.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(vec![0.0; 64], 10.0, 1.5, 2.0, 0.75, 1.0, 0.5, 0.5)
            .expect("default Amari field must be valid")
    }

    /// Construct a validated configurable field.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        u: Vec<f64>,
        tau: f64,
        a_exc: f64,
        a_width: f64,
        b_inh: f64,
        b_width: f64,
        dx: f64,
        dt: f64,
    ) -> Result<Self, &'static str> {
        let n = u.len();
        let values = [tau, a_exc, a_width, b_inh, b_width, dx, dt];
        if n < 2
            || !u.iter().all(|value| value.is_finite())
            || !values.iter().all(|value| value.is_finite())
            || tau <= 0.0
            || a_exc < 0.0
            || a_width <= 0.0
            || b_inh < 0.0
            || b_width <= 0.0
            || dx <= 0.0
            || dt <= 0.0
        {
            return Err("invalid Amari field configuration");
        }
        let kernel: Vec<f64> = (0..n)
            .map(|offset| {
                let distance = offset.min(n - offset) as f64 * dx;
                a_exc * (-a_width * distance).exp() - b_inh * (-b_width * distance).exp()
            })
            .collect();
        if kernel[0] <= 0.0 || kernel[n / 2] >= 0.0 || !kernel.iter().all(|value| value.is_finite())
        {
            return Err("Amari kernel is not lateral inhibitory");
        }
        Ok(Self {
            u,
            tau,
            a_exc,
            a_width,
            b_inh,
            b_width,
            dx,
            dt,
            kernel,
        })
    }

    /// Advance a finite exact-length vector and return active-site fraction.
    ///
    /// No state is committed when validation or candidate generation fails.
    pub fn step(&mut self, input: &[f64]) -> Result<f64, &'static str> {
        let n = self.u.len();
        if input.len() != n || !input.iter().all(|value| value.is_finite()) {
            return Err("invalid Amari field input");
        }
        let mut candidate = vec![0.0; n];
        for (i, next) in candidate.iter_mut().enumerate() {
            let mut convolution = 0.0;
            for (j, state) in self.u.iter().enumerate() {
                if *state > 0.0 {
                    convolution += self.kernel[(i + n - j) % n];
                }
            }
            *next =
                self.u[i] + (-self.u[i] + convolution * self.dx + input[i]) * (self.dt / self.tau);
        }
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err("invalid Amari field candidate");
        }
        self.u = candidate;
        Ok(self.u.iter().filter(|value| **value > 0.0).count() as f64 / n as f64)
    }

    /// Zero dynamic state without changing configured physics.
    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

impl Default for AmariNeuralField {
    fn default() -> Self {
        Self::new()
    }
}

/// Return whether state and configured parameters remain in their valid domain.
#[must_use]
pub fn validate_amari_field(state: &AmariNeuralField) -> bool {
    !state.u.is_empty()
        && state.u.iter().all(|value| value.is_finite())
        && [
            state.tau,
            state.a_exc,
            state.a_width,
            state.b_inh,
            state.b_width,
            state.dx,
            state.dt,
        ]
        .iter()
        .all(|value| value.is_finite())
        && state.tau > 0.0
        && state.a_width > 0.0
        && state.b_width > 0.0
        && state.dx > 0.0
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_update_is_real_and_valid() {
        let mut state = AmariNeuralField::new();
        assert_eq!(state.step(&vec![0.5; 64]).unwrap(), 1.0);
        assert!(validate_amari_field(&state));
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut state = AmariNeuralField::new();
        let before = state.u.clone();
        assert!(state.step(&[f64::NAN; 64]).is_err());
        assert_eq!(state.u, before);
    }
}
