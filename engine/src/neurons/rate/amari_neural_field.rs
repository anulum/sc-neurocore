// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Amari 1977 single-layer lateral-inhibition neural field

//! Periodic-grid specialization of Amari's 1977 equation (3).

/// Finite periodic Amari field with source-level Heaviside activity.
#[derive(Clone, Debug)]
pub struct AmariNeuralField {
    /// Field potential at each periodic site.
    pub u: Vec<f64>,
    /// Number of sites.
    pub n: usize,
    /// Positive field time constant.
    pub tau: f64,
    /// Local excitation amplitude.
    pub a_exc: f64,
    /// Excitation inverse width.
    pub a_width: f64,
    /// Distal inhibition amplitude.
    pub b_inh: f64,
    /// Inhibition inverse width.
    pub b_width: f64,
    /// Spatial grid interval.
    pub dx: f64,
    /// Explicit-Euler interval.
    pub dt: f64,
    w: Vec<f64>,
}

impl AmariNeuralField {
    /// Construct the source-faithful default field with `n` periodic sites.
    ///
    /// # Panics
    ///
    /// Panics when fewer than two sites are requested. Use
    /// [`Self::with_config`] for a recoverable configuration path.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self::with_config(n, 10.0, 1.5, 2.0, 0.75, 1.0, 0.5, 0.5, vec![0.0; n])
            .expect("default Amari field configuration must be valid")
    }

    /// Construct a fully configured field, validating state and kernel shape.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        n: usize,
        tau: f64,
        a_exc: f64,
        a_width: f64,
        b_inh: f64,
        b_width: f64,
        dx: f64,
        dt: f64,
        u: Vec<f64>,
    ) -> Result<Self, String> {
        if n < 2 || u.len() != n {
            return Err("Amari field requires n >= 2 and an exact-length state".into());
        }
        let parameters = [tau, a_exc, a_width, b_inh, b_width, dx, dt];
        if !parameters.iter().all(|value| value.is_finite())
            || tau <= 0.0
            || a_exc < 0.0
            || a_width <= 0.0
            || b_inh < 0.0
            || b_width <= 0.0
            || dx <= 0.0
            || dt <= 0.0
            || !u.iter().all(|value| value.is_finite())
        {
            return Err("invalid Amari field numerical configuration".into());
        }
        let mut w = Vec::with_capacity(n);
        for offset in 0..n {
            let wrapped = offset.min(n - offset);
            let distance = wrapped as f64 * dx;
            w.push(a_exc * (-a_width * distance).exp() - b_inh * (-b_width * distance).exp());
        }
        if w[0] <= 0.0 || w[n / 2] >= 0.0 || !w.iter().all(|value| value.is_finite()) {
            return Err("Amari kernel must be locally excitatory and distally inhibitory".into());
        }
        Ok(Self {
            u,
            n,
            tau,
            a_exc,
            a_width,
            b_inh,
            b_width,
            dx,
            dt,
            w,
        })
    }

    /// Advance one simultaneous Euler step and return active-site fraction.
    ///
    /// Errors leave the dynamic state unchanged.
    pub fn step(&mut self, input: &[f64]) -> Result<f64, String> {
        if input.len() != self.n || !input.iter().all(|value| value.is_finite()) {
            return Err("Amari input must be a finite exact-length vector".into());
        }
        let activity: Vec<f64> = self.u.iter().map(|value| f64::from(*value > 0.0)).collect();
        let mut candidate = vec![0.0; self.n];
        for (i, next) in candidate.iter_mut().enumerate() {
            let mut convolution = 0.0;
            for (j, active) in activity.iter().enumerate() {
                convolution += self.w[(i + self.n - j) % self.n] * active;
            }
            *next =
                self.u[i] + (-self.u[i] + convolution * self.dx + input[i]) * (self.dt / self.tau);
        }
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err("Amari candidate state must remain finite".into());
        }
        self.u = candidate;
        Ok(self.u.iter().filter(|value| **value > 0.0).count() as f64 / self.n as f64)
    }

    /// Zero all dynamic sites while preserving configuration.
    pub fn reset(&mut self) {
        self.u.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_kernel_changes_sign() {
        let field = AmariNeuralField::new(16);
        assert!(field.w[0] > 0.0);
        assert!(field.w[8] < 0.0);
    }

    #[test]
    fn update_and_reset_are_vector_complete() {
        let mut field = AmariNeuralField::new(16);
        let rate = field.step(&vec![0.5; 16]).unwrap();
        assert_eq!(rate, 1.0);
        field.reset();
        assert!(field.u.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn failed_update_is_atomic() {
        let mut field = AmariNeuralField::new(8);
        let before = field.u.clone();
        assert!(field.step(&[f64::NAN; 8]).is_err());
        assert_eq!(field.u, before);
    }
}
