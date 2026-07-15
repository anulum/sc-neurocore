// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Jansen and Rit 1995 cortical-column neural mass

//! Equation-(6) dynamics and an atomic explicit-Euler batch implementation.

/// One Jansen–Rit cortical column with states `[y0, y1, y2, y3, y4, y5]`.
#[derive(Clone, Debug)]
pub struct JansenRitUnit {
    pub y: [f64; 6],
    pub a_exc: f64,
    pub b_exc: f64,
    pub a_rate: f64,
    pub b_rate: f64,
    pub c: f64,
    pub e0: f64,
    pub v0: f64,
    pub r: f64,
    pub dt: f64,
}

impl JansenRitUnit {
    /// Construct the published parameter set with a 0.1 ms Euler step.
    pub fn new() -> Self {
        Self {
            y: [0.0; 6],
            a_exc: 3.25,
            b_exc: 22.0,
            a_rate: 100.0,
            b_rate: 50.0,
            c: 135.0,
            e0: 2.5,
            v0: 6.0,
            r: 0.56,
            dt: 0.0001,
        }
    }

    /// Construct and validate one configured state.
    #[allow(clippy::too_many_arguments)]
    pub fn with_parameters(
        y0: f64,
        y3: f64,
        y1: f64,
        y4: f64,
        y2: f64,
        y5: f64,
        a_exc: f64,
        b_exc: f64,
        a_rate: f64,
        b_rate: f64,
        c: f64,
        e0: f64,
        v0: f64,
        r: f64,
        dt: f64,
    ) -> Result<Self, String> {
        let unit = Self {
            y: [y0, y1, y2, y3, y4, y5],
            a_exc,
            b_exc,
            a_rate,
            b_rate,
            c,
            e0,
            v0,
            r,
            dt,
        };
        unit.validate()?;
        Ok(unit)
    }

    fn validate(&self) -> Result<(), String> {
        let parameters = [
            self.a_exc,
            self.b_exc,
            self.a_rate,
            self.b_rate,
            self.c,
            self.e0,
            self.v0,
            self.r,
            self.dt,
        ];
        if !self
            .y
            .iter()
            .chain(parameters.iter())
            .all(|value| value.is_finite())
        {
            return Err("Jansen–Rit state and parameters must be finite".into());
        }
        if self.a_exc <= 0.0
            || self.b_exc <= 0.0
            || self.a_rate <= 0.0
            || self.b_rate <= 0.0
            || self.e0 <= 0.0
            || self.r <= 0.0
            || self.dt <= 0.0
        {
            return Err(
                "Jansen–Rit gains, rates, sigmoid scale, slope, and dt must be positive".into(),
            );
        }
        if self.c < 0.0 {
            return Err("Jansen–Rit connectivity must be non-negative".into());
        }
        Ok(())
    }

    #[inline]
    fn sigmoid(&self, voltage: f64) -> Result<f64, String> {
        if !voltage.is_finite() {
            return Err("Jansen–Rit sigmoid input must be finite".into());
        }
        let exponent = self.r * (self.v0 - voltage);
        let response = if exponent >= 0.0 {
            let exp_neg = (-exponent).exp();
            2.0 * self.e0 * exp_neg / (1.0 + exp_neg)
        } else {
            2.0 * self.e0 / (1.0 + exponent.exp())
        };
        if !response.is_finite() {
            return Err("Jansen–Rit sigmoid response must be finite".into());
        }
        Ok(response)
    }

    /// Advance one equation-(6) Euler step and return post-update `y1 - y2`.
    pub fn step(&mut self, p_ext: f64) -> Result<f64, String> {
        self.validate()?;
        if !p_ext.is_finite() {
            return Err("Jansen–Rit external drive must be finite".into());
        }
        let c1 = self.c;
        let c2 = 0.8 * c1;
        let c3 = 0.25 * c1;
        let c4 = 0.25 * c1;
        let s_pyramidal = self.sigmoid(self.y[1] - self.y[2])?;
        let s_excitatory = self.sigmoid(c1 * self.y[0])?;
        let s_inhibitory = self.sigmoid(c3 * self.y[0])?;
        let derivatives = [
            self.y[3],
            self.y[4],
            self.y[5],
            self.a_exc * self.a_rate * s_pyramidal
                - 2.0 * self.a_rate * self.y[3]
                - self.a_rate.powi(2) * self.y[0],
            self.a_exc * self.a_rate * (p_ext + c2 * s_excitatory)
                - 2.0 * self.a_rate * self.y[4]
                - self.a_rate.powi(2) * self.y[1],
            self.b_exc * self.b_rate * c4 * s_inhibitory
                - 2.0 * self.b_rate * self.y[5]
                - self.b_rate.powi(2) * self.y[2],
        ];
        let mut candidate = self.y;
        for (next, derivative) in candidate.iter_mut().zip(derivatives) {
            *next += self.dt * derivative;
        }
        if !candidate.iter().all(|value| value.is_finite()) {
            return Err("Jansen–Rit candidate state must remain finite".into());
        }
        self.y = candidate;
        Ok(self.y[1] - self.y[2])
    }

    /// Restore all dynamic states while preserving parameters.
    pub fn reset(&mut self) {
        self.y = [0.0; 6];
    }
}

impl Default for JansenRitUnit {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-step state and EEG traces returned by the batch implementation.
pub struct JansenRitTrace {
    pub y0: Vec<f64>,
    pub y3: Vec<f64>,
    pub y1: Vec<f64>,
    pub y4: Vec<f64>,
    pub y2: Vec<f64>,
    pub y5: Vec<f64>,
    pub eeg: Vec<f64>,
    pub final_state: [f64; 6],
}

/// Simulate a complete external-drive batch.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    y0: f64,
    y3: f64,
    y1: f64,
    y4: f64,
    y2: f64,
    y5: f64,
    a_exc: f64,
    b_exc: f64,
    a_rate: f64,
    b_rate: f64,
    c: f64,
    e0: f64,
    v0: f64,
    r: f64,
    dt: f64,
    p_ext: &[f64],
) -> Result<JansenRitTrace, String> {
    let mut unit = JansenRitUnit::with_parameters(
        y0, y3, y1, y4, y2, y5, a_exc, b_exc, a_rate, b_rate, c, e0, v0, r, dt,
    )?;
    let mut trace = JansenRitTrace {
        y0: Vec::with_capacity(p_ext.len()),
        y3: Vec::with_capacity(p_ext.len()),
        y1: Vec::with_capacity(p_ext.len()),
        y4: Vec::with_capacity(p_ext.len()),
        y2: Vec::with_capacity(p_ext.len()),
        y5: Vec::with_capacity(p_ext.len()),
        eeg: Vec::with_capacity(p_ext.len()),
        final_state: unit.y,
    };
    for drive in p_ext {
        let eeg = unit.step(*drive)?;
        trace.y0.push(unit.y[0]);
        trace.y3.push(unit.y[3]);
        trace.y1.push(unit.y[1]);
        trace.y4.push(unit.y[4]);
        trace.y2.push(unit.y[2]);
        trace.y5.push(unit.y[5]);
        trace.eeg.push(eeg);
    }
    trace.final_state = unit.y;
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_step_uses_c1_inside_and_c2_outside_excitatory_sigmoid() {
        let mut unit = JansenRitUnit::with_parameters(
            0.1, 0.2, 0.3, -0.4, -0.1, 0.5, 3.25, 22.0, 100.0, 50.0, 135.0, 2.5, 6.0, 0.56, 0.0001,
        )
        .unwrap();
        let old = unit.y;
        let se = unit.sigmoid(135.0 * old[0]).unwrap();
        let expected_y4 = old[4]
            + 0.0001
                * (3.25 * 100.0 * (220.0 + 0.8 * 135.0 * se)
                    - 2.0 * 100.0 * old[4]
                    - 100.0_f64.powi(2) * old[1]);
        unit.step(220.0).unwrap();
        assert_eq!(unit.y[4], expected_y4);
    }

    #[test]
    fn batch_matches_scalar_and_preserves_empty_initial_state() {
        let empty = simulate(
            0.1,
            0.2,
            0.3,
            -0.4,
            -0.1,
            0.5,
            3.25,
            22.0,
            100.0,
            50.0,
            135.0,
            2.5,
            6.0,
            0.56,
            0.0001,
            &[],
        )
        .unwrap();
        assert!(empty.eeg.is_empty());
        assert_eq!(empty.final_state, [0.1, 0.3, -0.1, 0.2, -0.4, 0.5]);

        let drives = [120.0, 220.0, 320.0];
        let batch = simulate(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.25, 22.0, 100.0, 50.0, 135.0, 2.5, 6.0, 0.56, 0.0001,
            &drives,
        )
        .unwrap();
        let mut scalar = JansenRitUnit::new();
        for drive in drives {
            scalar.step(drive).unwrap();
        }
        assert_eq!(batch.final_state, scalar.y);
    }

    #[test]
    fn invalid_input_does_not_mutate_state() {
        let mut unit = JansenRitUnit::new();
        let before = unit.y;
        assert!(unit.step(f64::NAN).is_err());
        assert_eq!(unit.y, before);
    }
}
