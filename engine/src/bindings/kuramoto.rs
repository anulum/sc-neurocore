// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Kuramoto solver PyO3 binding

//! Python binding and validation contracts for the Kuramoto solver.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{extract_matrix_f64, scpn};

/// Register the Kuramoto solver with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyKuramotoSolver>()?;
    Ok(())
}

#[pyclass(
    name = "KuramotoSolver",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PyKuramotoSolver {
    inner: scpn::KuramotoSolver,
}

fn validate_kuramoto_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} values must be finite"
        )))
    }
}

fn validate_kuramoto_dt(dt: f64) -> PyResult<()> {
    if dt.is_finite() && dt > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err("dt must be finite and positive"))
    }
}

fn validate_kuramoto_matrix_shape(
    name: &str,
    values_len: usize,
    rows: usize,
    cols: usize,
    n: usize,
) -> PyResult<()> {
    let is_absent = rows == 0 && cols == 0 && values_len == 0;
    let is_flat = rows == 1 && values_len == n * n;
    let is_square = rows == n && cols == n;
    if is_absent || is_flat || is_square {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be shape ({n}, {n}) or flat length {}",
            n * n
        )))
    }
}

#[pymethods]
impl PyKuramotoSolver {
    #[getter]
    fn phases(&self) -> Vec<f64> {
        self.inner.phases.clone()
    }
    #[new]
    #[pyo3(signature = (omega, coupling, phases, noise_amp=0.1))]
    fn new(
        omega: Vec<f64>,
        coupling: &Bound<'_, PyAny>,
        phases: Vec<f64>,
        noise_amp: f64,
    ) -> PyResult<Self> {
        let n = omega.len();
        if n == 0 {
            return Err(PyValueError::new_err("omega must not be empty."));
        }
        if phases.len() != n {
            return Err(PyValueError::new_err(format!(
                "phases length mismatch: got {}, expected {}.",
                phases.len(),
                n
            )));
        }
        validate_kuramoto_finite("omega", &omega)?;
        validate_kuramoto_finite("initial_phases", &phases)?;
        if !(noise_amp.is_finite() && noise_amp >= 0.0) {
            return Err(PyValueError::new_err(
                "noise_amp must be finite and non-negative",
            ));
        }

        let (coupling_flat, rows, cols) = extract_matrix_f64(coupling, "coupling")?;
        if rows == 1 {
            if coupling_flat.len() != n * n {
                return Err(PyValueError::new_err(format!(
                    "Flat coupling length mismatch: got {}, expected {}.",
                    coupling_flat.len(),
                    n * n
                )));
            }
        } else if rows != n || cols != n {
            return Err(PyValueError::new_err(format!(
                "coupling must be shape ({}, {}) or flat length {}, got ({}, {}).",
                n,
                n,
                n * n,
                rows,
                cols
            )));
        }
        validate_kuramoto_finite("coupling", &coupling_flat)?;

        Ok(Self {
            inner: scpn::KuramotoSolver::new(omega, coupling_flat, phases, noise_amp),
        })
    }

    #[pyo3(signature = (dt, seed=0))]
    fn step(&mut self, dt: f64, seed: u64) -> PyResult<f64> {
        validate_kuramoto_dt(dt)?;
        Ok(self.inner.step(dt, seed))
    }

    #[pyo3(signature = (n_steps, dt, seed=0))]
    fn run(&mut self, n_steps: usize, dt: f64, seed: u64) -> PyResult<Vec<f64>> {
        validate_kuramoto_dt(dt)?;
        Ok(self.inner.run(n_steps, dt, seed))
    }

    fn set_field_pressure(&mut self, f: f64) -> PyResult<()> {
        if !f.is_finite() {
            return Err(PyValueError::new_err("field_pressure must be finite"));
        }
        self.inner.set_field_pressure(f);
        Ok(())
    }

    #[pyo3(signature = (
        dt,
        seed=0,
        W=None,
        sigma_g=0.0,
        h_munu=None,
        pgbo_weight=0.0,
    ))]
    #[allow(non_snake_case)]
    fn step_ssgf(
        &mut self,
        dt: f64,
        seed: u64,
        W: Option<&Bound<'_, PyAny>>,
        sigma_g: f64,
        h_munu: Option<&Bound<'_, PyAny>>,
        pgbo_weight: f64,
    ) -> PyResult<f64> {
        validate_kuramoto_dt(dt)?;
        if !sigma_g.is_finite() {
            return Err(PyValueError::new_err("sigma_g must be finite"));
        }
        if !pgbo_weight.is_finite() {
            return Err(PyValueError::new_err("pgbo_weight must be finite"));
        }
        let (w_flat, w_rows, w_cols) = match W {
            Some(w) => extract_matrix_f64(w, "W")?,
            None => (vec![], 0, 0),
        };
        let (h_flat, h_rows, h_cols) = match h_munu {
            Some(h) => extract_matrix_f64(h, "h_munu")?,
            None => (vec![], 0, 0),
        };
        validate_kuramoto_matrix_shape("W", w_flat.len(), w_rows, w_cols, self.inner.n)?;
        validate_kuramoto_matrix_shape("h_munu", h_flat.len(), h_rows, h_cols, self.inner.n)?;
        validate_kuramoto_finite("w_flat", &w_flat)?;
        validate_kuramoto_finite("h_flat", &h_flat)?;
        Ok(self
            .inner
            .step_ssgf(dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight))
    }

    #[pyo3(signature = (
        n_steps,
        dt,
        seed=0,
        W=None,
        sigma_g=0.0,
        h_munu=None,
        pgbo_weight=0.0,
    ))]
    #[allow(clippy::too_many_arguments, non_snake_case)]
    fn run_ssgf(
        &mut self,
        n_steps: usize,
        dt: f64,
        seed: u64,
        W: Option<&Bound<'_, PyAny>>,
        sigma_g: f64,
        h_munu: Option<&Bound<'_, PyAny>>,
        pgbo_weight: f64,
    ) -> PyResult<Vec<f64>> {
        validate_kuramoto_dt(dt)?;
        if !sigma_g.is_finite() {
            return Err(PyValueError::new_err("sigma_g must be finite"));
        }
        if !pgbo_weight.is_finite() {
            return Err(PyValueError::new_err("pgbo_weight must be finite"));
        }
        let (w_flat, w_rows, w_cols) = match W {
            Some(w) => extract_matrix_f64(w, "W")?,
            None => (vec![], 0, 0),
        };
        let (h_flat, h_rows, h_cols) = match h_munu {
            Some(h) => extract_matrix_f64(h, "h_munu")?,
            None => (vec![], 0, 0),
        };
        validate_kuramoto_matrix_shape("W", w_flat.len(), w_rows, w_cols, self.inner.n)?;
        validate_kuramoto_matrix_shape("h_munu", h_flat.len(), h_rows, h_cols, self.inner.n)?;
        validate_kuramoto_finite("w_flat", &w_flat)?;
        validate_kuramoto_finite("h_flat", &h_flat)?;
        Ok(self
            .inner
            .run_ssgf(n_steps, dt, seed, &w_flat, sigma_g, &h_flat, pgbo_weight))
    }

    fn order_parameter(&self) -> f64 {
        self.inner.order_parameter()
    }

    fn apply_phases(&mut self, phases: Vec<f64>) -> PyResult<()> {
        if phases.len() != self.inner.n {
            return Err(PyValueError::new_err(format!(
                "phases length mismatch: got {}, expected {}.",
                phases.len(),
                self.inner.n
            )));
        }
        validate_kuramoto_finite("phases", &phases)?;
        self.inner.set_phases(phases);
        Ok(())
    }

    fn set_phases(&mut self, phases: Vec<f64>) -> PyResult<()> {
        self.apply_phases(phases)
    }

    #[setter(phases)]
    fn set_phases_attr(&mut self, phases: Vec<f64>) -> PyResult<()> {
        self.apply_phases(phases)
    }

    fn set_coupling(&mut self, coupling: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.n;
        let (coupling_flat, rows, cols) = extract_matrix_f64(coupling, "coupling")?;
        if rows == 1 {
            if coupling_flat.len() != n * n {
                return Err(PyValueError::new_err(format!(
                    "Flat coupling length mismatch: got {}, expected {}.",
                    coupling_flat.len(),
                    n * n
                )));
            }
        } else if rows != n || cols != n {
            return Err(PyValueError::new_err(format!(
                "coupling must be shape ({}, {}) or flat length {}, got ({}, {}).",
                n,
                n,
                n * n,
                rows,
                cols
            )));
        }
        validate_kuramoto_finite("coupling", &coupling_flat)?;
        self.inner.set_coupling(coupling_flat);
        Ok(())
    }
}
