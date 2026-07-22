// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PING circuit PyO3 binding

//! Python binding for the Börgers-Kopell PING circuit step kernel.

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

use crate::ping;

/// Register the PING circuit step kernel with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_ping_step, module)?)?;
    Ok(())
}

/// Advance excitatory and inhibitory PING populations by one time step.
///
/// The caller supplies per-instance state arrays and pre-drawn noise samples
/// so Python and Rust preserve the same seeded random sequence. State and
/// spike arrays are updated in place; the return value contains the two
/// population spike counts needed for the caller's conductance update.
#[pyfunction]
#[pyo3(signature = (
    v_e, g_ampa_e, g_gaba_e, refrac_e, i_drive_e, xi_e, spikes_e_out,
    v_i, g_ampa_i, g_gaba_i, refrac_i, i_drive_i, xi_i, spikes_i_out,
    e_l, e_ampa, e_gaba, g_l, c_m, v_threshold, v_reset, t_refrac,
    tau_ampa, tau_gaba, sigma_e, sigma_i, dt,
))]
#[allow(clippy::too_many_arguments)]
fn py_ping_step<'py>(
    _py: Python<'py>,
    v_e: PyReadwriteArray1<'_, f64>,
    g_ampa_e: PyReadwriteArray1<'_, f64>,
    g_gaba_e: PyReadwriteArray1<'_, f64>,
    refrac_e: PyReadwriteArray1<'_, f64>,
    i_drive_e: PyReadonlyArray1<'_, f64>,
    xi_e: PyReadonlyArray1<'_, f64>,
    spikes_e_out: PyReadwriteArray1<'_, u8>,
    v_i: PyReadwriteArray1<'_, f64>,
    g_ampa_i: PyReadwriteArray1<'_, f64>,
    g_gaba_i: PyReadwriteArray1<'_, f64>,
    refrac_i: PyReadwriteArray1<'_, f64>,
    i_drive_i: PyReadonlyArray1<'_, f64>,
    xi_i: PyReadonlyArray1<'_, f64>,
    spikes_i_out: PyReadwriteArray1<'_, u8>,
    e_l: f64,
    e_ampa: f64,
    e_gaba: f64,
    g_l: f64,
    c_m: f64,
    v_threshold: f64,
    v_reset: f64,
    t_refrac: f64,
    tau_ampa: f64,
    tau_gaba: f64,
    sigma_e: f64,
    sigma_i: f64,
    dt: f64,
) -> PyResult<(u32, u32)> {
    let mut v_e = v_e;
    let mut g_ampa_e = g_ampa_e;
    let mut g_gaba_e = g_gaba_e;
    let mut refrac_e = refrac_e;
    let mut spikes_e_out = spikes_e_out;
    let mut v_i = v_i;
    let mut g_ampa_i = g_ampa_i;
    let mut g_gaba_i = g_gaba_i;
    let mut refrac_i = refrac_i;
    let mut spikes_i_out = spikes_i_out;
    let (ne, ni) = ping::step_kernel(
        v_e.as_slice_mut()?,
        g_ampa_e.as_slice_mut()?,
        g_gaba_e.as_slice_mut()?,
        refrac_e.as_slice_mut()?,
        i_drive_e.as_slice()?,
        xi_e.as_slice()?,
        spikes_e_out.as_slice_mut()?,
        v_i.as_slice_mut()?,
        g_ampa_i.as_slice_mut()?,
        g_gaba_i.as_slice_mut()?,
        refrac_i.as_slice_mut()?,
        i_drive_i.as_slice()?,
        xi_i.as_slice()?,
        spikes_i_out.as_slice_mut()?,
        e_l,
        e_ampa,
        e_gaba,
        g_l,
        c_m,
        v_threshold,
        v_reset,
        t_refrac,
        tau_ampa,
        tau_gaba,
        sigma_e,
        sigma_i,
        dt,
    );
    Ok((ne, ni))
}
