// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-compartment PyO3 binding composition

use pyo3::prelude::*;

mod astrocyte_lif;
mod booth_rinzel;
mod dendrify;
mod dendritic_nmda;
mod hay_l5_pyramidal;
mod marder_stg;
mod multicompartment_mcn;
mod pinsky_rinzel;
mod rall_cable;
mod two_compartment_lif;

pub use astrocyte_lif::PyAstrocyteLIFNeuron;
pub use booth_rinzel::PyBoothRinzelNeuron;
pub use dendrify::PyDendrifyNeuron;
pub use dendritic_nmda::PyDendriticNMDANeuron;
pub use hay_l5_pyramidal::PyHayL5PyramidalNeuron;
pub use marder_stg::PyMarderSTGNeuron;
pub use multicompartment_mcn::PyMulticompartmentMCNNeuron;
pub use pinsky_rinzel::PyPinskyRinzelNeuron;
pub use rall_cable::PyRallCableNeuron;
pub use two_compartment_lif::{PySCExponentialTwoCompartmentLIF, PyTwoCompartmentLIFNeuron};

/// Register the multi-compartment neuron classes in their stable order.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    pinsky_rinzel::register(module)?;
    hay_l5_pyramidal::register(module)?;
    marder_stg::register(module)?;
    rall_cable::register(module)?;
    booth_rinzel::register(module)?;
    dendrify::register(module)?;
    two_compartment_lif::register(module)?;
    dendritic_nmda::register(module)?;
    multicompartment_mcn::register(module)?;
    astrocyte_lif::register(module)?;
    Ok(())
}
