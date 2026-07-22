// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rate-model PyO3 binding composition

use pyo3::prelude::*;

mod amari_neural_field;
mod astrocyte_model;
mod compte_wm;
mod fractional_lif;
mod leaky_compete_fire;
mod liquid_time_constant;
mod parallel_spiking;
mod siegert_transfer_function;
mod tsodyks_markram;

pub use amari_neural_field::PyAmariNeuralField;
pub use astrocyte_model::PyAstrocyteModel;
pub use compte_wm::PyCompteWMNeuron;
pub use fractional_lif::PyFractionalLIFNeuron;
pub use leaky_compete_fire::PyLeakyCompeteFireNeuron;
pub use liquid_time_constant::PyLiquidTimeConstantNeuron;
pub use parallel_spiking::PyParallelSpikingNeuron;
pub use siegert_transfer_function::PySiegertTransferFunction;
pub use tsodyks_markram::PyTsodyksMarkramNeuron;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    astrocyte_model::register(module)?;
    tsodyks_markram::register(module)?;
    liquid_time_constant::register(module)?;
    compte_wm::register(module)?;
    siegert_transfer_function::register(module)?;
    fractional_lif::register(module)?;
    parallel_spiking::register(module)?;
    amari_neural_field::register(module)?;
    leaky_compete_fire::register(module)?;
    Ok(())
}
