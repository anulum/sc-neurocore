// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rate-model compatibility facade

//! Compatibility facade for bounded rate-model modules.

mod amari_neural_field;
mod astrocyte_model;
mod compte_wm;
mod fractional_lif;
mod leaky_compete_fire;
mod liquid_time_constant;
mod mcculloch_pitts;
mod parallel_spiking;
mod siegert_transfer_function;
mod sigmoid_rate;
mod threshold_linear_rate;
mod tsodyks_markram;

pub use amari_neural_field::AmariNeuralField;
pub use astrocyte_model::AstrocyteModel;
pub use compte_wm::CompteWMNeuron;
pub use fractional_lif::FractionalLIFNeuron;
pub use leaky_compete_fire::LeakyCompeteFireNeuron;
pub use liquid_time_constant::LiquidTimeConstantNeuron;
pub use mcculloch_pitts::McCullochPittsNeuron;
pub use parallel_spiking::ParallelSpikingNeuron;
pub use siegert_transfer_function::SiegertTransferFunction;
pub use sigmoid_rate::SigmoidRateNeuron;
pub use threshold_linear_rate::ThresholdLinearRateNeuron;
pub use tsodyks_markram::TsodyksMarkramNeuron;
