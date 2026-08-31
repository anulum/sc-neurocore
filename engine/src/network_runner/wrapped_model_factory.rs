// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NetworkRunner wrapped-model construction

//! Constructors for models normalized through the input-adapter layer.

use super::super::input_adapters::*;
use super::super::NeuronVariant;

pub(super) fn create_wrapped_neuron(name: &str) -> Option<Result<NeuronVariant, String>> {
    let result = match name {
        // Multi-input spiking models.
        "AlphaNeuron" | "Alpha" => Ok(NeuronVariant::WrAlphaCell(WrAlpha::new())),
        "COBALIFNeuron" | "COBALIF" => Ok(NeuronVariant::WrCOBALIFCell(WrCOBALIF::new())),
        "CompteWMNeuron" | "CompteWM" => Ok(NeuronVariant::WrCompteWMCell(WrCompteWM::new())),
        "TsodyksMarkramNeuron" | "TsodyksMarkram" => {
            Ok(NeuronVariant::WrTsodyksMarkramCell(WrTsodyksMarkram::new()))
        }
        "PinskyRinzelNeuron" | "PinskyRinzel" => {
            Ok(NeuronVariant::WrPinskyRinzelCell(WrPinskyRinzel::new()))
        }
        "HayL5PyramidalNeuron" | "HayL5" => Ok(NeuronVariant::WrHayL5Cell(WrHayL5::new())),
        "TwoCompartmentLIFNeuron" | "TwoCompLIF" => {
            Ok(NeuronVariant::WrTwoCompLIFCell(WrTwoCompLIF::new()))
        }
        // Hardware models with integer inputs.
        "LoihiCUBANeuron" | "LoihiCUBA" => Ok(NeuronVariant::WrLoihiCUBACell(WrLoihiCUBA::new())),
        "Loihi2Neuron" | "Loihi2" => Ok(NeuronVariant::WrLoihi2Cell(WrLoihi2::new())),
        "SpiNNaker2Neuron" | "SpiNNaker2" => {
            Ok(NeuronVariant::WrSpiNNaker2Cell(WrSpiNNaker2::new()))
        }
        "TrueNorthNeuron" | "TrueNorth" => Ok(NeuronVariant::WrTrueNorthCell(WrTrueNorth::new())),
        "IntegerQIFNeuron" | "IntegerQIF" => {
            Ok(NeuronVariant::WrIntegerQIFCell(WrIntegerQIF::new()))
        }
        "McCullochPittsNeuron" | "McCullochPitts" => {
            Ok(NeuronVariant::WrMcCullochPittsCell(WrMcCullochPitts::new()))
        }
        // Graded and rate-output models.
        "SigmoidRateNeuron" | "SigmoidRate" => {
            Ok(NeuronVariant::WrSigmoidRateCell(WrSigmoidRate::new()))
        }
        "ThresholdLinearRateNeuron" | "ThresholdLinearRate" => Ok(
            NeuronVariant::WrThresholdLinearCell(WrThresholdLinear::new()),
        ),
        "AstrocyteModel" | "Astrocyte" => Ok(NeuronVariant::WrAstrocyteCell(WrAstrocyte::new())),
        "InnerHairCell" | "IHC" => Ok(NeuronVariant::WrInnerHairCellCell(WrInnerHairCell::new())),
        "OuterHairCell" | "OHC" => Ok(NeuronVariant::WrOuterHairCellCell(WrOuterHairCell::new())),
        "RodPhotoreceptor" | "Rod" => Ok(NeuronVariant::WrRodPhotoreceptorCell(
            WrRodPhotoreceptor::new(),
        )),
        "ConePhotoreceptor" | "Cone" => Ok(NeuronVariant::WrConePhotoreceptorCell(
            WrConePhotoreceptor::new(),
        )),
        "TasteReceptorCell" | "TasteReceptor" => {
            Ok(NeuronVariant::WrTasteReceptorCell(WrTasteReceptor::new()))
        }
        _ => return None,
    };
    Some(result)
}
