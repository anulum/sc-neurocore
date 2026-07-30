// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner model factory

//! Canonical model-name catalogue and construction for network execution.

use super::input_adapters::*;
use super::{NeuronVariant, PopulationRunner};
use crate::neuron::*;
use crate::neurons::*;

/// Create a population of `n` identical neurons by model name.
pub fn create_population(model_name: &str, n: usize) -> Result<PopulationRunner, String> {
    let neurons: Vec<NeuronVariant> = (0..n)
        .map(|_| create_neuron(model_name))
        .collect::<Result<_, _>>()?;
    Ok(PopulationRunner::new(neurons))
}

pub fn create_neuron(name: &str) -> Result<NeuronVariant, String> {
    match name {
        "Izhikevich" => Ok(NeuronVariant::Izhikevich(Izhikevich::regular_spiking())),
        "AdEx" | "AdExNeuron" => Ok(NeuronVariant::AdEx(AdExNeuron::new())),
        "ExpIF" | "ExpIfNeuron" => Ok(NeuronVariant::ExpIF(ExpIfNeuron::new())),
        "Lapicque" | "LapicqueNeuron" => Ok(NeuronVariant::Lapicque(LapicqueNeuron::new(
            20.0, 1.0, 1.0, 1.0,
        ))),
        "HomeostaticLif" => Ok(NeuronVariant::HomeostaticLif(
            HomeostaticLif::with_defaults(),
        )),
        "HodgkinHuxley" | "HodgkinHuxleyNeuron" => {
            Ok(NeuronVariant::HodgkinHuxley(HodgkinHuxleyNeuron::new()))
        }
        "TraubMiles" | "TraubMilesNeuron" => Ok(NeuronVariant::TraubMiles(TraubMilesNeuron::new())),
        "WangBuzsaki" | "WangBuzsakiNeuron" => {
            Ok(NeuronVariant::WangBuzsaki(WangBuzsakiNeuron::new()))
        }
        "ConnorStevens" | "ConnorStevensNeuron" => {
            Ok(NeuronVariant::ConnorStevens(ConnorStevensNeuron::new()))
        }
        "DestexheThalamic" | "DestexheThalamicNeuron" => Ok(NeuronVariant::DestexheThalamic(
            DestexheThalamicNeuron::new(),
        )),
        "HuberBraun" | "HuberBraunNeuron" => Ok(NeuronVariant::HuberBraun(HuberBraunNeuron::new())),
        "GolombFS" | "GolombFSNeuron" => Ok(NeuronVariant::GolombFS(GolombFSNeuron::new())),
        "Pospischil" | "PospischilNeuron" => Ok(NeuronVariant::Pospischil(PospischilNeuron::new())),
        "MainenSejnowski" | "MainenSejnowskiNeuron" => {
            Ok(NeuronVariant::MainenSejnowski(MainenSejnowskiNeuron::new()))
        }
        "DeSchutterPurkinje" | "DeSchutterPurkinjeNeuron" => Ok(NeuronVariant::DeSchutterPurkinje(
            DeSchutterPurkinjeNeuron::new(),
        )),
        "PlantR15" | "PlantR15Neuron" => Ok(NeuronVariant::PlantR15(PlantR15Neuron::new())),
        "Prescott" | "PrescottNeuron" => Ok(NeuronVariant::Prescott(PrescottNeuron::new())),
        "MihalasNiebur" | "MihalasNieburNeuron" => {
            Ok(NeuronVariant::MihalasNiebur(MihalasNieburNeuron::new()))
        }
        "GLIF" | "GLIFNeuron" => Ok(NeuronVariant::GLIF(GLIFNeuron::new())),
        "GIFPopulation" | "GIFPopulationNeuron" => {
            Ok(NeuronVariant::GIFPopulation(GIFPopulationNeuron::new(42)))
        }
        "AvRonCardiac" | "AvRonCardiacNeuron" => {
            Ok(NeuronVariant::AvRonCardiac(AvRonCardiacNeuron::new()))
        }
        "DurstewitzDopamine" | "DurstewitzDopamineNeuron" => Ok(NeuronVariant::DurstewitzDopamine(
            DurstewitzDopamineNeuron::new(),
        )),
        "HillTononi" | "HillTononiNeuron" => Ok(NeuronVariant::HillTononi(HillTononiNeuron::new())),
        "BertramPhantom" | "BertramPhantomBurster" => {
            Ok(NeuronVariant::BertramPhantom(BertramPhantomBurster::new()))
        }
        "Yamada" | "YamadaNeuron" => Ok(NeuronVariant::Yamada(YamadaNeuron::new())),
        "FitzHughNagumo" | "FitzHughNagumoNeuron" => {
            Ok(NeuronVariant::FitzHughNagumo(FitzHughNagumoNeuron::new()))
        }
        "MorrisLecar" | "MorrisLecarNeuron" => {
            Ok(NeuronVariant::MorrisLecar(MorrisLecarNeuron::new()))
        }
        "HindmarshRose" | "HindmarshRoseNeuron" => {
            Ok(NeuronVariant::HindmarshRose(HindmarshRoseNeuron::new()))
        }
        "ResonateAndFire" | "ResonateAndFireNeuron" => {
            Ok(NeuronVariant::ResonateAndFire(ResonateAndFireNeuron::new()))
        }
        "BalancedResonateAndFire" | "BalancedResonateAndFireNeuron" => Ok(
            NeuronVariant::BalancedResonateAndFire(BalancedResonateAndFireNeuron::new()),
        ),
        "FitzHughRinzel" | "FitzHughRinzelNeuron" => {
            Ok(NeuronVariant::FitzHughRinzel(FitzHughRinzelNeuron::new()))
        }
        "McKean" | "McKeanNeuron" => Ok(NeuronVariant::McKean(McKeanNeuron::new())),
        "SCTriangularMcKean" | "SCTriangularMcKeanNeuron" => Ok(NeuronVariant::SCTriangularMcKean(
            SCTriangularMcKeanNeuron::new(),
        )),
        "TermanWang" | "TermanWangOscillator" => {
            Ok(NeuronVariant::TermanWang(TermanWangOscillator::new()))
        }
        "GutkinErmentrout" | "GutkinErmentroutNeuron" => Ok(NeuronVariant::GutkinErmentrout(
            GutkinErmentroutNeuron::new(),
        )),
        "WilsonHR" | "WilsonHRNeuron" => Ok(NeuronVariant::WilsonHR(WilsonHRNeuron::new())),
        "Chay" | "ChayNeuron" => Ok(NeuronVariant::Chay(ChayNeuron::new())),
        "ChayKeizer" | "ChayKeizerNeuron" => Ok(NeuronVariant::ChayKeizer(ChayKeizerNeuron::new())),
        "ShermanRinzelKeizer" | "ShermanRinzelKeizerNeuron" => Ok(
            NeuronVariant::ShermanRinzelKeizer(ShermanRinzelKeizerNeuron::new()),
        ),
        "ButeraRespiratory" | "ButeraRespiratoryNeuron" => Ok(NeuronVariant::ButeraRespiratory(
            ButeraRespiratoryNeuron::new(),
        )),
        "EPropALIF" | "EPropALIFNeuron" => Ok(NeuronVariant::EPropALIF(EPropALIFNeuron::default())),
        "SuperSpike" | "SuperSpikeNeuron" => {
            Ok(NeuronVariant::SuperSpike(SuperSpikeNeuron::default()))
        }
        "LearnableNeuron" | "LearnableNeuronModel" => {
            Ok(NeuronVariant::LearnableNeuron(LearnableNeuronModel::new()))
        }
        "Pernarowski" | "PernarowskiNeuron" => {
            Ok(NeuronVariant::Pernarowski(PernarowskiNeuron::new()))
        }
        "QuadraticIF" | "QuadraticIFNeuron" => {
            Ok(NeuronVariant::QuadraticIF(QuadraticIFNeuron::default()))
        }
        "Theta" | "ThetaNeuron" => Ok(NeuronVariant::Theta(ThetaNeuron::default())),
        "PerfectIntegrator" | "PerfectIntegratorNeuron" => Ok(NeuronVariant::PerfectIntegrator(
            PerfectIntegratorNeuron::default(),
        )),
        "GatedLIF" | "GatedLIFNeuron" => Ok(NeuronVariant::GatedLIF(GatedLIFNeuron::default())),
        "NonlinearLIF" | "NonlinearLIFNeuron" => {
            Ok(NeuronVariant::NonlinearLIF(NonlinearLIFNeuron::new()))
        }
        "SFA" | "SFANeuron" => Ok(NeuronVariant::SFA(SFANeuron::new())),
        "MAT" | "MATNeuron" => Ok(NeuronVariant::MAT(MATNeuron::new())),
        "SCResettingMAT" | "SCResettingMATNeuron" => {
            Ok(NeuronVariant::SCResettingMAT(SCResettingMATNeuron::new()))
        }
        "KLIF" | "KLIFNeuron" => Ok(NeuronVariant::KLIF(KLIFNeuron::default())),
        "InhibitoryLIF" | "InhibitoryLIFNeuron" => {
            Ok(NeuronVariant::InhibitoryLIF(InhibitoryLIFNeuron::default()))
        }
        "ComplementaryLIF" | "ComplementaryLIFNeuron" => Ok(NeuronVariant::ComplementaryLIF(
            ComplementaryLIFNeuron::default(),
        )),
        "ParametricLIF" | "ParametricLIFNeuron" => {
            Ok(NeuronVariant::ParametricLIF(ParametricLIFNeuron::default()))
        }
        "NonResettingLIF" | "NonResettingLIFNeuron" => {
            Ok(NeuronVariant::NonResettingLIF(NonResettingLIFNeuron::new()))
        }
        "SCNonResettingAdaptiveLIF" | "SCNonResettingAdaptiveLIFNeuron" => Ok(
            NeuronVariant::SCNonResettingAdaptiveLIF(SCNonResettingAdaptiveLIFNeuron::new()),
        ),
        "AdaptiveThresholdIF" | "AdaptiveThresholdIFNeuron" => Ok(
            NeuronVariant::AdaptiveThresholdIF(AdaptiveThresholdIFNeuron::new()),
        ),
        "SigmaDelta" | "SigmaDeltaNeuron" => {
            Ok(NeuronVariant::SigmaDelta(SigmaDeltaNeuron::default()))
        }
        "SCSigmaDeltaAccumulator" | "SCSigmaDeltaAccumulatorNeuron" => Ok(
            NeuronVariant::SCSigmaDeltaAccumulator(SCSigmaDeltaAccumulatorNeuron::default()),
        ),
        "EnergyLIF" | "EnergyLIFNeuron" => Ok(NeuronVariant::EnergyLIF(EnergyLIFNeuron::new())),
        "SCNormalizedEnergyLIF" | "SCNormalizedEnergyLIFNeuron" => Ok(
            NeuronVariant::SCNormalizedEnergyLIF(SCNormalizedEnergyLIFNeuron::new()),
        ),
        "ClosedFormContinuous" | "ClosedFormContinuousNeuron" => Ok(
            NeuronVariant::ClosedFormContinuous(ClosedFormContinuousNeuron::new()),
        ),
        "ChialvoMap" | "ChialvoMapNeuron" => Ok(NeuronVariant::ChialvoMap(ChialvoMapNeuron::new())),
        "RulkovMap" | "RulkovMapNeuron" => Ok(NeuronVariant::RulkovMap(RulkovMapNeuron::new())),
        "IbarzTanakaMap" | "IbarzTanakaMapNeuron" => {
            Ok(NeuronVariant::IbarzTanakaMap(IbarzTanakaMapNeuron::new()))
        }
        "MedvedevMap" | "MedvedevMapNeuron" => {
            Ok(NeuronVariant::MedvedevMap(MedvedevMapNeuron::default()))
        }
        "CazellesMap" | "CazellesMapNeuron" => {
            Ok(NeuronVariant::CazellesMap(CazellesMapNeuron::new()))
        }
        "CourageNekorkinMap" | "CourageNekorkinMapNeuron" => Ok(NeuronVariant::CourageNekorkinMap(
            CourageNekorkinMapNeuron::new(),
        )),
        "AiharaMap" | "AiharaMapNeuron" => Ok(NeuronVariant::AiharaMap(AiharaMapNeuron::new())),
        "SCChaoticMap" | "SCChaoticMapNeuron" => {
            Ok(NeuronVariant::SCChaoticMap(SCChaoticMapNeuron::new()))
        }
        "NagumoSatoMap" | "NagumoSatoMapNeuron" => {
            Ok(NeuronVariant::NagumoSatoMap(NagumoSatoMapNeuron::new()))
        }
        "SCAdaptiveThresholdMap"
        | "SCAdaptiveThresholdMapNeuron"
        | "KilincBhattMap"
        | "KilincBhattMapNeuron" => Ok(NeuronVariant::SCAdaptiveThresholdMap(
            SCAdaptiveThresholdMapNeuron::new(),
        )),
        "ErmentroutKopellMap" | "ErmentroutKopellMapNeuron" => Ok(
            NeuronVariant::ErmentroutKopellMap(ErmentroutKopellMapNeuron::new()),
        ),
        "BrainScaleSAdEx" | "BrainScaleSAdExNeuron" => {
            Ok(NeuronVariant::BrainScaleSAdEx(BrainScaleSAdExNeuron::new()))
        }
        "SpiNNakerLIF" | "SpiNNakerLIFNeuron" => {
            Ok(NeuronVariant::SpiNNakerLIF(SpiNNakerLIFNeuron::new()))
        }
        "NeuroGrid" | "NeuroGridNeuron" => Ok(NeuronVariant::NeuroGrid(NeuroGridNeuron::new())),
        "DPI" | "DPINeuron" => Ok(NeuronVariant::DPI(DPINeuron::new())),
        "MarderSTG" | "MarderSTGNeuron" => Ok(NeuronVariant::MarderSTG(MarderSTGNeuron::new())),
        "RallCable" | "RallCableNeuron" => Ok(NeuronVariant::RallCable(RallCableNeuron::new(5))),
        "BoothRinzel" | "BoothRinzelNeuron" => {
            Ok(NeuronVariant::BoothRinzel(BoothRinzelNeuron::new()))
        }
        "Dendrify" | "DendrifyNeuron" => Ok(NeuronVariant::Dendrify(DendrifyNeuron::new())),
        "Akida" | "AkidaNeuron" => Ok(NeuronVariant::Akida(AkidaNeuron::new(100))),
        "StochasticLIF" | "StochasticLIFNeuron" => {
            Ok(NeuronVariant::StochasticLIF(StochasticLIFNeuron::new(42)))
        }
        "LiquidTimeConstant" | "LiquidTimeConstantNeuron" => Ok(NeuronVariant::LiquidTimeConstant(
            LiquidTimeConstantNeuron::new(),
        )),
        "ParallelSpiking" | "ParallelSpikingNeuron" => Ok(NeuronVariant::ParallelSpiking(
            ParallelSpikingNeuron::new(4, 0.5),
        )),
        "FractionalLIF" | "FractionalLIFNeuron" => Ok(NeuronVariant::FractionalLIF(
            FractionalLIFNeuron::new(0.8, 50),
        )),
        "StochasticIF" | "StochasticIFNeuron" => {
            Ok(NeuronVariant::StochasticIF(StochasticIFNeuron::new(42)))
        }
        "GalvesLocherbach" | "GalvesLocherbachNeuron" => Ok(NeuronVariant::GalvesLocherbach(
            GalvesLocherbachNeuron::new(42),
        )),
        "SpikeResponse" | "SpikeResponseNeuron" => {
            Ok(NeuronVariant::SpikeResponse(SpikeResponseNeuron::new()))
        }
        "GLM" | "GLMNeuron" => Ok(NeuronVariant::GLM(GLMNeuron::new(5, 10, 42))),
        "Arcane" | "ArcaneNeuron" => Ok(NeuronVariant::Arcane(ArcaneNeuron::new())),
        // newly wired
        "MultiTimescale" | "MultiTimescaleNeuron" => {
            Ok(NeuronVariant::MultiTimescale(MultiTimescaleNeuron::new()))
        }
        "AttentionGated" | "AttentionGatedNeuron" => {
            Ok(NeuronVariant::AttentionGated(AttentionGatedNeuron::new()))
        }
        "PredictiveCoding" | "PredictiveCodingNeuron" => Ok(NeuronVariant::PredictiveCoding(
            PredictiveCodingNeuron::new(),
        )),
        "SelfReferential" | "SelfReferentialNeuron" => {
            Ok(NeuronVariant::SelfReferential(SelfReferentialNeuron::new()))
        }
        "CompositionalBinding" | "CompositionalBindingNeuron" => Ok(
            NeuronVariant::CompositionalBinding(CompositionalBindingNeuron::new()),
        ),
        "DifferentiableSurrogate" | "DifferentiableSurrogateNeuron" => Ok(
            NeuronVariant::DifferentiableSurrogate(DifferentiableSurrogateNeuron::new()),
        ),
        "ContinuousAttractor" | "ContinuousAttractorNeuron" => Ok(
            NeuronVariant::ContinuousAttractor(ContinuousAttractorNeuron::new(8)),
        ),
        "MetaPlastic" | "MetaPlasticNeuron" => {
            Ok(NeuronVariant::MetaPlastic(MetaPlasticNeuron::new()))
        }
        "BendaHerz" | "BendaHerzNeuron" => Ok(NeuronVariant::BendaHerz(BendaHerzNeuron::new(42))),
        "BrunelWang" | "BrunelWangNeuron" => Ok(NeuronVariant::BrunelWang(BrunelWangNeuron::new())),
        "Poisson" | "PoissonNeuron" => {
            Ok(NeuronVariant::Poisson(PoissonNeuron::new(50.0, 1.0, 42)))
        }
        "InhomogeneousPoisson" | "InhomogeneousPoissonNeuron" => Ok(
            NeuronVariant::InhomogeneousPoisson(InhomogeneousPoissonNeuron::new(1.0, 42)),
        ),
        "GammaRenewal" | "GammaRenewalNeuron" => Ok(NeuronVariant::GammaRenewal(
            GammaRenewalNeuron::new(50.0, 3, 42),
        )),
        "EscapeRate" | "EscapeRateNeuron" => {
            Ok(NeuronVariant::EscapeRate(EscapeRateNeuron::new(42)))
        }
        // interneurons
        "PVFastSpiking" | "PVFastSpikingNeuron" => {
            Ok(NeuronVariant::PVFastSpiking(PVFastSpikingNeuron::new()))
        }
        "SST" | "SSTNeuron" => Ok(NeuronVariant::SST(SSTNeuron::new())),
        "VIP" | "VIPNeuron" => Ok(NeuronVariant::VIP(VIPNeuron::new())),
        "Chandelier" | "ChandelierNeuron" => Ok(NeuronVariant::Chandelier(ChandelierNeuron::new())),
        "CerebellarBasket" | "CerebellarBasketNeuron" => Ok(NeuronVariant::CerebellarBasket(
            CerebellarBasketNeuron::new(),
        )),
        "Martinotti" | "MartinottiNeuron" => Ok(NeuronVariant::Martinotti(MartinottiNeuron::new())),
        // motor
        "AlphaMotor" | "AlphaMotorNeuron" => Ok(NeuronVariant::AlphaMotor(AlphaMotorNeuron::new())),
        "GammaMotor" | "GammaMotorNeuron" => Ok(NeuronVariant::GammaMotor(GammaMotorNeuron::new())),
        "UpperMotor" | "UpperMotorNeuron" => Ok(NeuronVariant::UpperMotor(UpperMotorNeuron::new())),
        "Renshaw" | "RenshawCell" => Ok(NeuronVariant::Renshaw(RenshawCell::new())),
        "MotorUnit" => Ok(NeuronVariant::MotorUnitCell(MotorUnit::new())),
        // sensory (spiking)
        "RetinalGanglion" | "RetinalGanglionCell" => {
            Ok(NeuronVariant::RetinalGanglion(RetinalGanglionCell::new()))
        }
        "Merkel" | "MerkelCell" => Ok(NeuronVariant::Merkel(MerkelCell::new())),
        "Pacinian" | "PacinianCorpuscle" => Ok(NeuronVariant::Pacinian(PacinianCorpuscle::new())),
        "Nociceptor" => Ok(NeuronVariant::NociceptorCell(Nociceptor::new())),
        "OlfactoryReceptor" | "OlfactoryReceptorNeuron" => Ok(NeuronVariant::OlfactoryReceptor(
            OlfactoryReceptorNeuron::new(),
        )),
        // cerebellar
        "GranuleCell" | "Granule" => Ok(NeuronVariant::Granule(GranuleCell::new())),
        "GolgiCell" | "Golgi" => Ok(NeuronVariant::Golgi(GolgiCell::new())),
        "StellateCell" | "Stellate" => Ok(NeuronVariant::Stellate(StellateCell::new())),
        "LugaroCell" | "Lugaro" => Ok(NeuronVariant::Lugaro(LugaroCell::new())),
        "UnipolarBrushCell" | "UBC" => Ok(NeuronVariant::UnipolarBrush(UnipolarBrushCell::new())),
        "DCNNeuron" | "DCN" => Ok(NeuronVariant::DCN(DCNNeuron::new())),
        // channels
        "PersistentNa" | "PersistentNaNeuron" => {
            Ok(NeuronVariant::PersistentNa(PersistentNaNeuron::new()))
        }
        "Ih" | "IhNeuron" => Ok(NeuronVariant::Ih(IhNeuron::new())),
        "TTypeCa" | "TTypeCaNeuron" => Ok(NeuronVariant::TTypeCa(TTypeCaNeuron::new())),
        "ATypeK" | "ATypeKNeuron" => Ok(NeuronVariant::ATypeK(ATypeKNeuron::new())),
        "BK" | "BKNeuron" => Ok(NeuronVariant::BK(BKNeuron::new())),
        "SK" | "SKNeuron" => Ok(NeuronVariant::SK(SKNeuron::new())),
        "NMDA" | "NMDANeuron" => Ok(NeuronVariant::NMDA(NMDANeuron::new())),
        // population
        "MontbrioMeanField" | "MPR" => Ok(NeuronVariant::MontbrioMPR(MontbrioMeanField::new())),
        "BrunelNetwork" | "Brunel" => Ok(NeuronVariant::Brunel(BrunelNetwork::new())),
        "TUMNetwork" | "TUM" => Ok(NeuronVariant::TUM(TUMNetwork::new())),
        "ElBoustaniNetwork" | "ElBoustani" => {
            Ok(NeuronVariant::ElBoustani(ElBoustaniNetwork::new()))
        }
        // misc
        "GradedSynapseNeuron" | "GradedSynapse" => {
            Ok(NeuronVariant::GradedSynapse(GradedSynapseNeuron::new()))
        }
        "GapJunctionNeuron" | "GapJunction" => {
            Ok(NeuronVariant::GapJunction(GapJunctionNeuron::new()))
        }
        "FrankenhaeUserHuxleyAxon" | "FHAxon" => {
            Ok(NeuronVariant::FHAxon(FrankenhaeUserHuxleyAxon::new()))
        }
        "NodeOfRanvier" => Ok(NeuronVariant::NodeOfRanvier(NodeOfRanvier::new())),
        "MyelinatedAxon" | "MyelinAxon" => Ok(NeuronVariant::MyelinAxon(MyelinatedAxon::new())),
        "CardiacPurkinjeFibre" | "CardiacPurkinje" => {
            Ok(NeuronVariant::CardiacPurkinje(CardiacPurkinjeFibre::new()))
        }
        "SmoothMuscleCell" | "SmoothMuscle" => {
            Ok(NeuronVariant::SmoothMuscle(SmoothMuscleCell::new()))
        }
        "EndocrineBetaCell" | "BetaCell" => Ok(NeuronVariant::BetaCell(EndocrineBetaCell::new())),
        // Wrapped multi-input spiking
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
        // Wrapped hardware integer-input
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
        // Wrapped graded/rate output
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
        _ => Err(format!("Unsupported model: '{name}'")),
    }
}

/// List all supported model names.
pub fn supported_models() -> Vec<&'static str> {
    vec![
        "Izhikevich",
        "AdEx",
        "ExpIF",
        "Lapicque",
        "HomeostaticLif",
        "HodgkinHuxley",
        "TraubMiles",
        "WangBuzsaki",
        "ConnorStevens",
        "DestexheThalamic",
        "HuberBraun",
        "GolombFS",
        "Pospischil",
        "MainenSejnowski",
        "DeSchutterPurkinje",
        "PlantR15",
        "Prescott",
        "MihalasNiebur",
        "GLIF",
        "GIFPopulation",
        "AvRonCardiac",
        "DurstewitzDopamine",
        "HillTononi",
        "BertramPhantom",
        "Yamada",
        "FitzHughNagumo",
        "MorrisLecar",
        "HindmarshRose",
        "ResonateAndFire",
        "BalancedResonateAndFire",
        "FitzHughRinzel",
        "McKean",
        "SCTriangularMcKean",
        "TermanWang",
        "GutkinErmentrout",
        "WilsonHR",
        "Akida",
        "StochasticLIF",
        "Chay",
        "ChayKeizer",
        "ShermanRinzelKeizer",
        "ButeraRespiratory",
        "EPropALIF",
        "SuperSpike",
        "LearnableNeuron",
        "Pernarowski",
        "QuadraticIF",
        "Theta",
        "PerfectIntegrator",
        "GatedLIF",
        "NonlinearLIF",
        "SFA",
        "MAT",
        "KLIF",
        "InhibitoryLIF",
        "ComplementaryLIF",
        "ParametricLIF",
        "NonResettingLIF",
        "SCNonResettingAdaptiveLIF",
        "AdaptiveThresholdIF",
        "SigmaDelta",
        "SCSigmaDeltaAccumulator",
        "EnergyLIF",
        "SCNormalizedEnergyLIF",
        "ClosedFormContinuous",
        "ChialvoMap",
        "RulkovMap",
        "IbarzTanakaMap",
        "MedvedevMap",
        "CazellesMap",
        "CourageNekorkinMap",
        "AiharaMap",
        "SCChaoticMap",
        "KilincBhattMap",
        "ErmentroutKopellMap",
        "BrainScaleSAdEx",
        "SpiNNakerLIF",
        "NeuroGrid",
        "DPI",
        "MarderSTG",
        "RallCable",
        "BoothRinzel",
        "Dendrify",
        "LiquidTimeConstant",
        "ParallelSpiking",
        "FractionalLIF",
        "StochasticIF",
        "GalvesLocherbach",
        "SpikeResponse",
        "GLM",
        "ArcaneNeuron",
        // advanced
        "MultiTimescale",
        "AttentionGated",
        "PredictiveCoding",
        "SelfReferential",
        "CompositionalBinding",
        "DifferentiableSurrogate",
        "ContinuousAttractor",
        "MetaPlastic",
        "BendaHerz",
        // point-process
        "Poisson",
        "InhomogeneousPoisson",
        "GammaRenewal",
        "EscapeRate",
        "BrunelWangNeuron",
        // interneurons
        "PVFastSpiking",
        "SST",
        "VIP",
        "Chandelier",
        "CerebellarBasket",
        "Martinotti",
        // motor
        "AlphaMotor",
        "GammaMotor",
        "UpperMotor",
        "Renshaw",
        "MotorUnit",
        // sensory (spiking)
        "RetinalGanglion",
        "Merkel",
        "Pacinian",
        "Nociceptor",
        "OlfactoryReceptor",
        // cerebellar
        "GranuleCell",
        "GolgiCell",
        "StellateCell",
        "LugaroCell",
        "UnipolarBrushCell",
        "DCNNeuron",
        // channels
        "PersistentNa",
        "Ih",
        "TTypeCa",
        "ATypeK",
        "BK",
        "SK",
        "NMDA",
        // population
        "MontbrioMeanField",
        "BrunelNetwork",
        "TUMNetwork",
        "ElBoustaniNetwork",
        // misc
        "GradedSynapseNeuron",
        "GapJunctionNeuron",
        "FrankenhaeUserHuxleyAxon",
        "NodeOfRanvier",
        "MyelinatedAxon",
        "CardiacPurkinjeFibre",
        "SmoothMuscleCell",
        "EndocrineBetaCell",
        // wrapped multi-input spiking
        "AlphaNeuron",
        "COBALIFNeuron",
        "CompteWMNeuron",
        "TsodyksMarkramNeuron",
        "PinskyRinzelNeuron",
        "HayL5PyramidalNeuron",
        "TwoCompartmentLIFNeuron",
        // wrapped hardware integer-input
        "LoihiCUBANeuron",
        "Loihi2Neuron",
        "SpiNNaker2Neuron",
        "TrueNorthNeuron",
        "IntegerQIFNeuron",
        "McCullochPittsNeuron",
        // wrapped graded/rate output
        "SigmoidRateNeuron",
        "ThresholdLinearRateNeuron",
        "AstrocyteModel",
        "InnerHairCell",
        "OuterHairCell",
        "RodPhotoreceptor",
        "ConePhotoreceptor",
        "TasteReceptorCell",
    ]
}
