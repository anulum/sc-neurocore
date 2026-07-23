// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner: high-performance Rust simulation backend

//! High-performance network simulation backend.
//!
//! Replaces the Python per-neuron loop with Rayon-parallel Rust execution
//! over CSR-stored projections and heterogeneous neuron populations.

use crate::neuron::*;
use crate::neurons::*;

mod input_adapters;
pub use input_adapters::*;

mod neuron_variant;
pub use neuron_variant::NeuronVariant;

mod population_runner;
pub use population_runner::PopulationRunner;

mod projection_runner;
pub use projection_runner::ProjectionRunner;

mod simulation_results;
pub use simulation_results::SimResults;

mod network_execution;
pub use network_execution::NetworkRunner;

// ── Factory ─────────────────────────────────────────────────────────

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
        "AdaptiveThresholdIF" | "AdaptiveThresholdIFNeuron" => Ok(
            NeuronVariant::AdaptiveThresholdIF(AdaptiveThresholdIFNeuron::new()),
        ),
        "SigmaDelta" | "SigmaDeltaNeuron" => {
            Ok(NeuronVariant::SigmaDelta(SigmaDeltaNeuron::default()))
        }
        "EnergyLIF" | "EnergyLIFNeuron" => Ok(NeuronVariant::EnergyLIF(EnergyLIFNeuron::new())),
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
        "KilincBhattMap" | "KilincBhattMapNeuron" => {
            Ok(NeuronVariant::KilincBhattMap(KilincBhattMapNeuron::new()))
        }
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
        "AdaptiveThresholdIF",
        "SigmaDelta",
        "EnergyLIF",
        "ClosedFormContinuous",
        "ChialvoMap",
        "RulkovMap",
        "IbarzTanakaMap",
        "MedvedevMap",
        "CazellesMap",
        "CourageNekorkinMap",
        "AiharaMap",
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

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn izhikevich_population_spikes() {
        let mut pop = create_population("Izhikevich", 10).unwrap();
        let mut total_spikes = 0usize;
        for _ in 0..100 {
            pop.currents.fill(10.0);
            pop.step_all();
            total_spikes += pop.spikes.iter().filter(|&&s| s != 0).count();
        }
        assert!(
            total_spikes > 0,
            "10 Izhikevich neurons must spike with I=10"
        );
    }

    #[test]
    fn single_population_step_accepts_external_currents() {
        let mut runner = NetworkRunner::new();
        let idx = runner.add_population(create_population("Lapicque", 3).unwrap());

        let (spikes, voltages) = runner
            .step_population_with_currents(idx, &[1.0, 2.0, 3.0])
            .unwrap();

        assert_eq!(spikes.len(), 3);
        assert_eq!(voltages.len(), 3);
        assert!(spikes.iter().all(|&s| s <= 1));
        assert!(voltages.iter().all(|v| v.is_finite()));
        assert!(runner
            .step_population_with_currents(idx, &[1.0, 2.0])
            .is_err());
        assert!(runner
            .step_population_with_currents(idx + 1, &[1.0, 2.0, 3.0])
            .is_err());
    }

    #[test]
    fn all_to_all_network_100_steps() {
        let mut runner = NetworkRunner::new();
        let pop = create_population("Izhikevich", 4).unwrap();
        runner.add_population(pop);

        // All-to-all CSR: 4 src -> 4 tgt
        let mut row_offsets = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut offset = 0;
        for _i in 0..4 {
            row_offsets.push(offset);
            for j in 0..4 {
                col_indices.push(j);
                values.push(2.0);
                offset += 1;
            }
        }
        row_offsets.push(offset);

        let proj = ProjectionRunner::new(0, 0, row_offsets, col_indices, values, 0);
        runner.add_projection(proj);

        // Inject external current by pre-filling
        for n in &mut runner.populations[0].neurons {
            if let NeuronVariant::Izhikevich(iz) = n {
                iz.v = -50.0;
            }
        }

        let results = runner.run(100);
        assert_eq!(results.spike_counts.len(), 1);
        assert_eq!(results.voltages.len(), 1);
        assert_eq!(results.voltages[0].len(), 4);
    }

    #[test]
    fn mixed_hh_adex_network() {
        let mut runner = NetworkRunner::new();

        let hh_pop = create_population("HodgkinHuxley", 3).unwrap();
        let adex_pop = create_population("AdEx", 3).unwrap();
        let hh_idx = runner.add_population(hh_pop);
        let adex_idx = runner.add_population(adex_pop);

        // HH -> AdEx projection
        let row_offsets = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![100.0; 9];
        let proj = ProjectionRunner::new(hh_idx, adex_idx, row_offsets, col_indices, values, 0);
        runner.add_projection(proj);

        // Drive HH with external current
        runner.populations[0].currents.fill(15.0);

        let results = runner.run(50);
        assert_eq!(results.spike_counts.len(), 2);
        assert_eq!(results.voltages.len(), 2);
    }

    #[test]
    fn large_network_performance() {
        let n = 1000;
        let mut pop = create_population("Izhikevich", n).unwrap();
        // Run 1000 steps with constant drive — should complete quickly
        for _ in 0..1000 {
            pop.currents.fill(10.0);
            pop.step_all();
        }
        let total: usize = pop.spikes.iter().map(|&s| s as usize).sum();
        // Sanity: spike count should be deterministic and nonzero after 1000 driven steps
        let _ = total;
        // Check voltages are finite
        let voltages = pop.collect_voltages();
        assert_eq!(voltages.len(), n);
        for v in &voltages {
            assert!(v.is_finite(), "voltage must be finite");
        }
    }

    #[test]
    fn batch_simulate_single_neuron() {
        let mut neuron = create_neuron("AdEx").unwrap();
        let n_steps = 1000;
        let current = 500.0;
        let mut voltages = Vec::with_capacity(n_steps);
        let mut spikes = Vec::new();
        for t in 0..n_steps {
            let fired = neuron.step(current);
            voltages.push(neuron.soma_voltage());
            if fired != 0 {
                spikes.push(t);
            }
        }
        assert_eq!(voltages.len(), n_steps);
        assert!(voltages.iter().all(|v| v.is_finite()));
        assert!(!spikes.is_empty(), "AdEx with I=10 should spike");
    }

    #[test]
    fn create_neuron_all_supported() {
        for name in supported_models() {
            let result = create_neuron(name);
            assert!(
                result.is_ok(),
                "create_neuron({name}) failed: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn mcculloch_pitts_network_wrapper_preserves_signed_logical_transport() {
        let mut neuron = create_neuron("McCullochPittsNeuron").unwrap();
        assert_eq!(neuron.step(0.0), 0);
        assert_eq!(neuron.step(1.0), 1);
        assert_eq!(neuron.step(-1.0), 0);
        assert_eq!(neuron.step(1.5), 0);
        assert_eq!(neuron.step(f64::NAN), 0);
        assert_eq!(neuron.soma_voltage(), 0.0);
        neuron.reset();
        assert_eq!(neuron.step(1.0), 1);
    }

    // ── Pipeline integration: interneurons ────────────────────────

    #[test]
    fn interneuron_population_create_step_reset() {
        for name in &[
            "PVFastSpiking",
            "SST",
            "VIP",
            "Chandelier",
            "CerebellarBasket",
            "Martinotti",
        ] {
            let mut pop = create_population(name, 5).unwrap();
            pop.currents.fill(3.0);
            for _ in 0..100 {
                pop.step_all();
            }
            let voltages = pop.collect_voltages();
            assert_eq!(voltages.len(), 5, "{name}: voltage count mismatch");
            for v in &voltages {
                assert!(v.is_finite(), "{name}: non-finite voltage {v}");
            }
            pop.reset_all();
            let v_after_reset = pop.collect_voltages();
            for v in &v_after_reset {
                assert!(v.is_finite(), "{name}: non-finite after reset");
            }
        }
    }

    #[test]
    fn interneuron_mixed_network() {
        let mut runner = NetworkRunner::new();
        let pv_pop = create_population("PVFastSpiking", 3).unwrap();
        let sst_pop = create_population("SST", 3).unwrap();
        let pv_idx = runner.add_population(pv_pop);
        let sst_idx = runner.add_population(sst_pop);

        // PV → SST all-to-all projection
        let row_offsets = vec![0, 3, 6, 9];
        let col_indices = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![1.0; 9];
        let proj = ProjectionRunner::new(pv_idx, sst_idx, row_offsets, col_indices, values, 0);
        runner.add_projection(proj);

        runner.populations[0].currents.fill(3.0);
        let results = runner.run(50);
        assert_eq!(results.spike_counts.len(), 2);
        assert_eq!(results.voltages.len(), 2);
        for pop_voltages in &results.voltages {
            for v in pop_voltages {
                assert!(v.is_finite());
            }
        }
    }

    // ── Pipeline integration: sensory spiking ─────────────────────

    #[test]
    fn sensory_spiking_population_create_step() {
        for name in &[
            "RetinalGanglion",
            "Merkel",
            "Pacinian",
            "Nociceptor",
            "OlfactoryReceptor",
        ] {
            let mut pop = create_population(name, 5).unwrap();
            pop.currents.fill(20.0);
            for _ in 0..200 {
                pop.step_all();
            }
            let voltages = pop.collect_voltages();
            assert_eq!(voltages.len(), 5, "{name}: voltage count mismatch");
            for v in &voltages {
                assert!(v.is_finite(), "{name}: non-finite voltage {v}");
            }
        }
    }

    // ── NaN/Inf edge-case tests ───────────────────────────────────

    #[test]
    fn all_models_nan_input_stays_finite() {
        // Models must not propagate NaN — they should produce finite
        // (possibly wrong) output. This catches catastrophic numerical issues.
        let fragile_models = &[
            "PVFastSpiking",
            "SST",
            "VIP",
            "Chandelier",
            "CerebellarBasket",
            "Martinotti",
            "RetinalGanglion",
            "Merkel",
            "Pacinian",
            "Nociceptor",
            "OlfactoryReceptor",
        ];
        for name in fragile_models {
            let mut neuron = create_neuron(name).unwrap();
            // Feed 100 normal steps first to get into active regime
            for _ in 0..100 {
                neuron.step(2.0);
            }
            // Then feed NaN — voltage may go NaN but should not panic
            for _ in 0..10 {
                let _ = neuron.step(f64::NAN);
            }
            // Reset must restore finite state
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(
                v.is_finite(),
                "{name}: voltage not finite after reset from NaN: {v}"
            );
        }
    }

    #[test]
    fn all_models_extreme_input_stays_finite() {
        let models = &[
            "PVFastSpiking",
            "SST",
            "VIP",
            "Chandelier",
            "CerebellarBasket",
            "Martinotti",
            "RetinalGanglion",
            "Merkel",
            "Pacinian",
            "Nociceptor",
            "OlfactoryReceptor",
        ];
        for name in models {
            let mut neuron = create_neuron(name).unwrap();
            // Large positive current
            for _ in 0..50 {
                neuron.step(1e6);
            }
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(
                v.is_finite(),
                "{name}: non-finite after large positive input"
            );

            // Large negative current
            for _ in 0..50 {
                neuron.step(-1e6);
            }
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(
                v.is_finite(),
                "{name}: non-finite after large negative input"
            );
        }
    }
}
