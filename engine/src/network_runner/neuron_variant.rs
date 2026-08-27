// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner neuron dispatch

//! Runtime dispatch over every neuron model supported by the network runner.

use super::input_adapters::*;
use crate::neuron::*;
use crate::neurons::*;

/// Enum dispatch across all neuron models.
///
/// Models with non-standard signatures are wrapped via `Wr*` types
/// to normalise to `step(f64) -> i32`.
#[allow(clippy::large_enum_variant)]
pub enum NeuronVariant {
    // neuron.rs
    Izhikevich(Izhikevich),
    AdEx(AdExNeuron),
    ExpIF(ExpIfNeuron),
    Lapicque(LapicqueNeuron),
    HomeostaticLif(HomeostaticLif),

    // biophysical/ model modules
    HodgkinHuxley(HodgkinHuxleyNeuron),
    TraubMiles(TraubMilesNeuron),
    WangBuzsaki(WangBuzsakiNeuron),
    ConnorStevens(ConnorStevensNeuron),
    DestexheThalamic(DestexheThalamicNeuron),
    HuberBraun(HuberBraunNeuron),
    GolombFS(GolombFSNeuron),
    Pospischil(PospischilNeuron),
    MainenSejnowski(MainenSejnowskiNeuron),
    DeSchutterPurkinje(DeSchutterPurkinjeNeuron),
    PlantR15(PlantR15Neuron),
    Prescott(PrescottNeuron),
    MihalasNiebur(MihalasNieburNeuron),
    GLIF(GLIFNeuron),
    GIFPopulation(GIFPopulationNeuron),
    AvRonCardiac(AvRonCardiacNeuron),
    DurstewitzDopamine(DurstewitzDopamineNeuron),
    HillTononi(HillTononiNeuron),
    BertramPhantom(BertramPhantomBurster),
    SCThreeStatePhantom(SCThreeStatePhantomBurster),
    Yamada(YamadaNeuron),

    // simple_spiking.rs
    FitzHughNagumo(FitzHughNagumoNeuron),
    MorrisLecar(MorrisLecarNeuron),
    HindmarshRose(HindmarshRoseNeuron),
    ResonateAndFire(ResonateAndFireNeuron),
    BalancedResonateAndFire(BalancedResonateAndFireNeuron),
    FitzHughRinzel(FitzHughRinzelNeuron),
    McKean(McKeanNeuron),
    SCTriangularMcKean(SCTriangularMcKeanNeuron),
    TermanWang(TermanWangOscillator),
    GutkinErmentrout(GutkinErmentroutNeuron),
    WilsonHR(WilsonHRNeuron),
    Chay(ChayNeuron),
    ChayKeizer(ChayKeizerNeuron),
    ShermanRinzelKeizer(ShermanRinzelKeizerNeuron),
    ButeraRespiratory(ButeraRespiratoryNeuron),
    EPropALIF(EPropALIFNeuron),
    SuperSpike(SuperSpikeNeuron),
    LearnableNeuron(LearnableNeuronModel),
    Pernarowski(PernarowskiNeuron),

    // trivial.rs (simple IF variants)
    QuadraticIF(QuadraticIFNeuron),
    Theta(ThetaNeuron),
    PerfectIntegrator(PerfectIntegratorNeuron),
    GatedLIF(GatedLIFNeuron),
    NonlinearLIF(NonlinearLIFNeuron),
    SFA(SFANeuron),
    MAT(MATNeuron),
    SCResettingMAT(SCResettingMATNeuron),
    KLIF(KLIFNeuron),
    InhibitoryLIF(InhibitoryLIFNeuron),
    ComplementaryLIF(ComplementaryLIFNeuron),
    ParametricLIF(ParametricLIFNeuron),
    NonResettingLIF(NonResettingLIFNeuron),
    SCNonResettingAdaptiveLIF(SCNonResettingAdaptiveLIFNeuron),
    AdaptiveThresholdIF(AdaptiveThresholdIFNeuron),
    SigmaDelta(SigmaDeltaNeuron),
    SCSigmaDeltaAccumulator(SCSigmaDeltaAccumulatorNeuron),
    EnergyLIF(EnergyLIFNeuron),
    SCNormalizedEnergyLIF(SCNormalizedEnergyLIFNeuron),
    ClosedFormContinuous(ClosedFormContinuousNeuron),

    // maps.rs
    ChialvoMap(ChialvoMapNeuron),
    RulkovMap(RulkovMapNeuron),
    IbarzTanakaMap(IbarzTanakaMapNeuron),
    MedvedevMap(MedvedevMapNeuron),
    CazellesMap(CazellesMapNeuron),
    CourageNekorkinMap(CourageNekorkinMapNeuron),
    AiharaMap(AiharaMapNeuron),
    SCChaoticMap(SCChaoticMapNeuron),
    NagumoSatoMap(NagumoSatoMapNeuron),
    SCAdaptiveThresholdMap(SCAdaptiveThresholdMapNeuron),
    ErmentroutKopellMap(ErmentroutKopellMapNeuron),

    // hardware.rs (f64 input subset)
    BrainScaleSAdEx(BrainScaleSAdExNeuron),
    SpiNNakerLIF(SpiNNakerLIFNeuron),
    NeuroGrid(NeuroGridNeuron),
    DPI(DPINeuron),
    Akida(AkidaNeuron),
    StochasticLIF(StochasticLIFNeuron),

    // multi_compartment.rs (single-f64-input subset)
    MarderSTG(MarderSTGNeuron),
    RallCable(RallCableNeuron),
    BoothRinzel(BoothRinzelNeuron),
    Dendrify(DendrifyNeuron),

    // rate.rs (spiking subset with step(f64)->i32)
    LiquidTimeConstant(LiquidTimeConstantNeuron),
    ParallelSpiking(ParallelSpikingNeuron),
    FractionalLIF(FractionalLIFNeuron),

    // special.rs (step(f64)->i32 subset)
    StochasticIF(StochasticIFNeuron),
    GalvesLocherbach(GalvesLocherbachNeuron),
    SpikeResponse(SpikeResponseNeuron),
    GLM(GLMNeuron),
    Arcane(ArcaneNeuron),

    // --- newly wired (step(f64)->i32 compatible) ---
    // ai_optimized.rs
    MultiTimescale(MultiTimescaleNeuron),
    AttentionGated(AttentionGatedNeuron),
    PredictiveCoding(PredictiveCodingNeuron),
    SelfReferential(SelfReferentialNeuron),
    CompositionalBinding(CompositionalBindingNeuron),
    DifferentiableSurrogate(DifferentiableSurrogateNeuron),
    ContinuousAttractor(ContinuousAttractorNeuron),
    MetaPlastic(MetaPlasticNeuron),
    // simple_spiking.rs
    BendaHerz(BendaHerzNeuron),
    SCStochasticRateAdaptation(SCStochasticRateAdaptationNeuron),
    BrunelWang(BrunelWangNeuron),
    // special.rs
    Poisson(PoissonNeuron),
    InhomogeneousPoisson(InhomogeneousPoissonNeuron),
    GammaRenewal(GammaRenewalNeuron),
    EscapeRate(EscapeRateNeuron),
    // interneurons.rs (step(f64)->i32)
    PVFastSpiking(PVFastSpikingNeuron),
    SST(SSTNeuron),
    VIP(VIPNeuron),
    Chandelier(ChandelierNeuron),
    CerebellarBasket(CerebellarBasketNeuron),
    Martinotti(MartinottiNeuron),

    // motor.rs (step(f64)->i32)
    AlphaMotor(AlphaMotorNeuron),
    GammaMotor(GammaMotorNeuron),
    UpperMotor(UpperMotorNeuron),
    Renshaw(RenshawCell),
    MotorUnitCell(MotorUnit),

    // sensory.rs (spiking subset: step(f64)->i32)
    RetinalGanglion(RetinalGanglionCell),
    Merkel(MerkelCell),
    Pacinian(PacinianCorpuscle),
    NociceptorCell(Nociceptor),
    OlfactoryReceptor(OlfactoryReceptorNeuron),

    // cerebellar.rs (step(f64)->i32)
    Granule(GranuleCell),
    Golgi(GolgiCell),
    Stellate(StellateCell),
    Lugaro(LugaroCell),
    UnipolarBrush(UnipolarBrushCell),
    DCN(DCNNeuron),

    // channels.rs (step(f64)->i32)
    PersistentNa(PersistentNaNeuron),
    Ih(IhNeuron),
    TTypeCa(TTypeCaNeuron),
    ATypeK(ATypeKNeuron),
    BK(BKNeuron),
    SK(SKNeuron),
    NMDA(NMDANeuron),

    // population.rs (step(f64)->i32)
    MontbrioMPR(MontbrioMeanField),
    Brunel(BrunelNetwork),
    TUM(TUMNetwork),
    ElBoustani(ElBoustaniNetwork),

    // misc.rs (step(f64)->i32)
    GradedSynapse(GradedSynapseNeuron),
    GapJunction(GapJunctionNeuron),
    FHAxon(FrankenhaeUserHuxleyAxon),
    NodeOfRanvier(NodeOfRanvier),
    MyelinAxon(MyelinatedAxon),
    CardiacPurkinje(CardiacPurkinjeFibre),
    SmoothMuscle(SmoothMuscleCell),
    BetaCell(EndocrineBetaCell),

    // Wrapped models: non-standard interfaces normalised to step(f64)->i32
    // Multi-input spiking
    WrAlphaCell(WrAlpha),
    WrCOBALIFCell(WrCOBALIF),
    WrCompteWMCell(WrCompteWM),
    WrTsodyksMarkramCell(WrTsodyksMarkram),
    WrPinskyRinzelCell(WrPinskyRinzel),
    WrHayL5Cell(WrHayL5),
    WrTwoCompLIFCell(WrTwoCompLIF),
    // Hardware integer-input
    WrLoihiCUBACell(WrLoihiCUBA),
    WrLoihi2Cell(WrLoihi2),
    WrSpiNNaker2Cell(WrSpiNNaker2),
    WrTrueNorthCell(WrTrueNorth),
    WrIntegerQIFCell(WrIntegerQIF),
    WrMcCullochPittsCell(WrMcCullochPitts),
    // Graded/rate output
    WrSigmoidRateCell(WrSigmoidRate),
    WrThresholdLinearCell(WrThresholdLinear),
    WrAstrocyteCell(WrAstrocyte),
    WrInnerHairCellCell(WrInnerHairCell),
    WrOuterHairCellCell(WrOuterHairCell),
    WrRodPhotoreceptorCell(WrRodPhotoreceptor),
    WrConePhotoreceptorCell(WrConePhotoreceptor),
    WrTasteReceptorCell(WrTasteReceptor),
}

macro_rules! dispatch_step {
    ($self:expr, $current:expr,
     $($variant:ident),* $(,)?) => {
        match $self {
            $(NeuronVariant::$variant(n) => n.step($current),)*
        }
    };
}

macro_rules! dispatch_reset {
    ($self:expr,
     $($variant:ident),* $(,)?) => {
        match $self {
            $(NeuronVariant::$variant(n) => n.reset(),)*
        }
    };
}

/// All variant names in one place for the dispatch macros.
macro_rules! all_variants {
    ($mac:ident, $($args:tt)*) => {
        $mac!($($args)*
            Izhikevich, AdEx, ExpIF, Lapicque, HomeostaticLif,
            HodgkinHuxley, TraubMiles, WangBuzsaki, ConnorStevens,
            DestexheThalamic, HuberBraun, GolombFS,
            Pospischil, MainenSejnowski, DeSchutterPurkinje,
            PlantR15, Prescott, MihalasNiebur, GLIF, GIFPopulation,
            AvRonCardiac, DurstewitzDopamine, HillTononi, BertramPhantom, SCThreeStatePhantom, Yamada, Akida, StochasticLIF,
            FitzHughNagumo, MorrisLecar, HindmarshRose, ResonateAndFire, BalancedResonateAndFire,
            FitzHughRinzel, McKean, SCTriangularMcKean, TermanWang, GutkinErmentrout, WilsonHR,
            Chay, ChayKeizer, ShermanRinzelKeizer, ButeraRespiratory,
            EPropALIF, SuperSpike, LearnableNeuron, Pernarowski,
            QuadraticIF, Theta, PerfectIntegrator, GatedLIF, NonlinearLIF,
            SFA, MAT, SCResettingMAT, KLIF, InhibitoryLIF, ComplementaryLIF, ParametricLIF,
            NonResettingLIF, SCNonResettingAdaptiveLIF, AdaptiveThresholdIF, SigmaDelta, SCSigmaDeltaAccumulator, EnergyLIF, SCNormalizedEnergyLIF,
            ClosedFormContinuous,
            ChialvoMap, RulkovMap, IbarzTanakaMap, MedvedevMap,
            CazellesMap, CourageNekorkinMap, AiharaMap, SCChaoticMap,
            NagumoSatoMap, SCAdaptiveThresholdMap, ErmentroutKopellMap,
            BrainScaleSAdEx, SpiNNakerLIF, NeuroGrid, DPI,
            MarderSTG, RallCable, BoothRinzel, Dendrify,
            LiquidTimeConstant, ParallelSpiking, FractionalLIF,
            StochasticIF, GalvesLocherbach, SpikeResponse, GLM,
            Arcane,
            MultiTimescale, AttentionGated, PredictiveCoding,
            SelfReferential, CompositionalBinding, DifferentiableSurrogate,
            ContinuousAttractor, MetaPlastic,
            BendaHerz, SCStochasticRateAdaptation, BrunelWang,
            Poisson, InhomogeneousPoisson, GammaRenewal, EscapeRate,
            PVFastSpiking, SST, VIP, Chandelier, CerebellarBasket, Martinotti,
            AlphaMotor, GammaMotor, UpperMotor, Renshaw, MotorUnitCell,
            RetinalGanglion, Merkel, Pacinian, NociceptorCell, OlfactoryReceptor,
            Granule, Golgi, Stellate, Lugaro, UnipolarBrush, DCN,
            PersistentNa, Ih, TTypeCa, ATypeK, BK, SK, NMDA,
            MontbrioMPR, Brunel, TUM, ElBoustani,
            GradedSynapse, GapJunction, FHAxon, NodeOfRanvier, MyelinAxon, CardiacPurkinje,
            SmoothMuscle, BetaCell,
            WrAlphaCell, WrCOBALIFCell, WrCompteWMCell, WrTsodyksMarkramCell,
            WrPinskyRinzelCell, WrHayL5Cell, WrTwoCompLIFCell,
            WrLoihiCUBACell, WrLoihi2Cell, WrSpiNNaker2Cell, WrTrueNorthCell, WrIntegerQIFCell,
            WrMcCullochPittsCell,
            WrSigmoidRateCell, WrThresholdLinearCell, WrAstrocyteCell,
            WrInnerHairCellCell, WrOuterHairCellCell,
            WrRodPhotoreceptorCell, WrConePhotoreceptorCell, WrTasteReceptorCell,
        )
    };
}

impl NeuronVariant {
    pub fn step(&mut self, current: f64) -> i32 {
        all_variants!(dispatch_step, self, current,)
    }

    pub fn reset(&mut self) {
        all_variants!(dispatch_reset, self,)
    }

    pub fn soma_voltage(&self) -> f64 {
        match self {
            NeuronVariant::Izhikevich(n) => n.v,
            NeuronVariant::AdEx(n) => n.v,
            NeuronVariant::ExpIF(n) => n.v,
            NeuronVariant::Lapicque(n) => n.v,
            NeuronVariant::HomeostaticLif(n) => n.v,
            NeuronVariant::HodgkinHuxley(n) => n.v,
            NeuronVariant::TraubMiles(n) => n.v,
            NeuronVariant::WangBuzsaki(n) => n.v,
            NeuronVariant::ConnorStevens(n) => n.v,
            NeuronVariant::DestexheThalamic(n) => n.v,
            NeuronVariant::HuberBraun(n) => n.v,
            NeuronVariant::GolombFS(n) => n.v,
            NeuronVariant::Pospischil(n) => n.v,
            NeuronVariant::MainenSejnowski(n) => n.vs,
            NeuronVariant::DeSchutterPurkinje(n) => n.v,
            NeuronVariant::PlantR15(n) => n.v,
            NeuronVariant::Prescott(n) => n.v,
            NeuronVariant::MihalasNiebur(n) => n.v,
            NeuronVariant::GLIF(n) => n.v,
            NeuronVariant::GIFPopulation(n) => n.v,
            NeuronVariant::AvRonCardiac(n) => n.v,
            NeuronVariant::DurstewitzDopamine(n) => n.v,
            NeuronVariant::HillTononi(n) => n.v,
            NeuronVariant::BertramPhantom(n) => n.v,
            NeuronVariant::SCThreeStatePhantom(n) => n.v,
            NeuronVariant::Yamada(n) => n.v,
            NeuronVariant::FitzHughNagumo(n) => n.v,
            NeuronVariant::MorrisLecar(n) => n.v,
            NeuronVariant::HindmarshRose(n) => n.x,
            NeuronVariant::ResonateAndFire(n) => n.y,
            NeuronVariant::BalancedResonateAndFire(n) => n.x,
            NeuronVariant::FitzHughRinzel(n) => n.v,
            NeuronVariant::McKean(n) => n.v,
            NeuronVariant::SCTriangularMcKean(n) => n.v,
            NeuronVariant::TermanWang(n) => n.v,
            NeuronVariant::GutkinErmentrout(n) => n.v,
            NeuronVariant::WilsonHR(n) => n.v,
            NeuronVariant::Chay(n) => n.v,
            NeuronVariant::ChayKeizer(n) => n.v,
            NeuronVariant::ShermanRinzelKeizer(n) => n.v,
            NeuronVariant::ButeraRespiratory(n) => n.v,
            NeuronVariant::EPropALIF(n) => n.v,
            NeuronVariant::SuperSpike(n) => n.v,
            NeuronVariant::LearnableNeuron(n) => n.v,
            NeuronVariant::Pernarowski(n) => n.v,
            NeuronVariant::QuadraticIF(n) => n.v,
            NeuronVariant::Theta(n) => n.theta,
            NeuronVariant::PerfectIntegrator(n) => n.v,
            NeuronVariant::GatedLIF(n) => n.v,
            NeuronVariant::NonlinearLIF(n) => n.v,
            NeuronVariant::SFA(n) => n.v,
            NeuronVariant::MAT(n) => n.v,
            NeuronVariant::SCResettingMAT(n) => n.v,
            NeuronVariant::KLIF(n) => n.v,
            NeuronVariant::InhibitoryLIF(n) => n.v,
            NeuronVariant::ComplementaryLIF(n) => n.v_pos,
            NeuronVariant::ParametricLIF(n) => n.v,
            NeuronVariant::NonResettingLIF(n) => n.v,
            NeuronVariant::SCNonResettingAdaptiveLIF(n) => n.v,
            NeuronVariant::AdaptiveThresholdIF(n) => n.v,
            NeuronVariant::SigmaDelta(n) => n.sigma,
            NeuronVariant::SCSigmaDeltaAccumulator(n) => n.sigma,
            NeuronVariant::EnergyLIF(n) => n.v,
            NeuronVariant::SCNormalizedEnergyLIF(n) => n.v,
            NeuronVariant::ClosedFormContinuous(n) => n.x,
            NeuronVariant::ChialvoMap(n) => n.x,
            NeuronVariant::RulkovMap(n) => n.x,
            NeuronVariant::IbarzTanakaMap(n) => n.v,
            NeuronVariant::MedvedevMap(n) => n.u,
            NeuronVariant::CazellesMap(n) => n.x,
            NeuronVariant::CourageNekorkinMap(n) => n.x,
            NeuronVariant::AiharaMap(n) => n.output(),
            NeuronVariant::SCChaoticMap(n) => n.x,
            NeuronVariant::NagumoSatoMap(n) => f64::from(n.output()),
            NeuronVariant::SCAdaptiveThresholdMap(n) => n.x,
            NeuronVariant::ErmentroutKopellMap(n) => n.theta,
            NeuronVariant::BrainScaleSAdEx(n) => n.v,
            NeuronVariant::SpiNNakerLIF(n) => n.v,
            NeuronVariant::NeuroGrid(n) => n.v_s,
            NeuronVariant::DPI(n) => n.i_mem,
            NeuronVariant::MarderSTG(n) => n.v,
            NeuronVariant::RallCable(n) => n.v.first().copied().unwrap_or(0.0),
            NeuronVariant::BoothRinzel(n) => n.vs,
            NeuronVariant::Dendrify(n) => n.v_s,
            NeuronVariant::LiquidTimeConstant(n) => n.x,
            NeuronVariant::ParallelSpiking(n) => n.hidden,
            NeuronVariant::FractionalLIF(n) => n.v,
            NeuronVariant::StochasticIF(n) => n.v,
            NeuronVariant::GalvesLocherbach(n) => n.v,
            NeuronVariant::SpikeResponse(n) => n.v,
            NeuronVariant::GLM(n) => n.mu,
            NeuronVariant::Arcane(n) => n.v_fast,
            // newly wired — default voltage field
            NeuronVariant::MultiTimescale(n) => n.v_fast,
            NeuronVariant::AttentionGated(n) => n.v,
            NeuronVariant::PredictiveCoding(n) => n.v,
            NeuronVariant::SelfReferential(n) => n.v,
            NeuronVariant::CompositionalBinding(_) => 0.0,
            NeuronVariant::DifferentiableSurrogate(n) => n.v,
            NeuronVariant::ContinuousAttractor(_) => 0.0,
            NeuronVariant::MetaPlastic(n) => n.v,
            NeuronVariant::BendaHerz(n) => n.a,
            NeuronVariant::SCStochasticRateAdaptation(n) => n.a,
            NeuronVariant::BrunelWang(n) => n.v,
            NeuronVariant::Poisson(_) => 0.0,
            NeuronVariant::InhomogeneousPoisson(_) => 0.0,
            NeuronVariant::GammaRenewal(_) => 0.0,
            NeuronVariant::EscapeRate(n) => n.v,
            NeuronVariant::Akida(n) => n.v as f64,
            NeuronVariant::StochasticLIF(n) => n.v,
            // interneurons
            NeuronVariant::PVFastSpiking(n) => n.v,
            NeuronVariant::SST(n) => n.v,
            NeuronVariant::VIP(n) => n.v,
            NeuronVariant::Chandelier(n) => n.v,
            NeuronVariant::CerebellarBasket(n) => n.v,
            NeuronVariant::Martinotti(n) => n.v,
            // motor
            NeuronVariant::AlphaMotor(n) => n.v,
            NeuronVariant::GammaMotor(n) => n.v,
            NeuronVariant::UpperMotor(n) => n.v,
            NeuronVariant::Renshaw(n) => n.v,
            NeuronVariant::MotorUnitCell(n) => n.v,
            // sensory (spiking)
            NeuronVariant::RetinalGanglion(n) => n.baseline, // GLM: no membrane V
            NeuronVariant::Merkel(n) => n.v,
            NeuronVariant::Pacinian(n) => n.v,
            NeuronVariant::NociceptorCell(n) => n.v,
            NeuronVariant::OlfactoryReceptor(n) => n.v,
            // cerebellar
            NeuronVariant::Granule(n) => n.v,
            NeuronVariant::Golgi(n) => n.v,
            NeuronVariant::Stellate(n) => n.v,
            NeuronVariant::Lugaro(n) => n.v,
            NeuronVariant::UnipolarBrush(n) => n.v,
            NeuronVariant::DCN(n) => n.v,
            // channels
            NeuronVariant::PersistentNa(n) => n.v,
            NeuronVariant::Ih(n) => n.v,
            NeuronVariant::TTypeCa(n) => n.v,
            NeuronVariant::ATypeK(n) => n.v,
            NeuronVariant::BK(n) => n.v,
            NeuronVariant::SK(n) => n.v,
            NeuronVariant::NMDA(n) => n.v,
            // population
            NeuronVariant::MontbrioMPR(n) => n.v,
            NeuronVariant::Brunel(n) => n.r_e,
            NeuronVariant::TUM(n) => n.r,
            NeuronVariant::ElBoustani(n) => n.r_e,
            // misc
            NeuronVariant::GradedSynapse(n) => n.v,
            NeuronVariant::GapJunction(n) => n.v,
            NeuronVariant::FHAxon(n) => n.v,
            NeuronVariant::NodeOfRanvier(n) => n.v,
            NeuronVariant::MyelinAxon(n) => n.v(),
            NeuronVariant::CardiacPurkinje(n) => n.v,
            NeuronVariant::SmoothMuscle(n) => n.v,
            NeuronVariant::BetaCell(n) => n.v,
            // Wrapped models — voltage via wrapper .v()
            NeuronVariant::WrAlphaCell(n) => n.v(),
            NeuronVariant::WrCOBALIFCell(n) => n.v(),
            NeuronVariant::WrCompteWMCell(n) => n.v(),
            NeuronVariant::WrTsodyksMarkramCell(n) => n.v(),
            NeuronVariant::WrPinskyRinzelCell(n) => n.v(),
            NeuronVariant::WrHayL5Cell(n) => n.v(),
            NeuronVariant::WrTwoCompLIFCell(n) => n.v(),
            NeuronVariant::WrLoihiCUBACell(n) => n.v(),
            NeuronVariant::WrLoihi2Cell(n) => n.v(),
            NeuronVariant::WrSpiNNaker2Cell(n) => n.v(),
            NeuronVariant::WrTrueNorthCell(n) => n.v(),
            NeuronVariant::WrIntegerQIFCell(n) => n.v(),
            NeuronVariant::WrMcCullochPittsCell(n) => n.v(),
            NeuronVariant::WrSigmoidRateCell(n) => n.v(),
            NeuronVariant::WrThresholdLinearCell(n) => n.v(),
            NeuronVariant::WrAstrocyteCell(n) => n.v(),
            NeuronVariant::WrInnerHairCellCell(n) => n.v(),
            NeuronVariant::WrOuterHairCellCell(n) => n.v(),
            NeuronVariant::WrRodPhotoreceptorCell(n) => n.v(),
            NeuronVariant::WrConePhotoreceptorCell(n) => n.v(),
            NeuronVariant::WrTasteReceptorCell(n) => n.v(),
        }
    }
}
