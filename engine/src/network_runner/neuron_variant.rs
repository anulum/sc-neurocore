// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner neuron dispatch

//! Runtime dispatch over every neuron model supported by the network runner.

#![deny(missing_docs)]

use super::input_adapters::*;
use crate::neuron::*;
use crate::neurons::*;
use crate::rk4_neurons::Izhikevich2007Rk4;

/// Enum dispatch across all neuron models.
///
/// Models with non-standard signatures are wrapped via `Wr*` types
/// to normalise to `step(f64) -> i32`.
#[allow(clippy::large_enum_variant)]
pub enum NeuronVariant {
    // neuron.rs
    /// The [`Izhikevich`] model.
    Izhikevich(Izhikevich),
    /// The [`Izhikevich2007Rk4`] model.
    Izhikevich2007(Izhikevich2007Rk4),
    /// The [`AdExNeuron`] model.
    AdEx(AdExNeuron),
    /// The [`ExpIfNeuron`] model.
    ExpIF(ExpIfNeuron),
    /// The [`LapicqueNeuron`] model.
    Lapicque(LapicqueNeuron),
    /// The [`HomeostaticLif`] model.
    HomeostaticLif(HomeostaticLif),

    // biophysical/ model modules
    /// The [`HodgkinHuxleyNeuron`] model.
    HodgkinHuxley(HodgkinHuxleyNeuron),
    /// The [`TraubMilesNeuron`] model.
    TraubMiles(TraubMilesNeuron),
    /// The [`WangBuzsakiNeuron`] model.
    WangBuzsaki(WangBuzsakiNeuron),
    /// The [`ConnorStevensNeuron`] model.
    ConnorStevens(ConnorStevensNeuron),
    /// The [`DestexheThalamicNeuron`] model.
    DestexheThalamic(DestexheThalamicNeuron),
    /// The [`HuberBraunNeuron`] model.
    HuberBraun(HuberBraunNeuron),
    /// The [`GolombFSNeuron`] model.
    GolombFS(GolombFSNeuron),
    /// The [`PospischilNeuron`] model.
    Pospischil(PospischilNeuron),
    /// The [`MainenSejnowskiNeuron`] model.
    MainenSejnowski(MainenSejnowskiNeuron),
    /// The [`DeSchutterPurkinjeNeuron`] model.
    DeSchutterPurkinje(DeSchutterPurkinjeNeuron),
    /// The [`PlantR15Neuron`] model.
    PlantR15(PlantR15Neuron),
    /// The [`PrescottNeuron`] model.
    Prescott(PrescottNeuron),
    /// The [`MihalasNieburNeuron`] model.
    MihalasNiebur(MihalasNieburNeuron),
    /// The [`SCScaledResetAdaptiveIFNeuron`] model.
    SCScaledResetAdaptiveIF(SCScaledResetAdaptiveIFNeuron),
    /// The [`GLIFNeuron`] model.
    GLIF(GLIFNeuron),
    /// The [`SCFourStateGLIFNeuron`] model.
    SCFourStateGLIF(SCFourStateGLIFNeuron),
    /// The [`GIFPopulationNeuron`] model.
    GIFPopulation(GIFPopulationNeuron),
    /// The [`AvRonCardiacNeuron`] model.
    AvRonCardiac(AvRonCardiacNeuron),
    /// The [`DurstewitzDopamineNeuron`] model.
    DurstewitzDopamine(DurstewitzDopamineNeuron),
    /// The [`HillTononiNeuron`] model.
    HillTononi(HillTononiNeuron),
    /// The [`BertramPhantomBurster`] model.
    BertramPhantom(BertramPhantomBurster),
    /// The [`SCThreeStatePhantomBurster`] model.
    SCThreeStatePhantom(SCThreeStatePhantomBurster),
    /// The [`YamadaNeuron`] model.
    Yamada(YamadaNeuron),

    // simple_spiking.rs
    /// The [`FitzHughNagumoNeuron`] model.
    FitzHughNagumo(FitzHughNagumoNeuron),
    /// The [`MorrisLecarNeuron`] model.
    MorrisLecar(MorrisLecarNeuron),
    /// The [`HindmarshRoseNeuron`] model.
    HindmarshRose(HindmarshRoseNeuron),
    /// The [`ResonateAndFireNeuron`] model.
    ResonateAndFire(ResonateAndFireNeuron),
    /// The [`BalancedResonateAndFireNeuron`] model.
    BalancedResonateAndFire(BalancedResonateAndFireNeuron),
    /// The [`FitzHughRinzelNeuron`] model.
    FitzHughRinzel(FitzHughRinzelNeuron),
    /// The [`McKeanNeuron`] model.
    McKean(McKeanNeuron),
    /// The [`SCTriangularMcKeanNeuron`] model.
    SCTriangularMcKean(SCTriangularMcKeanNeuron),
    /// The [`TermanWangOscillator`] model.
    TermanWang(TermanWangOscillator),
    /// The [`GutkinErmentroutNeuron`] model.
    GutkinErmentrout(GutkinErmentroutNeuron),
    /// The [`WilsonHRNeuron`] model.
    WilsonHR(WilsonHRNeuron),
    /// The [`SCResettingWilsonHRNeuron`] model.
    SCResettingWilsonHR(SCResettingWilsonHRNeuron),
    /// The [`ChayNeuron`] model.
    Chay(ChayNeuron),
    /// The [`ChayKeizerNeuron`] model.
    ChayKeizer(ChayKeizerNeuron),
    /// The [`ShermanRinzelKeizerNeuron`] model.
    ShermanRinzelKeizer(ShermanRinzelKeizerNeuron),
    /// The [`ButeraRespiratoryNeuron`] model.
    ButeraRespiratory(ButeraRespiratoryNeuron),
    /// The [`EPropALIFNeuron`] model.
    EPropALIF(EPropALIFNeuron),
    /// The [`SuperSpikeNeuron`] model.
    SuperSpike(SuperSpikeNeuron),
    /// The [`LearnableNeuronModel`] model.
    LearnableNeuron(LearnableNeuronModel),
    /// The [`PernarowskiNeuron`] model.
    Pernarowski(PernarowskiNeuron),

    // trivial.rs (simple IF variants)
    /// The [`QuadraticIFNeuron`] model.
    QuadraticIF(QuadraticIFNeuron),
    /// The [`ThetaNeuron`] model.
    Theta(ThetaNeuron),
    /// The [`PerfectIntegratorNeuron`] model.
    PerfectIntegrator(PerfectIntegratorNeuron),
    /// The [`GatedLIFNeuron`] model.
    GatedLIF(GatedLIFNeuron),
    /// The [`NonlinearLIFNeuron`] model.
    NonlinearLIF(NonlinearLIFNeuron),
    /// The [`SFANeuron`] model.
    SFA(SFANeuron),
    /// The [`MATNeuron`] model.
    MAT(MATNeuron),
    /// The [`SCResettingMATNeuron`] model.
    SCResettingMAT(SCResettingMATNeuron),
    /// The [`KLIFNeuron`] model.
    KLIF(KLIFNeuron),
    /// The [`InhibitoryLIFNeuron`] model.
    InhibitoryLIF(InhibitoryLIFNeuron),
    /// The [`ComplementaryLIFNeuron`] model.
    ComplementaryLIF(ComplementaryLIFNeuron),
    /// The [`ParametricLIFNeuron`] model.
    ParametricLIF(ParametricLIFNeuron),
    /// The [`NonResettingLIFNeuron`] model.
    NonResettingLIF(NonResettingLIFNeuron),
    /// The [`SCNonResettingAdaptiveLIFNeuron`] model.
    SCNonResettingAdaptiveLIF(SCNonResettingAdaptiveLIFNeuron),
    /// The [`AdaptiveThresholdIFNeuron`] model.
    AdaptiveThresholdIF(AdaptiveThresholdIFNeuron),
    /// The [`SigmaDeltaNeuron`] model.
    SigmaDelta(SigmaDeltaNeuron),
    /// The [`SCSigmaDeltaAccumulatorNeuron`] model.
    SCSigmaDeltaAccumulator(SCSigmaDeltaAccumulatorNeuron),
    /// The [`EnergyLIFNeuron`] model.
    EnergyLIF(EnergyLIFNeuron),
    /// The [`SCNormalizedEnergyLIFNeuron`] model.
    SCNormalizedEnergyLIF(SCNormalizedEnergyLIFNeuron),
    /// The [`ClosedFormContinuousNeuron`] model.
    ClosedFormContinuous(ClosedFormContinuousNeuron),

    // maps.rs
    /// The [`ChialvoMapNeuron`] model.
    ChialvoMap(ChialvoMapNeuron),
    /// The [`RulkovMapNeuron`] model.
    RulkovMap(RulkovMapNeuron),
    /// The [`SCUpwardCrossingRulkovMapNeuron`] model.
    SCUpwardCrossingRulkovMap(SCUpwardCrossingRulkovMapNeuron),
    /// The [`IbarzTanakaMapNeuron`] model.
    IbarzTanakaMap(IbarzTanakaMapNeuron),
    /// The [`MedvedevMapNeuron`] model.
    MedvedevMap(MedvedevMapNeuron),
    /// The [`CazellesMapNeuron`] model.
    CazellesMap(CazellesMapNeuron),
    /// The [`SCClippedLogisticBurstingMapNeuron`] model.
    SCClippedLogisticBurstingMap(SCClippedLogisticBurstingMapNeuron),
    /// The [`SCClippedRationalRecoveryMapNeuron`] model.
    SCClippedRationalRecoveryMap(SCClippedRationalRecoveryMapNeuron),
    /// The [`CourageNekorkinMapNeuron`] model.
    CourageNekorkinMap(CourageNekorkinMapNeuron),
    /// The [`AiharaMapNeuron`] model.
    AiharaMap(AiharaMapNeuron),
    /// The [`SCChaoticMapNeuron`] model.
    SCChaoticMap(SCChaoticMapNeuron),
    /// The [`NagumoSatoMapNeuron`] model.
    NagumoSatoMap(NagumoSatoMapNeuron),
    /// The [`SCAdaptiveThresholdMapNeuron`] model.
    SCAdaptiveThresholdMap(SCAdaptiveThresholdMapNeuron),
    /// The [`ErmentroutKopellMapNeuron`] model.
    ErmentroutKopellMap(ErmentroutKopellMapNeuron),

    // hardware.rs (f64 input subset)
    /// The [`BrainScaleSAdExNeuron`] model.
    BrainScaleSAdEx(BrainScaleSAdExNeuron),
    /// The [`SpiNNakerLIFNeuron`] model.
    SpiNNakerLIF(SpiNNakerLIFNeuron),
    /// The [`NeuroGridNeuron`] model.
    NeuroGrid(NeuroGridNeuron),
    /// The [`DPINeuron`] model.
    DPI(DPINeuron),
    /// The [`AkidaNeuron`] model.
    Akida(AkidaNeuron),
    /// The [`StochasticLIFNeuron`] model.
    StochasticLIF(StochasticLIFNeuron),

    // multi_compartment.rs (single-f64-input subset)
    /// The [`MarderSTGNeuron`] model.
    MarderSTG(MarderSTGNeuron),
    /// The [`RallCableNeuron`] model.
    RallCable(RallCableNeuron),
    /// The [`BoothRinzelNeuron`] model.
    BoothRinzel(BoothRinzelNeuron),
    /// The [`DendrifyNeuron`] model.
    Dendrify(DendrifyNeuron),

    // rate.rs (spiking subset with step(f64)->i32)
    /// The [`LiquidTimeConstantNeuron`] model.
    LiquidTimeConstant(LiquidTimeConstantNeuron),
    /// The [`ParallelSpikingNeuron`] model.
    ParallelSpiking(ParallelSpikingNeuron),
    /// The [`FractionalLIFNeuron`] model.
    FractionalLIF(FractionalLIFNeuron),

    // special.rs (step(f64)->i32 subset)
    /// The [`StochasticIFNeuron`] model.
    StochasticIF(StochasticIFNeuron),
    /// The [`GalvesLocherbachNeuron`] model.
    GalvesLocherbach(GalvesLocherbachNeuron),
    /// The [`SpikeResponseNeuron`] model.
    SpikeResponse(SpikeResponseNeuron),
    /// The [`GLMNeuron`] model.
    GLM(GLMNeuron),
    /// The [`ArcaneNeuron`] model.
    Arcane(ArcaneNeuron),

    // --- newly wired (step(f64)->i32 compatible) ---
    // ai_optimized.rs
    /// The [`MultiTimescaleNeuron`] model.
    MultiTimescale(MultiTimescaleNeuron),
    /// The [`AttentionGatedNeuron`] model.
    AttentionGated(AttentionGatedNeuron),
    /// The [`PredictiveCodingNeuron`] model.
    PredictiveCoding(PredictiveCodingNeuron),
    /// The [`SelfReferentialNeuron`] model.
    SelfReferential(SelfReferentialNeuron),
    /// The [`CompositionalBindingNeuron`] model.
    CompositionalBinding(CompositionalBindingNeuron),
    /// The [`DifferentiableSurrogateNeuron`] model.
    DifferentiableSurrogate(DifferentiableSurrogateNeuron),
    /// The [`ContinuousAttractorNeuron`] model.
    ContinuousAttractor(ContinuousAttractorNeuron),
    /// The [`MetaPlasticNeuron`] model.
    MetaPlastic(MetaPlasticNeuron),
    // simple_spiking.rs
    /// The [`BendaHerzNeuron`] model.
    BendaHerz(BendaHerzNeuron),
    /// The [`SCStochasticRateAdaptationNeuron`] model.
    SCStochasticRateAdaptation(SCStochasticRateAdaptationNeuron),
    /// The [`BrunelWangNeuron`] model.
    BrunelWang(BrunelWangNeuron),
    // special.rs
    /// The [`PoissonNeuron`] model.
    Poisson(PoissonNeuron),
    /// The [`InhomogeneousPoissonNeuron`] model.
    InhomogeneousPoisson(InhomogeneousPoissonNeuron),
    /// The [`GammaRenewalNeuron`] model.
    GammaRenewal(GammaRenewalNeuron),
    /// The [`EscapeRateNeuron`] model.
    EscapeRate(EscapeRateNeuron),
    // interneurons.rs (step(f64)->i32)
    /// The [`PVFastSpikingNeuron`] model.
    PVFastSpiking(PVFastSpikingNeuron),
    /// The [`SSTNeuron`] model.
    SST(SSTNeuron),
    /// The [`VIPNeuron`] model.
    VIP(VIPNeuron),
    /// The [`ChandelierNeuron`] model.
    Chandelier(ChandelierNeuron),
    /// The [`CerebellarBasketNeuron`] model.
    CerebellarBasket(CerebellarBasketNeuron),
    /// The [`MartinottiNeuron`] model.
    Martinotti(MartinottiNeuron),

    // motor.rs (step(f64)->i32)
    /// The [`AlphaMotorNeuron`] model.
    AlphaMotor(AlphaMotorNeuron),
    /// The [`GammaMotorNeuron`] model.
    GammaMotor(GammaMotorNeuron),
    /// The [`UpperMotorNeuron`] model.
    UpperMotor(UpperMotorNeuron),
    /// The [`RenshawCell`] model.
    Renshaw(RenshawCell),
    /// The [`MotorUnit`] model.
    MotorUnitCell(MotorUnit),

    // sensory.rs (spiking subset: step(f64)->i32)
    /// The [`RetinalGanglionCell`] model.
    RetinalGanglion(RetinalGanglionCell),
    /// The [`MerkelCell`] model.
    Merkel(MerkelCell),
    /// The [`PacinianCorpuscle`] model.
    Pacinian(PacinianCorpuscle),
    /// The [`Nociceptor`] model.
    NociceptorCell(Nociceptor),
    /// The [`OlfactoryReceptorNeuron`] model.
    OlfactoryReceptor(OlfactoryReceptorNeuron),

    // cerebellar.rs (step(f64)->i32)
    /// The [`GranuleCell`] model.
    Granule(GranuleCell),
    /// The [`GolgiCell`] model.
    Golgi(GolgiCell),
    /// The [`StellateCell`] model.
    Stellate(StellateCell),
    /// The [`LugaroCell`] model.
    Lugaro(LugaroCell),
    /// The [`UnipolarBrushCell`] model.
    UnipolarBrush(UnipolarBrushCell),
    /// The [`DCNNeuron`] model.
    DCN(DCNNeuron),

    // channels.rs (step(f64)->i32)
    /// The [`PersistentNaNeuron`] model.
    PersistentNa(PersistentNaNeuron),
    /// The [`IhNeuron`] model.
    Ih(IhNeuron),
    /// The [`TTypeCaNeuron`] model.
    TTypeCa(TTypeCaNeuron),
    /// The [`ATypeKNeuron`] model.
    ATypeK(ATypeKNeuron),
    /// The [`BKNeuron`] model.
    BK(BKNeuron),
    /// The [`SKNeuron`] model.
    SK(SKNeuron),
    /// The [`NMDANeuron`] model.
    NMDA(NMDANeuron),
    /// The [`SCWBNMDAMagnesiumBlockNeuron`] model.
    SCWBNMDAMagnesiumBlock(SCWBNMDAMagnesiumBlockNeuron),

    // population.rs (step(f64)->i32)
    /// The [`MontbrioMeanField`] model.
    MontbrioMPR(MontbrioMeanField),
    /// The [`BrunelNetwork`] model.
    Brunel(BrunelNetwork),
    /// The [`TUMNetwork`] model.
    TUM(TUMNetwork),
    /// The [`ElBoustaniNetwork`] model.
    ElBoustani(ElBoustaniNetwork),

    // misc.rs (step(f64)->i32)
    /// The [`GradedSynapseNeuron`] model.
    GradedSynapse(GradedSynapseNeuron),
    /// The [`GapJunctionNeuron`] model.
    GapJunction(GapJunctionNeuron),
    /// The [`FrankenhaeUserHuxleyAxon`] model.
    FHAxon(FrankenhaeUserHuxleyAxon),
    /// The [`NodeOfRanvier`] model.
    NodeOfRanvier(NodeOfRanvier),
    /// The [`MyelinatedAxon`] model.
    MyelinAxon(MyelinatedAxon),
    /// The [`CardiacPurkinjeFibre`] model.
    CardiacPurkinje(CardiacPurkinjeFibre),
    /// The [`SmoothMuscleCell`] model.
    SmoothMuscle(SmoothMuscleCell),
    /// The [`EndocrineBetaCell`] model.
    BetaCell(EndocrineBetaCell),

    // Wrapped models: non-standard interfaces normalised to step(f64)->i32
    // Multi-input spiking
    /// The [`WrAlpha`] adapter, normalised to `step(f64) -> i32`.
    WrAlphaCell(WrAlpha),
    /// The [`WrCOBALIF`] adapter, normalised to `step(f64) -> i32`.
    WrCOBALIFCell(WrCOBALIF),
    /// The [`WrCompteWM`] adapter, normalised to `step(f64) -> i32`.
    WrCompteWMCell(WrCompteWM),
    /// The [`WrTsodyksMarkram`] adapter, normalised to `step(f64) -> i32`.
    WrTsodyksMarkramCell(WrTsodyksMarkram),
    /// The [`WrPinskyRinzel`] adapter, normalised to `step(f64) -> i32`.
    WrPinskyRinzelCell(WrPinskyRinzel),
    /// The [`WrHayL5`] adapter, normalised to `step(f64) -> i32`.
    WrHayL5Cell(WrHayL5),
    /// The [`WrTwoCompLIF`] adapter, normalised to `step(f64) -> i32`.
    WrTwoCompLIFCell(WrTwoCompLIF),
    // Hardware integer-input
    /// The [`WrLoihiCUBA`] adapter, normalised to `step(f64) -> i32`.
    WrLoihiCUBACell(WrLoihiCUBA),
    /// The [`WrLoihi2`] adapter, normalised to `step(f64) -> i32`.
    WrLoihi2Cell(WrLoihi2),
    /// The [`WrSpiNNaker2`] adapter, normalised to `step(f64) -> i32`.
    WrSpiNNaker2Cell(WrSpiNNaker2),
    /// The [`WrTrueNorth`] adapter, normalised to `step(f64) -> i32`.
    WrTrueNorthCell(WrTrueNorth),
    /// The [`WrIntegerQIF`] adapter, normalised to `step(f64) -> i32`.
    WrIntegerQIFCell(WrIntegerQIF),
    /// The [`WrMcCullochPitts`] adapter, normalised to `step(f64) -> i32`.
    WrMcCullochPittsCell(WrMcCullochPitts),
    // Graded/rate output
    /// The [`WrSigmoidRate`] adapter, normalised to `step(f64) -> i32`.
    WrSigmoidRateCell(WrSigmoidRate),
    /// The [`WrThresholdLinear`] adapter, normalised to `step(f64) -> i32`.
    WrThresholdLinearCell(WrThresholdLinear),
    /// The [`WrAstrocyte`] adapter, normalised to `step(f64) -> i32`.
    WrAstrocyteCell(WrAstrocyte),
    /// The [`WrInnerHairCell`] adapter, normalised to `step(f64) -> i32`.
    WrInnerHairCellCell(WrInnerHairCell),
    /// The [`WrOuterHairCell`] adapter, normalised to `step(f64) -> i32`.
    WrOuterHairCellCell(WrOuterHairCell),
    /// The [`WrRodPhotoreceptor`] adapter, normalised to `step(f64) -> i32`.
    WrRodPhotoreceptorCell(WrRodPhotoreceptor),
    /// The [`WrConePhotoreceptor`] adapter, normalised to `step(f64) -> i32`.
    WrConePhotoreceptorCell(WrConePhotoreceptor),
    /// The [`WrTasteReceptor`] adapter, normalised to `step(f64) -> i32`.
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
            Izhikevich, Izhikevich2007, AdEx, ExpIF, Lapicque, HomeostaticLif,
            HodgkinHuxley, TraubMiles, WangBuzsaki, ConnorStevens,
            DestexheThalamic, HuberBraun, GolombFS,
            Pospischil, MainenSejnowski, DeSchutterPurkinje,
            PlantR15, Prescott, MihalasNiebur, SCScaledResetAdaptiveIF, GLIF, SCFourStateGLIF, GIFPopulation,
            AvRonCardiac, DurstewitzDopamine, HillTononi, BertramPhantom, SCThreeStatePhantom, Yamada, Akida, StochasticLIF,
            FitzHughNagumo, MorrisLecar, HindmarshRose, ResonateAndFire, BalancedResonateAndFire,
            FitzHughRinzel, McKean, SCTriangularMcKean, TermanWang, GutkinErmentrout, WilsonHR, SCResettingWilsonHR,
            Chay, ChayKeizer, ShermanRinzelKeizer, ButeraRespiratory,
            EPropALIF, SuperSpike, LearnableNeuron, Pernarowski,
            QuadraticIF, Theta, PerfectIntegrator, GatedLIF, NonlinearLIF,
            SFA, MAT, SCResettingMAT, KLIF, InhibitoryLIF, ComplementaryLIF, ParametricLIF,
            NonResettingLIF, SCNonResettingAdaptiveLIF, AdaptiveThresholdIF, SigmaDelta, SCSigmaDeltaAccumulator, EnergyLIF, SCNormalizedEnergyLIF,
            ClosedFormContinuous,
            ChialvoMap, RulkovMap, SCUpwardCrossingRulkovMap, IbarzTanakaMap, MedvedevMap,
            CazellesMap, SCClippedLogisticBurstingMap, SCClippedRationalRecoveryMap, CourageNekorkinMap, AiharaMap, SCChaoticMap,
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
            PersistentNa, Ih, TTypeCa, ATypeK, BK, SK, NMDA, SCWBNMDAMagnesiumBlock,
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
    /// Advance the wrapped model one step with `current` and return what its `step` returns.
    pub fn step(&mut self, current: f64) -> i32 {
        all_variants!(dispatch_step, self, current,)
    }

    /// Reset the wrapped model with its own `reset`.
    pub fn reset(&mut self) {
        all_variants!(dispatch_reset, self,)
    }

    /// Return the state the runner records as the soma voltage.
    ///
    /// That is `v` for most models. Others give a state of their own: a map's `x` or output, a
    /// phase such as `theta`, a population rate, an adaptation, accumulator or current
    /// variable, a compartment's somatic potential, or a wrapper's `v()`. The retinal
    /// ganglion cell gives its constant `baseline`. The point processes and the
    /// continuous-attractor and compositional-binding models give 0.
    pub fn soma_voltage(&self) -> f64 {
        match self {
            NeuronVariant::Izhikevich(n) => n.v,
            NeuronVariant::Izhikevich2007(n) => n.v,
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
            NeuronVariant::SCScaledResetAdaptiveIF(n) => n.v,
            NeuronVariant::GLIF(n) => n.v,
            NeuronVariant::SCFourStateGLIF(n) => n.v,
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
            NeuronVariant::SCResettingWilsonHR(n) => n.v,
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
            NeuronVariant::SCUpwardCrossingRulkovMap(n) => n.x,
            NeuronVariant::IbarzTanakaMap(n) => n.v,
            NeuronVariant::MedvedevMap(n) => n.u,
            NeuronVariant::CazellesMap(n) => n.x,
            NeuronVariant::SCClippedLogisticBurstingMap(n) => n.x,
            NeuronVariant::SCClippedRationalRecoveryMap(n) => n.x,
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
            NeuronVariant::SCWBNMDAMagnesiumBlock(n) => n.v,
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
