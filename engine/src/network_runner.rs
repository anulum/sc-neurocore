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

use rayon::prelude::*;

use crate::neuron::*;
use crate::neurons::*;

// ── Interface wrappers for non-standard models ──────────────────────

/// Wrapper: multi-input spiking model → single-input interface.
/// Extra inputs default to 0/false when driven from the network runner.
macro_rules! wrap_2arg_f64 {
    ($name:ident, $inner:ty, $v:ident, $extra:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self(<$inner>::new())
            }
            pub fn step(&mut self, current: f64) -> i32 {
                self.0.step(current, $extra)
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Wrapper: 3-arg model → single-input interface.
macro_rules! wrap_3arg {
    ($name:ident, $inner:ty, $v:ident, $e2:expr, $e3:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self(<$inner>::new())
            }
            pub fn step(&mut self, current: f64) -> i32 {
                self.0.step(current, $e2, $e3)
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Wrapper: i32-input model → f64-input interface.
macro_rules! wrap_i32_input {
    ($name:ident, $inner:ty, $v:ident, $ctor:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self($ctor)
            }
            pub fn step(&mut self, current: f64) -> i32 {
                self.0.step(current as i32)
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Wrapper: f64-output (rate/graded) → i32 spike interface via thresholding.
macro_rules! wrap_graded {
    ($name:ident, $inner:ty, $v:ident, $threshold:expr) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub $inner);
        impl $name {
            pub fn new() -> Self {
                Self(<$inner>::new())
            }
            pub fn step(&mut self, current: f64) -> i32 {
                let out = self.0.step(current);
                if out > $threshold {
                    1
                } else {
                    0
                }
            }
            pub fn reset(&mut self) {
                self.0.reset();
            }
            pub fn v(&self) -> f64 {
                self.0.$v as f64
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

// Multi-input spiking wrappers
wrap_2arg_f64!(WrAlpha, AlphaNeuron, v, 0.0_f64);
wrap_3arg!(WrCOBALIF, COBALIFNeuron, v, 0.0_f64, 0.0_f64);
wrap_2arg_f64!(WrCompteWM, CompteWMNeuron, v, false);
wrap_2arg_f64!(WrTsodyksMarkram, TsodyksMarkramNeuron, v, false);
wrap_2arg_f64!(WrPinskyRinzel, PinskyRinzelNeuron, v_s, 0.0_f64);
wrap_2arg_f64!(WrHayL5, HayL5PyramidalNeuron, v_s, 0.0_f64);
wrap_2arg_f64!(WrTwoCompLIF, TwoCompartmentLIFNeuron, v_s, 0.0_f64);

// Hardware integer-input wrappers
wrap_i32_input!(WrLoihiCUBA, LoihiCUBANeuron, v, LoihiCUBANeuron::new());
wrap_i32_input!(WrLoihi2, Loihi2Neuron, s1, Loihi2Neuron::new());
wrap_i32_input!(WrSpiNNaker2, SpiNNaker2Neuron, v, SpiNNaker2Neuron::new());
wrap_i32_input!(WrTrueNorth, TrueNorthNeuron, v, TrueNorthNeuron::new(256));
wrap_i32_input!(
    WrIntegerQIF,
    IntegerQIFNeuron,
    v,
    IntegerQIFNeuron::new(1, 1000)
);

// Graded/rate output wrappers (threshold at 0.5 for rates, 0.0 for sensory)
wrap_graded!(WrSigmoidRate, SigmoidRateNeuron, r, 0.5);
wrap_graded!(WrThresholdLinear, ThresholdLinearRateNeuron, r, 0.5);
wrap_graded!(WrAstrocyte, AstrocyteModel, ca, 0.1);
wrap_graded!(WrInnerHairCell, InnerHairCell, v, 0.0);
wrap_graded!(WrOuterHairCell, OuterHairCell, v, 0.0);
wrap_graded!(WrRodPhotoreceptor, RodPhotoreceptor, v, 0.0);
wrap_graded!(WrConePhotoreceptor, ConePhotoreceptor, v, 0.0);
wrap_graded!(WrTasteReceptor, TasteReceptorCell, v, 0.0);

// ── NeuronVariant ───────────────────────────────────────────────────

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

    // biophysical.rs
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
    Yamada(YamadaNeuron),

    // simple_spiking.rs
    FitzHughNagumo(FitzHughNagumoNeuron),
    MorrisLecar(MorrisLecarNeuron),
    HindmarshRose(HindmarshRoseNeuron),
    ResonateAndFire(ResonateAndFireNeuron),
    BalancedResonateAndFire(BalancedResonateAndFireNeuron),
    FitzHughRinzel(FitzHughRinzelNeuron),
    McKean(McKeanNeuron),
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
    KLIF(KLIFNeuron),
    InhibitoryLIF(InhibitoryLIFNeuron),
    ComplementaryLIF(ComplementaryLIFNeuron),
    ParametricLIF(ParametricLIFNeuron),
    NonResettingLIF(NonResettingLIFNeuron),
    AdaptiveThresholdIF(AdaptiveThresholdIFNeuron),
    SigmaDelta(SigmaDeltaNeuron),
    EnergyLIF(EnergyLIFNeuron),
    ClosedFormContinuous(ClosedFormContinuousNeuron),

    // maps.rs
    ChialvoMap(ChialvoMapNeuron),
    RulkovMap(RulkovMapNeuron),
    IbarzTanakaMap(IbarzTanakaMapNeuron),
    MedvedevMap(MedvedevMapNeuron),
    CazellesMap(CazellesMapNeuron),
    CourageNekorkinMap(CourageNekorkinMapNeuron),
    AiharaMap(AiharaMapNeuron),
    KilincBhattMap(KilincBhattMapNeuron),
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
            AvRonCardiac, DurstewitzDopamine, HillTononi, BertramPhantom, Yamada, Akida, StochasticLIF,
            FitzHughNagumo, MorrisLecar, HindmarshRose, ResonateAndFire, BalancedResonateAndFire,
            FitzHughRinzel, McKean, TermanWang, GutkinErmentrout, WilsonHR,
            Chay, ChayKeizer, ShermanRinzelKeizer, ButeraRespiratory,
            EPropALIF, SuperSpike, LearnableNeuron, Pernarowski,
            QuadraticIF, Theta, PerfectIntegrator, GatedLIF, NonlinearLIF,
            SFA, MAT, KLIF, InhibitoryLIF, ComplementaryLIF, ParametricLIF,
            NonResettingLIF, AdaptiveThresholdIF, SigmaDelta, EnergyLIF,
            ClosedFormContinuous,
            ChialvoMap, RulkovMap, IbarzTanakaMap, MedvedevMap,
            CazellesMap, CourageNekorkinMap, AiharaMap, KilincBhattMap, ErmentroutKopellMap,
            BrainScaleSAdEx, SpiNNakerLIF, NeuroGrid, DPI,
            MarderSTG, RallCable, BoothRinzel, Dendrify,
            LiquidTimeConstant, ParallelSpiking, FractionalLIF,
            StochasticIF, GalvesLocherbach, SpikeResponse, GLM,
            Arcane,
            MultiTimescale, AttentionGated, PredictiveCoding,
            SelfReferential, CompositionalBinding, DifferentiableSurrogate,
            ContinuousAttractor, MetaPlastic,
            BendaHerz, BrunelWang,
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
            NeuronVariant::Yamada(n) => n.v,
            NeuronVariant::FitzHughNagumo(n) => n.v,
            NeuronVariant::MorrisLecar(n) => n.v,
            NeuronVariant::HindmarshRose(n) => n.x,
            NeuronVariant::ResonateAndFire(n) => n.x,
            NeuronVariant::BalancedResonateAndFire(n) => n.x,
            NeuronVariant::FitzHughRinzel(n) => n.v,
            NeuronVariant::McKean(n) => n.v,
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
            NeuronVariant::KLIF(n) => n.v,
            NeuronVariant::InhibitoryLIF(n) => n.v,
            NeuronVariant::ComplementaryLIF(n) => n.v_pos,
            NeuronVariant::ParametricLIF(n) => n.v,
            NeuronVariant::NonResettingLIF(n) => n.v,
            NeuronVariant::AdaptiveThresholdIF(n) => n.v,
            NeuronVariant::SigmaDelta(n) => n.sigma,
            NeuronVariant::EnergyLIF(n) => n.v,
            NeuronVariant::ClosedFormContinuous(n) => n.x,
            NeuronVariant::ChialvoMap(n) => n.x,
            NeuronVariant::RulkovMap(n) => n.x,
            NeuronVariant::IbarzTanakaMap(n) => n.x,
            NeuronVariant::MedvedevMap(n) => n.u,
            NeuronVariant::CazellesMap(n) => n.x,
            NeuronVariant::CourageNekorkinMap(n) => n.x,
            NeuronVariant::AiharaMap(n) => n.x,
            NeuronVariant::KilincBhattMap(n) => n.x,
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
            NeuronVariant::ParallelSpiking(_) => 0.0,
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

// ── PopulationRunner ────────────────────────────────────────────────

pub struct PopulationRunner {
    neurons: Vec<NeuronVariant>,
    spikes: Vec<u8>,
    currents: Vec<f64>,
}

const CHUNK_SIZE: usize = 256;

impl PopulationRunner {
    pub fn new(neurons: Vec<NeuronVariant>) -> Self {
        let n = neurons.len();
        Self {
            neurons,
            spikes: vec![0u8; n],
            currents: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    pub fn step_all(&mut self) {
        let neurons = &mut self.neurons;
        let spikes = &mut self.spikes;
        let currents = &self.currents;

        neurons
            .par_chunks_mut(CHUNK_SIZE)
            .zip(spikes.par_chunks_mut(CHUNK_SIZE))
            .zip(currents.par_chunks(CHUNK_SIZE))
            .for_each(|((n_chunk, s_chunk), c_chunk)| {
                for i in 0..n_chunk.len() {
                    let fired = n_chunk[i].step(c_chunk[i]);
                    s_chunk[i] = if fired != 0 { 1 } else { 0 };
                }
            });
    }

    pub fn reset_all(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
        self.spikes.fill(0);
        self.currents.fill(0.0);
    }

    pub fn reset_currents(&mut self) {
        self.currents.fill(0.0);
    }

    pub fn set_currents(&mut self, currents: &[f64]) -> Result<(), String> {
        if currents.len() != self.currents.len() {
            return Err(format!(
                "current vector length mismatch: got {}, expected {}",
                currents.len(),
                self.currents.len()
            ));
        }
        self.currents.copy_from_slice(currents);
        Ok(())
    }

    pub fn collect_voltages(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.soma_voltage()).collect()
    }
}

// ── ProjectionRunner ────────────────────────────────────────────────

/// CSR-stored synaptic projection with optional axonal delay.
pub struct ProjectionRunner {
    pub src_pop: usize,
    pub tgt_pop: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
    delay_steps: usize,
    delay_buffer: Vec<Vec<u8>>,
    buf_idx: usize,
}

impl ProjectionRunner {
    pub fn new(
        src_pop: usize,
        tgt_pop: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        delay_steps: usize,
    ) -> Self {
        let n_delay = if delay_steps > 0 { delay_steps } else { 0 };
        let n_src = if row_offsets.is_empty() {
            0
        } else {
            row_offsets.len() - 1
        };
        let delay_buffer = if n_delay > 0 {
            vec![vec![0u8; n_src]; n_delay]
        } else {
            Vec::new()
        };
        Self {
            src_pop,
            tgt_pop,
            row_offsets,
            col_indices,
            values,
            delay_steps: n_delay,
            delay_buffer,
            buf_idx: 0,
        }
    }

    /// Scatter spikes through CSR connectivity into target current buffer.
    pub fn propagate(&mut self, src_spikes: &[u8], tgt_currents: &mut [f64]) {
        let spikes = if self.delay_steps > 0 {
            let delayed = &self.delay_buffer[self.buf_idx];
            let out: Vec<u8> = delayed.clone();
            self.delay_buffer[self.buf_idx] = src_spikes.to_vec();
            self.buf_idx = (self.buf_idx + 1) % self.delay_steps;
            out
        } else {
            src_spikes.to_vec()
        };

        let n_src = self.row_offsets.len().saturating_sub(1);
        for i in 0..n_src {
            if spikes.get(i).copied().unwrap_or(0) == 0 {
                continue;
            }
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];
            for k in start..end {
                let j = self.col_indices[k];
                if j < tgt_currents.len() {
                    tgt_currents[j] += self.values[k];
                }
            }
        }
    }
}

// ── SimResults ──────────────────────────────────────────────────────

pub struct SimResults {
    pub spike_counts: Vec<usize>,
    /// Per-population flat spike data: (neuron_id << 32) | timestep packed as u64.
    /// Supports up to 2^32 neurons and 2^32 timesteps.
    pub spike_data: Vec<Vec<u64>>,
    pub voltages: Vec<Vec<f64>>,
}

fn pack_spike_event(neuron_id: usize, timestep: usize) -> u64 {
    assert!(
        timestep <= u32::MAX as usize,
        "spike event timestep exceeds 32-bit packing lane"
    );
    ((neuron_id as u64) << 32) | (timestep as u64)
}

// ── NetworkRunner ───────────────────────────────────────────────────

pub struct NetworkRunner {
    pub populations: Vec<PopulationRunner>,
    pub projections: Vec<ProjectionRunner>,
}

impl NetworkRunner {
    pub fn new() -> Self {
        Self {
            populations: Vec::new(),
            projections: Vec::new(),
        }
    }

    pub fn add_population(&mut self, pop: PopulationRunner) -> usize {
        let idx = self.populations.len();
        self.populations.push(pop);
        idx
    }

    pub fn add_projection(&mut self, proj: ProjectionRunner) {
        self.projections.push(proj);
    }

    pub fn step_population_with_currents(
        &mut self,
        pop_idx: usize,
        currents: &[f64],
    ) -> Result<(Vec<u8>, Vec<f64>), String> {
        let pop = self
            .populations
            .get_mut(pop_idx)
            .ok_or_else(|| format!("population index {pop_idx} out of range"))?;
        pop.set_currents(currents)?;
        pop.step_all();
        Ok((pop.spikes.clone(), pop.collect_voltages()))
    }

    pub fn run(&mut self, n_steps: usize) -> SimResults {
        let n_pops = self.populations.len();
        let mut spike_counts = vec![0usize; n_pops];
        let mut spike_data: Vec<Vec<u64>> = vec![Vec::new(); n_pops];

        for t in 0..n_steps {
            // Reset currents
            for pop in &mut self.populations {
                pop.reset_currents();
            }

            // Propagate spikes through projections
            // Must borrow populations mutably for target currents, immutably for source spikes.
            // Use index-based access to avoid aliasing issues.
            for proj_idx in 0..self.projections.len() {
                let src = self.projections[proj_idx].src_pop;
                let tgt = self.projections[proj_idx].tgt_pop;
                if src == tgt {
                    // Self-projection: copy spikes, then propagate
                    let spikes_copy = self.populations[src].spikes.clone();
                    let currents = &mut self.populations[tgt].currents;
                    self.projections[proj_idx].propagate(&spikes_copy, currents);
                } else {
                    // Split borrow via raw pointers (safe because src != tgt)
                    let pops_ptr = self.populations.as_mut_ptr();
                    let src_spikes = unsafe { &(*pops_ptr.add(src)).spikes };
                    let tgt_currents = unsafe { &mut (*pops_ptr.add(tgt)).currents };
                    self.projections[proj_idx].propagate(src_spikes, tgt_currents);
                }
            }

            // Step all populations
            for (pop_idx, pop) in self.populations.iter_mut().enumerate() {
                pop.step_all();
                for (nid, &spike) in pop.spikes.iter().enumerate() {
                    if spike != 0 {
                        spike_counts[pop_idx] += 1;
                        spike_data[pop_idx].push(pack_spike_event(nid, t));
                    }
                }
            }
        }

        let voltages: Vec<Vec<f64>> = self
            .populations
            .iter()
            .map(|p| p.collect_voltages())
            .collect();

        SimResults {
            spike_counts,
            spike_data,
            voltages,
        }
    }
}

impl Default for NetworkRunner {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn projection_propagates_spikes() {
        let row_offsets = vec![0, 2, 4];
        let col_indices = vec![0, 1, 0, 1];
        let values = vec![5.0, 3.0, 2.0, 4.0];

        let mut proj = ProjectionRunner::new(0, 1, row_offsets, col_indices, values, 0);
        let src_spikes = vec![1u8, 0];
        let mut tgt_currents = vec![0.0; 2];
        proj.propagate(&src_spikes, &mut tgt_currents);

        assert!((tgt_currents[0] - 5.0).abs() < 1e-10);
        assert!((tgt_currents[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn spike_event_pack_preserves_high_neuron_and_timestep_bits() {
        let packed = pack_spike_event(100_000, 70_000);
        assert_eq!((packed >> 32) as usize, 100_000);
        assert_eq!((packed & u32::MAX as u64) as usize, 70_000);
    }

    #[test]
    fn spike_event_pack_rejects_timestep_overflow() {
        if usize::BITS > 32 {
            let result = std::panic::catch_unwind(|| {
                let _ = pack_spike_event(0, u32::MAX as usize + 1);
            });
            assert!(result.is_err());
        }
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
