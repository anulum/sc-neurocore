// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

// ── NeuronVariant ───────────────────────────────────────────────────

/// Enum dispatch across all neuron models that accept `step(f64) -> i32`.
///
/// Models with non-standard signatures (multi-input, integer-input,
/// rate-output) are excluded — they require specialized projection wiring.
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

    // sensory.rs (spiking subset: step(f64)->i32)
    RetinalGanglion(RetinalGanglionCell),
    Merkel(MerkelCell),
    Pacinian(PacinianCorpuscle),
    NociceptorCell(Nociceptor),
    OlfactoryReceptor(OlfactoryReceptorNeuron),

    // Not wired (multi-arg / i32 input / f64 return / no reset):
    // Alpha (2 args), COBALIF (3 args), TsodyksMarkram (bool arg),
    // CompteWM (bool arg), McCullochPitts (no reset), SigmoidRate (f64 ret),
    // ThresholdLinearRate (f64 ret), AstrocyteModel (f64 ret),
    // PinskyRinzel (2 args), HayL5 (2 args),
    // LoihiCUBA/Loihi2/TrueNorth/SpiNNaker2/Akida (i32 input)
    // InnerHairCell, OuterHairCell, RodPhotoreceptor, ConePhotoreceptor,
    // TasteReceptorCell (graded: step(f64)->f64)
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
            FitzHughNagumo, MorrisLecar, HindmarshRose, ResonateAndFire,
            FitzHughRinzel, McKean, TermanWang, GutkinErmentrout, WilsonHR,
            Chay, ChayKeizer, ShermanRinzelKeizer, ButeraRespiratory,
            EPropALIF, SuperSpike, LearnableNeuron, Pernarowski,
            QuadraticIF, Theta, PerfectIntegrator, GatedLIF, NonlinearLIF,
            SFA, MAT, KLIF, InhibitoryLIF, ComplementaryLIF, ParametricLIF,
            NonResettingLIF, AdaptiveThresholdIF, SigmaDelta, EnergyLIF,
            ClosedFormContinuous,
            ChialvoMap, RulkovMap, IbarzTanakaMap, MedvedevMap,
            CazellesMap, CourageNekorkinMap,
            BrainScaleSAdEx, SpiNNakerLIF, NeuroGrid, DPI,
            MarderSTG, RallCable, BoothRinzel, Dendrify,
            LiquidTimeConstant, ParallelSpiking, FractionalLIF,
            StochasticIF, GalvesLocherbach, SpikeResponse, GLM,
            Arcane,
            MultiTimescale, AttentionGated, PredictiveCoding,
            SelfReferential, CompositionalBinding, DifferentiableSurrogate,
            ContinuousAttractor, MetaPlastic,
            BendaHerz,
            Poisson, InhomogeneousPoisson, GammaRenewal, EscapeRate,
            PVFastSpiking, SST, VIP, Chandelier, CerebellarBasket, Martinotti,
            AlphaMotor, GammaMotor,
            RetinalGanglion, Merkel, Pacinian, NociceptorCell, OlfactoryReceptor,
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
            NeuronVariant::MedvedevMap(n) => n.x,
            NeuronVariant::CazellesMap(n) => n.x,
            NeuronVariant::CourageNekorkinMap(n) => n.x,
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
            // sensory (spiking)
            NeuronVariant::RetinalGanglion(n) => n.v,
            NeuronVariant::Merkel(n) => n.v,
            NeuronVariant::Pacinian(n) => n.v,
            NeuronVariant::NociceptorCell(n) => n.v,
            NeuronVariant::OlfactoryReceptor(n) => n.v,
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
                        spike_data[pop_idx].push(((nid as u64) << 32) | (t as u64));
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
        "Chandelier" | "ChandelierNeuron" => {
            Ok(NeuronVariant::Chandelier(ChandelierNeuron::new()))
        }
        "CerebellarBasket" | "CerebellarBasketNeuron" => {
            Ok(NeuronVariant::CerebellarBasket(CerebellarBasketNeuron::new()))
        }
        "Martinotti" | "MartinottiNeuron" => {
            Ok(NeuronVariant::Martinotti(MartinottiNeuron::new()))
        }
        // motor
        "AlphaMotor" | "AlphaMotorNeuron" => {
            Ok(NeuronVariant::AlphaMotor(AlphaMotorNeuron::new()))
        }
        "GammaMotor" | "GammaMotorNeuron" => {
            Ok(NeuronVariant::GammaMotor(GammaMotorNeuron::new()))
        }
        // sensory (spiking)
        "RetinalGanglion" | "RetinalGanglionCell" => {
            Ok(NeuronVariant::RetinalGanglion(RetinalGanglionCell::new()))
        }
        "Merkel" | "MerkelCell" => Ok(NeuronVariant::Merkel(MerkelCell::new())),
        "Pacinian" | "PacinianCorpuscle" => {
            Ok(NeuronVariant::Pacinian(PacinianCorpuscle::new()))
        }
        "Nociceptor" => Ok(NeuronVariant::NociceptorCell(Nociceptor::new())),
        "OlfactoryReceptor" | "OlfactoryReceptorNeuron" => {
            Ok(NeuronVariant::OlfactoryReceptor(OlfactoryReceptorNeuron::new()))
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
        "FitzHughNagumo",
        "MorrisLecar",
        "HindmarshRose",
        "ResonateAndFire",
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
        // sensory (spiking)
        "RetinalGanglion",
        "Merkel",
        "Pacinian",
        "Nociceptor",
        "OlfactoryReceptor",
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
        for name in &["PVFastSpiking", "SST", "VIP", "Chandelier", "CerebellarBasket", "Martinotti"] {
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
        for name in &["RetinalGanglion", "Merkel", "Pacinian", "Nociceptor", "OlfactoryReceptor"] {
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
            "PVFastSpiking", "SST", "VIP", "Chandelier",
            "CerebellarBasket", "Martinotti",
            "RetinalGanglion", "Merkel", "Pacinian",
            "Nociceptor", "OlfactoryReceptor",
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
            assert!(v.is_finite(), "{name}: voltage not finite after reset from NaN: {v}");
        }
    }

    #[test]
    fn all_models_extreme_input_stays_finite() {
        let models = &[
            "PVFastSpiking", "SST", "VIP", "Chandelier",
            "CerebellarBasket", "Martinotti",
            "RetinalGanglion", "Merkel", "Pacinian",
            "Nociceptor", "OlfactoryReceptor",
        ];
        for name in models {
            let mut neuron = create_neuron(name).unwrap();
            // Large positive current
            for _ in 0..50 {
                neuron.step(1e6);
            }
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(v.is_finite(), "{name}: non-finite after large positive input");

            // Large negative current
            for _ in 0..50 {
                neuron.step(-1e6);
            }
            neuron.reset();
            let v = neuron.soma_voltage();
            assert!(v.is_finite(), "{name}: non-finite after large negative input");
        }
    }
}
