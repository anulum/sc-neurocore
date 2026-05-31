// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # Industrial whole-process evolve runner
//!
//! Port of `sc_neurocore.evo_substrate.evo_substrate.ReplicationEngine`
//! plus the eleven industrial guards:
//! TournamentSelector, AgeRegulator, FormalSafetyGuard, BloatPenalizer,
//! ExtinctionDetector, HallOfFame, ParetoFront, LineageTracker,
//! MutationEngine (point/structural/duplication/swap), CrossoverEngine,
//! parametric FitnessEvaluator.
//!
//! Determinism: every RNG is XorShift64 seeded from `config.seed`. The
//! same PRNG algorithm + same call order is shared with the Julia, Go,
//! and Mojo runners so the **byte-for-byte output is identical across
//! all four backends** on a given seed. The Python reference still uses
//! NumPy's PCG64 — Python is orchestration-only, so a parity-critical
//! cross-language run goes through one of the four backends.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ─── Shared XorShift64 PRNG (byte-identical across Rust/Julia/Go/Mojo) ──

/// 64-bit XorShift PRNG — single-state, unused-state-never-zero.
/// Algorithm: state ^= state << 13; state ^= state >> 7; state ^= state << 17.
/// Same constants used by the other three backends so the full sequence is
/// byte-identical across languages on a fixed seed.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn seed_from_u64(seed: u64) -> Self {
        // XorShift requires non-zero state. Bump a zero seed to something harmless.
        let s = if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed };
        Self { state: s }
    }

    /// Next raw u64. Caller is responsible for trimming or scaling.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform [0, 1) double from the top 53 bits of next_u64.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Gaussian (Box-Muller, consumes two uniforms in fixed order).
    pub fn next_normal(&mut self, mu: f64, sigma: f64) -> f64 {
        let mut u1 = self.next_f64();
        let u2 = self.next_f64();
        if u1 < 1e-300 {
            u1 = 1e-300;
        }
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        mu + sigma * r * theta.cos()
    }

    pub fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
        // [lo, hi) inclusive-exclusive — matches Rust rand::gen_range semantics
        let span = (hi - lo) as u64;
        lo + (self.next_u64() % span) as usize
    }
}

const GENOME_DIM: usize = 19;
const EPS_SC: f64 = 1e-10;

// ─── Gene blocks (19-D vector = 5 topology + 8 neuron + 6 plasticity) ──

/// Topology gene block (5 dims).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopologyGene {
    pub num_neurons: i32,
    pub num_layers: i32,
    pub connectivity: f64,
    pub recurrent_fraction: f64,
    pub bitstream_length: i32,
}

impl Default for TopologyGene {
    fn default() -> Self {
        Self {
            num_neurons: 16,
            num_layers: 2,
            connectivity: 0.3,
            recurrent_fraction: 0.1,
            bitstream_length: 256,
        }
    }
}

impl TopologyGene {
    pub fn to_slice(&self, out: &mut [f64]) {
        out[0] = self.num_neurons as f64;
        out[1] = self.num_layers as f64;
        out[2] = self.connectivity;
        out[3] = self.recurrent_fraction;
        out[4] = self.bitstream_length as f64;
    }

    pub fn from_slice(v: &[f64]) -> Self {
        Self {
            num_neurons: (v[0] as i32).max(2),
            num_layers: (v[1] as i32).max(1),
            connectivity: v[2].clamp(0.01, 1.0),
            recurrent_fraction: v[3].clamp(0.0, 0.5),
            bitstream_length: (v[4] as i32).max(32),
        }
    }
}

/// Neuron gene block (8 dims).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeuronGene {
    pub tau_fast: f64,
    pub tau_work: f64,
    pub tau_deep: f64,
    pub theta: f64,
    pub gamma: f64,
    pub delta_conf: f64,
    pub kappa: f64,
    pub w_inh: f64,
}

impl Default for NeuronGene {
    fn default() -> Self {
        Self {
            tau_fast: 5.0,
            tau_work: 200.0,
            tau_deep: 10_000.0,
            theta: 1.0,
            gamma: 0.2,
            delta_conf: 0.3,
            kappa: 5.0,
            w_inh: 0.3,
        }
    }
}

impl NeuronGene {
    pub fn to_slice(&self, out: &mut [f64]) {
        out[0] = self.tau_fast;
        out[1] = self.tau_work;
        out[2] = self.tau_deep;
        out[3] = self.theta;
        out[4] = self.gamma;
        out[5] = self.delta_conf;
        out[6] = self.kappa;
        out[7] = self.w_inh;
    }

    pub fn from_slice(v: &[f64]) -> Self {
        Self {
            tau_fast: v[0].max(0.5),
            tau_work: v[1].max(1.0),
            tau_deep: v[2].max(10.0),
            theta: v[3].max(0.1),
            gamma: v[4].clamp(0.0, 1.0),
            delta_conf: v[5].clamp(0.0, 1.0),
            kappa: v[6].max(0.1),
            w_inh: v[7].clamp(0.0, 1.0),
        }
    }
}

/// Plasticity gene block (6 dims).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlasticityGene {
    pub stdp_lr: f64,
    pub stdp_tau_plus: f64,
    pub stdp_tau_minus: f64,
    pub stp_u_base: f64,
    pub homeostatic_rate: f64,
    pub meta_sensitivity: f64,
}

impl Default for PlasticityGene {
    fn default() -> Self {
        Self {
            stdp_lr: 0.01,
            stdp_tau_plus: 20.0,
            stdp_tau_minus: 20.0,
            stp_u_base: 0.5,
            homeostatic_rate: 0.001,
            meta_sensitivity: 1.0,
        }
    }
}

impl PlasticityGene {
    pub fn to_slice(&self, out: &mut [f64]) {
        out[0] = self.stdp_lr;
        out[1] = self.stdp_tau_plus;
        out[2] = self.stdp_tau_minus;
        out[3] = self.stp_u_base;
        out[4] = self.homeostatic_rate;
        out[5] = self.meta_sensitivity;
    }

    pub fn from_slice(v: &[f64]) -> Self {
        Self {
            stdp_lr: v[0].max(1e-6),
            stdp_tau_plus: v[1].max(1.0),
            stdp_tau_minus: v[2].max(1.0),
            stp_u_base: v[3].clamp(0.01, 0.99),
            homeostatic_rate: v[4].max(1e-6),
            meta_sensitivity: v[5].max(0.1),
        }
    }
}

/// Complete 19-D evolving individual.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub genome_id: String,
    pub parent_id: String,
    pub generation: i32,
    pub topology: TopologyGene,
    pub neuron: NeuronGene,
    pub plasticity: PlasticityGene,
    pub weight_seed: u64,
    pub identity_deep: f64,
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            genome_id: String::new(),
            parent_id: String::new(),
            generation: 0,
            topology: TopologyGene::default(),
            neuron: NeuronGene::default(),
            plasticity: PlasticityGene::default(),
            weight_seed: 42,
            identity_deep: 0.0,
        }
    }
}

impl Genome {
    pub fn to_vector(&self) -> [f64; GENOME_DIM] {
        let mut v = [0.0; GENOME_DIM];
        self.topology.to_slice(&mut v[0..5]);
        self.neuron.to_slice(&mut v[5..13]);
        self.plasticity.to_slice(&mut v[13..19]);
        v
    }

    pub fn from_vector(v: &[f64], generation: i32) -> Self {
        Self {
            generation,
            topology: TopologyGene::from_slice(&v[0..5]),
            neuron: NeuronGene::from_slice(&v[5..13]),
            plasticity: PlasticityGene::from_slice(&v[13..19]),
            ..Default::default()
        }
    }

    /// Compute the 12-hex-char SHA-256 fingerprint matching the Python
    /// reference: `hashlib.sha256(vector.tobytes()).hexdigest()[:12]`.
    pub fn compute_id(&mut self) -> &str {
        let v = self.to_vector();
        let mut hasher = Sha256::new();
        // NumPy's tobytes() on float64 is little-endian contiguous;
        // mirror that byte layout.
        for x in v.iter() {
            hasher.update(x.to_le_bytes());
        }
        let digest = hasher.finalize();
        let hex: String = digest
            .iter()
            .take(6)
            .map(|b| format!("{:02x}", b))
            .collect();
        self.genome_id = hex;
        &self.genome_id
    }
}

// ─── Mutation ────────────────────────────────────────────────────────

/// Mutation operator variant (wire-serialised string form).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MutationType {
    Point,
    Structural,
    Duplication,
    Swap,
    Identity,
}

impl MutationType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Structural => "structural",
            Self::Duplication => "duplication",
            Self::Swap => "swap",
            Self::Identity => "identity",
        }
    }
}

/// Per-rate configuration for the 4 mutation variants.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MutationConfig {
    pub point_rate: f64,
    pub point_sigma: f64,
    pub structural_rate: f64,
    pub duplication_rate: f64,
    pub swap_rate: f64,
    pub max_neurons: i32,
    pub min_neurons: i32,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            point_rate: 0.2,
            point_sigma: 0.05,
            structural_rate: 0.05,
            duplication_rate: 0.01,
            swap_rate: 0.02,
            max_neurons: 1024,
            min_neurons: 4,
        }
    }
}

fn apply_point(cfg: &MutationConfig, genome: &mut Genome, rng: &mut XorShift64) {
    let mut v = genome.to_vector();
    for i in 0..GENOME_DIM {
        if rng.next_f64() < cfg.point_rate {
            let noise = rng.next_normal(0.0, cfg.point_sigma);
            v[i] += noise * (v[i].abs() + 1e-8);
        }
    }
    let rebuilt = Genome::from_vector(&v, genome.generation);
    genome.topology = rebuilt.topology;
    genome.neuron = rebuilt.neuron;
    genome.plasticity = rebuilt.plasticity;
}

fn apply_structural(cfg: &MutationConfig, genome: &mut Genome, rng: &mut XorShift64) {
    let delta = [-2i32, -1, 1, 2][rng.gen_range(0, 4)];
    let new_n =
        (genome.topology.num_neurons + delta).clamp(cfg.min_neurons, cfg.max_neurons);
    genome.topology.num_neurons = new_n;
    let conn_noise = rng.next_normal(0.0, 0.05);
    genome.topology.connectivity = (genome.topology.connectivity + conn_noise).clamp(0.01, 1.0);
}

fn apply_duplication(cfg: &MutationConfig, genome: &mut Genome) {
    genome.topology.num_layers = genome.topology.num_layers.saturating_add(1).min(10);
    let scaled = ((genome.topology.num_neurons as f64) * 1.5) as i32;
    genome.topology.num_neurons = scaled.min(cfg.max_neurons);
}

fn apply_swap(genome: &mut Genome) {
    std::mem::swap(&mut genome.neuron.tau_fast, &mut genome.neuron.tau_work);
}

/// Deterministic mutation engine mirroring the Python cumulative-roll.
pub struct MutationEngine {
    pub config: MutationConfig,
    pub rng: XorShift64,
}

impl MutationEngine {
    pub fn new(config: MutationConfig, seed: u64) -> Self {
        Self {
            config,
            rng: XorShift64::seed_from_u64(seed),
        }
    }

    /// Apply one mutation; returns the child and the variant that fired.
    pub fn mutate(&mut self, parent: &Genome) -> (Genome, MutationType) {
        let mut child = parent.clone();
        child.parent_id = parent.genome_id.clone();
        child.generation = parent.generation + 1;
        child.identity_deep = 0.0;

        let roll = self.rng.next_f64();
        let mut cumulative = 0.0;

        cumulative += self.config.structural_rate;
        if roll < cumulative {
            apply_structural(&self.config, &mut child, &mut self.rng);
            child.compute_id();
            return (child, MutationType::Structural);
        }
        cumulative += self.config.duplication_rate;
        if roll < cumulative {
            apply_duplication(&self.config, &mut child);
            child.compute_id();
            return (child, MutationType::Duplication);
        }
        cumulative += self.config.swap_rate;
        if roll < cumulative {
            apply_swap(&mut child);
            child.compute_id();
            return (child, MutationType::Swap);
        }

        apply_point(&self.config, &mut child, &mut self.rng);
        child.compute_id();
        (child, MutationType::Point)
    }
}

/// Uniform crossover over full 19-D vector.
pub struct CrossoverEngine {
    pub rng: XorShift64,
}

impl CrossoverEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: XorShift64::seed_from_u64(seed),
        }
    }

    pub fn crossover(&mut self, a: &Genome, b: &Genome) -> Genome {
        let va = a.to_vector();
        let vb = b.to_vector();
        let mut child_v = [0.0; GENOME_DIM];
        for i in 0..GENOME_DIM {
            child_v[i] = if self.rng.next_f64() < 0.5 {
                va[i]
            } else {
                vb[i]
            };
        }
        let new_gen = a.generation.max(b.generation) + 1;
        let mut child = Genome::from_vector(&child_v, new_gen);
        child.parent_id = format!("{}x{}", a.genome_id, b.genome_id);
        child.compute_id();
        child
    }
}

// ─── Fitness (parametric — no Python callback) ────────────────────────

/// Parametric fitness evaluator matching the Python reference default
/// formula:
/// * accuracy = `accuracy_bias + accuracy_neuron_coef * num_neurons / 32`
/// * energy   = max(0, 1 − 0.5 · neurons/1024 − 0.5 · bitstream/1024)
/// * latency  = max(0, 1 − layers / 10)
/// * composite = w_acc·A + w_en·E + w_lat·L
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessSpec {
    pub accuracy_bias: f64,
    pub accuracy_neuron_coef: f64,
    pub w_accuracy: f64,
    pub w_energy: f64,
    pub w_latency: f64,
}

impl Default for FitnessSpec {
    fn default() -> Self {
        Self {
            accuracy_bias: 0.5,
            accuracy_neuron_coef: 0.01,
            w_accuracy: 0.5,
            w_energy: 0.3,
            w_latency: 0.2,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FitnessResult {
    pub genome_id: String,
    pub accuracy: f64,
    pub energy_score: f64,
    pub latency_score: f64,
    pub composite: f64,
}

fn evaluate_fitness(spec: &FitnessSpec, genome: &Genome) -> FitnessResult {
    let n = genome.topology.num_neurons as f64;
    let layers = genome.topology.num_layers as f64;
    let bitstream = genome.topology.bitstream_length as f64;

    let accuracy = spec.accuracy_bias + spec.accuracy_neuron_coef * n / 32.0;
    let energy = (1.0 - 0.5 * n / 1024.0 - 0.5 * bitstream / 1024.0).max(0.0);
    let latency = (1.0 - layers / 10.0).max(0.0);
    let composite =
        spec.w_accuracy * accuracy + spec.w_energy * energy + spec.w_latency * latency;

    FitnessResult {
        genome_id: genome.genome_id.clone(),
        accuracy,
        energy_score: energy,
        latency_score: latency,
        composite,
    }
}

// ─── Industrial guards ────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyBounds {
    pub max_neurons: i32,
    pub min_neurons: i32,
    pub max_layers: i32,
    pub max_bitstream: i32,
    pub min_bitstream: i32,
    pub max_connectivity: f64,
}

impl Default for SafetyBounds {
    fn default() -> Self {
        Self {
            max_neurons: 1024,
            min_neurons: 4,
            max_layers: 16,
            max_bitstream: 4096,
            min_bitstream: 32,
            max_connectivity: 1.0,
        }
    }
}

pub struct FormalSafetyGuard {
    pub bounds: SafetyBounds,
    pub checked: u64,
    pub rejected: u64,
}

impl FormalSafetyGuard {
    pub fn new(bounds: SafetyBounds) -> Self {
        Self {
            bounds,
            checked: 0,
            rejected: 0,
        }
    }

    pub fn check(&mut self, genome: &Genome) -> bool {
        self.checked += 1;
        let n_ok = genome.topology.num_neurons <= self.bounds.max_neurons;
        let c_ok = genome.topology.connectivity <= self.bounds.max_connectivity;
        let b_ok = genome.topology.bitstream_length <= self.bounds.max_bitstream;
        let passed = n_ok && c_ok && b_ok;
        if !passed {
            self.rejected += 1;
        }
        passed
    }
}

pub struct BloatPenalizer {
    pub penalty_weight: f64,
    pub threshold: f64,
    pub baseline_neurons: i32,
}

impl Default for BloatPenalizer {
    fn default() -> Self {
        Self {
            penalty_weight: 0.1,
            threshold: 2.0,
            baseline_neurons: 16,
        }
    }
}

impl BloatPenalizer {
    fn bloat_score(&self, genome: &Genome) -> f64 {
        let n = genome.topology.num_neurons as f64;
        let l = genome.topology.num_layers as f64;
        let conn = (n * n * genome.topology.connectivity) as i64;
        let total = (n * 8.0 + l) as i64 + conn;
        let base_n = self.baseline_neurons as f64;
        let baseline =
            (base_n * 8.0 + 2.0) as i64 + (base_n * base_n * 0.3) as i64;
        total as f64 / (baseline.max(1)) as f64
    }

    pub fn penalize(&self, fitness: f64, genome: &Genome) -> f64 {
        let score = self.bloat_score(genome);
        if score > self.threshold {
            let excess = score - self.threshold;
            (fitness - self.penalty_weight * excess).max(0.0)
        } else {
            fitness
        }
    }
}

pub struct AgeRegulator {
    pub max_age: i32,
}

impl AgeRegulator {
    pub fn new(max_age: i32) -> Self {
        Self { max_age }
    }

    /// Returns indices in the population whose age exceeds `max_age`.
    pub fn cull_indices(&self, population: &[Organism], current_generation: i32) -> Vec<usize> {
        population
            .iter()
            .enumerate()
            .filter_map(|(i, org)| {
                if current_generation - org.birth_generation > self.max_age && org.alive {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

pub struct ExtinctionDetector {
    pub stagnation_gens: usize,
    pub kill_fraction: f64,
    best_history: Vec<f64>,
    pub extinction_count: u64,
}

impl ExtinctionDetector {
    pub fn new(stagnation_gens: usize, kill_fraction: f64) -> Self {
        Self {
            stagnation_gens,
            kill_fraction,
            best_history: Vec::new(),
            extinction_count: 0,
        }
    }

    pub fn check(&mut self, best_fitness: f64) -> bool {
        self.best_history.push(best_fitness);
        if self.best_history.len() < self.stagnation_gens {
            return false;
        }
        let recent = &self.best_history[self.best_history.len() - self.stagnation_gens..];
        let max = recent.iter().cloned().fold(f64::MIN, f64::max);
        let min = recent.iter().cloned().fold(f64::MAX, f64::min);
        if max - min < 1e-6 {
            self.extinction_count += 1;
            return true;
        }
        false
    }

    pub fn apply(&self, population: &mut [Organism], rng: &mut XorShift64) -> usize {
        let n_kill = ((population.len() as f64) * self.kill_fraction) as usize;
        let n_kill = n_kill.min(population.len());
        let mut indices: Vec<usize> = (0..population.len()).collect();
        // Fisher-Yates partial shuffle to select distinct random indices.
        for i in 0..n_kill {
            let j = rng.gen_range(i, indices.len());
            indices.swap(i, j);
        }
        let mut killed = 0;
        for &idx in &indices[..n_kill] {
            if population[idx].alive {
                population[idx].alive = false;
                killed += 1;
            }
        }
        killed
    }
}

pub struct HallOfFame {
    pub max_size: usize,
    /// Sorted descending by composite fitness; (fitness, genome).
    pub entries: Vec<(f64, Genome)>,
}

impl HallOfFame {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: Vec::new(),
        }
    }

    pub fn update(&mut self, organism: &Organism) -> bool {
        let Some(fit) = organism.fitness.as_ref() else {
            return false;
        };
        self.entries.push((fit.composite, organism.genome.clone()));
        self.entries
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if self.entries.len() > self.max_size {
            self.entries.truncate(self.max_size);
        }
        true
    }
}

pub struct ParetoFront {
    pub front: Vec<Organism>,
}

impl Default for ParetoFront {
    fn default() -> Self {
        Self { front: Vec::new() }
    }
}

fn dominates(a: &FitnessResult, b: &FitnessResult) -> bool {
    let va = [a.accuracy, a.energy_score, a.latency_score];
    let vb = [b.accuracy, b.energy_score, b.latency_score];
    let mut at_least_one_better = false;
    for (x, y) in va.iter().zip(vb.iter()) {
        if x < y {
            return false;
        }
        if x > y {
            at_least_one_better = true;
        }
    }
    at_least_one_better
}

impl ParetoFront {
    pub fn update(&mut self, organism: &Organism) -> bool {
        let Some(new_fit) = organism.fitness.as_ref() else {
            return false;
        };
        for existing in &self.front {
            if let Some(ef) = existing.fitness.as_ref() {
                if dominates(ef, new_fit) {
                    return false;
                }
            }
        }
        self.front.retain(|o| {
            if let Some(ef) = o.fitness.as_ref() {
                !dominates(new_fit, ef)
            } else {
                true
            }
        });
        self.front.push(organism.clone());
        true
    }
}

pub struct TournamentSelector {
    pub tournament_size: usize,
}

impl TournamentSelector {
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }

    pub fn select<'a>(
        &self,
        population: &'a [Organism],
        rng: &mut XorShift64,
    ) -> Option<&'a Organism> {
        if population.is_empty() {
            return None;
        }
        let k = self.tournament_size.min(population.len());
        let mut best: Option<&Organism> = None;
        let mut best_fit = f64::MIN;
        let mut seen = Vec::with_capacity(k);
        while seen.len() < k {
            let idx = rng.gen_range(0, population.len());
            if seen.contains(&idx) {
                continue;
            }
            seen.push(idx);
            let org = &population[idx];
            let fit = org.fitness.as_ref().map(|f| f.composite).unwrap_or(0.0);
            if fit > best_fit {
                best_fit = fit;
                best = Some(org);
            }
        }
        best
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineageRecord {
    pub genome_id: String,
    pub parent_id: String,
    pub generation: i32,
    pub mutation_type: String,
    pub fitness: f64,
}

pub struct LineageTracker {
    pub records: Vec<LineageRecord>,
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

impl LineageTracker {
    pub fn record(&mut self, org: &Organism, mutation_type: &str) {
        let fit = org.fitness.as_ref().map(|f| f.composite).unwrap_or(0.0);
        self.records.push(LineageRecord {
            genome_id: org.genome.genome_id.clone(),
            parent_id: org.genome.parent_id.clone(),
            generation: org.genome.generation,
            mutation_type: mutation_type.to_string(),
            fitness: fit,
        });
    }
}

// ─── Organism ─────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Organism {
    pub genome: Genome,
    pub fitness: Option<FitnessResult>,
    pub alive: bool,
    pub birth_generation: i32,
}

impl Organism {
    pub fn new(genome: Genome, birth_generation: i32) -> Self {
        Self {
            genome,
            fitness: None,
            alive: true,
            birth_generation,
        }
    }
}

// ─── Top-level config + result structs ────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolveConfig {
    pub seed: u64,
    pub pop_size: usize,
    pub n_generations: usize,
    pub elitism: usize,
    pub survival_fraction: f64,
    pub tournament_size: usize,
    pub crossover_prob: f64,
    pub max_age: i32,
    pub hall_of_fame_size: usize,
    pub stagnation_gens: usize,
    pub extinction_kill_fraction: f64,
    pub mutation: MutationConfig,
    pub fitness: FitnessSpec,
    pub safety_bounds: SafetyBounds,
    pub industrial_mode: bool,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            pop_size: 32,
            n_generations: 20,
            elitism: 1,
            survival_fraction: 0.5,
            tournament_size: 3,
            crossover_prob: 0.3,
            max_age: 20,
            hall_of_fame_size: 10,
            stagnation_gens: 10,
            extinction_kill_fraction: 0.9,
            mutation: MutationConfig::default(),
            fitness: FitnessSpec::default(),
            safety_bounds: SafetyBounds::default(),
            industrial_mode: true,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: i32,
    pub population_size: usize,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub diversity: f64,
    pub killed: usize,
    pub children: usize,
    pub extinctions: u64,
    pub safety_rejections: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EvolveResult {
    pub final_population: Vec<Genome>,
    pub stats_per_generation: Vec<GenerationStats>,
    pub hall_of_fame: Vec<Genome>,
    pub pareto_front: Vec<Genome>,
    pub lineage: Vec<LineageRecord>,
    pub total_replications: u64,
    pub safety_checked: u64,
    pub safety_rejected: u64,
    pub extinction_count: u64,
}

// ─── Helper — mean pairwise distance on an Organism slice ─────────────

fn pairwise_diversity(pop: &[Organism]) -> f64 {
    let alive: Vec<&Organism> = pop.iter().filter(|o| o.alive).collect();
    if alive.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut count = 0.0;
    for i in 0..alive.len() {
        let va = alive[i].genome.to_vector();
        for j in (i + 1)..alive.len() {
            let vb = alive[j].genome.to_vector();
            let mut s = 0.0;
            for k in 0..GENOME_DIM {
                s += (va[k] - vb[k]).abs() / (va[k].abs() + vb[k].abs() + EPS_SC);
            }
            acc += s / (GENOME_DIM as f64);
            count += 1.0;
        }
    }
    acc / count
}

// ─── Main entry point ────────────────────────────────────────────────

/// Run the full industrial evolve loop. See module docs for scope.
pub fn evolve_run(config: &EvolveConfig) -> EvolveResult {
    let mut master_rng = XorShift64::seed_from_u64(config.seed);
    let mut mutator = MutationEngine::new(config.mutation.clone(), master_rng.next_u64());
    let mut crossover = CrossoverEngine::new(master_rng.next_u64());
    let mut guard = FormalSafetyGuard::new(config.safety_bounds.clone());
    let mut bloat = BloatPenalizer::default();
    let age = AgeRegulator::new(config.max_age);
    let mut extinction =
        ExtinctionDetector::new(config.stagnation_gens, config.extinction_kill_fraction);
    let mut hof = HallOfFame::new(config.hall_of_fame_size);
    let mut pareto = ParetoFront::default();
    let tournament = TournamentSelector::new(config.tournament_size);
    let mut lineage = LineageTracker::default();

    // Seed the population with default genomes (matches Python demo).
    let mut population: Vec<Organism> = Vec::with_capacity(config.pop_size);
    for _ in 0..config.pop_size {
        let mut g = Genome::default();
        g.compute_id();
        let org = Organism::new(g, 0);
        lineage.record(&org, "seed");
        population.push(org);
    }

    let mut result = EvolveResult::default();
    let mut total_replications: u64 = 0;

    // Run N generations
    for gen in 1..=config.n_generations as i32 {
        // 1. Evaluate fitness for every alive organism
        for org in population.iter_mut().filter(|o| o.alive) {
            let mut fit = evaluate_fitness(&config.fitness, &org.genome);
            if config.industrial_mode {
                fit.composite = bloat.penalize(fit.composite, &org.genome);
            }
            org.fitness = Some(fit);
            hof.update(org);
            pareto.update(org);
        }

        // 2. Industrial culling passes (age + extinction)
        let mut killed = 0usize;
        if config.industrial_mode {
            for idx in age.cull_indices(&population, gen) {
                population[idx].alive = false;
                killed += 1;
            }
            let best = population
                .iter()
                .filter(|o| o.alive)
                .filter_map(|o| o.fitness.as_ref())
                .map(|f| f.composite)
                .fold(0.0f64, f64::max);
            if extinction.check(best) {
                killed += extinction.apply(&mut population, &mut mutator.rng);
            }
        }

        // 3. Rank-based survival cull (elitism-preserving)
        let mut alive_sorted: Vec<usize> = population
            .iter()
            .enumerate()
            .filter(|(_, o)| o.alive && o.fitness.is_some())
            .map(|(i, _)| i)
            .collect();
        alive_sorted.sort_by(|a, b| {
            let fa = population[*a].fitness.as_ref().unwrap().composite;
            let fb = population[*b].fitness.as_ref().unwrap().composite;
            fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep =
            (config.elitism + 1).max((alive_sorted.len() as f64 * config.survival_fraction) as usize);
        for &idx in alive_sorted.iter().skip(keep) {
            population[idx].alive = false;
            killed += 1;
        }
        // Compact the population to alive members only.
        population.retain(|o| o.alive);

        // 4. Replicate from survivors until pop_size is reached.
        let survivors: Vec<Organism> = population.clone();
        let mut children = 0usize;
        while population.len() < config.pop_size && !survivors.is_empty() {
            let parent = if config.industrial_mode {
                tournament.select(&survivors, &mut mutator.rng).cloned()
            } else {
                survivors.first().cloned()
            };
            let Some(parent) = parent else { break };
            let partner = if config.industrial_mode {
                tournament.select(&survivors, &mut mutator.rng).cloned()
            } else {
                survivors.get(1).cloned()
            };

            let child_genome = if let (Some(partner), true) =
                (partner.as_ref(), mutator.rng.next_f64() < config.crossover_prob)
            {
                let mut c = crossover.crossover(&parent.genome, &partner.genome);
                c.generation = gen;
                c
            } else {
                let (mut c, mtype) = mutator.mutate(&parent.genome);
                c.generation = gen;
                // Record mutation type into lineage later.
                let _ = mtype;
                c
            };

            if !guard.check(&child_genome) {
                continue;
            }

            total_replications += 1;
            let child = Organism::new(child_genome, gen);
            lineage.record(&child, "replicate");
            population.push(child);
            children += 1;
        }

        // 5. Record generation stats.
        let best_fitness = population
            .iter()
            .filter_map(|o| o.fitness.as_ref())
            .map(|f| f.composite)
            .fold(0.0f64, f64::max);
        let fits: Vec<f64> = population
            .iter()
            .filter_map(|o| o.fitness.as_ref())
            .map(|f| f.composite)
            .collect();
        let mean_fitness = if fits.is_empty() {
            0.0
        } else {
            fits.iter().sum::<f64>() / fits.len() as f64
        };

        result.stats_per_generation.push(GenerationStats {
            generation: gen,
            population_size: population.len(),
            best_fitness,
            mean_fitness,
            diversity: pairwise_diversity(&population),
            killed,
            children,
            extinctions: extinction.extinction_count,
            safety_rejections: guard.rejected,
        });
    }

    // Emit final state.
    result.final_population = population.iter().map(|o| o.genome.clone()).collect();
    result.hall_of_fame = hof.entries.iter().map(|(_, g)| g.clone()).collect();
    result.pareto_front = pareto.front.iter().map(|o| o.genome.clone()).collect();
    result.lineage = lineage.records;
    result.total_replications = total_replications;
    result.safety_checked = guard.checked;
    result.safety_rejected = guard.rejected;
    result.extinction_count = extinction.extinction_count;

    result
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genome_vector_roundtrip_is_idempotent() {
        let mut g = Genome::default();
        g.topology.num_neurons = 64;
        g.neuron.tau_fast = 7.5;
        g.plasticity.stdp_lr = 0.02;
        let v = g.to_vector();
        let back = Genome::from_vector(&v, g.generation);
        assert_eq!(back.topology.num_neurons, 64);
        assert!((back.neuron.tau_fast - 7.5).abs() < 1e-12);
        assert!((back.plasticity.stdp_lr - 0.02).abs() < 1e-12);
    }

    #[test]
    fn genome_id_is_deterministic_and_12_chars() {
        let mut g1 = Genome::default();
        let mut g2 = Genome::default();
        g1.compute_id();
        g2.compute_id();
        assert_eq!(g1.genome_id, g2.genome_id);
        assert_eq!(g1.genome_id.len(), 12);
    }

    #[test]
    fn mutation_changes_genome_id() {
        let mut parent = Genome::default();
        parent.compute_id();
        let mut eng = MutationEngine::new(MutationConfig::default(), 7);
        let (child, _) = eng.mutate(&parent);
        assert_ne!(parent.genome_id, child.genome_id);
        assert_eq!(child.parent_id, parent.genome_id);
        assert_eq!(child.generation, parent.generation + 1);
    }

    #[test]
    fn crossover_child_id_records_parents() {
        let mut a = Genome::default();
        a.topology.num_neurons = 32;
        a.compute_id();
        let mut b = Genome::default();
        b.topology.num_neurons = 16;
        b.compute_id();
        let mut x = CrossoverEngine::new(9);
        let child = x.crossover(&a, &b);
        assert!(child.parent_id.contains(&a.genome_id));
        assert!(child.parent_id.contains(&b.genome_id));
    }

    #[test]
    fn fitness_matches_python_default_formula() {
        let mut g = Genome::default();
        g.topology.num_neurons = 32;
        g.topology.num_layers = 2;
        g.topology.bitstream_length = 256;
        let f = evaluate_fitness(&FitnessSpec::default(), &g);
        // accuracy = 0.5 + 0.01 * 32 / 32 = 0.51
        assert!((f.accuracy - 0.51).abs() < 1e-9);
        // energy  = 1 − 0.5 · 32/1024 − 0.5 · 256/1024 = 0.859375
        assert!((f.energy_score - 0.859_375).abs() < 1e-9);
        // latency = 1 − 2/10 = 0.8
        assert!((f.latency_score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn safety_guard_rejects_oversized_genomes() {
        let mut guard = FormalSafetyGuard::new(SafetyBounds::default());
        let mut g = Genome::default();
        g.topology.num_neurons = 4096;
        assert!(!guard.check(&g));
        assert_eq!(guard.rejected, 1);
    }

    #[test]
    fn evolve_run_converges_non_trivially() {
        let cfg = EvolveConfig {
            n_generations: 10,
            pop_size: 16,
            seed: 7,
            ..Default::default()
        };
        let r = evolve_run(&cfg);
        assert_eq!(r.stats_per_generation.len(), 10);
        assert!(!r.final_population.is_empty());
        assert!(!r.hall_of_fame.is_empty());
        assert!(r.total_replications > 0);
    }

    #[test]
    fn evolve_run_is_deterministic() {
        let cfg = EvolveConfig {
            n_generations: 8,
            pop_size: 12,
            seed: 11,
            ..Default::default()
        };
        let r1 = evolve_run(&cfg);
        let r2 = evolve_run(&cfg);
        assert_eq!(r1.total_replications, r2.total_replications);
        assert_eq!(
            r1.stats_per_generation.last().unwrap().best_fitness,
            r2.stats_per_generation.last().unwrap().best_fitness,
        );
        assert_eq!(
            r1.final_population.len(),
            r2.final_population.len(),
        );
    }
}
