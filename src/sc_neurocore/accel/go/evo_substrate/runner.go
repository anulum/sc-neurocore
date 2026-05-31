// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
//
// SC-NeuroCore — Go industrial whole-process evolve runner.
//
// Port of `crates/evo_substrate_core/src/runner.rs` to Go.
// Same industrial guards (TournamentSelector, AgeRegulator,
// FormalSafetyGuard, BloatPenalizer, ExtinctionDetector, HallOfFame,
// ParetoFront, LineageTracker, MutationEngine × 4 variants,
// CrossoverEngine, parametric FitnessEvaluator). Same 19-D genome +
// SHA-256 id (12 hex chars). Same JSON wire contract.
//
// Build via `go build -o evo_runner runner.go` from this directory;
// the resulting binary reads an EvolveConfig JSON on stdin and writes
// an EvolveResult JSON on stdout.

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

const genomeDim = 19
const epsSC = 1e-10


// ─── Shared XorShift64 PRNG (byte-identical across Rust/Julia/Go/Mojo) ──

type XorShift64 struct {
	state uint64
}

func newXorShift64(seed uint64) *XorShift64 {
	s := seed
	if s == 0 {
		s = 0xDEADBEEFCAFEBABE
	}
	return &XorShift64{state: s}
}

func (r *XorShift64) nextU64() uint64 {
	x := r.state
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	r.state = x
	return x
}

func (r *XorShift64) nextF64() float64 {
	return float64(r.nextU64()>>11) / float64(uint64(1)<<53)
}

func (r *XorShift64) nextNormal(mu, sigma float64) float64 {
	u1 := r.nextF64()
	u2 := r.nextF64()
	if u1 < 1e-300 {
		u1 = 1e-300
	}
	radius := math.Sqrt(-2.0 * math.Log(u1))
	theta := 2.0 * math.Pi * u2
	return mu + sigma*radius*math.Cos(theta)
}

func (r *XorShift64) genRange(lo, hi int) int {
	span := uint64(hi - lo)
	return lo + int(r.nextU64()%span)
}

// ─── Gene blocks + Genome ─────────────────────────────────────────

type TopologyGene struct {
	NumNeurons        int32   `json:"num_neurons"`
	NumLayers         int32   `json:"num_layers"`
	Connectivity      float64 `json:"connectivity"`
	RecurrentFraction float64 `json:"recurrent_fraction"`
	BitstreamLength   int32   `json:"bitstream_length"`
}

type NeuronGene struct {
	TauFast   float64 `json:"tau_fast"`
	TauWork   float64 `json:"tau_work"`
	TauDeep   float64 `json:"tau_deep"`
	Theta     float64 `json:"theta"`
	Gamma     float64 `json:"gamma"`
	DeltaConf float64 `json:"delta_conf"`
	Kappa     float64 `json:"kappa"`
	WInh      float64 `json:"w_inh"`
}

type PlasticityGene struct {
	StdpLr         float64 `json:"stdp_lr"`
	StdpTauPlus    float64 `json:"stdp_tau_plus"`
	StdpTauMinus   float64 `json:"stdp_tau_minus"`
	StpUBase       float64 `json:"stp_u_base"`
	HomeostaticRate float64 `json:"homeostatic_rate"`
	MetaSensitivity float64 `json:"meta_sensitivity"`
}

type Genome struct {
	GenomeID      string         `json:"genome_id"`
	ParentID      string         `json:"parent_id"`
	Generation    int32          `json:"generation"`
	Topology      TopologyGene   `json:"topology,omitempty"`
	Neuron        NeuronGene     `json:"neuron,omitempty"`
	Plasticity    PlasticityGene `json:"plasticity,omitempty"`
	WeightSeed    uint64         `json:"weight_seed"`
	IdentityDeep  float64        `json:"identity_deep"`
	// Flat fields for the JSON result (mirrors Rust output shape).
	NumNeurons      int32   `json:"num_neurons,omitempty"`
	NumLayers       int32   `json:"num_layers,omitempty"`
	ConnectivityOut float64 `json:"connectivity,omitempty"`
	BitstreamOut    int32   `json:"bitstream_length,omitempty"`
	TauFastOut      float64 `json:"tau_fast,omitempty"`
	TauWorkOut      float64 `json:"tau_work,omitempty"`
	TauDeepOut      float64 `json:"tau_deep,omitempty"`
}

func defaultTopology() TopologyGene {
	return TopologyGene{NumNeurons: 16, NumLayers: 2, Connectivity: 0.3,
		RecurrentFraction: 0.1, BitstreamLength: 256}
}

func defaultNeuron() NeuronGene {
	return NeuronGene{TauFast: 5.0, TauWork: 200.0, TauDeep: 10000.0, Theta: 1.0,
		Gamma: 0.2, DeltaConf: 0.3, Kappa: 5.0, WInh: 0.3}
}

func defaultPlasticity() PlasticityGene {
	return PlasticityGene{StdpLr: 0.01, StdpTauPlus: 20.0, StdpTauMinus: 20.0,
		StpUBase: 0.5, HomeostaticRate: 0.001, MetaSensitivity: 1.0}
}

func defaultGenome() Genome {
	return Genome{
		Topology:     defaultTopology(),
		Neuron:       defaultNeuron(),
		Plasticity:   defaultPlasticity(),
		WeightSeed:   42,
		IdentityDeep: 0.0,
	}
}

func (g *Genome) toVector() [genomeDim]float64 {
	return [genomeDim]float64{
		float64(g.Topology.NumNeurons), float64(g.Topology.NumLayers),
		g.Topology.Connectivity, g.Topology.RecurrentFraction,
		float64(g.Topology.BitstreamLength),
		g.Neuron.TauFast, g.Neuron.TauWork, g.Neuron.TauDeep, g.Neuron.Theta,
		g.Neuron.Gamma, g.Neuron.DeltaConf, g.Neuron.Kappa, g.Neuron.WInh,
		g.Plasticity.StdpLr, g.Plasticity.StdpTauPlus, g.Plasticity.StdpTauMinus,
		g.Plasticity.StpUBase, g.Plasticity.HomeostaticRate, g.Plasticity.MetaSensitivity,
	}
}

func clampInt32(x, lo, hi int32) int32 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func clampF64(x, lo, hi float64) float64 {
	if x < lo {
		return lo
	}
	if x > hi {
		return hi
	}
	return x
}

func fromVector(v *[genomeDim]float64, generation int32) Genome {
	g := defaultGenome()
	g.Generation = generation
	g.Topology = TopologyGene{
		NumNeurons:        int32(math.Max(2, math.Floor(v[0]))),
		NumLayers:         int32(math.Max(1, math.Floor(v[1]))),
		Connectivity:      clampF64(v[2], 0.01, 1.0),
		RecurrentFraction: clampF64(v[3], 0.0, 0.5),
		BitstreamLength:   int32(math.Max(32, math.Floor(v[4]))),
	}
	g.Neuron = NeuronGene{
		TauFast: math.Max(0.5, v[5]), TauWork: math.Max(1.0, v[6]),
		TauDeep: math.Max(10.0, v[7]), Theta: math.Max(0.1, v[8]),
		Gamma: clampF64(v[9], 0.0, 1.0), DeltaConf: clampF64(v[10], 0.0, 1.0),
		Kappa: math.Max(0.1, v[11]), WInh: clampF64(v[12], 0.0, 1.0),
	}
	g.Plasticity = PlasticityGene{
		StdpLr: math.Max(1e-6, v[13]), StdpTauPlus: math.Max(1.0, v[14]),
		StdpTauMinus: math.Max(1.0, v[15]),
		StpUBase:     clampF64(v[16], 0.01, 0.99),
		HomeostaticRate: math.Max(1e-6, v[17]),
		MetaSensitivity: math.Max(0.1, v[18]),
	}
	return g
}

func (g *Genome) computeID() string {
	v := g.toVector()
	buf := make([]byte, genomeDim*8)
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(x))
	}
	sum := sha256.Sum256(buf)
	g.GenomeID = hex.EncodeToString(sum[:6])
	return g.GenomeID
}

// ─── Mutation ─────────────────────────────────────────────────────

type MutationConfig struct {
	PointRate        float64 `json:"point_rate"`
	PointSigma       float64 `json:"point_sigma"`
	StructuralRate   float64 `json:"structural_rate"`
	DuplicationRate  float64 `json:"duplication_rate"`
	SwapRate         float64 `json:"swap_rate"`
	MaxNeurons       int32   `json:"max_neurons"`
	MinNeurons       int32   `json:"min_neurons"`
}

func applyPoint(cfg *MutationConfig, g *Genome, rng *XorShift64) {
	v := g.toVector()
	for i := 0; i < genomeDim; i++ {
		if rng.nextF64() < cfg.PointRate {
			noise := rng.nextNormal(0.0, cfg.PointSigma)
			v[i] += noise * (math.Abs(v[i]) + 1e-8)
		}
	}
	rebuilt := fromVector(&v, g.Generation)
	g.Topology = rebuilt.Topology
	g.Neuron = rebuilt.Neuron
	g.Plasticity = rebuilt.Plasticity
}

func applyStructural(cfg *MutationConfig, g *Genome, rng *XorShift64) {
	deltas := []int32{-2, -1, 1, 2}
	delta := deltas[rng.genRange(0, 4)]
	g.Topology.NumNeurons = clampInt32(g.Topology.NumNeurons+delta, cfg.MinNeurons, cfg.MaxNeurons)
	g.Topology.Connectivity = clampF64(g.Topology.Connectivity+rng.nextNormal(0.0, 0.05), 0.01, 1.0)
}

func applyDuplication(cfg *MutationConfig, g *Genome) {
	g.Topology.NumLayers = clampInt32(g.Topology.NumLayers+1, 1, 10)
	scaled := int32(math.Floor(float64(g.Topology.NumNeurons) * 1.5))
	if scaled > cfg.MaxNeurons {
		scaled = cfg.MaxNeurons
	}
	g.Topology.NumNeurons = scaled
}

func applySwap(g *Genome) {
	g.Neuron.TauFast, g.Neuron.TauWork = g.Neuron.TauWork, g.Neuron.TauFast
}

type MutationEngine struct {
	Config MutationConfig
	RNG    *XorShift64
}

func newMutationEngine(cfg MutationConfig, seed uint64) *MutationEngine {
	return &MutationEngine{Config: cfg, RNG: newXorShift64(seed)}
}

func deepCopyGenome(g *Genome) Genome {
	return Genome{
		GenomeID:     g.GenomeID,
		ParentID:     g.ParentID,
		Generation:   g.Generation,
		Topology:     g.Topology,
		Neuron:       g.Neuron,
		Plasticity:   g.Plasticity,
		WeightSeed:   g.WeightSeed,
		IdentityDeep: g.IdentityDeep,
	}
}

func (e *MutationEngine) mutate(parent *Genome) (Genome, string) {
	child := deepCopyGenome(parent)
	child.ParentID = parent.GenomeID
	child.Generation = parent.Generation + 1
	child.IdentityDeep = 0.0

	roll := e.RNG.nextF64()
	cumulative := 0.0

	cumulative += e.Config.StructuralRate
	if roll < cumulative {
		applyStructural(&e.Config, &child, e.RNG)
		child.computeID()
		return child, "structural"
	}
	cumulative += e.Config.DuplicationRate
	if roll < cumulative {
		applyDuplication(&e.Config, &child)
		child.computeID()
		return child, "duplication"
	}
	cumulative += e.Config.SwapRate
	if roll < cumulative {
		applySwap(&child)
		child.computeID()
		return child, "swap"
	}
	applyPoint(&e.Config, &child, e.RNG)
	child.computeID()
	return child, "point"
}

// ─── Crossover ────────────────────────────────────────────────────

type CrossoverEngine struct {
	RNG *XorShift64
}

func newCrossoverEngine(seed uint64) *CrossoverEngine {
	return &CrossoverEngine{RNG: newXorShift64(seed)}
}

func (x *CrossoverEngine) crossover(a, b *Genome) Genome {
	va := a.toVector()
	vb := b.toVector()
	var childV [genomeDim]float64
	for i := 0; i < genomeDim; i++ {
		if x.RNG.nextF64() < 0.5 {
			childV[i] = va[i]
		} else {
			childV[i] = vb[i]
		}
	}
	gen := a.Generation
	if b.Generation > gen {
		gen = b.Generation
	}
	child := fromVector(&childV, gen+1)
	child.ParentID = fmt.Sprintf("%sx%s", a.GenomeID, b.GenomeID)
	child.computeID()
	return child
}

// ─── Fitness ──────────────────────────────────────────────────────

type FitnessSpec struct {
	AccuracyBias        float64 `json:"accuracy_bias"`
	AccuracyNeuronCoef  float64 `json:"accuracy_neuron_coef"`
	WAccuracy           float64 `json:"w_accuracy"`
	WEnergy             float64 `json:"w_energy"`
	WLatency            float64 `json:"w_latency"`
}

type FitnessResult struct {
	GenomeID     string  `json:"genome_id"`
	Accuracy     float64 `json:"accuracy"`
	EnergyScore  float64 `json:"energy_score"`
	LatencyScore float64 `json:"latency_score"`
	Composite    float64 `json:"composite"`
}

func evaluateFitness(spec *FitnessSpec, g *Genome) FitnessResult {
	n := float64(g.Topology.NumNeurons)
	layers := float64(g.Topology.NumLayers)
	bitstream := float64(g.Topology.BitstreamLength)
	accuracy := spec.AccuracyBias + spec.AccuracyNeuronCoef*n/32.0
	energy := math.Max(0.0, 1.0-0.5*n/1024.0-0.5*bitstream/1024.0)
	latency := math.Max(0.0, 1.0-layers/10.0)
	composite := spec.WAccuracy*accuracy + spec.WEnergy*energy + spec.WLatency*latency
	return FitnessResult{GenomeID: g.GenomeID, Accuracy: accuracy,
		EnergyScore: energy, LatencyScore: latency, Composite: composite}
}

// ─── Guards ───────────────────────────────────────────────────────

type SafetyBounds struct {
	MaxNeurons      int32   `json:"max_neurons"`
	MinNeurons      int32   `json:"min_neurons"`
	MaxLayers       int32   `json:"max_layers"`
	MaxBitstream    int32   `json:"max_bitstream"`
	MinBitstream    int32   `json:"min_bitstream"`
	MaxConnectivity float64 `json:"max_connectivity"`
}

type FormalSafetyGuard struct {
	Bounds   SafetyBounds
	Checked  int64
	Rejected int64
}

func (g *FormalSafetyGuard) check(genome *Genome) bool {
	g.Checked++
	nOK := genome.Topology.NumNeurons <= g.Bounds.MaxNeurons
	cOK := genome.Topology.Connectivity <= g.Bounds.MaxConnectivity
	bOK := genome.Topology.BitstreamLength <= g.Bounds.MaxBitstream
	passed := nOK && cOK && bOK
	if !passed {
		g.Rejected++
	}
	return passed
}

type BloatPenalizer struct {
	PenaltyWeight    float64
	Threshold        float64
	BaselineNeurons  int32
}

func (b *BloatPenalizer) bloatScore(g *Genome) float64 {
	n := float64(g.Topology.NumNeurons)
	l := float64(g.Topology.NumLayers)
	conn := int64(n * n * g.Topology.Connectivity)
	total := int64(n*8+l) + conn
	baseN := float64(b.BaselineNeurons)
	baseline := int64(baseN*8+2) + int64(baseN*baseN*0.3)
	if baseline < 1 {
		baseline = 1
	}
	return float64(total) / float64(baseline)
}

func (b *BloatPenalizer) penalize(fitness float64, g *Genome) float64 {
	score := b.bloatScore(g)
	if score > b.Threshold {
		return math.Max(0.0, fitness-b.PenaltyWeight*(score-b.Threshold))
	}
	return fitness
}

type AgeRegulator struct {
	MaxAge int32
}

type ExtinctionDetector struct {
	StagnationGens  int
	KillFraction    float64
	BestHistory     []float64
	ExtinctionCount int64
}

func (d *ExtinctionDetector) check(best float64) bool {
	d.BestHistory = append(d.BestHistory, best)
	if len(d.BestHistory) < d.StagnationGens {
		return false
	}
	recent := d.BestHistory[len(d.BestHistory)-d.StagnationGens:]
	mn, mx := recent[0], recent[0]
	for _, v := range recent[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	if mx-mn < 1e-6 {
		d.ExtinctionCount++
		return true
	}
	return false
}

func (d *ExtinctionDetector) apply(pop []Organism, rng *XorShift64) int {
	nKill := int(float64(len(pop)) * d.KillFraction)
	if nKill > len(pop) {
		nKill = len(pop)
	}
	indices := make([]int, len(pop))
	for i := range indices {
		indices[i] = i
	}
	// Fisher-Yates partial shuffle matching the Rust runner's loop.
	for i := 0; i < nKill; i++ {
		j := rng.genRange(i, len(indices))
		indices[i], indices[j] = indices[j], indices[i]
	}
	killed := 0
	for _, idx := range indices[:nKill] {
		if pop[idx].Alive {
			pop[idx].Alive = false
			killed++
		}
	}
	return killed
}

type hofEntry struct {
	Fitness float64
	Genome  Genome
}

type HallOfFame struct {
	MaxSize int
	Entries []hofEntry
}

func (h *HallOfFame) update(o *Organism) bool {
	if o.Fitness == nil {
		return false
	}
	h.Entries = append(h.Entries, hofEntry{Fitness: o.Fitness.Composite, Genome: deepCopyGenome(&o.Genome)})
	sort.SliceStable(h.Entries, func(i, j int) bool { return h.Entries[i].Fitness > h.Entries[j].Fitness })
	if len(h.Entries) > h.MaxSize {
		h.Entries = h.Entries[:h.MaxSize]
	}
	return true
}

type ParetoFront struct {
	Front []Organism
}

func dominates(a, b *FitnessResult) bool {
	va := [3]float64{a.Accuracy, a.EnergyScore, a.LatencyScore}
	vb := [3]float64{b.Accuracy, b.EnergyScore, b.LatencyScore}
	atLeastOne := false
	for i := 0; i < 3; i++ {
		if va[i] < vb[i] {
			return false
		}
		if va[i] > vb[i] {
			atLeastOne = true
		}
	}
	return atLeastOne
}

func (p *ParetoFront) update(o *Organism) bool {
	if o.Fitness == nil {
		return false
	}
	for _, existing := range p.Front {
		if existing.Fitness != nil && dominates(existing.Fitness, o.Fitness) {
			return false
		}
	}
	filtered := make([]Organism, 0, len(p.Front))
	for _, existing := range p.Front {
		if existing.Fitness == nil || !dominates(o.Fitness, existing.Fitness) {
			filtered = append(filtered, existing)
		}
	}
	p.Front = append(filtered, *o)
	return true
}

type TournamentSelector struct {
	TournamentSize int
}

func (t *TournamentSelector) selectOne(pop []Organism, rng *XorShift64) *Organism {
	if len(pop) == 0 {
		return nil
	}
	k := t.TournamentSize
	if k > len(pop) {
		k = len(pop)
	}
	var best *Organism
	bestFit := math.Inf(-1)
	// Linear-scan seen slice, matching the Rust runner's Vec<usize>.
	seen := make([]int, 0, k)
	for len(seen) < k {
		idx := rng.genRange(0, len(pop))
		dupe := false
		for _, sIdx := range seen {
			if sIdx == idx {
				dupe = true
				break
			}
		}
		if dupe {
			continue
		}
		seen = append(seen, idx)
		o := &pop[idx]
		fit := 0.0
		if o.Fitness != nil {
			fit = o.Fitness.Composite
		}
		if fit > bestFit {
			bestFit = fit
			best = o
		}
	}
	return best
}

// ─── Organism + Lineage ───────────────────────────────────────────

type Organism struct {
	Genome          Genome
	Fitness         *FitnessResult
	Alive           bool
	BirthGeneration int32
}

type LineageRecord struct {
	GenomeID     string  `json:"genome_id"`
	ParentID     string  `json:"parent_id"`
	Generation   int32   `json:"generation"`
	MutationType string  `json:"mutation_type"`
	Fitness      float64 `json:"fitness"`
}

type LineageTracker struct {
	Records []LineageRecord
}

func (l *LineageTracker) record(o *Organism, mtype string) {
	fit := 0.0
	if o.Fitness != nil {
		fit = o.Fitness.Composite
	}
	l.Records = append(l.Records, LineageRecord{
		GenomeID: o.Genome.GenomeID, ParentID: o.Genome.ParentID,
		Generation: o.Genome.Generation, MutationType: mtype, Fitness: fit,
	})
}

func cullIndices(age *AgeRegulator, pop []Organism, currentGen int32) []int {
	out := []int{}
	for i, o := range pop {
		if o.Alive && currentGen-o.BirthGeneration > age.MaxAge {
			out = append(out, i)
		}
	}
	return out
}

func pairwiseDiversity(pop []Organism) float64 {
	alive := []Organism{}
	for _, o := range pop {
		if o.Alive {
			alive = append(alive, o)
		}
	}
	if len(alive) < 2 {
		return 0.0
	}
	acc := 0.0
	count := 0.0
	for i := 0; i < len(alive); i++ {
		va := alive[i].Genome.toVector()
		for j := i + 1; j < len(alive); j++ {
			vb := alive[j].Genome.toVector()
			s := 0.0
			for k := 0; k < genomeDim; k++ {
				s += math.Abs(va[k]-vb[k]) / (math.Abs(va[k]) + math.Abs(vb[k]) + epsSC)
			}
			acc += s / float64(genomeDim)
			count++
		}
	}
	return acc / count
}

// ─── Config + Result ──────────────────────────────────────────────

type EvolveConfig struct {
	Seed                    uint64          `json:"seed"`
	PopSize                 int             `json:"pop_size"`
	NGenerations            int             `json:"n_generations"`
	Elitism                 int             `json:"elitism"`
	SurvivalFraction        float64         `json:"survival_fraction"`
	TournamentSize          int             `json:"tournament_size"`
	CrossoverProb           float64         `json:"crossover_prob"`
	MaxAge                  int32           `json:"max_age"`
	HallOfFameSize          int             `json:"hall_of_fame_size"`
	StagnationGens          int             `json:"stagnation_gens"`
	ExtinctionKillFraction  float64         `json:"extinction_kill_fraction"`
	Mutation                MutationConfig  `json:"mutation"`
	Fitness                 FitnessSpec     `json:"fitness"`
	SafetyBoundsCfg         SafetyBounds    `json:"safety_bounds"`
	IndustrialMode          bool            `json:"industrial_mode"`
}

type GenerationStats struct {
	Generation       int32   `json:"generation"`
	PopulationSize   int     `json:"population_size"`
	BestFitness      float64 `json:"best_fitness"`
	MeanFitness      float64 `json:"mean_fitness"`
	Diversity        float64 `json:"diversity"`
	Killed           int     `json:"killed"`
	Children         int     `json:"children"`
	Extinctions      int64   `json:"extinctions"`
	SafetyRejections int64   `json:"safety_rejections"`
}

type GenomeOut struct {
	GenomeID        string  `json:"genome_id"`
	ParentID        string  `json:"parent_id"`
	Generation      int32   `json:"generation"`
	NumNeurons      int32   `json:"num_neurons"`
	NumLayers       int32   `json:"num_layers"`
	Connectivity    float64 `json:"connectivity"`
	BitstreamLength int32   `json:"bitstream_length"`
	TauFast         float64 `json:"tau_fast"`
	TauWork         float64 `json:"tau_work"`
	TauDeep         float64 `json:"tau_deep"`
}

func toOut(g *Genome) GenomeOut {
	return GenomeOut{
		GenomeID: g.GenomeID, ParentID: g.ParentID, Generation: g.Generation,
		NumNeurons: g.Topology.NumNeurons, NumLayers: g.Topology.NumLayers,
		Connectivity: g.Topology.Connectivity, BitstreamLength: g.Topology.BitstreamLength,
		TauFast: g.Neuron.TauFast, TauWork: g.Neuron.TauWork, TauDeep: g.Neuron.TauDeep,
	}
}

type EvolveResult struct {
	FinalPopulation    []GenomeOut       `json:"final_population"`
	StatsPerGeneration []GenerationStats `json:"stats_per_generation"`
	HallOfFame         []GenomeOut       `json:"hall_of_fame"`
	ParetoFrontOut     []GenomeOut       `json:"pareto_front"`
	Lineage            []LineageRecord   `json:"lineage"`
	TotalReplications  int64             `json:"total_replications"`
	SafetyChecked      int64             `json:"safety_checked"`
	SafetyRejected     int64             `json:"safety_rejected"`
	ExtinctionCount    int64             `json:"extinction_count"`
}

// ─── Main runner ──────────────────────────────────────────────────

func evolveRun(cfg *EvolveConfig) EvolveResult {
	master := newXorShift64(cfg.Seed)
	mutator := newMutationEngine(cfg.Mutation, master.nextU64())
	xover := newCrossoverEngine(master.nextU64())
	guard := &FormalSafetyGuard{Bounds: cfg.SafetyBoundsCfg}
	bloat := &BloatPenalizer{PenaltyWeight: 0.1, Threshold: 2.0, BaselineNeurons: 16}
	age := &AgeRegulator{MaxAge: cfg.MaxAge}
	extinction := &ExtinctionDetector{
		StagnationGens: cfg.StagnationGens, KillFraction: cfg.ExtinctionKillFraction,
	}
	hof := &HallOfFame{MaxSize: cfg.HallOfFameSize}
	pareto := &ParetoFront{}
	tournament := &TournamentSelector{TournamentSize: cfg.TournamentSize}
	lineage := &LineageTracker{}

	population := make([]Organism, 0, cfg.PopSize)
	for i := 0; i < cfg.PopSize; i++ {
		g := defaultGenome()
		g.computeID()
		o := Organism{Genome: g, Alive: true, BirthGeneration: 0}
		lineage.record(&o, "seed")
		population = append(population, o)
	}

	var totalReplications int64
	stats := make([]GenerationStats, 0, cfg.NGenerations)

	for gen := int32(1); gen <= int32(cfg.NGenerations); gen++ {
		for i := range population {
			if !population[i].Alive {
				continue
			}
			fit := evaluateFitness(&cfg.Fitness, &population[i].Genome)
			if cfg.IndustrialMode {
				fit.Composite = bloat.penalize(fit.Composite, &population[i].Genome)
			}
			population[i].Fitness = &fit
			hof.update(&population[i])
			pareto.update(&population[i])
		}

		killed := 0
		if cfg.IndustrialMode {
			for _, idx := range cullIndices(age, population, gen) {
				population[idx].Alive = false
				killed++
			}
			best := 0.0
			for _, o := range population {
				if o.Alive && o.Fitness != nil && o.Fitness.Composite > best {
					best = o.Fitness.Composite
				}
			}
			if extinction.check(best) {
				killed += extinction.apply(population, mutator.RNG)
			}
		}

		// survival cull
		aliveIndices := []int{}
		for i, o := range population {
			if o.Alive && o.Fitness != nil {
				aliveIndices = append(aliveIndices, i)
			}
		}
		sort.SliceStable(aliveIndices, func(i, j int) bool {
			return population[aliveIndices[i]].Fitness.Composite >
				population[aliveIndices[j]].Fitness.Composite
		})
		keep := cfg.Elitism + 1
		if int(float64(len(aliveIndices))*cfg.SurvivalFraction) > keep {
			keep = int(float64(len(aliveIndices)) * cfg.SurvivalFraction)
		}
		for _, idx := range aliveIndices[keep:] {
			population[idx].Alive = false
			killed++
		}
		compacted := population[:0]
		for _, o := range population {
			if o.Alive {
				compacted = append(compacted, o)
			}
		}
		population = compacted

		// replicate
		survivors := make([]Organism, len(population))
		copy(survivors, population)
		children := 0
		for len(population) < cfg.PopSize && len(survivors) > 0 {
			var parent *Organism
			if cfg.IndustrialMode {
				parent = tournament.selectOne(survivors, mutator.RNG)
			} else {
				parent = &survivors[0]
			}
			if parent == nil {
				break
			}
			var partner *Organism
			if cfg.IndustrialMode {
				partner = tournament.selectOne(survivors, mutator.RNG)
			} else if len(survivors) > 1 {
				partner = &survivors[1]
			}

			var childGenome Genome
			// Consume RNG unconditionally so Go matches Rust's tuple-pattern
			// eager evaluation; otherwise Go's && short-circuit skips an RNG
			// call when `partner` is nil and the RNG state diverges.
			roll := mutator.RNG.nextF64()
			if partner != nil && roll < cfg.CrossoverProb {
				c := xover.crossover(&parent.Genome, &partner.Genome)
				c.Generation = gen
				childGenome = c
			} else {
				c, _ := mutator.mutate(&parent.Genome)
				c.Generation = gen
				childGenome = c
			}

			if !guard.check(&childGenome) {
				continue
			}
			totalReplications++
			child := Organism{Genome: childGenome, Alive: true, BirthGeneration: gen}
			lineage.record(&child, "replicate")
			population = append(population, child)
			children++
		}

		bestFitness := 0.0
		var fits []float64
		for _, o := range population {
			if o.Fitness != nil {
				fits = append(fits, o.Fitness.Composite)
				if o.Fitness.Composite > bestFitness {
					bestFitness = o.Fitness.Composite
				}
			}
		}
		meanFitness := 0.0
		if len(fits) > 0 {
			s := 0.0
			for _, f := range fits {
				s += f
			}
			meanFitness = s / float64(len(fits))
		}
		stats = append(stats, GenerationStats{
			Generation: gen, PopulationSize: len(population),
			BestFitness: bestFitness, MeanFitness: meanFitness,
			Diversity: pairwiseDiversity(population), Killed: killed, Children: children,
			Extinctions: extinction.ExtinctionCount, SafetyRejections: guard.Rejected,
		})
	}

	result := EvolveResult{
		StatsPerGeneration: stats,
		Lineage:            lineage.Records,
		TotalReplications:  totalReplications,
		SafetyChecked:      guard.Checked,
		SafetyRejected:     guard.Rejected,
		ExtinctionCount:    extinction.ExtinctionCount,
	}
	for i := range population {
		result.FinalPopulation = append(result.FinalPopulation, toOut(&population[i].Genome))
	}
	for _, e := range hof.Entries {
		result.HallOfFame = append(result.HallOfFame, toOut(&e.Genome))
	}
	for i := range pareto.Front {
		result.ParetoFrontOut = append(result.ParetoFrontOut, toOut(&pareto.Front[i].Genome))
	}
	return result
}

func runnerMain() {
	var cfg EvolveConfig
	if err := json.NewDecoder(os.Stdin).Decode(&cfg); err != nil {
		fmt.Fprintln(os.Stderr, "config decode error:", err)
		os.Exit(1)
	}
	result := evolveRun(&cfg)
	if err := json.NewEncoder(os.Stdout).Encode(&result); err != nil {
		fmt.Fprintln(os.Stderr, "result encode error:", err)
		os.Exit(1)
	}
}
