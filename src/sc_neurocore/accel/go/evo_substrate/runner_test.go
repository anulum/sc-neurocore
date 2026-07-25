// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go unit tests for evo runner.

package main

import (
	"math"
	"testing"
)

func TestXorShift64Determinism(t *testing.T) {
	r := newXorShift64(7)
	// Reference values from the Python XorShift64 fixed-seed configuration in
	// tests/test_evo_substrate/multilang_parity_support.py.
	if v := r.nextU64(); v != 7575888327 {
		t.Fatalf("first state = %d, want 7575888327", v)
	}
	if v := r.nextU64(); v != 8070950887952051652 {
		t.Fatalf("second state = %d, want 8070950887952051652", v)
	}
}

func TestXorShift64ZeroSeedFallback(t *testing.T) {
	r := newXorShift64(0)
	if r.state != 0xDEADBEEFCAFEBABE {
		t.Fatalf("zero-seed fallback = %x, want DEADBEEFCAFEBABE", r.state)
	}
}

func TestXorShift64UniformInUnitInterval(t *testing.T) {
	r := newXorShift64(42)
	for i := 0; i < 10_000; i++ {
		v := r.nextF64()
		if v < 0.0 || v >= 1.0 {
			t.Fatalf("nextF64 produced %g outside [0,1)", v)
		}
	}
}

func TestGenomeVectorRoundtrip(t *testing.T) {
	g := defaultGenome()
	g.Topology.NumNeurons = 64
	g.Neuron.TauFast = 7.5
	g.Plasticity.StdpLr = 0.02
	v := g.toVector()
	back := fromVector(&v, g.Generation)
	if back.Topology.NumNeurons != 64 {
		t.Fatalf("num_neurons roundtrip = %d, want 64", back.Topology.NumNeurons)
	}
	if math.Abs(back.Neuron.TauFast-7.5) > 1e-12 {
		t.Fatalf("tau_fast roundtrip = %g, want 7.5", back.Neuron.TauFast)
	}
	if math.Abs(back.Plasticity.StdpLr-0.02) > 1e-12 {
		t.Fatalf("stdp_lr roundtrip = %g, want 0.02", back.Plasticity.StdpLr)
	}
}

func TestGenomeIDIs12HexChars(t *testing.T) {
	g1 := defaultGenome()
	g2 := defaultGenome()
	g1.computeID()
	g2.computeID()
	if g1.GenomeID != g2.GenomeID {
		t.Fatalf("same default genome produced different ids: %q vs %q", g1.GenomeID, g2.GenomeID)
	}
	if len(g1.GenomeID) != 12 {
		t.Fatalf("id length = %d, want 12", len(g1.GenomeID))
	}
}

func TestFitnessMatchesPythonReferenceFormula(t *testing.T) {
	g := defaultGenome()
	g.Topology.NumNeurons = 32
	g.Topology.NumLayers = 2
	g.Topology.BitstreamLength = 256
	spec := FitnessSpec{
		AccuracyBias: 0.5, AccuracyNeuronCoef: 0.01,
		WAccuracy: 0.5, WEnergy: 0.3, WLatency: 0.2,
	}
	f := evaluateFitness(&spec, &g)
	if math.Abs(f.Accuracy-0.51) > 1e-9 {
		t.Fatalf("accuracy = %g, want 0.51", f.Accuracy)
	}
	if math.Abs(f.EnergyScore-0.859375) > 1e-9 {
		t.Fatalf("energy = %g, want 0.859375", f.EnergyScore)
	}
	if math.Abs(f.LatencyScore-0.8) > 1e-9 {
		t.Fatalf("latency = %g, want 0.8", f.LatencyScore)
	}
}

func TestFormalSafetyGuardRejectsOversizedGenomes(t *testing.T) {
	guard := &FormalSafetyGuard{Bounds: SafetyBounds{
		MaxNeurons: 1024, MinNeurons: 4, MaxLayers: 16,
		MaxBitstream: 4096, MinBitstream: 32, MaxConnectivity: 1.0,
	}}
	g := defaultGenome()
	g.Topology.NumNeurons = 4096
	if guard.check(&g) {
		t.Fatalf("guard accepted oversized genome")
	}
	if guard.Rejected != 1 {
		t.Fatalf("rejected counter = %d, want 1", guard.Rejected)
	}
}

func TestEvolveRunIsDeterministic(t *testing.T) {
	cfg := &EvolveConfig{
		Seed: 11, PopSize: 8, NGenerations: 5,
		Elitism: 1, SurvivalFraction: 0.5, TournamentSize: 3,
		CrossoverProb: 0.3, MaxAge: 20, HallOfFameSize: 5,
		StagnationGens: 10, ExtinctionKillFraction: 0.9,
		Mutation: MutationConfig{
			PointRate: 0.2, PointSigma: 0.05,
			StructuralRate: 0.05, DuplicationRate: 0.01,
			SwapRate: 0.02, MaxNeurons: 1024, MinNeurons: 4,
		},
		Fitness: FitnessSpec{
			AccuracyBias: 0.5, AccuracyNeuronCoef: 0.01,
			WAccuracy: 0.5, WEnergy: 0.3, WLatency: 0.2,
		},
		SafetyBoundsCfg: SafetyBounds{
			MaxNeurons: 1024, MinNeurons: 4, MaxLayers: 16,
			MaxBitstream: 4096, MinBitstream: 32, MaxConnectivity: 1.0,
		},
		IndustrialMode: true,
	}
	a := evolveRun(cfg)
	b := evolveRun(cfg)
	if a.TotalReplications != b.TotalReplications {
		t.Fatalf("total_replications diverges: %d vs %d", a.TotalReplications, b.TotalReplications)
	}
	if len(a.FinalPopulation) != len(b.FinalPopulation) {
		t.Fatalf("final population size diverges")
	}
	for i := range a.FinalPopulation {
		if a.FinalPopulation[i].GenomeID != b.FinalPopulation[i].GenomeID {
			t.Fatalf("genome %d diverges: %q vs %q", i, a.FinalPopulation[i].GenomeID, b.FinalPopulation[i].GenomeID)
		}
	}
}
