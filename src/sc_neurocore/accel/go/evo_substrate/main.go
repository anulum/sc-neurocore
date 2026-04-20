// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go reference implementation for evo_substrate hot paths
//
// Minimal, bit-exact reference of the three compute kernels:
//   GenomicDistance, CrossoverUniform, PointMutation
// for multi-language parity + benchmarking vs Rust, Julia, Mojo, Python.

package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"
)

const epsilon = 1e-10

// GenomicDistance is the scale-invariant L1 distance
//
//	(1/D) · Σ |aᵢ−bᵢ| / (|aᵢ|+|bᵢ|+ε)
//
// matching the Python / Rust / Julia / Mojo references.
func GenomicDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("genome length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	var acc float64
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		norm := math.Abs(a[i]) + math.Abs(b[i]) + epsilon
		acc += diff / norm
	}
	return acc / float64(len(a))
}

// CrossoverUniform runs Syswerda uniform crossover: out[i]=a[i] if mask[i]!=0 else b[i].
func CrossoverUniform(a, b []float64, mask []uint8) []float64 {
	if len(a) != len(b) || len(a) != len(mask) {
		panic("length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		if mask[i] != 0 {
			out[i] = a[i]
		} else {
			out[i] = b[i]
		}
	}
	return out
}

// PointMutation applies Gaussian multiplicative perturbation in-place.
// Caller provides `noise` so the kernel stays pure / deterministic.
func PointMutation(gene []float64, mask []uint8, noise []float64) {
	if len(gene) != len(mask) || len(gene) != len(noise) {
		panic("length mismatch")
	}
	for i := range gene {
		if mask[i] != 0 {
			gene[i] += noise[i] * (math.Abs(gene[i]) + 1e-8)
		}
	}
}

// PopulationDiversity returns the mean pairwise genomic_distance over
// the rows of an n × d population (row-major).
func PopulationDiversity(population []float64, n, d int) float64 {
	if n < 2 {
		return 0
	}
	var acc, count float64
	for i := 0; i < n; i++ {
		rowI := population[i*d : (i+1)*d]
		for j := i + 1; j < n; j++ {
			rowJ := population[j*d : (j+1)*d]
			acc += GenomicDistance(rowI, rowJ)
			count++
		}
	}
	return acc / count
}

func main() {
	// `--runner` dispatches to the full industrial evolve runner in
	// runner.go; no flag runs the per-kernel benchmark.
	if len(os.Args) > 1 && os.Args[1] == "--runner" {
		runnerMain()
		return
	}
	runBench()
}

func runBench() {
	const iters = 100_000
	const d = 19

	a := make([]float64, d)
	b := make([]float64, d)
	mask := make([]uint8, d)
	noise := make([]float64, d)
	for i := 0; i < d; i++ {
		a[i] = float64(i+1) * 0.1
		b[i] = float64(i+1) * 0.2
		mask[i] = uint8(i % 2)
		noise[i] = 0.01
	}

	// Warm-up
	_ = GenomicDistance(a, b)
	_ = CrossoverUniform(a, b, mask)
	PointMutation(append([]float64{}, a...), mask, noise)

	start := time.Now()
	for i := 0; i < iters; i++ {
		_ = GenomicDistance(a, b)
	}
	dGenomic := float64(time.Since(start).Nanoseconds()) / float64(iters)

	start = time.Now()
	for i := 0; i < iters; i++ {
		_ = CrossoverUniform(a, b, mask)
	}
	dCross := float64(time.Since(start).Nanoseconds()) / float64(iters)

	start = time.Now()
	for i := 0; i < iters; i++ {
		g := append([]float64{}, a...)
		PointMutation(g, mask, noise)
	}
	dMut := float64(time.Since(start).Nanoseconds()) / float64(iters)

	results := map[string]float64{
		"genomic_distance_ns_per_call":  dGenomic,
		"crossover_uniform_ns_per_call": dCross,
		"point_mutation_ns_per_call":    dMut,
	}

	// Print human-readable table + emit JSON on stderr for the driver.
	fmt.Printf("%-36s %10.1f ns/call\n", "genomic_distance", dGenomic)
	fmt.Printf("%-36s %10.1f ns/call\n", "crossover_uniform", dCross)
	fmt.Printf("%-36s %10.1f ns/call\n", "point_mutation", dMut)

	enc := json.NewEncoder(os.Stderr)
	if err := enc.Encode(results); err != nil {
		fmt.Fprintln(os.Stderr, "json encode failed:", err)
	}
}
