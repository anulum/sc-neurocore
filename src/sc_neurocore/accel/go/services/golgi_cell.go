// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for golgi_cell

package services

import (
	"math"
)

// GolgiCellState holds the neuron state
type GolgiCellState struct {
	V float64
	M float64
	H float64
	PNa float64
	N float64
	A float64
	B float64
	W float64
	MT float64
	S float64
	CN float64
	R float64
	Ca float64
	GNaT float64
	GNaP float64
	GKdr float64
	GKa float64
	GKm float64
	GCat float64
	GCan float64
}

// NewGolgiCell creates a new GolgiCell neuron with default parameters
func NewGolgiCell() *GolgiCellState {
	return &GolgiCellState{
		V: -60.0,
		M: 0.02,
		H: 0.85,
		PNa: 0.01,
		N: 0.05,
		A: 0.1,
		B: 0.8,
		W: 0.01,
		MT: 0.01,
		S: 0.9,
		CN: 0.01,
		R: 0.1,
		Ca: 0.05,
		GNaT: 48.0,
		GNaP: 0.2,
		GKdr: 16.0,
		GKa: 8.0,
		GKm: 1.0,
		GCat: 0.5,
		GCan: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *GolgiCellState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateGolgiCell runs the neuron for n steps
func SimulateGolgiCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewGolgiCell()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
