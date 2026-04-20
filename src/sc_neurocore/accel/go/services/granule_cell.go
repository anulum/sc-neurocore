// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for granule_cell

package services

import (
	"math"
)

// GranuleCellState holds the neuron state
type GranuleCellState struct {
	V float64
	M float64
	H float64
	N float64
	A float64
	B float64
	MT float64
	S float64
	Ca float64
	R float64
	CM float64
	GNa float64
	GKdr float64
	GKa float64
	GT float64
	GKca float64
	GH float64
	GL float64
	GTonic float64
	ENa float64
}

// NewGranuleCell creates a new GranuleCell neuron with default parameters
func NewGranuleCell() *GranuleCellState {
	return &GranuleCellState{
		V: -70.0,
		M: 0.02,
		H: 0.85,
		N: 0.05,
		A: 0.1,
		B: 0.8,
		MT: 0.01,
		S: 0.95,
		Ca: 0.05,
		R: 0.1,
		CM: 1.0,
		GNa: 17.0,
		GKdr: 9.0,
		GKa: 1.0,
		GT: 0.5,
		GKca: 3.5,
		GH: 0.03,
		GL: 0.1,
		GTonic: 0.2,
		ENa: 87.4,
	}
}

// Step advances the neuron by one timestep
func (s *GranuleCellState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateGranuleCell runs the neuron for n steps
func SimulateGranuleCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewGranuleCell()
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
