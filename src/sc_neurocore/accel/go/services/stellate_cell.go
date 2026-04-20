// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for stellate_cell

package services

import (
	"math"
)

// StellateCellState holds the neuron state
type StellateCellState struct {
	V float64
	H float64
	N float64
	P float64
	GNa float64
	GK float64
	GKv3 float64
	GL float64
	ENa float64
	EK float64
	EL float64
	CM float64
	Phi float64
	Dt float64
	VThreshold float64
	Gain float64
	SubSteps float64
}

// NewStellateCell creates a new StellateCell neuron with default parameters
func NewStellateCell() *StellateCellState {
	return &StellateCellState{
		V: -65.0,
		H: 0.6,
		N: 0.32,
		P: 0.0,
		GNa: 35.0,
		GK: 9.0,
		GKv3: 3.0,
		GL: 0.1,
		ENa: 55.0,
		EK: -90.0,
		EL: -65.0,
		CM: 0.5,
		Phi: 5.0,
		Dt: 0.5,
		VThreshold: -20.0,
		Gain: 1.0,
		SubSteps: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *StellateCellState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -65.0
		return 1
	}
	return 0
}

// SimulateStellateCell runs the neuron for n steps
func SimulateStellateCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewStellateCell()
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
