// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for unipolar_brush_cell

package services

import (
	"math"
)

// UnipolarBrushCellState holds the neuron state
type UnipolarBrushCellState struct {
	V float64
	Persistent float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	TauPersistent float64
	PersistentGain float64
	Gain float64
	Dt float64
}

// NewUnipolarBrushCell creates a new UnipolarBrushCell neuron with default parameters
func NewUnipolarBrushCell() *UnipolarBrushCellState {
	return &UnipolarBrushCellState{
		V: -65.0,
		Persistent: 0.0,
		VRest: -65.0,
		VReset: -70.0,
		VThreshold: -50.0,
		TauM: 8.0,
		TauPersistent: 200.0,
		PersistentGain: 0.5,
		Gain: 2.5,
		Dt: 0.5,
	}
}

// Step advances the neuron by one timestep
func (s *UnipolarBrushCellState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateUnipolarBrushCell runs the neuron for n steps
func SimulateUnipolarBrushCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewUnipolarBrushCell()
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
