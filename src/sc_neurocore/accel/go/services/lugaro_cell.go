// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lugaro_cell

package services

import (
	"math"
)

// LugaroCellState holds the neuron state
type LugaroCellState struct {
	V float64
	Adapt float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	TauAdapt float64
	AAdapt float64
	Gain float64
	Serotonin float64
	Dt float64
}

// NewLugaroCell creates a new LugaroCell neuron with default parameters
func NewLugaroCell() *LugaroCellState {
	return &LugaroCellState{
		V: -55.0,
		Adapt: 0.0,
		VRest: -55.0,
		VReset: -65.0,
		VThreshold: -48.0,
		TauM: 10.0,
		TauAdapt: 150.0,
		AAdapt: 0.05,
		Gain: 2.0,
		Serotonin: 0.0,
		Dt: 0.5,
	}
}

// Step advances the neuron by one timestep
func (s *LugaroCellState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateLugaroCell runs the neuron for n steps
func SimulateLugaroCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewLugaroCell()
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
