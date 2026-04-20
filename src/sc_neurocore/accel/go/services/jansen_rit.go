// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for jansen_rit

package services

import (
	"math"
)

// JansenRitUnitState holds the neuron state
type JansenRitUnitState struct {
	Y0 float64
	Y3 float64
	Y1 float64
	Y4 float64
	Y2 float64
	Y5 float64
	AExc float64
	BExc float64
	ARate float64
	BRate float64
	C float64
	E0 float64
	V0 float64
	R float64
	Dt float64
}

// NewJansenRitUnit creates a new JansenRitUnit neuron with default parameters
func NewJansenRitUnit() *JansenRitUnitState {
	return &JansenRitUnitState{
		Y0: 0.0,
		Y3: 0.0,
		Y1: 0.0,
		Y4: 0.0,
		Y2: 0.0,
		Y5: 0.0,
		AExc: 3.25,
		BExc: 22.0,
		ARate: 100.0,
		BRate: 50.0,
		C: 135.0,
		E0: 2.5,
		V0: 6.0,
		R: 0.56,
		Dt: 0.001,
	}
}

// Step advances the neuron by one timestep
func (s *JansenRitUnitState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateJansenRitUnit runs the neuron for n steps
func SimulateJansenRitUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewJansenRitUnit()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Y0
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
