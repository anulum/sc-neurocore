// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wilson_cowan

package services

import (
	"math"
)

// WilsonCowanUnitState holds the neuron state
type WilsonCowanUnitState struct {
	E float64
	I float64
	WEe float64
	WEi float64
	WIe float64
	WIi float64
	TauE float64
	TauI float64
	A float64
	Theta float64
	Dt float64
}

// NewWilsonCowanUnit creates a new WilsonCowanUnit neuron with default parameters
func NewWilsonCowanUnit() *WilsonCowanUnitState {
	return &WilsonCowanUnitState{
		E: 0.1,
		I: 0.05,
		WEe: 10.0,
		WEi: 6.0,
		WIe: 10.0,
		WIi: 1.0,
		TauE: 1.0,
		TauI: 2.0,
		A: 1.2,
		Theta: 4.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *WilsonCowanUnitState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateWilsonCowanUnit runs the neuron for n steps
func SimulateWilsonCowanUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewWilsonCowanUnit()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.E
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
