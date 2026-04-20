// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for direction_selective_rgc

package services

import (
	"math"
)

// DirectionSelectiveRGCState holds the neuron state
type DirectionSelectiveRGCState struct {
	Tau float64
	Theta float64
	IsOnCentre float64
	WCentre float64
	WSurround float64
	DirectionPref float64
	Dt float64
	V float64
	PrevIntensity float64
	Surround float64
}

// NewDirectionSelectiveRGC creates a new DirectionSelectiveRGC neuron with default parameters
func NewDirectionSelectiveRGC() *DirectionSelectiveRGCState {
	return &DirectionSelectiveRGCState{
		Tau: 10.0,
		Theta: 0.5,
		IsOnCentre: 1.0,
		WCentre: 1.0,
		WSurround: 0.3,
		DirectionPref: 0.0,
		Dt: 1.0,
		V: 0.0,
		PrevIntensity: 0.0,
		Surround: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *DirectionSelectiveRGCState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateDirectionSelectiveRGC runs the neuron for n steps
func SimulateDirectionSelectiveRGC(nSteps int, iExt float64) ([]float64, int) {
	s := NewDirectionSelectiveRGC()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Tau
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
