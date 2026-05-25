// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fitzhugh_nagumo

package services

import (
	"math"
)

// FitzHughNagumoNeuronState holds the neuron state
type FitzHughNagumoNeuronState struct {
	V          float64
	W          float64
	A          float64
	B          float64
	Epsilon    float64
	Dt         float64
	VThreshold float64
}

// NewFitzHughNagumoNeuron creates a new FitzHughNagumoNeuron neuron with default parameters
func NewFitzHughNagumoNeuron() *FitzHughNagumoNeuronState {
	return &FitzHughNagumoNeuronState{
		V:          -1.0,
		W:          -0.5,
		A:          0.7,
		B:          0.8,
		Epsilon:    0.08,
		Dt:         0.1,
		VThreshold: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *FitzHughNagumoNeuronState) Step(iExt float64) int {
	if !math.IsInf(s.V, 0) && !math.IsNaN(s.V) && !math.IsInf(s.W, 0) && !math.IsNaN(s.W) && !math.IsInf(iExt, 0) && !math.IsNaN(iExt) {
		// valid
	} else {
		panic("FitzHugh-Nagumo state/current must be finite")
	}
	vPrev := s.V
	dv := (s.V - math.Pow(s.V, 3.0)/3.0 - s.W + iExt) * s.Dt
	dw := s.Epsilon * (s.V + s.A - s.B*s.W) * s.Dt
	newV := s.V + dv
	newW := s.W + dw
	if math.IsInf(newV, 0) || math.IsNaN(newV) || math.IsInf(newW, 0) || math.IsNaN(newW) {
		panic("FitzHugh-Nagumo state became non-finite")
	}
	s.V = newV
	s.W = newW
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateFitzHughNagumoNeuron runs the neuron for n steps
func SimulateFitzHughNagumoNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFitzHughNagumoNeuron()
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
