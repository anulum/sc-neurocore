// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for lapicque

package services

import (
	"math"
)

// LapicqueNeuronState holds the neuron state
type LapicqueNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	Tau float64
	Resistance float64
	Dt float64
}

// NewLapicqueNeuron creates a new LapicqueNeuron neuron with default parameters
func NewLapicqueNeuron() *LapicqueNeuronState {
	return &LapicqueNeuronState{
		V: 0.0,
		VRest: 0.0,
		VReset: 0.0,
		VThreshold: 1.0,
		Tau: 20.0,
		Resistance: 1.0,
		Dt: 1.0,
	}
}

// Step advances the neuron by one timestep
func (s *LapicqueNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateLapicqueNeuron runs the neuron for n steps
func SimulateLapicqueNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLapicqueNeuron()
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
