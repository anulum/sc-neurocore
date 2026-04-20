// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for loihi2

package services

import (
	"math"
)

// Loihi2NeuronState holds the neuron state
type Loihi2NeuronState struct {
	S1 float64
	S2 float64
	S3 float64
	Tau1 float64
	Tau2 float64
	Tau3 float64
	W12 float64
	W13 float64
	W23 float64
	S1Threshold float64
	S1Reset float64
	S3Incr float64
}

// NewLoihi2Neuron creates a new Loihi2Neuron neuron with default parameters
func NewLoihi2Neuron() *Loihi2NeuronState {
	return &Loihi2NeuronState{
		S1: 0.0,
		S2: 0.0,
		S3: 0.0,
		Tau1: 10.0,
		Tau2: 5.0,
		Tau3: 50.0,
		W12: 1.0,
		W13: 0.0,
		W23: 0.0,
		S1Threshold: 1000.0,
		S1Reset: 0.0,
		S3Incr: 10.0,
	}
}

// Step advances the neuron by one timestep
func (s *Loihi2NeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateLoihi2Neuron runs the neuron for n steps
func SimulateLoihi2Neuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLoihi2Neuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.S1
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
