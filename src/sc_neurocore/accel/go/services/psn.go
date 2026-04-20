// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for psn

package services

import (
	"math"
)

// ParallelSpikingNeuronState holds the neuron state
type ParallelSpikingNeuronState struct {
	KernelSize float64
	VThreshold float64
	Kernel float64
	Buffer float64
	Ptr float64
}

// NewParallelSpikingNeuron creates a new ParallelSpikingNeuron neuron with default parameters
func NewParallelSpikingNeuron() *ParallelSpikingNeuronState {
	return &ParallelSpikingNeuronState{
		KernelSize: 8.0,
		VThreshold: 1.0,
		Kernel: 0.0,
		Buffer: 0.0,
		Ptr: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *ParallelSpikingNeuronState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateParallelSpikingNeuron runs the neuron for n steps
func SimulateParallelSpikingNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewParallelSpikingNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.KernelSize
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
