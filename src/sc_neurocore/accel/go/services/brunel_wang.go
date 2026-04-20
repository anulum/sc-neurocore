// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for brunel_wang

package services

import (
	"math"
)

// BrunelWangNeuronState holds the neuron state
type BrunelWangNeuronState struct {
	V float64
	VRest float64
	VReset float64
	VThreshold float64
	TauM float64
	TauRef float64
	TauAmpa float64
	TauNmdaRise float64
	TauNmdaDecay float64
	TauGaba float64
	GAmpaExt float64
	GAmpaRec float64
	GNmda float64
	GGaba float64
	VAmpa float64
	VNmda float64
	VGaba float64
	CM float64
	MgConc float64
	Dt float64
}

// NewBrunelWangNeuron creates a new BrunelWangNeuron neuron with default parameters
func NewBrunelWangNeuron() *BrunelWangNeuronState {
	return &BrunelWangNeuronState{
		V: -70.0,
		VRest: -70.0,
		VReset: -55.0,
		VThreshold: -50.0,
		TauM: 20.0,
		TauRef: 2.0,
		TauAmpa: 2.0,
		TauNmdaRise: 2.0,
		TauNmdaDecay: 100.0,
		TauGaba: 5.0,
		GAmpaExt: 2.1,
		GAmpaRec: 0.05,
		GNmda: 0.165,
		GGaba: 1.3,
		VAmpa: 0.0,
		VNmda: 0.0,
		VGaba: -70.0,
		CM: 0.5,
		MgConc: 1.0,
		Dt: 0.1,
	}
}

// Step advances the neuron by one timestep
func (s *BrunelWangNeuronState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = s.VReset
		return 1
	}
	return 0
}

// SimulateBrunelWangNeuron runs the neuron for n steps
func SimulateBrunelWangNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewBrunelWangNeuron()
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
