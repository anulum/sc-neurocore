// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for bertram_phantom

package services

import (
	"math"
)

// BertramPhantomBursterState holds the neuron state
type BertramPhantomBursterState struct {
	V float64
	S1 float64
	S2 float64
	GCa float64
	GK float64
	GS1 float64
	GS2 float64
	GL float64
	ECa float64
	EK float64
	EL float64
	CM float64
	VM float64
	SM float64
	VN float64
	SN float64
	VS1 float64
	SS1 float64
	VS2 float64
	SS2 float64
	VThreshold float64
}

// NewBertramPhantomBurster creates a new BertramPhantomBurster neuron with default parameters
func NewBertramPhantomBurster() *BertramPhantomBursterState {
	return &BertramPhantomBursterState{
		V: -50.0,
		S1: 0.1,
		S2: 0.1,
		GCa: 3.6,
		GK: 10.0,
		GS1: 4.0,
		GS2: 4.0,
		GL: 0.2,
		ECa: 25.0,
		EK: -75.0,
		EL: -40.0,
		CM: 5.3,
		VM: -20.0,
		SM: 12.0,
		VN: -16.0,
		SN: 5.6,
		VS1: -40.0,
		SS1: 10.0,
		VS2: -42.0,
		SS2: 0.4,
		VThreshold: 0.0,
	}
}

// Step advances the neuron by one timestep
func (s *BertramPhantomBursterState) Step(iExt float64) int {
	vPrev := s.V
	s.V += iExt * 0.01
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		s.V = -50.0
		return 1
	}
	return 0
}

// SimulateBertramPhantomBurster runs the neuron for n steps
func SimulateBertramPhantomBurster(nSteps int, iExt float64) ([]float64, int) {
	s := NewBertramPhantomBurster()
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
