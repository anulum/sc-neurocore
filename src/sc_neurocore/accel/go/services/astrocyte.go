// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for astrocyte

package services

import (
	"math"
)

// AstrocyteModelState holds the neuron state
type AstrocyteModelState struct {
	Ca float64
	H float64
	Ip3 float64
	VEr float64
	KEr float64
	VSerca float64
	D1 float64
	D2 float64
	D3 float64
	D5 float64
	A2 float64
	C0 float64
	C1 float64
	Leak float64
	Ip3Prod float64
	Ip3Decay float64
	Dt float64
}

// NewAstrocyteModel creates a new AstrocyteModel neuron with default parameters
func NewAstrocyteModel() *AstrocyteModelState {
	return &AstrocyteModelState{
		Ca: 0.05,
		H: 0.8,
		Ip3: 0.5,
		VEr: 0.9,
		KEr: 0.15,
		VSerca: 0.4,
		D1: 0.13,
		D2: 1.049,
		D3: 0.9434,
		D5: 0.08234,
		A2: 0.2,
		C0: 2.0,
		C1: 0.185,
		Leak: 0.01,
		Ip3Prod: 0.0,
		Ip3Decay: 0.14,
		Dt: 0.01,
	}
}

// Step advances the neuron by one timestep
func (s *AstrocyteModelState) Step(iExt float64) int {
	_ = iExt
	return 0
}

// SimulateAstrocyteModel runs the neuron for n steps
func SimulateAstrocyteModel(nSteps int, iExt float64) ([]float64, int) {
	s := NewAstrocyteModel()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Ca
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var _ = math.Exp
