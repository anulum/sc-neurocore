// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for poisson

package services

import (
	"math"
)

// PoissonNeuronState holds the neuron state
type PoissonNeuronState struct {
	RateHz float64
	DtMs   float64
	Rng    float64
}

// NewPoissonNeuron creates a new PoissonNeuron neuron with default parameters
func NewPoissonNeuron() *PoissonNeuronState {
	return &PoissonNeuronState{
		RateHz: 100.0,
		DtMs:   1.0,
		Rng:    0.0,
	}
}

// Step advances the neuron by one timestep
func (s *PoissonNeuronState) Step(iExt float64) int {
	if !ValidatePoisson(s) || !finite(iExt) {
		return 0
	}
	rateHz := s.RateHz
	if iExt >= 0.0 {
		rateHz = iExt
	}
	if !finite(rateHz) || rateHz < 0.0 {
		return 0
	}
	pSpike := -math.Expm1(-(rateHz * s.DtMs / 1000.0))
	if pSpike >= 1.0 {
		return 1
	}
	return 0
}

// ValidatePoisson enforces finite, physically valid rate parameters.
func ValidatePoisson(s *PoissonNeuronState) bool {
	if s == nil {
		return false
	}
	return finite(s.RateHz) && s.RateHz >= 0.0 && finite(s.DtMs) && s.DtMs > 0.0
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// SimulatePoissonNeuron runs the neuron for n steps
func SimulatePoissonNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPoissonNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.RateHz
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
