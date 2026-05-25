// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for poisson

package services

import (
	"errors"
	"math"
)

var (
	ErrPoissonInvalidInput       = errors.New("poisson rate override must be finite")
	ErrPoissonInvalidState       = errors.New("poisson rate and timestep must be finite with non-negative rate and positive timestep")
	ErrPoissonNonFiniteHazard    = errors.New("poisson interval hazard must remain finite and non-negative")
	ErrPoissonInvalidProbability = errors.New("poisson spike probability must remain finite and bounded")
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

// Step advances the neuron by one timestep.
func (s *PoissonNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) {
		return 0, ErrPoissonInvalidInput
	}
	if !ValidatePoisson(s) {
		return 0, ErrPoissonInvalidState
	}
	rateHz := s.RateHz
	if iExt >= 0.0 {
		rateHz = iExt
	}
	if !finite(rateHz) || rateHz < 0.0 {
		return 0, ErrPoissonInvalidInput
	}
	hazard := rateHz * s.DtMs / 1000.0
	if !finite(hazard) || hazard < 0.0 {
		return 0, ErrPoissonNonFiniteHazard
	}
	pSpike := -math.Expm1(-hazard)
	if !finite(pSpike) || pSpike < 0.0 || pSpike > 1.0 {
		return 0, ErrPoissonInvalidProbability
	}
	if pSpike >= 1.0 {
		return 1, nil
	}
	return 0, nil
}

// ValidatePoisson enforces finite, physically valid rate parameters.
func ValidatePoisson(s *PoissonNeuronState) bool {
	if s == nil {
		return false
	}
	return finite(s.RateHz) && s.RateHz >= 0.0 && finite(s.DtMs) && s.DtMs > 0.0
}

// SimulatePoissonNeuron runs the neuron for n steps
func SimulatePoissonNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPoissonNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.RateHz
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
