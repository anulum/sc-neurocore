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
	ErrPoissonInvalidSteps       = errors.New("poisson step count must be non-negative")
)

// PoissonNeuronState holds the neuron state
type PoissonNeuronState struct {
	RateHz      float64
	DtMs        float64
	RNGState    uint16
	InitialSeed uint16
}

// NewPoissonNeuron creates a new PoissonNeuron neuron with default parameters
func NewPoissonNeuron() *PoissonNeuronState {
	return NewPoissonNeuronWithSeed(0xACE1)
}

// NewPoissonNeuronWithSeed creates a replayable canonical LFSR16 stream.
func NewPoissonNeuronWithSeed(seed uint16) *PoissonNeuronState {
	if seed == 0 {
		seed = 0xACE1
	}
	return &PoissonNeuronState{
		RateHz:      100.0,
		DtMs:        1.0,
		RNGState:    seed,
		InitialSeed: seed,
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
	sample := s.RNGState
	for advance := 0; advance < 8; advance++ {
		feedback := ((sample >> 0) ^ (sample >> 2) ^ (sample >> 3) ^ (sample >> 5)) & 1
		sample = (sample >> 1) | (feedback << 15)
	}
	threshold := uint32(0)
	if pSpike >= 1.0 {
		threshold = 65536
	} else if pSpike > 0.0 {
		threshold = uint32(math.Floor(pSpike*65535.0)) + 1
	}
	s.RNGState = sample
	if uint32(sample) < threshold {
		return 1, nil
	}
	return 0, nil
}

// ValidatePoisson enforces finite, physically valid rate parameters.
func ValidatePoisson(s *PoissonNeuronState) bool {
	if s == nil {
		return false
	}
	return finite(s.RateHz) && s.RateHz >= 0.0 && finite(s.DtMs) && s.DtMs > 0.0 &&
		s.RNGState != 0 && s.InitialSeed != 0
}

// Reset restores the exact initial RNG seed.
func (s *PoissonNeuronState) Reset() {
	s.RNGState = s.InitialSeed
}

// SimulatePoissonTrace runs a complete state without mutating the caller's value.
func SimulatePoissonTrace(
	state PoissonNeuronState,
	nSteps int,
	rateOverride float64,
) ([]uint8, PoissonNeuronState, error) {
	if nSteps < 0 {
		return nil, state, ErrPoissonInvalidSteps
	}
	if !finite(rateOverride) {
		return nil, state, ErrPoissonInvalidInput
	}
	if !ValidatePoisson(&state) {
		return nil, state, ErrPoissonInvalidState
	}
	events := make([]uint8, nSteps)
	for index := 0; index < nSteps; index++ {
		spike, err := state.Step(rateOverride)
		if err != nil {
			return nil, state, err
		}
		events[index] = uint8(spike)
	}
	return events, state, nil
}

// SimulatePoissonNeuron runs the neuron for n steps
func SimulatePoissonNeuron(nSteps int, iExt float64) ([]float64, int) {
	initial := *NewPoissonNeuron()
	events, _, err := SimulatePoissonTrace(initial, nSteps, iExt)
	if err != nil {
		panic(err)
	}
	trace := make([]float64, nSteps)
	spikes := 0
	for index, event := range events {
		trace[index] = float64(event)
		spikes += int(event)
	}
	return trace, spikes
}
