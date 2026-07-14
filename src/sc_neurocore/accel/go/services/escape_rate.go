// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for escape_rate

package services

import (
	"errors"
	"math"
)

var (
	ErrEscapeRateInvalidInput       = errors.New("escape rate input current must be finite")
	ErrEscapeRateInvalidState       = errors.New("escape rate state parameters must be finite and positive")
	ErrEscapeRateNonFiniteUpdate    = errors.New("escape rate membrane candidate must remain finite")
	ErrEscapeRateNonFiniteHazard    = errors.New("escape rate hazard must remain finite and non-negative")
	ErrEscapeRateInvalidProbability = errors.New("escape rate spike probability must remain finite and bounded")
	ErrEscapeRateInvalidSteps       = errors.New("escape rate step count must be non-negative")
)

// EscapeRateNeuronState holds the neuron state
type EscapeRateNeuronState struct {
	V           float64
	VRest       float64
	VReset      float64
	VThreshold  float64
	TauM        float64
	Rho0        float64
	DeltaU      float64
	Resistance  float64
	Dt          float64
	RNGState    uint16
	InitialSeed uint16
}

// NewEscapeRateNeuron creates a new EscapeRateNeuron neuron with default parameters
func NewEscapeRateNeuron() *EscapeRateNeuronState {
	return NewEscapeRateNeuronWithSeed(0xACE1)
}

// NewEscapeRateNeuronWithSeed creates a replayable canonical LFSR16 stream.
func NewEscapeRateNeuronWithSeed(seed uint16) *EscapeRateNeuronState {
	if seed == 0 {
		seed = 0xACE1
	}
	return &EscapeRateNeuronState{
		V:           -70.0,
		VRest:       -70.0,
		VReset:      -70.0,
		VThreshold:  -50.0,
		TauM:        10.0,
		Rho0:        0.001,
		DeltaU:      3.0,
		Resistance:  1.0,
		Dt:          1.0,
		RNGState:    seed,
		InitialSeed: seed,
	}
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *EscapeRateNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) {
		return 0, ErrEscapeRateInvalidInput
	}
	if !ValidateEscapeRate(s) {
		return 0, ErrEscapeRateInvalidState
	}

	vInf := s.VRest + s.Resistance*iExt
	decay := math.Exp(-s.Dt / s.TauM)
	nextV := vInf + (s.V-vInf)*decay
	if !finite(vInf) || !finite(decay) || !finite(nextV) {
		return 0, ErrEscapeRateNonFiniteUpdate
	}
	hazard := s.Rho0 * safeExp((nextV-s.VThreshold)/s.DeltaU) * s.Dt
	if !finite(hazard) || hazard < 0.0 {
		return 0, ErrEscapeRateNonFiniteHazard
	}
	pSpike := -math.Expm1(-hazard)
	if !finite(pSpike) || pSpike < 0.0 || pSpike > 1.0 {
		return 0, ErrEscapeRateInvalidProbability
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
		s.V = s.VReset
		return 1, nil
	}
	s.V = nextV
	return 0, nil
}

// ValidateEscapeRate enforces finite, physically valid state parameters.
func ValidateEscapeRate(s *EscapeRateNeuronState) bool {
	if s == nil {
		return false
	}
	return finite(s.V) && finite(s.VRest) && finite(s.VReset) && finite(s.VThreshold) &&
		finite(s.TauM) && s.TauM > 0.0 &&
		finite(s.Rho0) && s.Rho0 > 0.0 &&
		finite(s.DeltaU) && s.DeltaU > 0.0 &&
		finite(s.Resistance) && s.Resistance > 0.0 &&
		finite(s.Dt) && s.Dt > 0.0 &&
		s.RNGState != 0 && s.InitialSeed != 0
}

// Reset restores the membrane and exact initial RNG seed.
func (s *EscapeRateNeuronState) Reset() {
	s.V = s.VRest
	s.RNGState = s.InitialSeed
}

// SimulateEscapeRateTrace runs a complete state without mutating the caller's value.
func SimulateEscapeRateTrace(
	state EscapeRateNeuronState,
	nSteps int,
	iExt float64,
) ([]float64, []uint8, EscapeRateNeuronState, error) {
	if nSteps < 0 {
		return nil, nil, state, ErrEscapeRateInvalidSteps
	}
	if !finite(iExt) {
		return nil, nil, state, ErrEscapeRateInvalidInput
	}
	if !ValidateEscapeRate(&state) {
		return nil, nil, state, ErrEscapeRateInvalidState
	}
	trace := make([]float64, nSteps)
	events := make([]uint8, nSteps)
	for index := 0; index < nSteps; index++ {
		spike, err := state.Step(iExt)
		if err != nil {
			return nil, nil, state, err
		}
		trace[index] = state.V
		events[index] = uint8(spike)
	}
	return trace, events, state, nil
}

func safeExp(x float64) float64 {
	return math.Exp(math.Max(-700.0, math.Min(700.0, x)))
}

// SimulateEscapeRateNeuron runs the neuron for n steps
func SimulateEscapeRateNeuron(nSteps int, iExt float64) ([]float64, int) {
	initial := *NewEscapeRateNeuron()
	trace, events, _, err := SimulateEscapeRateTrace(initial, nSteps, iExt)
	if err != nil {
		panic(err)
	}
	spikes := 0
	for _, event := range events {
		if event > 0 {
			spikes++
		}
	}
	return trace, spikes
}
