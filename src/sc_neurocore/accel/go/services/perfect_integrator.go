// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for perfect_integrator

package services

import (
	"errors"
	"math"
)

// PerfectIntegratorNeuronState holds the neuron state.
type PerfectIntegratorNeuronState struct {
	V          float64
	CM         float64
	VThreshold float64
	VReset     float64
	Dt         float64
}

// NewPerfectIntegratorNeuron creates a new PerfectIntegratorNeuron neuron with default parameters.
func NewPerfectIntegratorNeuron() *PerfectIntegratorNeuronState {
	return &PerfectIntegratorNeuronState{
		V:          0.0,
		CM:         1.0,
		VThreshold: 1.0,
		VReset:     0.0,
		Dt:         0.1,
	}
}

// Valid reports whether the state satisfies the non-leaky integration contract.
func (s PerfectIntegratorNeuronState) Valid() bool {
	return finitePerfectIntegrator(s.V) &&
		finitePerfectIntegrator(s.CM) && s.CM > 0.0 &&
		finitePerfectIntegrator(s.VThreshold) &&
		finitePerfectIntegrator(s.VReset) && s.VThreshold > s.VReset &&
		s.V < s.VThreshold &&
		finitePerfectIntegrator(s.Dt) && s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *PerfectIntegratorNeuronState) Step(iExt float64) (int, error) {
	if !finitePerfectIntegrator(iExt) || !s.Valid() {
		return 0, ErrPerfectIntegratorInvalidState
	}

	voltageIncrement := iExt / s.CM * s.Dt
	nextV := s.V + voltageIncrement
	if !finitePerfectIntegrator(voltageIncrement) || !finitePerfectIntegrator(nextV) {
		return 0, ErrPerfectIntegratorNonFiniteUpdate
	}

	s.V = nextV
	if s.V >= s.VThreshold {
		s.V = s.VReset
		return 1, nil
	}
	return 0, nil
}

// Reset restores dynamic state without changing parameters.
func (s *PerfectIntegratorNeuronState) Reset() {
	s.V = s.VReset
}

// SimulatePerfectIntegratorNeuron runs the neuron for n steps.
func SimulatePerfectIntegratorNeuron(nSteps int, iExt float64) ([]float64, int) {
	trace, spikes, err := SimulatePerfectIntegratorTrace(
		*NewPerfectIntegratorNeuron(), nSteps, iExt,
	)
	if err != nil {
		panic(err)
	}
	return trace, spikes
}

// SimulatePerfectIntegratorTrace executes a complete state/parameter contract.
func SimulatePerfectIntegratorTrace(
	initial PerfectIntegratorNeuronState,
	nSteps int,
	iExt float64,
) ([]float64, int, error) {
	if nSteps < 0 || !finitePerfectIntegrator(iExt) || !initial.Valid() {
		return nil, 0, ErrPerfectIntegratorInvalidState
	}
	state := initial
	trace := make([]float64, nSteps)
	spikes := 0
	for index := 0; index < nSteps; index++ {
		spike, err := state.Step(iExt)
		if err != nil {
			return nil, 0, err
		}
		spikes += spike
		trace[index] = state.V
	}
	return trace, spikes, nil
}

func finitePerfectIntegrator(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

var (
	ErrPerfectIntegratorInvalidState    = errors.New("perfect-integrator state/current must be finite and physically ordered")
	ErrPerfectIntegratorNonFiniteUpdate = errors.New("perfect-integrator voltage increment became non-finite")
)
