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
	// SourceProfile selects Naud-Gerstner's strict > threshold boundary.
	SourceProfile bool
}

// NewNaudGerstnerPerfectIntegrator creates the strict source-equation profile.
func NewNaudGerstnerPerfectIntegrator() *PerfectIntegratorNeuronState {
	state := NewPerfectIntegratorNeuron()
	state.SourceProfile = true
	return state
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
		((s.SourceProfile && s.V <= s.VThreshold) ||
			(!s.SourceProfile && s.V < s.VThreshold)) &&
		finitePerfectIntegrator(s.Dt) && s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *PerfectIntegratorNeuronState) Step(iExt float64) (int, error) {
	if !finitePerfectIntegrator(iExt) || !s.Valid() {
		return 0, ErrPerfectIntegratorInvalidState
	}

	voltageIncrement := iExt * s.Dt / s.CM
	nextV := s.V + voltageIncrement
	if !finitePerfectIntegrator(voltageIncrement) || !finitePerfectIntegrator(nextV) {
		return 0, ErrPerfectIntegratorNonFiniteUpdate
	}

	crossed := nextV >= s.VThreshold
	if s.SourceProfile {
		crossed = nextV > s.VThreshold
	}
	if crossed {
		s.V = s.VReset
		return 1, nil
	}
	s.V = nextV
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

// SimulatePerfectIntegratorComplete returns an aligned, failure-atomic packet.
func SimulatePerfectIntegratorComplete(
	initial PerfectIntegratorNeuronState,
	nSteps int,
	iExt float64,
) ([]float64, []uint8, float64, error) {
	if nSteps < 0 || !finitePerfectIntegrator(iExt) || !initial.Valid() {
		return nil, nil, initial.V, ErrPerfectIntegratorInvalidState
	}
	state := initial
	trace := make([]float64, nSteps)
	events := make([]uint8, nSteps)
	for index := 0; index < nSteps; index++ {
		event, err := state.Step(iExt)
		if err != nil {
			return nil, nil, initial.V, err
		}
		trace[index] = state.V
		events[index] = uint8(event)
	}
	return trace, events, state.V, nil
}

func finitePerfectIntegrator(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

var (
	ErrPerfectIntegratorInvalidState    = errors.New("perfect-integrator state/current must be finite and physically ordered")
	ErrPerfectIntegratorNonFiniteUpdate = errors.New("perfect-integrator voltage increment became non-finite")
)
