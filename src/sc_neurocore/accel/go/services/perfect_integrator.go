// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for perfect_integrator

package services

import "errors"

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
	return finite(s.V) &&
		finite(s.CM) && s.CM > 0.0 &&
		finite(s.VThreshold) &&
		finite(s.VReset) && s.VThreshold > s.VReset &&
		s.V < s.VThreshold &&
		finite(s.Dt) && s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *PerfectIntegratorNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) || !s.Valid() {
		return 0, ErrPerfectIntegratorInvalidState
	}

	voltageIncrement := iExt / s.CM * s.Dt
	nextV := s.V + voltageIncrement
	if !finite(voltageIncrement) || !finite(nextV) {
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
	s := NewPerfectIntegratorNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var (
	ErrPerfectIntegratorInvalidState    = errors.New("perfect-integrator state/current must be finite and physically ordered")
	ErrPerfectIntegratorNonFiniteUpdate = errors.New("perfect-integrator voltage increment became non-finite")
)
