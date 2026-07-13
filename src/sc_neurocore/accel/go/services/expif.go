// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Fourcaud-Trocmé ExpIF recurrence

package services

import (
	"errors"
	"math"
)

var (
	ErrExpIFInvalidInput    = errors.New("expif input current must be finite")
	ErrExpIFInvalidState    = errors.New("expif state or parameters violate the runtime contract")
	ErrExpIFNonFiniteUpdate = errors.New("expif rk4 update must remain finite")
)

// ExpIFNeuronState holds the full deterministic and refractory state.
type ExpIFNeuronState struct {
	V                   float64
	VRest               float64
	VReset              float64
	VThreshold          float64
	VRh                 float64
	DeltaT              float64
	Tau                 float64
	Dt                  float64
	RefractoryPeriod    float64
	RefractoryRemaining float64
}

// NewExpIFNeuron returns the source-bound deterministic catalogue defaults.
func NewExpIFNeuron() *ExpIFNeuronState {
	return &ExpIFNeuronState{
		V:                   -65.0,
		VRest:               -65.0,
		VReset:              -68.0,
		VThreshold:          30.0,
		VRh:                 -59.9,
		DeltaT:              3.48,
		Tau:                 10.0,
		Dt:                  0.02,
		RefractoryPeriod:    0.0,
		RefractoryRemaining: 0.0,
	}
}

// Valid reports whether every precondition of Step holds.
func (s *ExpIFNeuronState) Valid() bool {
	return finiteExpIF(
		s.V,
		s.VRest,
		s.VReset,
		s.VThreshold,
		s.VRh,
		s.DeltaT,
		s.Tau,
		s.Dt,
		s.RefractoryPeriod,
		s.RefractoryRemaining,
	) && s.DeltaT > 0.0 && s.Tau > 0.0 && s.Dt > 0.0 &&
		s.RefractoryPeriod >= 0.0 && s.RefractoryRemaining >= 0.0 &&
		s.RefractoryRemaining <= s.RefractoryPeriod && s.VThreshold > s.VRh &&
		s.V < s.VThreshold && s.VRest < s.VThreshold && s.VReset < s.VThreshold
}

func finiteExpIF(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *ExpIFNeuronState) rhs(v float64, current float64) float64 {
	boundedV := math.Min(v, s.VThreshold)
	arg := (boundedV - s.VRh) / s.DeltaT
	expTerm := s.DeltaT * math.Exp(arg)
	return (-(boundedV - s.VRest) + expTerm + current) / s.Tau
}

// Step advances one candidate-first RK4 update without partial mutation.
func (s *ExpIFNeuronState) Step(current float64) (int, error) {
	if !finiteExpIF(current) {
		return 0, ErrExpIFInvalidInput
	}
	if !s.Valid() {
		return 0, ErrExpIFInvalidState
	}

	if s.RefractoryRemaining > 0.0 {
		s.RefractoryRemaining = math.Max(0.0, s.RefractoryRemaining-s.Dt)
		s.V = s.VReset
		return 0, nil
	}

	k1 := s.rhs(s.V, current)
	k2 := s.rhs(s.V+0.5*s.Dt*k1, current)
	k3 := s.rhs(s.V+0.5*s.Dt*k2, current)
	k4 := s.rhs(s.V+s.Dt*k3, current)
	nextV := s.V + (s.Dt/6.0)*(k1+2.0*k2+2.0*k3+k4)
	if !finiteExpIF(k1, k2, k3, k4, nextV) {
		return 0, ErrExpIFNonFiniteUpdate
	}

	if nextV >= s.VThreshold {
		s.V = s.VReset
		s.RefractoryRemaining = s.RefractoryPeriod
		return 1, nil
	}
	s.V = nextV
	return 0, nil
}

// Reset restores resting voltage and clears the refractory hold.
func (s *ExpIFNeuronState) Reset() {
	s.V = s.VRest
	s.RefractoryRemaining = 0.0
}

// SimulateExpIFNeuron runs the default recurrence and returns a post-step trace.
func SimulateExpIFNeuron(nSteps int, current float64) ([]float64, int, error) {
	if nSteps < 0 {
		return nil, 0, ErrExpIFInvalidState
	}
	state := NewExpIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for index := range trace {
		spike, err := state.Step(current)
		if err != nil {
			return nil, 0, err
		}
		spikes += spike
		trace[index] = state.V
	}
	return trace, spikes, nil
}
