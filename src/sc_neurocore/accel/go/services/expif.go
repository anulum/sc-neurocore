// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for expif

package services

import (
	"errors"
	"math"
)

var (
	ErrExpIFInvalidInput    = errors.New("expif input current must be finite")
	ErrExpIFInvalidState    = errors.New("expif state parameters must be finite with positive delta_t, tau, and dt")
	ErrExpIFNonFiniteUpdate = errors.New("expif rk4 update must remain finite")
)

// ExpIFNeuronState holds the neuron state
type ExpIFNeuronState struct {
	V          float64
	VRest      float64
	VReset     float64
	VThreshold float64
	VRh        float64
	DeltaT     float64
	Tau        float64
	Dt         float64
}

// NewExpIFNeuron creates a new ExpIFNeuron neuron with default parameters
func NewExpIFNeuron() *ExpIFNeuronState {
	return &ExpIFNeuronState{
		V:          -65.0,
		VRest:      -65.0,
		VReset:     -68.0,
		VThreshold: -50.0,
		VRh:        -55.0,
		DeltaT:     2.0,
		Tau:        20.0,
		Dt:         0.1,
	}
}

func expIFFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *ExpIFNeuronState) Valid() bool {
	return expIFFinite(s.V, s.VRest, s.VReset, s.VThreshold, s.VRh, s.DeltaT, s.Tau, s.Dt) &&
		s.DeltaT > 0.0 &&
		s.Tau > 0.0 &&
		s.Dt > 0.0
}

func (s *ExpIFNeuronState) rhs(v float64, iExt float64) (float64, bool) {
	arg := (v - s.VRh) / s.DeltaT
	if arg < -20.0 {
		arg = -20.0
	} else if arg > 20.0 {
		arg = 20.0
	}
	expTerm := s.DeltaT * math.Exp(arg)
	rhs := (-(v - s.VRest) + expTerm + iExt) / s.Tau
	return rhs, expIFFinite(expTerm, rhs)
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *ExpIFNeuronState) Step(iExt float64) (int, error) {
	if !expIFFinite(iExt) {
		return 0, ErrExpIFInvalidInput
	}
	if !s.Valid() {
		return 0, ErrExpIFInvalidState
	}

	k1, ok1 := s.rhs(s.V, iExt)
	k2, ok2 := s.rhs(s.V+0.5*s.Dt*k1, iExt)
	k3, ok3 := s.rhs(s.V+0.5*s.Dt*k2, iExt)
	k4, ok4 := s.rhs(s.V+s.Dt*k3, iExt)
	nextV := s.V + s.Dt*(k1+2.0*k2+2.0*k3+k4)/6.0
	if !ok1 || !ok2 || !ok3 || !ok4 || !expIFFinite(nextV) {
		return 0, ErrExpIFNonFiniteUpdate
	}

	s.V = nextV
	if s.V >= s.VThreshold {
		s.V = s.VReset
		return 1, nil
	}
	return 0, nil
}

// SimulateExpIFNeuron runs the neuron for n steps
func SimulateExpIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewExpIFNeuron()
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
