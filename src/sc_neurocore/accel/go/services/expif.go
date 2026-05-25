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
	ErrExpIFNonFiniteUpdate = errors.New("expif euler update must remain finite")
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

// Step advances the neuron by one timestep
func (s *ExpIFNeuronState) Step(iExt float64) (int, error) {
	if !expIFFinite(iExt) {
		return 0, ErrExpIFInvalidInput
	}
	if !s.Valid() {
		return 0, ErrExpIFInvalidState
	}

	arg := (s.V - s.VRh) / s.DeltaT
	if arg < -20.0 {
		arg = -20.0
	} else if arg > 20.0 {
		arg = 20.0
	}
	expTerm := s.DeltaT * math.Exp(arg)
	dv := (-(s.V - s.VRest) + expTerm + iExt) / s.Tau * s.Dt
	nextV := s.V + dv
	if !expIFFinite(expTerm, dv, nextV) {
		return 0, ErrExpIFNonFiniteUpdate
	}

	vPrev := s.V
	s.V = nextV
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
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
