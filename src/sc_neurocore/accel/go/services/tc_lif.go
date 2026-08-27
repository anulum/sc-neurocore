// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for TwoCompartmentLIFNeuron

package services

import (
	"errors"
	"math"
)

// TwoCompartmentLIFNeuronState holds the TC-LIF map state (Zhang et al.
// 2024, Eqs. 10-12); defaults are the published S-MNIST feedforward
// profile.
type TwoCompartmentLIFNeuronState struct {
	UD    float64
	US    float64
	SPrev float64
	Beta1 float64
	Beta2 float64
	Gamma float64
	VTh   float64
}

// NewTwoCompartmentLIFNeuron creates the S-MNIST feedforward profile.
func NewTwoCompartmentLIFNeuron() *TwoCompartmentLIFNeuronState {
	return &TwoCompartmentLIFNeuronState{
		UD: 0.0, US: 0.0, SPrev: 0.0,
		Beta1: -0.5, Beta2: 0.5, Gamma: 0.5, VTh: 1.0,
	}
}

// ValidTwoCompartmentLIF enforces the public descriptor and runtime bounds.
func ValidTwoCompartmentLIF(s *TwoCompartmentLIFNeuronState) bool {
	if s == nil {
		return false
	}
	for _, value := range []float64{s.UD, s.US, s.SPrev, s.Beta1, s.Beta2, s.Gamma, s.VTh} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return s.UD >= -1e6 && s.UD <= 1e6 &&
		s.US >= -1e6 && s.US <= 1e6 &&
		(s.SPrev == 0.0 || s.SPrev == 1.0) &&
		s.Beta1 > -1.0 && s.Beta1 < 0.0 &&
		s.Beta2 > 0.0 && s.Beta2 < 1.0 &&
		s.Gamma >= 0.0 && s.Gamma <= 10.0 &&
		s.VTh > 0.0 && s.VTh <= 100.0
}

// TryStep advances the TC-LIF map atomically or returns an error.
func (s *TwoCompartmentLIFNeuronState) TryStep(iExt float64) (int, error) {
	if math.IsNaN(iExt) || math.IsInf(iExt, 0) {
		return 0, errors.New("i_ext must be finite")
	}
	if !ValidTwoCompartmentLIF(s) {
		return 0, errors.New("TC-LIF state and parameters must satisfy the public bounds")
	}

	uD := s.UD + s.Beta1*s.US + iExt - s.Gamma*s.SPrev
	uS := s.US + s.Beta2*uD - s.VTh*s.SPrev
	if math.IsNaN(uD) || math.IsInf(uD, 0) || math.IsNaN(uS) || math.IsInf(uS, 0) {
		return 0, errors.New("TC-LIF candidate state became non-finite")
	}
	spike := 0
	if uS >= s.VTh {
		spike = 1
	}
	s.UD = uD
	s.US = uS
	s.SPrev = float64(spike)
	return spike, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *TwoCompartmentLIFNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores the dynamic state to zero, preserving configuration.
func (s *TwoCompartmentLIFNeuronState) Reset() {
	s.UD, s.US, s.SPrev = 0.0, 0.0, 0.0
}

// SimulateTwoCompartmentLIFNeuron runs the neuron for n steps.
func SimulateTwoCompartmentLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewTwoCompartmentLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.US
	}
	return trace, spikes
}
