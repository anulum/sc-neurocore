// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for adaptive_threshold_if

package services

import (
	"errors"
	"math"
)

var (
	ErrAdaptiveThresholdIFInvalidInput    = errors.New("adaptive threshold if input current must be finite")
	ErrAdaptiveThresholdIFInvalidState    = errors.New("adaptive threshold if state parameters must be finite and physically ordered")
	ErrAdaptiveThresholdIFNonFiniteUpdate = errors.New("adaptive threshold if exact relaxation update must remain finite")
)

// AdaptiveThresholdIFNeuronState holds the neuron state
type AdaptiveThresholdIFNeuronState struct {
	V          float64
	Theta      float64
	VRest      float64
	VReset     float64
	ThetaRest  float64
	DeltaTheta float64
	TauM       float64
	TauTheta   float64
	Dt         float64
}

// NewAdaptiveThresholdIFNeuron creates a new AdaptiveThresholdIFNeuron neuron with default parameters
func NewAdaptiveThresholdIFNeuron() *AdaptiveThresholdIFNeuronState {
	return &AdaptiveThresholdIFNeuronState{
		V:          -65.0,
		Theta:      -50.0,
		VRest:      -65.0,
		VReset:     -65.0,
		ThetaRest:  -50.0,
		DeltaTheta: 5.0,
		TauM:       10.0,
		TauTheta:   50.0,
		Dt:         0.1,
	}
}

// Step advances the neuron by one timestep
func (s *AdaptiveThresholdIFNeuronState) Step(iExt float64) (int, error) {
	if !adaptiveThresholdIFFinite(iExt) {
		return 0, ErrAdaptiveThresholdIFInvalidInput
	}
	if !s.Valid() {
		return 0, ErrAdaptiveThresholdIFInvalidState
	}

	nextV := s.exactRelaxation(s.V, s.VRest+iExt, s.TauM)
	nextTheta := s.exactRelaxation(s.Theta, s.ThetaRest, s.TauTheta)
	if !adaptiveThresholdIFFinite(nextV, nextTheta) {
		return 0, ErrAdaptiveThresholdIFNonFiniteUpdate
	}

	if nextV >= nextTheta {
		spikeTheta := nextTheta + s.DeltaTheta
		if !adaptiveThresholdIFFinite(spikeTheta) {
			return 0, ErrAdaptiveThresholdIFNonFiniteUpdate
		}
		s.V = s.VReset
		s.Theta = spikeTheta
		return 1, nil
	}
	s.V = nextV
	s.Theta = nextTheta
	return 0, nil
}

// Valid returns true when the state satisfies the adaptive-threshold IF physics contract.
func (s *AdaptiveThresholdIFNeuronState) Valid() bool {
	return adaptiveThresholdIFFinite(s.V) &&
		adaptiveThresholdIFFinite(s.Theta) &&
		adaptiveThresholdIFFinite(s.VRest) &&
		adaptiveThresholdIFFinite(s.VReset) &&
		adaptiveThresholdIFFinite(s.ThetaRest) &&
		adaptiveThresholdIFFinite(s.DeltaTheta) &&
		s.DeltaTheta >= 0.0 &&
		adaptiveThresholdIFFinite(s.TauM) &&
		s.TauM > 0.0 &&
		adaptiveThresholdIFFinite(s.TauTheta) &&
		s.TauTheta > 0.0 &&
		adaptiveThresholdIFFinite(s.Dt) &&
		s.Dt > 0.0 &&
		s.ThetaRest > s.VRest &&
		s.ThetaRest > s.VReset
}

func (s AdaptiveThresholdIFNeuronState) exactRelaxation(state float64, steadyState float64, tau float64) float64 {
	decay := math.Exp(-s.Dt / tau)
	return steadyState + (state-steadyState)*decay
}

func adaptiveThresholdIFFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// SimulateAdaptiveThresholdIFNeuron runs the neuron for n steps
func SimulateAdaptiveThresholdIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAdaptiveThresholdIFNeuron()
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
