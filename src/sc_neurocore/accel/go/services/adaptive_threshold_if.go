// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for composite reduced adaptive-threshold IF

package services

import (
	"errors"
	"math"
)

// AdaptiveThresholdIFNeuronState holds the membrane potential and adaptive threshold.
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

// NewAdaptiveThresholdIFNeuron returns the catalogue model-family defaults.
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

// Valid reports whether the complete numerical configuration is admissible.
func (s AdaptiveThresholdIFNeuronState) Valid() bool {
	return finite(s.V) &&
		finite(s.Theta) &&
		finite(s.VRest) &&
		finite(s.VReset) &&
		finite(s.ThetaRest) &&
		s.ThetaRest > s.VRest &&
		s.ThetaRest > s.VReset &&
		finite(s.DeltaTheta) && s.DeltaTheta >= 0.0 &&
		finite(s.TauM) && s.TauM > 0.0 &&
		finite(s.TauTheta) && s.TauTheta > 0.0 &&
		finite(s.Dt) && s.Dt > 0.0
}

// Step advances the exact constant-input relaxation. Rejected calls are atomic.
func (s *AdaptiveThresholdIFNeuronState) Step(iExt float64) (int, error) {
	if !finite(iExt) || !s.Valid() {
		return 0, ErrAdaptiveThresholdIFInvalidState
	}
	nextV, err := adaptiveThresholdIFExactRelaxation(s.V, s.VRest+iExt, s.TauM, s.Dt)
	if err != nil {
		return 0, err
	}
	nextTheta, err := adaptiveThresholdIFExactRelaxation(s.Theta, s.ThetaRest, s.TauTheta, s.Dt)
	if err != nil {
		return 0, err
	}
	if nextV >= nextTheta {
		spikeTheta := nextTheta + s.DeltaTheta
		if !finite(spikeTheta) {
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

// Reset restores the documented rest state without changing configuration.
func (s *AdaptiveThresholdIFNeuronState) Reset() {
	s.V = s.VRest
	s.Theta = s.ThetaRest
}

// SimulateAdaptiveThresholdIFNeuron returns the voltage trace and spike count.
func SimulateAdaptiveThresholdIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 || !finite(iExt) {
		return nil, 0
	}
	s := NewAdaptiveThresholdIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.V
		spikes += result
	}
	return trace, spikes
}

var (
	ErrAdaptiveThresholdIFInvalidState    = errors.New("adaptive-threshold state/current must be finite and well-formed")
	ErrAdaptiveThresholdIFNonFiniteUpdate = errors.New("adaptive-threshold exact-relaxation update became non-finite")
)

func adaptiveThresholdIFExactRelaxation(
	state float64,
	steadyState float64,
	tau float64,
	dt float64,
) (float64, error) {
	decay := math.Exp(-dt / tau)
	candidate := steadyState + (state-steadyState)*decay
	if !finite(candidate) {
		return 0.0, ErrAdaptiveThresholdIFNonFiniteUpdate
	}
	return candidate, nil
}
