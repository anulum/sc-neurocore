// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for non_resetting_lif

package services

import (
	"errors"
	"math"
)

// NonResettingLIFNeuronState holds the neuron state
type NonResettingLIFNeuronState struct {
	V          float64
	Theta      float64
	VRest      float64
	ThetaRest  float64
	DeltaTheta float64
	TauM       float64
	TauTheta   float64
	RM         float64
	Dt         float64
}

// NewNonResettingLIFNeuron creates a new NonResettingLIFNeuron neuron with default parameters
func NewNonResettingLIFNeuron() *NonResettingLIFNeuronState {
	return &NonResettingLIFNeuronState{
		V:          -65.0,
		Theta:      -50.0,
		VRest:      -65.0,
		ThetaRest:  -50.0,
		DeltaTheta: 5.0,
		TauM:       10.0,
		TauTheta:   50.0,
		RM:         1.0,
		Dt:         0.1,
	}
}

func (s NonResettingLIFNeuronState) Valid() bool {
	return nonResettingLIFFinite(s.V) &&
		nonResettingLIFFinite(s.Theta) &&
		nonResettingLIFFinite(s.VRest) &&
		nonResettingLIFFinite(s.ThetaRest) &&
		nonResettingLIFFinite(s.DeltaTheta) && s.DeltaTheta >= 0.0 &&
		nonResettingLIFFinite(s.TauM) && s.TauM > 0.0 &&
		nonResettingLIFFinite(s.TauTheta) && s.TauTheta > 0.0 &&
		nonResettingLIFFinite(s.RM) && s.RM >= 0.0 &&
		nonResettingLIFFinite(s.Dt) && s.Dt > 0.0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *NonResettingLIFNeuronState) Step(iExt float64) (int, error) {
	if !nonResettingLIFFinite(iExt) || !s.Valid() {
		return 0, ErrNonResettingLIFInvalidState
	}
	membraneSteadyState := s.VRest + s.RM*iExt
	if !nonResettingLIFFinite(membraneSteadyState) {
		return 0, ErrNonResettingLIFNonFiniteUpdate
	}
	nextV := nonResettingLIFExactRelaxation(s.V, membraneSteadyState, s.Dt, s.TauM)
	if !nonResettingLIFFinite(nextV) {
		return 0, ErrNonResettingLIFNonFiniteUpdate
	}
	nextTheta := nonResettingLIFExactRelaxation(s.Theta, s.ThetaRest, s.Dt, s.TauTheta)
	if !nonResettingLIFFinite(nextTheta) {
		return 0, ErrNonResettingLIFNonFiniteUpdate
	}
	spike := 0
	if nextV >= nextTheta {
		spike = 1
		nextTheta += s.DeltaTheta
		if !nonResettingLIFFinite(nextTheta) {
			return 0, ErrNonResettingLIFNonFiniteUpdate
		}
	}
	s.V = nextV
	s.Theta = nextTheta
	return spike, nil
}

// SimulateNonResettingLIFNeuron runs the neuron for n steps
func SimulateNonResettingLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewNonResettingLIFNeuron()
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
	ErrNonResettingLIFInvalidState    = errors.New("non-resetting-lif state/current must be finite and physically valid")
	ErrNonResettingLIFNonFiniteUpdate = errors.New("non-resetting-lif exact relaxation update became non-finite")
)

func nonResettingLIFExactRelaxation(state float64, steadyState float64, dt float64, tau float64) float64 {
	decay := math.Exp(-dt / tau)
	return decay*state + (1.0-decay)*steadyState
}

func nonResettingLIFFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
