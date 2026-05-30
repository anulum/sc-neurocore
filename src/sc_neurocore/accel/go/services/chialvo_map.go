// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for chialvo_map

package services

import (
	"errors"
	"math"
)

// ChialvoMapNeuronState holds the neuron state
type ChialvoMapNeuronState struct {
	X          float64
	Y          float64
	A          float64
	B          float64
	C          float64
	K          float64
	XThreshold float64
}

// NewChialvoMapNeuron creates a new ChialvoMapNeuron neuron with default parameters
func NewChialvoMapNeuron() *ChialvoMapNeuronState {
	return &ChialvoMapNeuronState{
		X:          0.0,
		Y:          0.0,
		A:          0.89,
		B:          0.6,
		C:          0.28,
		K:          0.04,
		XThreshold: 1.0,
	}
}

func finiteChialvoMap(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func safeExpChialvoMap(value float64) float64 {
	return math.Exp(math.Min(709.0, math.Max(-745.0, value)))
}

// ValidateChialvoMap checks discrete-map state and numerical parameters.
func ValidateChialvoMap(s *ChialvoMapNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteChialvoMap(s.X) &&
		finiteChialvoMap(s.Y) &&
		finiteChialvoMap(s.A) &&
		finiteChialvoMap(s.B) &&
		finiteChialvoMap(s.C) &&
		finiteChialvoMap(s.K) &&
		finiteChialvoMap(s.XThreshold)
}

// Step advances the neuron by one timestep
func (s *ChialvoMapNeuronState) Step(iExt float64) (int, error) {
	if !ValidateChialvoMap(s) {
		return 0, errors.New("invalid Chialvo map runtime state")
	}
	if !finiteChialvoMap(iExt) {
		return 0, errors.New("invalid Chialvo map current")
	}

	xPrev := s.X
	xNew := s.X*s.X*safeExpChialvoMap(s.Y-s.X) + s.K + iExt
	yNew := s.A*s.Y - s.B*s.X + s.C
	if !finiteChialvoMap(xNew) || !finiteChialvoMap(yNew) {
		return 0, errors.New("invalid Chialvo map candidate state")
	}
	s.X = xNew
	s.Y = yNew
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateChialvoMapNeuron runs the neuron for n steps
func SimulateChialvoMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewChialvoMapNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.X
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
