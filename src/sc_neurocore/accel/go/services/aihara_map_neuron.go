// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for aihara_map_neuron

package services

import (
	"errors"
	"math"
)

// AiharaMapNeuronState holds the neuron state
type AiharaMapNeuronState struct {
	X          float64
	Y          float64
	KF         float64
	KS         float64
	Alpha      float64
	Delta      float64
	XThreshold float64
}

// NewAiharaMapNeuron creates a new AiharaMapNeuron neuron with default parameters
func NewAiharaMapNeuron() *AiharaMapNeuronState {
	return &AiharaMapNeuronState{
		X:          0.0,
		Y:          0.0,
		KF:         0.7,
		KS:         0.95,
		Alpha:      2.0,
		Delta:      0.05,
		XThreshold: 0.5,
	}
}

func finiteAiharaMap(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func logisticAiharaMap(z float64) float64 {
	if z >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	expZ := math.Exp(z)
	return expZ / (1.0 + expZ)
}

// ValidateAiharaMap checks chaotic-map state and numerical parameters.
func ValidateAiharaMap(s *AiharaMapNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteAiharaMap(s.X) &&
		finiteAiharaMap(s.Y) &&
		finiteAiharaMap(s.KF) && s.KF >= 0.0 &&
		finiteAiharaMap(s.KS) &&
		finiteAiharaMap(s.Alpha) &&
		finiteAiharaMap(s.Delta) && s.Delta >= 0.0 &&
		finiteAiharaMap(s.XThreshold)
}

// Step advances the neuron by one timestep
func (s *AiharaMapNeuronState) Step(iExt float64) (int, error) {
	if !ValidateAiharaMap(s) {
		return 0, errors.New("invalid Aihara map runtime state")
	}
	if !finiteAiharaMap(iExt) {
		return 0, errors.New("invalid Aihara map current")
	}

	xPrev := s.X
	sigmoid := logisticAiharaMap(s.X + s.Alpha)
	xNew := s.KF*s.X*sigmoid - s.Y + iExt
	yNew := s.KS*s.Y + s.Delta*s.X
	if !finiteAiharaMap(xNew) || !finiteAiharaMap(yNew) {
		return 0, errors.New("invalid Aihara map candidate state")
	}
	s.X = math.Min(10.0, math.Max(-10.0, xNew))
	s.Y = math.Min(10.0, math.Max(-10.0, yNew))
	if s.X >= s.XThreshold && xPrev < s.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateAiharaMapNeuron runs the neuron for n steps
func SimulateAiharaMapNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewAiharaMapNeuron()
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
