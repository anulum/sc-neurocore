// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — source-faithful Go Aihara service

package services

import (
	"errors"
	"math"
)

// AiharaMapNeuronState is the one independent state and source parameters.
type AiharaMapNeuronState struct {
	Y       float64
	K       float64
	Alpha   float64
	Bias    float64
	Epsilon float64
}

func NewAiharaMapNeuron() *AiharaMapNeuronState {
	return &AiharaMapNeuronState{Y: 0.1, K: 0.7, Alpha: 1.0, Bias: 0.3968, Epsilon: 0.01}
}

func finiteAiharaMap(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func logisticAiharaMap(value, epsilon float64) float64 {
	argument := value / epsilon
	if argument >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-argument))
	}
	exponential := math.Exp(argument)
	return exponential / (1.0 + exponential)
}

func ValidateAiharaMap(state *AiharaMapNeuronState) bool {
	return state != nil && finiteAiharaMap(state.Y) && finiteAiharaMap(state.K) &&
		state.K >= 0.0 && state.K < 1.0 && finiteAiharaMap(state.Alpha) &&
		state.Alpha > 0.0 && finiteAiharaMap(state.Bias) &&
		finiteAiharaMap(state.Epsilon) && state.Epsilon > 0.0
}

func (state *AiharaMapNeuronState) Output() float64 {
	return logisticAiharaMap(state.Y, state.Epsilon)
}

// Step applies Eqs. 10-12; rejected updates do not mutate state.
func (state *AiharaMapNeuronState) Step(current float64) (int, error) {
	if !ValidateAiharaMap(state) {
		return 0, errors.New("invalid Aihara map runtime state")
	}
	if !finiteAiharaMap(current) {
		return 0, errors.New("invalid Aihara map current")
	}
	nextY := state.K*state.Y - state.Alpha*state.Output() + state.Bias + current
	if !finiteAiharaMap(nextY) {
		return 0, errors.New("invalid Aihara map candidate state")
	}
	state.Y = nextY
	if state.Output() >= 0.5 {
		return 1, nil
	}
	return 0, nil
}

func SimulateAiharaMapNeuron(nSteps int, current float64) ([]float64, int) {
	state := NewAiharaMapNeuron()
	trace := make([]float64, nSteps)
	events := 0
	for index := range trace {
		event, err := state.Step(current)
		if err != nil {
			trace[index] = math.NaN()
			continue
		}
		trace[index] = state.Y
		events += event
	}
	return trace, events
}
