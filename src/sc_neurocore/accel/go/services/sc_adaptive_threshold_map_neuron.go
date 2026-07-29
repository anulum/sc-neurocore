// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for the retained project adaptive-threshold map

package services

import (
	"errors"
	"math"
)

// SCAdaptiveThresholdMapNeuronState holds the project model state and parameters.
type SCAdaptiveThresholdMapNeuronState struct {
	X          float64
	Theta      float64
	K          float64
	Beta       float64
	Gamma      float64
	ThetaSpike float64
	XThreshold float64
}

// NewSCAdaptiveThresholdMapNeuron returns the project reference configuration.
func NewSCAdaptiveThresholdMapNeuron() *SCAdaptiveThresholdMapNeuronState {
	return &SCAdaptiveThresholdMapNeuronState{
		X: 0.0, Theta: 0.0, K: 1.5, Beta: 0.95, Gamma: 0.3, ThetaSpike: 0.8, XThreshold: 0.8,
	}
}

func scAdaptiveFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func scAdaptiveBounded(value, lower, upper float64) bool {
	return scAdaptiveFinite(value) && value >= lower && value <= upper
}

// Valid reports whether state and parameters satisfy the project bounds.
func (state *SCAdaptiveThresholdMapNeuronState) Valid() bool {
	return scAdaptiveBounded(state.X, -5.0, 5.0) && scAdaptiveBounded(state.Theta, -5.0, 5.0) &&
		scAdaptiveBounded(state.K, 0.0, 5.0) && scAdaptiveBounded(state.Beta, 0.0, 1.0) &&
		scAdaptiveBounded(state.Gamma, 0.0, 2.0) && scAdaptiveBounded(state.ThetaSpike, 0.0, 2.0) &&
		scAdaptiveBounded(state.XThreshold, 0.0, 2.0)
}

func scAdaptiveSigmoid(value float64) float64 {
	if value >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-value))
	}
	exponential := math.Exp(value)
	return exponential / (1.0 + exponential)
}

// TryStep advances the simultaneous project recurrence atomically.
func (state *SCAdaptiveThresholdMapNeuronState) TryStep(current float64) (int, error) {
	if !scAdaptiveFinite(current) {
		return 0, errors.New("current must be finite")
	}
	if !state.Valid() {
		return 0, errors.New("SC adaptive-map state and parameters must satisfy public bounds")
	}
	previousX := state.X
	activation := scAdaptiveSigmoid((state.X - state.Theta) * 4.0)
	nextX := -state.X + state.K*activation + current
	fired := 0.0
	if state.X >= state.ThetaSpike {
		fired = 1.0
	}
	nextTheta := state.Beta*state.Theta + state.Gamma*fired
	if !scAdaptiveFinite(nextX) || !scAdaptiveFinite(nextTheta) {
		return 0, errors.New("SC adaptive-map candidate state became non-finite")
	}
	state.X = math.Max(-5.0, math.Min(5.0, nextX))
	state.Theta = math.Max(-5.0, math.Min(5.0, nextTheta))
	if state.X >= state.XThreshold && previousX < state.XThreshold {
		return 1, nil
	}
	return 0, nil
}

// Step advances the map and fails closed for compatibility callers.
func (state *SCAdaptiveThresholdMapNeuronState) Step(current float64) int {
	event, err := state.TryStep(current)
	if err != nil {
		return 0
	}
	return event
}
