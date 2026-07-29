// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for source-faithful Nagumo–Sato dynamics

package services

import (
	"errors"
	"math"
)

// NagumoSatoMapNeuronState holds the source map state and parameters.
type NagumoSatoMapNeuronState struct {
	Y     float64
	K     float64
	Alpha float64
	Bias  float64
}

// NewNagumoSatoMapNeuron returns the primary-source operating configuration.
func NewNagumoSatoMapNeuron() *NagumoSatoMapNeuronState {
	return &NagumoSatoMapNeuronState{Y: 0.1, K: 0.6, Alpha: 1.0, Bias: 0.2}
}

func nagumoSatoFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// Valid reports whether state and parameters satisfy the source bounds.
func (state *NagumoSatoMapNeuronState) Valid() bool {
	return nagumoSatoFinite(state.Y) && nagumoSatoFinite(state.K) && state.K >= 0.0 && state.K < 1.0 &&
		nagumoSatoFinite(state.Alpha) && state.Alpha > 0.0 && nagumoSatoFinite(state.Bias)
}

// Output returns H(y), with the source convention H(0)=1.
func (state *NagumoSatoMapNeuronState) Output() int {
	if state.Y >= 0.0 {
		return 1
	}
	return 0
}

// TryStep advances the source equation atomically.
func (state *NagumoSatoMapNeuronState) TryStep(current float64) (int, error) {
	if !nagumoSatoFinite(current) {
		return 0, errors.New("current must be finite")
	}
	if !state.Valid() {
		return 0, errors.New("Nagumo-Sato state and parameters must satisfy source bounds")
	}
	nextY := state.K*state.Y - state.Alpha*float64(state.Output()) + state.Bias + current
	if !nagumoSatoFinite(nextY) {
		return 0, errors.New("Nagumo-Sato candidate state became non-finite")
	}
	state.Y = nextY
	return state.Output(), nil
}

// Step advances the map and fails closed for compatibility callers.
func (state *NagumoSatoMapNeuronState) Step(current float64) int {
	event, err := state.TryStep(current)
	if err != nil {
		return 0
	}
	return event
}
