// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for the Montbrió population mean field

package services

import (
	"errors"
	"math"
)

// ErmentroutKopellPopulationState holds the two MPR macroscopic states.
type ErmentroutKopellPopulationState struct {
	R      float64
	V      float64
	Tau    float64
	Delta  float64
	EtaBar float64
	J      float64
	Dt     float64
}

// NewErmentroutKopellPopulation returns the maintained source-paper phase point.
func NewErmentroutKopellPopulation() *ErmentroutKopellPopulationState {
	return &ErmentroutKopellPopulationState{
		R: 0.1, V: -2.0, Tau: 1.0, Delta: 1.0, EtaBar: -5.0, J: 15.0, Dt: 0.01,
	}
}

func finiteMPR(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// ValidateErmentroutKopellPopulation checks the complete numerical contract.
func ValidateErmentroutKopellPopulation(state *ErmentroutKopellPopulationState) bool {
	if state == nil {
		return false
	}
	values := [...]float64{state.R, state.V, state.Tau, state.Delta, state.EtaBar, state.J, state.Dt}
	for _, value := range values {
		if !finiteMPR(value) {
			return false
		}
	}
	return state.R >= 0.0 && state.Tau > 0.0 && state.Delta >= 0.0 && state.Dt > 0.0
}

// Step advances the R=tau*r, t'=t/tau restoration of equations (12a-b).
func (state *ErmentroutKopellPopulationState) Step(extInput float64) (float64, error) {
	if !ValidateErmentroutKopellPopulation(state) || !finiteMPR(extInput) {
		return 0.0, errors.New("invalid MPR state, parameter, or input")
	}
	scaledRate := math.Pi * state.Tau * state.R
	dr := state.Delta/(math.Pi*state.Tau*state.Tau) + 2.0*state.R*state.V/state.Tau
	dv := (state.V*state.V + state.EtaBar + extInput + state.J*state.Tau*state.R - scaledRate*scaledRate) / state.Tau
	nextR := state.R + state.Dt*dr
	nextV := state.V + state.Dt*dv
	if !finiteMPR(nextR) || !finiteMPR(nextV) || nextR < 0.0 {
		return 0.0, errors.New("invalid MPR candidate state")
	}
	state.R, state.V = nextR, nextV
	return state.R, nil
}

// SimulateErmentroutKopellPopulation returns complete post-update state traces.
func SimulateErmentroutKopellPopulation(extInput []float64) ([]float64, []float64, error) {
	state := NewErmentroutKopellPopulation()
	rTrace := make([]float64, len(extInput))
	vTrace := make([]float64, len(extInput))
	for index, drive := range extInput {
		if _, err := state.Step(drive); err != nil {
			return nil, nil, err
		}
		rTrace[index], vTrace[index] = state.R, state.V
	}
	return rTrace, vTrace, nil
}
