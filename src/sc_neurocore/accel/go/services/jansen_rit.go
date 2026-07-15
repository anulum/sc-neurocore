// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for the Jansen–Rit neural mass

package services

import (
	"errors"
	"math"
)

// JansenRitUnitState holds six equation-(6) states and their parameters.
type JansenRitUnitState struct {
	Y0    float64
	Y3    float64
	Y1    float64
	Y4    float64
	Y2    float64
	Y5    float64
	AExc  float64
	BExc  float64
	ARate float64
	BRate float64
	C     float64
	E0    float64
	V0    float64
	R     float64
	Dt    float64
}

// NewJansenRitUnit returns the published parameter set at a 0.1 ms Euler step.
func NewJansenRitUnit() *JansenRitUnitState {
	return &JansenRitUnitState{
		AExc: 3.25, BExc: 22.0, ARate: 100.0, BRate: 50.0,
		C: 135.0, E0: 2.5, V0: 6.0, R: 0.56, Dt: 0.0001,
	}
}

// Step advances atomically and returns the post-update EEG proxy y1-y2.
func (state *JansenRitUnitState) Step(pExt float64) (float64, error) {
	if !finiteJansenRit(pExt) || !ValidateJansenRitUnit(state) {
		return 0.0, errors.New("Jansen–Rit input, state, and parameters must be finite and physical")
	}
	c1 := state.C
	c2 := 0.8 * c1
	c3 := 0.25 * c1
	c4 := 0.25 * c1
	sPyramidal := state.sigmoid(state.Y1 - state.Y2)
	sExcitatory := state.sigmoid(c1 * state.Y0)
	sInhibitory := state.sigmoid(c3 * state.Y0)

	dy0 := state.Y3
	dy3 := state.AExc*state.ARate*sPyramidal - 2.0*state.ARate*state.Y3 - state.ARate*state.ARate*state.Y0
	dy1 := state.Y4
	dy4 := state.AExc*state.ARate*(pExt+c2*sExcitatory) - 2.0*state.ARate*state.Y4 - state.ARate*state.ARate*state.Y1
	dy2 := state.Y5
	dy5 := state.BExc*state.BRate*c4*sInhibitory - 2.0*state.BRate*state.Y5 - state.BRate*state.BRate*state.Y2

	next := *state
	next.Y0 += dy0 * state.Dt
	next.Y3 += dy3 * state.Dt
	next.Y1 += dy1 * state.Dt
	next.Y4 += dy4 * state.Dt
	next.Y2 += dy2 * state.Dt
	next.Y5 += dy5 * state.Dt
	if !ValidateJansenRitUnit(&next) {
		return 0.0, errors.New("Jansen–Rit candidate state became non-finite")
	}
	*state = next
	return state.Y1 - state.Y2, nil
}

// SimulateJansenRitUnit runs a complete caller-owned drive sequence.
func SimulateJansenRitUnit(pExt []float64) ([]float64, error) {
	state := NewJansenRitUnit()
	trace := make([]float64, len(pExt))
	for index, drive := range pExt {
		eeg, err := state.Step(drive)
		if err != nil {
			return nil, err
		}
		trace[index] = eeg
	}
	return trace, nil
}

func finiteJansenRit(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

// ValidateJansenRitUnit enforces the shared scalar state contract.
func ValidateJansenRitUnit(state *JansenRitUnitState) bool {
	if state == nil {
		return false
	}
	return finiteJansenRit(
		state.Y0, state.Y3, state.Y1, state.Y4, state.Y2, state.Y5,
		state.AExc, state.BExc, state.ARate, state.BRate, state.C,
		state.E0, state.V0, state.R, state.Dt,
	) && state.AExc > 0.0 && state.BExc > 0.0 &&
		state.ARate > 0.0 && state.BRate > 0.0 && state.C >= 0.0 &&
		state.E0 > 0.0 && state.R > 0.0 && state.Dt > 0.0
}

func (state *JansenRitUnitState) sigmoid(voltage float64) float64 {
	exponent := state.R * (state.V0 - voltage)
	if exponent >= 0.0 {
		expNeg := math.Exp(-exponent)
		return 2.0 * state.E0 * expNeg / (1.0 + expNeg)
	}
	return 2.0 * state.E0 / (1.0 + math.Exp(exponent))
}
