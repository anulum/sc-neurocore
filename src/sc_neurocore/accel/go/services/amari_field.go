// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go mirror of the Amari 1977 periodic neural field

package services

import (
	"errors"
	"math"
)

// AmariNeuralFieldState owns the complete periodic vector state and kernel.
type AmariNeuralFieldState struct {
	U                                       []float64
	Tau, AExc, AWidth, BInh, BWidth, Dx, Dt float64
	kernel                                  []float64
}

func finiteAmari(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// NewAmariNeuralField returns the maintained 64-site source-faithful defaults.
func NewAmariNeuralField() *AmariNeuralFieldState {
	state, err := NewAmariNeuralFieldWithConfig(make([]float64, 64), 10, 1.5, 2, 0.75, 1, 0.5, 0.5)
	if err != nil {
		panic(err)
	}
	return state
}

// NewAmariNeuralFieldWithConfig validates and copies a complete field configuration.
func NewAmariNeuralFieldWithConfig(
	u []float64, tau, aExc, aWidth, bInh, bWidth, dx, dt float64,
) (*AmariNeuralFieldState, error) {
	n := len(u)
	values := []float64{tau, aExc, aWidth, bInh, bWidth, dx, dt}
	if n < 2 || tau <= 0 || aExc < 0 || aWidth <= 0 || bInh < 0 || bWidth <= 0 || dx <= 0 || dt <= 0 {
		return nil, errors.New("invalid Amari field configuration")
	}
	for _, value := range append(append([]float64{}, values...), u...) {
		if !finiteAmari(value) {
			return nil, errors.New("non-finite Amari field configuration")
		}
	}
	kernel := make([]float64, n)
	for offset := range kernel {
		wrapped := offset
		if n-offset < wrapped {
			wrapped = n - offset
		}
		distance := float64(wrapped) * dx
		kernel[offset] = aExc*math.Exp(-aWidth*distance) - bInh*math.Exp(-bWidth*distance)
	}
	if kernel[0] <= 0 || kernel[n/2] >= 0 {
		return nil, errors.New("Amari kernel must be locally excitatory and distally inhibitory")
	}
	return &AmariNeuralFieldState{
		U: append([]float64{}, u...), Tau: tau, AExc: aExc, AWidth: aWidth,
		BInh: bInh, BWidth: bWidth, Dx: dx, Dt: dt, kernel: kernel,
	}, nil
}

// ValidateAmariNeuralField reports whether state and parameters are finite and in domain.
func ValidateAmariNeuralField(state *AmariNeuralFieldState) bool {
	if state == nil || len(state.U) < 2 || state.Tau <= 0 || state.AWidth <= 0 || state.BWidth <= 0 || state.Dx <= 0 || state.Dt <= 0 {
		return false
	}
	for _, value := range append([]float64{state.Tau, state.AExc, state.AWidth, state.BInh, state.BWidth, state.Dx, state.Dt}, state.U...) {
		if !finiteAmari(value) {
			return false
		}
	}
	return true
}

// Step advances one simultaneous Euler update and returns active-site fraction.
// Invalid inputs or candidates return an error without committing partial state.
func (state *AmariNeuralFieldState) Step(input []float64) (float64, error) {
	if !ValidateAmariNeuralField(state) || len(input) != len(state.U) {
		return 0, errors.New("invalid Amari runtime state or input length")
	}
	for _, value := range input {
		if !finiteAmari(value) {
			return 0, errors.New("non-finite Amari input")
		}
	}
	n := len(state.U)
	candidate := make([]float64, n)
	for i := range candidate {
		convolution := 0.0
		for j, value := range state.U {
			if value > 0 {
				convolution += state.kernel[(i+n-j)%n]
			}
		}
		candidate[i] = state.U[i] + (-state.U[i]+convolution*state.Dx+input[i])*(state.Dt/state.Tau)
		if !finiteAmari(candidate[i]) {
			return 0, errors.New("non-finite Amari candidate")
		}
	}
	state.U = candidate
	active := 0
	for _, value := range state.U {
		if value > 0 {
			active++
		}
	}
	return float64(active) / float64(n), nil
}

// Reset zeros dynamic field state while preserving configured physics.
func (state *AmariNeuralFieldState) Reset() { clear(state.U) }

// SimulateAmariNeuralField preserves the historical homogeneous-drive service API.
func SimulateAmariNeuralField(nSteps int, iExt float64) ([]float64, int) {
	state := NewAmariNeuralField()
	trace := make([]float64, nSteps)
	input := make([]float64, len(state.U))
	for index := range input {
		input[index] = iExt
	}
	for step := range trace {
		rate, err := state.Step(input)
		if err != nil {
			trace[step] = math.NaN()
		} else {
			trace[step] = rate
		}
	}
	return trace, 0
}
