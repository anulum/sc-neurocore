// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for fractional_lif

package services

import (
	"math"
)

// FractionalLIFNeuronState holds the neuron state
type FractionalLIFNeuronState struct {
	V          float64
	VRest      float64
	VReset     float64
	VThreshold float64
	Alpha      float64
	Resistance float64
	Dt         float64
	MaxHistory int
	History    []float64
	GLCoeffs   []float64
}

// NewFractionalLIFNeuron creates a new FractionalLIFNeuron neuron with default parameters
func NewFractionalLIFNeuron() *FractionalLIFNeuronState {
	return &FractionalLIFNeuronState{
		V:          0.0,
		VRest:      0.0,
		VReset:     0.0,
		VThreshold: 1.0,
		Alpha:      0.8,
		Resistance: 1.0,
		Dt:         1.0,
		MaxHistory: 100,
		History:    make([]float64, 100),
		GLCoeffs:   computeFractionalLIFGLCoefficients(0.8, 100),
	}
}

// Step advances the neuron by one timestep
func (s *FractionalLIFNeuronState) Step(iExt float64) int {
	if !s.Valid() || !isFiniteFractionalLIF(iExt) {
		return 0
	}

	rhs := -(s.V - s.VRest) + s.Resistance*iExt
	terms := minFractionalLIF(len(s.History), minFractionalLIF(s.MaxHistory, len(s.GLCoeffs)))
	glSum := 0.0
	for k := 1; k < terms; k++ {
		glSum += s.GLCoeffs[k] * s.History[len(s.History)-k]
	}
	s.V = rhs*math.Pow(s.Dt, s.Alpha) - glSum
	s.History = append(s.History, s.V)
	if len(s.History) > s.MaxHistory {
		s.History = s.History[1:]
	}
	if s.V >= s.VThreshold {
		s.V = s.VReset
		s.History[len(s.History)-1] = s.VReset
		return 1
	}
	return 0
}

// Valid returns true when the state satisfies the fractional-LIF physics contract.
func (s *FractionalLIFNeuronState) Valid() bool {
	return isFiniteFractionalLIF(s.V) &&
		isFiniteFractionalLIF(s.VRest) &&
		isFiniteFractionalLIF(s.VReset) &&
		isFiniteFractionalLIF(s.VThreshold) &&
		isFiniteFractionalLIF(s.Alpha) &&
		s.Alpha > 0.0 &&
		s.Alpha <= 1.0 &&
		isFiniteFractionalLIF(s.Resistance) &&
		s.Resistance > 0.0 &&
		isFiniteFractionalLIF(s.Dt) &&
		s.Dt > 0.0 &&
		s.MaxHistory > 1 &&
		len(s.History) == s.MaxHistory &&
		len(s.GLCoeffs) == s.MaxHistory &&
		allFiniteFractionalLIF(s.History) &&
		allFiniteFractionalLIF(s.GLCoeffs)
}

func computeFractionalLIFGLCoefficients(alpha float64, maxHistory int) []float64 {
	coeffs := make([]float64, maxHistory)
	coeffs[0] = 1.0
	for k := 1; k < maxHistory; k++ {
		coeffs[k] = coeffs[k-1] * (float64(k) - 1.0 - alpha) / float64(k)
	}
	return coeffs
}

func allFiniteFractionalLIF(values []float64) bool {
	for _, value := range values {
		if !isFiniteFractionalLIF(value) {
			return false
		}
	}
	return true
}

func isFiniteFractionalLIF(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func minFractionalLIF(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

// SimulateFractionalLIFNeuron runs the neuron for n steps
func SimulateFractionalLIFNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewFractionalLIFNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
