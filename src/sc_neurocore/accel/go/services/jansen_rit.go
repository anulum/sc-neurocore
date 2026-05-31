// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for jansen_rit

package services

import (
	"math"
)

// JansenRitUnitState holds the neuron state
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

// NewJansenRitUnit creates a new JansenRitUnit neuron with default parameters
func NewJansenRitUnit() *JansenRitUnitState {
	return &JansenRitUnitState{
		Y0:    0.0,
		Y3:    0.0,
		Y1:    0.0,
		Y4:    0.0,
		Y2:    0.0,
		Y5:    0.0,
		AExc:  3.25,
		BExc:  22.0,
		ARate: 100.0,
		BRate: 50.0,
		C:     135.0,
		E0:    2.5,
		V0:    6.0,
		R:     0.56,
		Dt:    0.001,
	}
}

// Step advances the neuron by one timestep
func (s *JansenRitUnitState) Step(iExt float64) int {
	if !finiteJansenRit(iExt) || !ValidateJansenRitUnit(s) {
		return -1
	}
	s1 := s.sigmoid(s.Y1 - s.Y2)
	s0 := s.sigmoid(s.C * 0.8 * s.Y0)
	s2 := s.sigmoid(s.C * 0.25 * s.Y0)

	dy0 := s.Y3
	dy3 := s.AExc*s.ARate*s1 - 2.0*s.ARate*s.Y3 - math.Pow(s.ARate, 2)*s.Y0
	dy1 := s.Y4
	dy4 := s.AExc*s.ARate*(iExt+s.C*0.8*s0) - 2.0*s.ARate*s.Y4 - math.Pow(s.ARate, 2)*s.Y1
	dy2 := s.Y5
	dy5 := s.BExc*s.BRate*s.C*0.25*s2 - 2.0*s.BRate*s.Y5 - math.Pow(s.BRate, 2)*s.Y2

	next := *s
	next.Y0 += dy0 * s.Dt
	next.Y3 += dy3 * s.Dt
	next.Y1 += dy1 * s.Dt
	next.Y4 += dy4 * s.Dt
	next.Y2 += dy2 * s.Dt
	next.Y5 += dy5 * s.Dt
	if !ValidateJansenRitUnit(&next) {
		return -1
	}
	*s = next
	return 0
}

// SimulateJansenRitUnit runs the neuron for n steps
func SimulateJansenRitUnit(nSteps int, iExt float64) ([]float64, int) {
	s := NewJansenRitUnit()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Y1 - s.Y2
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

func finiteJansenRit(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func ValidateJansenRitUnit(s *JansenRitUnitState) bool {
	if s == nil {
		return false
	}
	return finiteJansenRit(
		s.Y0, s.Y3, s.Y1, s.Y4, s.Y2, s.Y5,
		s.AExc, s.BExc, s.ARate, s.BRate, s.C, s.E0, s.V0, s.R, s.Dt,
	) &&
		s.AExc > 0.0 &&
		s.BExc > 0.0 &&
		s.ARate > 0.0 &&
		s.BRate > 0.0 &&
		s.C >= 0.0 &&
		s.E0 > 0.0 &&
		s.R > 0.0 &&
		s.Dt > 0.0
}

func (s *JansenRitUnitState) sigmoid(x float64) float64 {
	if !finiteJansenRit(x) {
		return math.NaN()
	}
	exponent := s.R * (s.V0 - x)
	if exponent >= 0.0 {
		expNeg := math.Exp(-exponent)
		return 2.0 * s.E0 * expNeg / (1.0 + expNeg)
	}
	return 2.0 * s.E0 / (1.0 + math.Exp(exponent))
}
