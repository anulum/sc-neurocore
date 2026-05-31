// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wendling

package services

import (
	"math"
)

// WendlingNeuronState holds the neuron state
type WendlingNeuronState struct {
	Y0    float64
	Y5    float64
	Y1    float64
	Y6    float64
	Y2    float64
	Y7    float64
	Y3    float64
	Y8    float64
	Y4    float64
	Y9    float64
	AExc  float64
	BFast float64
	GSlow float64
	ARate float64
	BRate float64
	GRate float64
	C     float64
	E0    float64
	V0    float64
	R     float64
	Dt    float64
}

// NewWendlingNeuron creates a new WendlingNeuron neuron with default parameters
func NewWendlingNeuron() *WendlingNeuronState {
	return &WendlingNeuronState{
		Y0:    0.0,
		Y5:    0.0,
		Y1:    0.0,
		Y6:    0.0,
		Y2:    0.0,
		Y7:    0.0,
		Y3:    0.0,
		Y8:    0.0,
		Y4:    0.0,
		Y9:    0.0,
		AExc:  3.25,
		BFast: 22.0,
		GSlow: 10.0,
		ARate: 100.0,
		BRate: 500.0,
		GRate: 20.0,
		C:     135.0,
		E0:    2.5,
		V0:    6.0,
		R:     0.56,
		Dt:    0.001,
	}
}

func finiteWendling(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func ValidateWendlingNeuron(s *WendlingNeuronState) bool {
	if s == nil {
		return false
	}
	return finiteWendling(
		s.Y0, s.Y5, s.Y1, s.Y6, s.Y2, s.Y7, s.Y3, s.Y8, s.Y4, s.Y9,
		s.AExc, s.BFast, s.GSlow, s.ARate, s.BRate, s.GRate, s.C, s.E0, s.V0, s.R, s.Dt,
	) &&
		s.AExc > 0.0 &&
		s.BFast > 0.0 &&
		s.GSlow > 0.0 &&
		s.ARate > 0.0 &&
		s.BRate > 0.0 &&
		s.GRate > 0.0 &&
		s.C >= 0.0 &&
		s.E0 > 0.0 &&
		s.R > 0.0 &&
		s.Dt > 0.0
}

func (s *WendlingNeuronState) sigmoid(x float64) float64 {
	if !finiteWendling(x) {
		return math.NaN()
	}
	exponent := s.R * (s.V0 - x)
	if exponent >= 0.0 {
		expNeg := math.Exp(-exponent)
		return 2.0 * s.E0 * expNeg / (1.0 + expNeg)
	}
	return 2.0 * s.E0 / (1.0 + math.Exp(exponent))
}

// Step advances the neuron by one timestep
func (s *WendlingNeuronState) Step(iExt float64) int {
	if !finiteWendling(iExt) || !ValidateWendlingNeuron(s) {
		return -1
	}
	sig123 := s.sigmoid(s.Y1 - s.Y2 - s.Y3)
	sig0 := s.sigmoid(s.C * 0.8 * s.Y0)
	sigFast := s.sigmoid(s.C * 0.25 * s.Y0)
	sigSlow := s.sigmoid(s.C * 0.1 * s.Y0)

	dy0 := s.Y5
	dy5 := s.AExc*s.ARate*sig123 - 2.0*s.ARate*s.Y5 - math.Pow(s.ARate, 2)*s.Y0
	dy1 := s.Y6
	dy6 := s.AExc*s.ARate*(iExt+s.C*0.8*sig0) - 2.0*s.ARate*s.Y6 - math.Pow(s.ARate, 2)*s.Y1
	dy2 := s.Y7
	dy7 := s.BFast*s.BRate*s.C*0.25*sigFast - 2.0*s.BRate*s.Y7 - math.Pow(s.BRate, 2)*s.Y2
	dy3 := s.Y8
	dy8 := s.GSlow*s.GRate*s.C*0.1*sigSlow - 2.0*s.GRate*s.Y8 - math.Pow(s.GRate, 2)*s.Y3

	next := *s
	next.Y0 += dy0 * s.Dt
	next.Y5 += dy5 * s.Dt
	next.Y1 += dy1 * s.Dt
	next.Y6 += dy6 * s.Dt
	next.Y2 += dy2 * s.Dt
	next.Y7 += dy7 * s.Dt
	next.Y3 += dy3 * s.Dt
	next.Y8 += dy8 * s.Dt
	if !ValidateWendlingNeuron(&next) {
		return -1
	}
	*s = next
	return 0
}

// SimulateWendlingNeuron runs the neuron for n steps
func SimulateWendlingNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewWendlingNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.Y1 - s.Y2 - s.Y3
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
