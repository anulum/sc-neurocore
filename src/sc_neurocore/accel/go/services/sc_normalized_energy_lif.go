// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for retained normalized energy LIF

package services

import "math"

// SCNormalizedEnergyLIFNeuronState holds the frozen project recurrence.
type SCNormalizedEnergyLIFNeuronState struct {
	V, Epsilon, VRest, VReset, VThreshold float64
	TauM, TauE, Alpha, Epsilon0           float64
	Resistance, Dt                        float64
}

// NewSCNormalizedEnergyLIFNeuron constructs the frozen project defaults.
func NewSCNormalizedEnergyLIFNeuron() *SCNormalizedEnergyLIFNeuronState {
	return &SCNormalizedEnergyLIFNeuronState{-70, 1, -70, -70, -50, 10, 500, .1, 1, 1, 1}
}

func (s *SCNormalizedEnergyLIFNeuronState) exactCandidate(current float64) (float64, float64) {
	md := math.Exp(-s.Dt / s.TauM)
	ed := math.Exp(-s.Dt / s.TauE)
	de := s.Epsilon - s.Epsilon0
	epsilon := s.Epsilon0 + de*ed
	steady := s.Epsilon0 * s.TauM * (1 - md)
	rate := 1/s.TauM - 1/s.TauE
	transient := de * md * s.Dt
	if math.Abs(rate) >= 1e-12 {
		transient = de * md * math.Expm1(rate*s.Dt) / rate
	}
	v := s.VRest + (s.V-s.VRest)*md + (s.Resistance*current/s.TauM)*(steady+transient)
	return v, epsilon
}

// Step advances one retained exact-flow sample, returning -1 on invalid input.
func (s *SCNormalizedEnergyLIFNeuronState) Step(current float64) int {
	if !s.Valid() || !isFiniteEnergyLIF(current) {
		return -1
	}
	v, epsilon := s.exactCandidate(current)
	if !isFiniteEnergyLIF(v) || v < -200 || v > 100 || !isFiniteEnergyLIF(epsilon) || epsilon < 0 || epsilon > s.Epsilon0 {
		return -1
	}
	if v >= s.VThreshold && epsilon > .1 {
		s.V, s.Epsilon = s.VReset, math.Max(0, epsilon-s.Alpha)
		return 1
	}
	s.V, s.Epsilon = v, epsilon
	return 0
}

// Valid reports whether the complete retained state is valid.
func (s *SCNormalizedEnergyLIFNeuronState) Valid() bool {
	values := []float64{s.V, s.Epsilon, s.VRest, s.VReset, s.VThreshold, s.TauM, s.TauE, s.Alpha, s.Epsilon0, s.Resistance, s.Dt}
	for _, value := range values {
		if !isFiniteEnergyLIF(value) {
			return false
		}
	}
	return s.V >= -200 && s.V <= 100 && s.VReset >= -200 && s.VReset <= 100 && s.Epsilon >= 0 &&
		s.Epsilon <= s.Epsilon0 && s.TauM > 0 && s.TauE > 0 && s.Alpha >= 0 && s.Epsilon0 >= 0 &&
		s.Resistance > 0 && s.Dt > 0 && s.Dt <= s.TauM && s.Dt <= s.TauE &&
		s.VThreshold > s.VRest && s.VThreshold > s.VReset
}
