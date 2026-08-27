// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Wang 1999 NMDA-autapse neuron

package services

import (
	"errors"
	"math"
)

// NMDANeuronState holds the complete Wang 1999 scalar source contract.
type NMDANeuronState struct {
	V, XNmda, SNmda, Ca, RefractoryRemaining float64
	CM, GL, VL, GNmda, ENmda, MgConc         float64
	AlphaX, TauX, AlphaS, TauS, KineticScale float64
	GAHP, VK, AlphaCa, TauCa                 float64
	Dt, VThreshold, VReset, RefractoryPeriod float64
}

// NewNMDANeuron creates the source NMDA-autapse profile.
func NewNMDANeuron() *NMDANeuronState {
	return &NMDANeuronState{
		V: -70, CM: 0.5, GL: 0.025, VL: -70, GNmda: 0.1,
		MgConc: 1, AlphaX: 1, TauX: 2, AlphaS: 1, TauS: 80, KineticScale: 1,
		VK: -85, AlphaCa: 0.2, TauCa: 80, Dt: 0.05,
		VThreshold: -52, VReset: -59, RefractoryPeriod: 2,
	}
}

func nmdaFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func nmdaBetween(value, lower, upper float64) bool { return value >= lower && value <= upper }

// ValidNMDANeuron validates all public state and configuration fields.
func ValidNMDANeuron(s *NMDANeuronState) bool {
	if s == nil || !nmdaFinite(
		s.V, s.XNmda, s.SNmda, s.Ca, s.RefractoryRemaining, s.CM, s.GL, s.VL, s.GNmda, s.ENmda,
		s.MgConc, s.AlphaX, s.TauX, s.AlphaS, s.TauS, s.KineticScale, s.GAHP, s.VK, s.AlphaCa,
		s.TauCa, s.Dt, s.VThreshold, s.VReset, s.RefractoryPeriod) {
		return false
	}
	return nmdaBetween(s.V, -120, 80) && s.XNmda >= 0 && nmdaBetween(s.SNmda, 0, 1) && s.Ca >= 0 &&
		nmdaBetween(s.RefractoryRemaining, 0, s.RefractoryPeriod) && nmdaBetween(s.CM, 0.01, 10) &&
		nmdaBetween(s.GL, 0, 1) && nmdaBetween(s.VL, -100, -40) && nmdaBetween(s.GNmda, 0, 2) &&
		nmdaBetween(s.ENmda, -10, 10) && nmdaBetween(s.MgConc, 0, 5) && nmdaBetween(s.AlphaX, 0, 10) &&
		nmdaBetween(s.TauX, 0.01, 100) && nmdaBetween(s.AlphaS, 0, 10) && nmdaBetween(s.TauS, 1, 1000) &&
		nmdaBetween(s.KineticScale, 0.01, 100) && nmdaBetween(s.GAHP, 0, 10) && nmdaBetween(s.VK, -120, -40) &&
		nmdaBetween(s.AlphaCa, 0, 10) && nmdaBetween(s.TauCa, 1, 1000) && s.Dt > 0 && s.Dt <= 0.05 &&
		nmdaBetween(s.VThreshold, -80, -30) && s.VReset >= -100 && s.VReset < s.VThreshold &&
		nmdaBetween(s.RefractoryPeriod, 0, 20)
}

func (s *NMDANeuronState) derivatives(v, x, g, ca, current float64) [4]float64 {
	block := 1 / (1 + s.MgConc*math.Exp(-0.062*v)/3.57)
	iL := s.GL * (v - s.VL)
	iAHP := s.GAHP * ca * (v - s.VK)
	iNmda := s.GNmda * g * block * (v - s.ENmda)
	return [4]float64{(-iL - iAHP - iNmda + current) / s.CM, s.KineticScale * (-x / s.TauX),
		s.KineticScale * (s.AlphaS*x*(1-g) - g/s.TauS), -ca / s.TauCa}
}

func (s *NMDANeuronState) rk2(v, current float64) [4]float64 {
	y := [4]float64{v, s.XNmda, s.SNmda, s.Ca}
	k1 := s.derivatives(y[0], y[1], y[2], y[3], current)
	h := 0.5 * s.Dt
	m := [4]float64{y[0] + h*k1[0], y[1] + h*k1[1], y[2] + h*k1[2], y[3] + h*k1[3]}
	k2 := s.derivatives(m[0], m[1], m[2], m[3], current)
	return [4]float64{y[0] + s.Dt*k2[0], y[1] + s.Dt*k2[1], y[2] + s.Dt*k2[2], y[3] + s.Dt*k2[3]}
}

// TryStep advances one source-grid step atomically.
func (s *NMDANeuronState) TryStep(current float64) (int, error) {
	if !nmdaFinite(current) {
		return 0, errors.New("current must be finite")
	}
	if !ValidNMDANeuron(s) {
		return 0, errors.New("NMDA state and parameters must satisfy the public bounds")
	}
	held := s.RefractoryRemaining > 0
	voltage := s.V
	if held {
		voltage = s.VReset
	}
	y := s.rk2(voltage, current)
	refractory := math.Max(0, s.RefractoryRemaining-s.Dt)
	event := 0
	if held {
		y[0] = s.VReset
	} else if y[0] >= s.VThreshold {
		event = 1
		y[0] = s.VReset
		refractory = s.RefractoryPeriod
		y[1] += s.KineticScale * s.AlphaX
		y[3] += s.AlphaCa
	}
	if !nmdaFinite(y[0], y[1], y[2], y[3], refractory) {
		return 0, errors.New("NMDA candidate state became non-finite")
	}
	s.V = math.Max(-120, math.Min(80, y[0]))
	s.XNmda = math.Max(0, y[1])
	s.SNmda = math.Max(0, math.Min(1, y[2]))
	s.Ca = math.Max(0, y[3])
	s.RefractoryRemaining = refractory
	return event, nil
}

// Step is the fail-closed legacy wrapper.
func (s *NMDANeuronState) Step(current float64) int {
	event, err := s.TryStep(current)
	if err != nil {
		return 0
	}
	return event
}

// Reset restores dynamic source state.
func (s *NMDANeuronState) Reset() {
	s.V = s.VL
	s.XNmda = 0
	s.SNmda = 0
	s.Ca = 0
	s.RefractoryRemaining = 0
}

// SimulateNMDANeuron runs a fresh source state.
func SimulateNMDANeuron(nSteps int, current float64) ([]float64, int) {
	s := NewNMDANeuron()
	trace := make([]float64, nSteps)
	events := 0
	for i := range trace {
		events += s.Step(current)
		trace[i] = s.V
	}
	return trace, events
}
