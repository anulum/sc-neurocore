// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for Prescott 2008 RK4 dynamics

package services

import "math"

// PrescottNeuronState holds the neuron state.
type PrescottNeuronState struct {
	V          float64
	W          float64
	GFast      float64
	GSlow      float64
	GL         float64
	EFast      float64
	ESlow      float64
	EL         float64
	BetaW      float64
	GammaW     float64
	TauW       float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewPrescottNeuron creates a new PrescottNeuron neuron with default parameters.
func NewPrescottNeuron() *PrescottNeuronState {
	return &PrescottNeuronState{
		V: -65.0, W: 0.0, GFast: 20.0, GSlow: 20.0, GL: 2.0,
		EFast: 50.0, ESlow: -100.0, EL: -70.0, BetaW: -21.0,
		GammaW: 15.0, TauW: 100.0, Phi: 0.15, Dt: 0.1, VThreshold: -20.0,
	}
}

func prescottSigmoid(x float64) float64 {
	if x >= 0.0 {
		z := math.Exp(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}

func prescottFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func prescottValidState(v float64, w float64) bool {
	return prescottFinite(v) && prescottFinite(w) && w >= 0.0 && w <= 1.0
}

func (s *PrescottNeuronState) validRuntime() bool {
	return prescottValidState(s.V, s.W) &&
		prescottFinite(s.GFast) && s.GFast >= 0.0 &&
		prescottFinite(s.GSlow) && s.GSlow >= 0.0 &&
		prescottFinite(s.GL) && s.GL >= 0.0 &&
		prescottFinite(s.EFast) && prescottFinite(s.ESlow) && prescottFinite(s.EL) &&
		prescottFinite(s.BetaW) && prescottFinite(s.GammaW) && s.GammaW > 0.0 &&
		prescottFinite(s.TauW) && s.TauW > 0.0 &&
		prescottFinite(s.Phi) && s.Phi >= 0.0 &&
		prescottFinite(s.Dt) && s.Dt > 0.0 && prescottFinite(s.VThreshold)
}

func (s *PrescottNeuronState) derivatives(v float64, w float64, iExt float64) (float64, float64, bool) {
	if !prescottValidState(v, w) {
		return 0.0, 0.0, false
	}
	mInf := prescottSigmoid((v + 20.0) / 15.0)
	wInf := prescottSigmoid((v - s.BetaW) / s.GammaW)
	iFast := s.GFast * mInf * (v - s.EFast)
	iSlow := s.GSlow * w * (v - s.ESlow)
	iL := s.GL * (v - s.EL)
	dv := -iFast - iSlow - iL + iExt
	dw := s.Phi * (wInf - w) / s.TauW
	return dv, dw, prescottFinite(dv) && prescottFinite(dw)
}

func (s *PrescottNeuronState) rk4Step(iExt float64) (float64, float64, bool) {
	dt := s.Dt
	k1V, k1W, ok := s.derivatives(s.V, s.W, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	k2V, k2W, ok := s.derivatives(s.V+0.5*dt*k1V, s.W+0.5*dt*k1W, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	k3V, k3W, ok := s.derivatives(s.V+0.5*dt*k2V, s.W+0.5*dt*k2W, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	k4V, k4W, ok := s.derivatives(s.V+dt*k3V, s.W+dt*k3W, iExt)
	if !ok {
		return 0.0, 0.0, false
	}
	nextV := s.V + dt*(k1V+2.0*k2V+2.0*k3V+k4V)/6.0
	nextW := s.W + dt*(k1W+2.0*k2W+2.0*k3W+k4W)/6.0
	return nextV, nextW, prescottValidState(nextV, nextW)
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *PrescottNeuronState) Step(iExt float64) int {
	if !prescottFinite(iExt) || !s.validRuntime() {
		return 0
	}
	vPrev := s.V
	nextV, nextW, ok := s.rk4Step(iExt)
	if !ok {
		return 0
	}
	s.V = nextV
	s.W = nextW
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulatePrescottNeuron runs the neuron for n steps.
func SimulatePrescottNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPrescottNeuron()
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
