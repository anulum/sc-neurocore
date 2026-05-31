// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for larter_breakspear

package services

import (
	"math"
)

// LarterBreakspearNeuronState holds the neuron state
type LarterBreakspearNeuronState struct {
	V    float64
	W    float64
	Z    float64
	GCa  float64
	GNa  float64
	GK   float64
	VCa  float64
	VNa  float64
	VK   float64
	VL   float64
	GL   float64
	Phi  float64
	TauK float64
	B    float64
	AEe  float64
	V0   float64
	IExt float64
	Dt   float64
}

// NewLarterBreakspearNeuron creates a new LarterBreakspearNeuron neuron with default parameters
func NewLarterBreakspearNeuron() *LarterBreakspearNeuronState {
	return &LarterBreakspearNeuronState{
		V:    -0.5,
		W:    0.0,
		Z:    0.0,
		GCa:  1.1,
		GNa:  6.7,
		GK:   2.0,
		VCa:  1.0,
		VNa:  0.53,
		VK:   -0.7,
		VL:   -0.5,
		GL:   0.5,
		Phi:  0.7,
		TauK: 1.0,
		B:    0.1,
		AEe:  0.36,
		V0:   0.0,
		IExt: 0.3,
		Dt:   0.01,
	}
}

// Step advances the neuron by one timestep
func (s *LarterBreakspearNeuronState) Step(coupling float64) float64 {
	if !validateLarterBreakspearState(s) || !finiteLarterBreakspear(coupling) {
		return math.NaN()
	}

	v0, w0, z0 := s.V, s.W, s.Z
	dt := s.Dt
	k1v, k1w, k1z := s.derivatives(v0, w0, z0, coupling)
	k2v, k2w, k2z := s.derivatives(v0+0.5*dt*k1v, w0+0.5*dt*k1w, z0+0.5*dt*k1z, coupling)
	k3v, k3w, k3z := s.derivatives(v0+0.5*dt*k2v, w0+0.5*dt*k2w, z0+0.5*dt*k2z, coupling)
	k4v, k4w, k4z := s.derivatives(v0+dt*k3v, w0+dt*k3w, z0+dt*k3z, coupling)

	next := *s
	next.V = v0 + (dt/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v)
	next.W = w0 + (dt/6.0)*(k1w+2.0*k2w+2.0*k3w+k4w)
	next.Z = z0 + (dt/6.0)*(k1z+2.0*k2z+2.0*k3z+k4z)
	if !validateLarterBreakspearState(&next) {
		return math.NaN()
	}
	*s = next
	return s.V
}

// SimulateLarterBreakspearNeuron runs the neuron for n steps
func SimulateLarterBreakspearNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewLarterBreakspearNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.V
		if math.IsNaN(result) {
			break
		}
	}
	return trace, spikes
}

func (s *LarterBreakspearNeuronState) mCa(v float64) float64 {
	return 0.5 * (1.0 + math.Tanh((v-(-0.01))/0.15))
}

func (s *LarterBreakspearNeuronState) mNa(v float64) float64 {
	return 0.5 * (1.0 + math.Tanh((v-0.12)/0.15))
}

func (s *LarterBreakspearNeuronState) mK(v float64) float64 {
	return 0.5 * (1.0 + math.Tanh((v-s.V0)/0.3))
}

func (s *LarterBreakspearNeuronState) derivatives(v float64, w float64, z float64, coupling float64) (float64, float64, float64) {
	iCa := s.GCa * s.mCa(v) * (v - s.VCa)
	iNa := s.GNa * s.mNa(v) * (v - s.VNa)
	iK := s.GK * w * (v - s.VK)
	iL := s.GL * (v - s.VL)

	dv := -iCa - iNa - iK - iL + s.IExt + coupling + s.AEe*v
	dw := s.Phi * (s.mK(v) - w) / s.TauK
	dz := s.B * (v + 0.5 - z)
	return dv, dw, dz
}

func finiteLarterBreakspear(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func validateLarterBreakspearState(s *LarterBreakspearNeuronState) bool {
	return finiteLarterBreakspear(s.V) &&
		finiteLarterBreakspear(s.W) &&
		finiteLarterBreakspear(s.Z) &&
		finiteLarterBreakspear(s.GCa) &&
		finiteLarterBreakspear(s.GNa) &&
		finiteLarterBreakspear(s.GK) &&
		finiteLarterBreakspear(s.VCa) &&
		finiteLarterBreakspear(s.VNa) &&
		finiteLarterBreakspear(s.VK) &&
		finiteLarterBreakspear(s.VL) &&
		finiteLarterBreakspear(s.GL) &&
		finiteLarterBreakspear(s.Phi) &&
		finiteLarterBreakspear(s.TauK) &&
		finiteLarterBreakspear(s.B) &&
		finiteLarterBreakspear(s.AEe) &&
		finiteLarterBreakspear(s.V0) &&
		finiteLarterBreakspear(s.IExt) &&
		finiteLarterBreakspear(s.Dt) &&
		s.Dt > 0.0 &&
		s.TauK > 0.0 &&
		s.Phi > 0.0 &&
		s.B > 0.0 &&
		s.GCa > 0.0 &&
		s.GNa > 0.0 &&
		s.GK > 0.0 &&
		s.GL > 0.0 &&
		s.W >= 0.0 &&
		s.W <= 1.0
}
