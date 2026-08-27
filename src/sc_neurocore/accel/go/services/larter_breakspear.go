// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Larter-Breakspear source dynamics

package services

import (
	"errors"
	"math"
)

// LarterBreakspearNeuronState holds the complete source state and configuration.
type LarterBreakspearNeuronState struct {
	V, W, Z                    float64
	GCa, GNa, GK, GL           float64
	VCa, VNa, VK, VL           float64
	TCa, TNa, TK               float64
	DeltaCa, DeltaNa, DeltaK   float64
	Phi, TauK, B               float64
	AEe, AEi, AIe, ANe, ANi    float64
	RNMDA, CouplingBalance     float64
	VT, ZT, DeltaV, DeltaZ     float64
	QVMax, QZMax, IExt, TScale float64
	Dt                         float64
}

// NewLarterBreakspearNeuron returns the maintained source-profile configuration.
func NewLarterBreakspearNeuron() *LarterBreakspearNeuronState {
	return &LarterBreakspearNeuronState{
		V: 0.1, W: 0.1, Z: 0.1,
		GCa: 1.1, GNa: 6.7, GK: 2.0, GL: 0.5,
		VCa: 1.0, VNa: 0.53, VK: -0.7, VL: -0.5,
		TCa: -0.01, TNa: 0.3, TK: 0.0,
		DeltaCa: 0.15, DeltaNa: 0.15, DeltaK: 0.3,
		Phi: 0.7, TauK: 1.0, B: 0.1,
		AEe: 0.4, AEi: 2.0, AIe: 2.0, ANe: 1.0, ANi: 0.4,
		RNMDA: 0.25, CouplingBalance: 0.1,
		VT: 0.0, ZT: 0.0, DeltaV: 0.65, DeltaZ: 0.7,
		QVMax: 1.0, QZMax: 1.0, IExt: 0.3, TScale: 1.0, Dt: 0.01,
	}
}

func lbSigmoid(value, threshold, width float64) float64 {
	return 0.5 * (1.0 + math.Tanh((value-threshold)/width))
}

func (s *LarterBreakspearNeuronState) derivatives(v, w, z, coupling float64) (float64, float64, float64) {
	mCa := lbSigmoid(v, s.TCa, s.DeltaCa)
	mNa := lbSigmoid(v, s.TNa, s.DeltaNa)
	mK := lbSigmoid(v, s.TK, s.DeltaK)
	qV := s.QVMax * lbSigmoid(v, s.VT, s.DeltaV)
	qZ := s.QZMax * lbSigmoid(z, s.ZT, s.DeltaZ)
	excitation := s.AEe * ((1.0-s.CouplingBalance)*qV + s.CouplingBalance*coupling)
	dv := -(s.GCa+s.RNMDA*excitation)*mCa*(v-s.VCa) - s.GK*w*(v-s.VK) - s.GL*(v-s.VL) - (s.GNa*mNa+excitation)*(v-s.VNa) - s.AIe*z*qZ + s.ANe*s.IExt
	dw := s.Phi * (mK - w) / s.TauK
	dz := s.B * (s.ANi*s.IExt + s.AEi*v*qV)
	return s.TScale * dv, s.TScale * dw, s.TScale * dz
}

// TryStep advances one classical-RK4 step and leaves state unchanged on failure.
func (s *LarterBreakspearNeuronState) TryStep(coupling float64) (float64, error) {
	if !validateLarterBreakspearState(s) || !finiteLarterBreakspear(coupling) {
		return math.NaN(), errors.New("invalid Larter-Breakspear state, configuration, or coupling")
	}
	v0, w0, z0, dt := s.V, s.W, s.Z, s.Dt
	k1v, k1w, k1z := s.derivatives(v0, w0, z0, coupling)
	k2v, k2w, k2z := s.derivatives(v0+0.5*dt*k1v, w0+0.5*dt*k1w, z0+0.5*dt*k1z, coupling)
	k3v, k3w, k3z := s.derivatives(v0+0.5*dt*k2v, w0+0.5*dt*k2w, z0+0.5*dt*k2z, coupling)
	k4v, k4w, k4z := s.derivatives(v0+dt*k3v, w0+dt*k3w, z0+dt*k3z, coupling)
	next := *s
	next.V = v0 + dt*(k1v+2*k2v+2*k3v+k4v)/6
	next.W = w0 + dt*(k1w+2*k2w+2*k3w+k4w)/6
	next.Z = z0 + dt*(k1z+2*k2z+2*k3z+k4z)/6
	if !validateLarterBreakspearState(&next) {
		return math.NaN(), errors.New("Larter-Breakspear candidate is invalid")
	}
	*s = next
	return s.V, nil
}

// Step is the compatibility facade; invalid inputs return NaN without mutation.
func (s *LarterBreakspearNeuronState) Step(coupling float64) float64 {
	value, err := s.TryStep(coupling)
	if err != nil {
		return math.NaN()
	}
	return value
}

// Reset restores source-profile dynamic state while preserving configuration.
func (s *LarterBreakspearNeuronState) Reset() { s.V, s.W, s.Z = 0.1, 0.1, 0.1 }

// SimulateLarterBreakspearNeuron returns the continuous voltage trace.
func SimulateLarterBreakspearNeuron(nSteps int, coupling float64) ([]float64, int) {
	state := NewLarterBreakspearNeuron()
	trace := make([]float64, nSteps)
	for index := range trace {
		trace[index] = state.Step(coupling)
	}
	return trace, 0
}

func finiteLarterBreakspear(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func validateLarterBreakspearState(s *LarterBreakspearNeuronState) bool {
	values := []float64{s.V, s.W, s.Z, s.GCa, s.GNa, s.GK, s.GL, s.VCa, s.VNa, s.VK, s.VL, s.TCa, s.TNa, s.TK, s.DeltaCa, s.DeltaNa, s.DeltaK, s.Phi, s.TauK, s.B, s.AEe, s.AEi, s.AIe, s.ANe, s.ANi, s.RNMDA, s.CouplingBalance, s.VT, s.ZT, s.DeltaV, s.DeltaZ, s.QVMax, s.QZMax, s.IExt, s.TScale, s.Dt}
	for _, value := range values {
		if !finiteLarterBreakspear(value) {
			return false
		}
	}
	return s.W >= 0 && s.W <= 1 && s.GCa >= 0 && s.GNa >= 0 && s.GK >= 0 && s.GL >= 0 && s.RNMDA >= 0 && s.QVMax >= 0 && s.QZMax >= 0 && s.DeltaCa > 0 && s.DeltaNa > 0 && s.DeltaK > 0 && s.DeltaV > 0 && s.DeltaZ > 0 && s.Phi > 0 && s.TauK > 0 && s.B > 0 && s.TScale > 0 && s.Dt > 0 && s.CouplingBalance >= 0 && s.CouplingBalance <= 1
}
