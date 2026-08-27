// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained three-state project ion-mass recurrence

package services

import (
	"errors"
	"math"
)

type SCDecoupledAdaptationIonMassNeuronState struct {
	V, W, Z           float64
	GCa, GNa, GK, GL  float64
	VCa, VNa, VK, VL  float64
	Phi, TauK, B, AEe float64
	V0, IExt, Dt      float64
}

func NewSCDecoupledAdaptationIonMassNeuron() *SCDecoupledAdaptationIonMassNeuronState {
	return &SCDecoupledAdaptationIonMassNeuronState{
		V: -0.5, W: 0.0, Z: 0.0, GCa: 1.1, GNa: 6.7, GK: 2.0, GL: 0.5,
		VCa: 1.0, VNa: 0.53, VK: -0.7, VL: -0.5, Phi: 0.7, TauK: 1.0,
		B: 0.1, AEe: 0.36, V0: 0.0, IExt: 0.3, Dt: 0.01,
	}
}

func (s *SCDecoupledAdaptationIonMassNeuronState) derivatives(v, w, z, coupling float64) (float64, float64, float64) {
	mCa := lbSigmoid(v, -0.01, 0.15)
	mNa := lbSigmoid(v, 0.12, 0.15)
	mK := lbSigmoid(v, s.V0, 0.3)
	dv := -s.GCa*mCa*(v-s.VCa) - s.GNa*mNa*(v-s.VNa) - s.GK*w*(v-s.VK) - s.GL*(v-s.VL) + s.IExt + coupling + s.AEe*v
	return dv, s.Phi * (mK - w) / s.TauK, s.B * (v + 0.5 - z)
}

func (s *SCDecoupledAdaptationIonMassNeuronState) valid() bool {
	values := []float64{s.V, s.W, s.Z, s.GCa, s.GNa, s.GK, s.GL, s.VCa, s.VNa, s.VK, s.VL, s.Phi, s.TauK, s.B, s.AEe, s.V0, s.IExt, s.Dt}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return s.W >= 0 && s.W <= 1 && s.GCa > 0 && s.GNa > 0 && s.GK > 0 && s.GL > 0 && s.Phi > 0 && s.TauK > 0 && s.B > 0 && s.Dt > 0
}

func (s *SCDecoupledAdaptationIonMassNeuronState) TryStep(coupling float64) (float64, error) {
	if !s.valid() || math.IsNaN(coupling) || math.IsInf(coupling, 0) {
		return math.NaN(), errors.New("invalid SC ion-mass input")
	}
	v, w, z, dt := s.V, s.W, s.Z, s.Dt
	k1v, k1w, k1z := s.derivatives(v, w, z, coupling)
	k2v, k2w, k2z := s.derivatives(v+0.5*dt*k1v, w+0.5*dt*k1w, z+0.5*dt*k1z, coupling)
	k3v, k3w, k3z := s.derivatives(v+0.5*dt*k2v, w+0.5*dt*k2w, z+0.5*dt*k2z, coupling)
	k4v, k4w, k4z := s.derivatives(v+dt*k3v, w+dt*k3w, z+dt*k3z, coupling)
	next := *s
	next.V = v + dt*(k1v+2*k2v+2*k3v+k4v)/6
	next.W = w + dt*(k1w+2*k2w+2*k3w+k4w)/6
	next.Z = z + dt*(k1z+2*k2z+2*k3z+k4z)/6
	if !next.valid() {
		return math.NaN(), errors.New("invalid SC ion-mass candidate")
	}
	*s = next
	return s.V, nil
}

func (s *SCDecoupledAdaptationIonMassNeuronState) Step(coupling float64) float64 {
	value, err := s.TryStep(coupling)
	if err != nil {
		return math.NaN()
	}
	return value
}

func (s *SCDecoupledAdaptationIonMassNeuronState) Reset() { s.V, s.W, s.Z = -0.5, 0.0, 0.0 }
