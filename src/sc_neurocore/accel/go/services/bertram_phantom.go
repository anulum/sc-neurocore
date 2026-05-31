// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for bertram_phantom

package services

import "math"

const bertramVoltageMin = -250.0
const bertramVoltageMax = 250.0
const bertramGateTol = 1e-9

type BertramPhantomBursterState struct {
	V          float64
	S1         float64
	S2         float64
	GCa        float64
	GK         float64
	GS1        float64
	GS2        float64
	GL         float64
	ECa        float64
	EK         float64
	EL         float64
	CM         float64
	VM         float64
	SM         float64
	VN         float64
	SN         float64
	VS1        float64
	SS1        float64
	VS2        float64
	SS2        float64
	TauS1      float64
	TauS2      float64
	Dt         float64
	VThreshold float64
}

func NewBertramPhantomBurster() *BertramPhantomBursterState {
	return &BertramPhantomBursterState{
		V: -50.0, S1: 0.1, S2: 0.1,
		GCa: 3.6, GK: 10.0, GS1: 4.0, GS2: 4.0, GL: 0.2,
		ECa: 25.0, EK: -75.0, EL: -40.0, CM: 5.3,
		VM: -20.0, SM: 12.0, VN: -16.0, SN: 5.6,
		VS1: -40.0, SS1: 10.0, VS2: -42.0, SS2: 0.4,
		TauS1: 20000.0, TauS2: 100000.0, Dt: 0.5, VThreshold: -20.0,
	}
}

func bertramFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func bertramPositive(value float64) bool { return bertramFinite(value) && value > 0.0 }

func bertramNonNegative(value float64) bool { return bertramFinite(value) && value >= 0.0 }

func (s *BertramPhantomBursterState) validState() bool {
	return bertramFinite(s.V) && s.V >= bertramVoltageMin && s.V <= bertramVoltageMax &&
		bertramFinite(s.S1) && s.S1 >= 0.0 && s.S1 <= 1.0 &&
		bertramFinite(s.S2) && s.S2 >= 0.0 && s.S2 <= 1.0 &&
		bertramNonNegative(s.GCa) && bertramNonNegative(s.GK) && bertramNonNegative(s.GS1) &&
		bertramNonNegative(s.GS2) && bertramNonNegative(s.GL) &&
		bertramFinite(s.ECa) && bertramFinite(s.EK) && bertramFinite(s.EL) &&
		bertramPositive(s.CM) && bertramFinite(s.VM) && bertramPositive(s.SM) &&
		bertramFinite(s.VN) && bertramPositive(s.SN) && bertramFinite(s.VS1) &&
		bertramPositive(s.SS1) && bertramFinite(s.VS2) && bertramPositive(s.SS2) &&
		bertramPositive(s.TauS1) && bertramPositive(s.TauS2) && bertramPositive(s.Dt) &&
		bertramFinite(s.VThreshold)
}

func bertramBoltz(v float64, vh float64, k float64) float64 {
	z := (vh - v) / k
	if z >= 0.0 {
		expNeg := math.Exp(-z)
		return expNeg / (1.0 + expNeg)
	}
	expPos := math.Exp(z)
	return 1.0 / (1.0 + expPos)
}

func (s *BertramPhantomBursterState) derivatives(v float64, s1 float64, s2 float64, iExt float64) (float64, float64, float64) {
	mInf := bertramBoltz(v, s.VM, s.SM)
	nInf := bertramBoltz(v, s.VN, s.SN)
	s1Inf := bertramBoltz(v, s.VS1, s.SS1)
	s2Inf := bertramBoltz(v, s.VS2, s.SS2)
	iCa := s.GCa * mInf * (v - s.ECa)
	iK := s.GK * nInf * (v - s.EK)
	iS1 := s.GS1 * s1 * (v - s.EK)
	iS2 := s.GS2 * s2 * (v - s.EK)
	iL := s.GL * (v - s.EL)
	dv := (-iCa - iK - iS1 - iS2 - iL + iExt) / s.CM
	ds1 := (s1Inf - s1) / s.TauS1
	ds2 := (s2Inf - s2) / s.TauS2
	return dv, ds1, ds2
}

func bertramCandidateValid(v float64, s1 float64, s2 float64) bool {
	return bertramFinite(v) && v >= bertramVoltageMin && v <= bertramVoltageMax &&
		bertramFinite(s1) && s1 >= -bertramGateTol && s1 <= 1.0+bertramGateTol &&
		bertramFinite(s2) && s2 >= -bertramGateTol && s2 <= 1.0+bertramGateTol
}

func clampBertramGate(value float64) float64 {
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

func (s *BertramPhantomBursterState) Step(iExt float64) int {
	if !bertramFinite(iExt) || !s.validState() {
		return 0
	}
	vPrev := s.V
	dt := s.Dt
	k1v, k1s1, k1s2 := s.derivatives(s.V, s.S1, s.S2, iExt)
	k2v, k2s1, k2s2 := s.derivatives(s.V+0.5*dt*k1v, s.S1+0.5*dt*k1s1, s.S2+0.5*dt*k1s2, iExt)
	k3v, k3s1, k3s2 := s.derivatives(s.V+0.5*dt*k2v, s.S1+0.5*dt*k2s1, s.S2+0.5*dt*k2s2, iExt)
	k4v, k4s1, k4s2 := s.derivatives(s.V+dt*k3v, s.S1+dt*k3s1, s.S2+dt*k3s2, iExt)
	v := s.V + dt*(k1v+2.0*k2v+2.0*k3v+k4v)/6.0
	s1 := s.S1 + dt*(k1s1+2.0*k2s1+2.0*k3s1+k4s1)/6.0
	s2 := s.S2 + dt*(k1s2+2.0*k2s2+2.0*k3s2+k4s2)/6.0
	if !bertramCandidateValid(v, s1, s2) {
		return 0
	}
	s.V = v
	s.S1 = clampBertramGate(s1)
	s.S2 = clampBertramGate(s2)
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

func SimulateBertramPhantomBurster(nSteps int, iExt float64) ([]float64, int) {
	s := NewBertramPhantomBurster()
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
