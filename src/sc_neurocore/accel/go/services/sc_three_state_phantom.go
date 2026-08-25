// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for retained three-state project phantom

package services

import "math"

const threeStatePhantomVoltageMin = -250.0
const threeStatePhantomVoltageMax = 250.0
const threeStatePhantomGateTol = 1e-9

type SCThreeStatePhantomState struct {
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

func NewSCThreeStatePhantom() *SCThreeStatePhantomState {
	return &SCThreeStatePhantomState{
		V: -50.0, S1: 0.1, S2: 0.1,
		GCa: 3.6, GK: 10.0, GS1: 4.0, GS2: 4.0, GL: 0.2,
		ECa: 25.0, EK: -75.0, EL: -40.0, CM: 5.3,
		VM: -20.0, SM: 12.0, VN: -16.0, SN: 5.6,
		VS1: -40.0, SS1: 10.0, VS2: -42.0, SS2: 0.4,
		TauS1: 20000.0, TauS2: 100000.0, Dt: 0.5, VThreshold: -20.0,
	}
}

func threeStatePhantomFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func threeStatePhantomPositive(value float64) bool {
	return threeStatePhantomFinite(value) && value > 0.0
}

func threeStatePhantomNonNegative(value float64) bool {
	return threeStatePhantomFinite(value) && value >= 0.0
}

func (s *SCThreeStatePhantomState) validState() bool {
	return threeStatePhantomFinite(s.V) && s.V >= threeStatePhantomVoltageMin && s.V <= threeStatePhantomVoltageMax &&
		threeStatePhantomFinite(s.S1) && s.S1 >= 0.0 && s.S1 <= 1.0 &&
		threeStatePhantomFinite(s.S2) && s.S2 >= 0.0 && s.S2 <= 1.0 &&
		threeStatePhantomNonNegative(s.GCa) && threeStatePhantomNonNegative(s.GK) && threeStatePhantomNonNegative(s.GS1) &&
		threeStatePhantomNonNegative(s.GS2) && threeStatePhantomNonNegative(s.GL) &&
		threeStatePhantomFinite(s.ECa) && threeStatePhantomFinite(s.EK) && threeStatePhantomFinite(s.EL) &&
		threeStatePhantomPositive(s.CM) && threeStatePhantomFinite(s.VM) && threeStatePhantomPositive(s.SM) &&
		threeStatePhantomFinite(s.VN) && threeStatePhantomPositive(s.SN) && threeStatePhantomFinite(s.VS1) &&
		threeStatePhantomPositive(s.SS1) && threeStatePhantomFinite(s.VS2) && threeStatePhantomPositive(s.SS2) &&
		threeStatePhantomPositive(s.TauS1) && threeStatePhantomPositive(s.TauS2) && threeStatePhantomPositive(s.Dt) &&
		threeStatePhantomFinite(s.VThreshold)
}

func threeStatePhantomBoltz(v float64, vh float64, k float64) float64 {
	z := (vh - v) / k
	if z >= 0.0 {
		expNeg := math.Exp(-z)
		return expNeg / (1.0 + expNeg)
	}
	expPos := math.Exp(z)
	return 1.0 / (1.0 + expPos)
}

func (s *SCThreeStatePhantomState) derivatives(v float64, s1 float64, s2 float64, iExt float64) (float64, float64, float64) {
	mInf := threeStatePhantomBoltz(v, s.VM, s.SM)
	nInf := threeStatePhantomBoltz(v, s.VN, s.SN)
	s1Inf := threeStatePhantomBoltz(v, s.VS1, s.SS1)
	s2Inf := threeStatePhantomBoltz(v, s.VS2, s.SS2)
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

func threeStatePhantomCandidateValid(v float64, s1 float64, s2 float64) bool {
	return threeStatePhantomFinite(v) && v >= threeStatePhantomVoltageMin && v <= threeStatePhantomVoltageMax &&
		threeStatePhantomFinite(s1) && s1 >= -threeStatePhantomGateTol && s1 <= 1.0+threeStatePhantomGateTol &&
		threeStatePhantomFinite(s2) && s2 >= -threeStatePhantomGateTol && s2 <= 1.0+threeStatePhantomGateTol
}

func clampThreeStatePhantomGate(value float64) float64 {
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

func (s *SCThreeStatePhantomState) Step(iExt float64) int {
	if !threeStatePhantomFinite(iExt) || !s.validState() {
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
	if !threeStatePhantomCandidateValid(v, s1, s2) {
		return 0
	}
	s.V = v
	s.S1 = clampThreeStatePhantomGate(s1)
	s.S2 = clampThreeStatePhantomGate(s2)
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

func SimulateSCThreeStatePhantom(nSteps int, iExt float64) ([]float64, int) {
	s := NewSCThreeStatePhantom()
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
