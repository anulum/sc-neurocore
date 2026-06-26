// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go candidate-first RK4 service for hay_l5

package services

import "math"

const hayL5Substeps = 4

// HayL5PyramidalNeuronState holds the reduced three-compartment Hay L5 state.
type HayL5PyramidalNeuronState struct {
	VS         float64
	HNa        float64
	NK         float64
	VT         float64
	MCa        float64
	HCa        float64
	MIh        float64
	VA         float64
	CaA        float64
	GNa        float64
	GK         float64
	GLS        float64
	ENa        float64
	EK         float64
	EL         float64
	GCaT       float64
	GIh        float64
	GLT        float64
	ECa        float64
	EIh        float64
	GCaA       float64
	GKca       float64
	GLA        float64
	GST        float64
	GTA        float64
	PS         float64
	PT         float64
	PA         float64
	CaDecay    float64
	FCa        float64
	Dt         float64
	VThreshold float64
	CM         float64
}

// NewHayL5PyramidalNeuron creates a new HayL5PyramidalNeuron neuron with default parameters.
func NewHayL5PyramidalNeuron() *HayL5PyramidalNeuronState {
	return &HayL5PyramidalNeuronState{
		VS:         -75.0,
		HNa:        0.9,
		NK:         0.1,
		VT:         -75.0,
		MCa:        0.0,
		HCa:        1.0,
		MIh:        0.0,
		VA:         -75.0,
		CaA:        0.0001,
		GNa:        300.0,
		GK:         40.0,
		GLS:        0.03,
		ENa:        50.0,
		EK:         -85.0,
		EL:         -75.0,
		GCaT:       2.0,
		GIh:        0.02,
		GLT:        0.03,
		ECa:        140.0,
		EIh:        -45.0,
		GCaA:       1.5,
		GKca:       2.5,
		GLA:        0.03,
		GST:        1.5,
		GTA:        0.8,
		PS:         0.15,
		PT:         0.25,
		PA:         0.60,
		CaDecay:    200.0,
		FCa:        0.0002,
		Dt:         0.025,
		VThreshold: -30.0,
		CM:         1.0,
	}
}

func hayL5Finite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *HayL5PyramidalNeuronState) valid() bool {
	return hayL5Finite(
		s.VS, s.HNa, s.NK, s.VT, s.MCa, s.HCa, s.MIh, s.VA, s.CaA,
		s.GNa, s.GK, s.GLS, s.ENa, s.EK, s.EL, s.GCaT, s.GIh, s.GLT, s.ECa, s.EIh,
		s.GCaA, s.GKca, s.GLA, s.GST, s.GTA, s.PS, s.PT, s.PA, s.CaDecay, s.FCa,
		s.Dt, s.VThreshold, s.CM,
	) &&
		s.CaA >= 0.0 &&
		s.GNa >= 0.0 && s.GK >= 0.0 && s.GLS >= 0.0 && s.GCaT >= 0.0 && s.GIh >= 0.0 &&
		s.GLT >= 0.0 && s.GCaA >= 0.0 && s.GKca >= 0.0 && s.GLA >= 0.0 &&
		s.GST >= 0.0 && s.GTA >= 0.0 && s.FCa >= 0.0 &&
		s.PS > 0.0 && s.PT > 0.0 && s.PA > 0.0 && s.CaDecay > 0.0 && s.Dt > 0.0 && s.CM > 0.0
}

func (s *HayL5PyramidalNeuronState) derivatives(state [9]float64, currentSoma float64, currentTuft float64) [9]float64 {
	vs, hNa, nK := state[0], state[1], state[2]
	vt, mCa, hCa, mIh := state[3], state[4], state[5], state[6]
	va, caA := state[7], math.Max(state[8], 0.0)

	mNaInf := 1.0 / (1.0 + math.Exp(-(vs+38.0)/7.0))
	hNaInf := 1.0 / (1.0 + math.Exp((vs+65.0)/6.0))
	nKInf := 1.0 / (1.0 + math.Exp(-(vs+25.0)/12.0))
	tauH := 0.5 + 14.0/(1.0+math.Exp((vs+35.0)/10.0))
	tauN := 1.0 + 5.0/(1.0+math.Exp((vs+30.0)/10.0))
	iNa := s.GNa * mNaInf * mNaInf * mNaInf * hNa * (vs - s.ENa)
	iK := s.GK * nK * nK * nK * nK * (vs - s.EK)
	iLS := s.GLS * (vs - s.EL)
	iST := s.GST * (vs - vt) / s.PS

	mCaInf := 1.0 / (1.0 + math.Exp(-(vt+27.0)/7.0))
	hCaInf := 1.0 / (1.0 + math.Exp((vt+52.0)/5.0))
	mIhInf := 1.0 / (1.0 + math.Exp((vt+75.0)/5.5))
	iCaT := s.GCaT * mCa * mCa * hCa * (vt - s.ECa)
	iIh := s.GIh * mIh * (vt - s.EIh)
	iLT := s.GLT * (vt - s.EL)
	iTS := s.GST * (vt - vs) / s.PT
	iTA := s.GTA * (vt - va) / s.PT

	mCaAInf := 1.0 / (1.0 + math.Exp(-(va+30.0)/5.0))
	kcaAct := caA / (caA + 0.001)
	iCaA := s.GCaA * mCaAInf * mCaAInf * (va - s.ECa)
	iKca := s.GKca * kcaAct * (va - s.EK)
	iLA := s.GLA * (va - s.EL)
	iAT := s.GTA * (va - vt) / s.PA

	return [9]float64{
		(-iNa - iK - iLS - iST + currentSoma/s.PS) / s.CM,
		(hNaInf - hNa) / tauH,
		(nKInf - nK) / tauN,
		(-iCaT - iIh - iLT - iTS - iTA) / s.CM,
		mCaInf - mCa,
		(hCaInf - hCa) / 20.0,
		(mIhInf - mIh) / 50.0,
		(-iCaA - iKca - iLA - iAT + currentTuft/s.PA) / s.CM,
		-s.FCa*iCaA - caA/s.CaDecay,
	}
}

func (s *HayL5PyramidalNeuronState) rk4Substep(state [9]float64, currentSoma float64, currentTuft float64) [9]float64 {
	dt := s.Dt
	k1 := s.derivatives(state, currentSoma, currentTuft)
	var s2 [9]float64
	var s3 [9]float64
	var s4 [9]float64
	for i := 0; i < 9; i++ {
		s2[i] = state[i] + 0.5*dt*k1[i]
	}
	k2 := s.derivatives(s2, currentSoma, currentTuft)
	for i := 0; i < 9; i++ {
		s3[i] = state[i] + 0.5*dt*k2[i]
	}
	k3 := s.derivatives(s3, currentSoma, currentTuft)
	for i := 0; i < 9; i++ {
		s4[i] = state[i] + dt*k3[i]
	}
	k4 := s.derivatives(s4, currentSoma, currentTuft)
	var next [9]float64
	for i := 0; i < 9; i++ {
		next[i] = state[i] + dt*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])/6.0
	}
	next[8] = math.Max(next[8], 0.0)
	return next
}

// Step advances the neuron by one candidate-first RK4 timestep.
func (s *HayL5PyramidalNeuronState) Step(currentSoma float64, currentTuft ...float64) int {
	tuft := 0.0
	if len(currentTuft) > 0 {
		tuft = currentTuft[0]
	}
	if !hayL5Finite(currentSoma, tuft) || !s.valid() {
		return 0
	}
	vPrev := s.VS
	state := [9]float64{s.VS, s.HNa, s.NK, s.VT, s.MCa, s.HCa, s.MIh, s.VA, s.CaA}
	for i := 0; i < hayL5Substeps; i++ {
		state = s.rk4Substep(state, currentSoma, tuft)
		if !hayL5Finite(state[:]...) {
			return 0
		}
	}
	s.VS = state[0]
	s.HNa = state[1]
	s.NK = state[2]
	s.VT = state[3]
	s.MCa = state[4]
	s.HCa = state[5]
	s.MIh = state[6]
	s.VA = state[7]
	s.CaA = state[8]
	if s.VS >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateHayL5PyramidalNeuron runs the neuron for n steps.
func SimulateHayL5PyramidalNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHayL5PyramidalNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.VS
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
