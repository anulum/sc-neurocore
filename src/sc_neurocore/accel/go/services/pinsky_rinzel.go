// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for pinsky_rinzel (PR1994, RK4)

// Package services implements the Pinsky-Rinzel 1994 two-compartment CA3 cell
// with a fourth-order Runge-Kutta integrator. It mirrors
// neurons/models/pinsky_rinzel.py: eight states (v_s, v_d, h, n, s, c, q, ca),
// chi(ca), and capacitance cm. Kinetics follow ModelDB 35358.
package services

import (
	"math"
)

// PinskyRinzelNeuronState holds the eight-state neuron and its parameters.
type PinskyRinzelNeuronState struct {
	VS         float64
	VD         float64
	H          float64
	N          float64
	S          float64
	C          float64
	Q          float64
	Ca         float64
	Cm         float64
	Gc         float64
	P          float64
	GNa        float64
	GKdr       float64
	GCa        float64
	GKahp      float64
	GKc        float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EL         float64
	Dt         float64
	VThreshold float64
}

// NewPinskyRinzelNeuron creates a PinskyRinzelNeuron with published defaults.
func NewPinskyRinzelNeuron() *PinskyRinzelNeuronState {
	return &PinskyRinzelNeuronState{
		VS:         -60.0,
		VD:         -60.0,
		H:          0.999,
		N:          0.001,
		S:          0.009,
		C:          0.007,
		Q:          0.01,
		Ca:         0.2,
		Cm:         3.0,
		Gc:         2.1,
		P:          0.5,
		GNa:        30.0,
		GKdr:       15.0,
		GCa:        10.0,
		GKahp:      0.8,
		GKc:        15.0,
		GL:         0.1,
		ENa:        60.0,
		EK:         -75.0,
		ECa:        80.0,
		EL:         -60.0,
		Dt:         0.02,
		VThreshold: -20.0,
	}
}

// Step advances the neuron by one timestep with somatic drive only.
func (s *PinskyRinzelNeuronState) Step(iExt float64) int {
	return s.StepDend(iExt, 0.0)
}

// derivatives returns d/dt of (v_s, v_d, h, n, s, c, q, ca).
func (s *PinskyRinzelNeuronState) derivatives(y [8]float64, iS float64, iD float64) [8]float64 {
	vS, vD, h, n, sg, c, q, ca := y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
	am := exprelMinusPinskyRinzel(0.32, vS+46.9, 4.0)
	bm := exprelPlusPinskyRinzel(0.28, vS+19.9, 5.0)
	mInf := 0.0
	if am+bm > 0.0 {
		mInf = am / (am + bm)
	}
	ah := 0.128 * math.Exp(-(vS+43.0)/18.0)
	bh := 4.0 / (1.0 + math.Exp(-(vS+20.0)/5.0))
	an := exprelMinusPinskyRinzel(0.016, vS+24.9, 5.0)
	bn := 0.25 * math.Exp(-1.0-0.025*vS)
	aS := 1.6 / (1.0 + math.Exp(-0.072*(vD-5.0)))
	bS := exprelPlusPinskyRinzel(0.02, vD+8.9, 5.0)
	var ac, bc float64
	if vD <= -10.0 {
		ac = math.Exp((vD+50.0)/11.0-(vD+53.5)/27.0) / 18.975
		bc = 2.0*math.Exp((-53.5-vD)/27.0) - ac
	} else {
		ac = 2.0 * math.Exp((-53.5-vD)/27.0)
		bc = 0.0
	}
	aq := math.Min(0.00002*ca, 0.01)
	bq := 0.001
	chi := math.Min(ca/250.0, 1.0)

	iNa := s.GNa * mInf * mInf * h * (vS - s.ENa)
	iKdr := s.GKdr * n * (vS - s.EK)
	iLs := s.GL * (vS - s.EL)
	iCa := s.GCa * sg * sg * (vD - s.ECa)
	iKahp := s.GKahp * q * (vD - s.EK)
	iKc := s.GKc * c * chi * (vD - s.EK)
	iLd := s.GL * (vD - s.EL)
	iCoupling := s.Gc * (vD - vS)

	dvS := (-iLs - iNa - iKdr + iCoupling/s.P + iS/s.P) / s.Cm
	dvD := (-iLd - iCa - iKahp - iKc - iCoupling/(1.0-s.P) + iD/(1.0-s.P)) / s.Cm
	return [8]float64{
		dvS,
		dvD,
		ah*(1.0-h) - bh*h,
		an*(1.0-n) - bn*n,
		aS*(1.0-sg) - bS*sg,
		ac*(1.0-c) - bc*c,
		aq*(1.0-q) - bq*q,
		-0.13*iCa - 0.075*ca,
	}
}

// StepDend advances the neuron by one RK4 timestep with somatic and dendritic drive.
func (s *PinskyRinzelNeuronState) StepDend(currentSoma float64, currentDend float64) int {
	if !validatePinskyRinzelState(s) || !finitePinskyRinzel(currentSoma) || !finitePinskyRinzel(currentDend) {
		return -1
	}
	vPrev := s.VS
	y := [8]float64{s.VS, s.VD, s.H, s.N, s.S, s.C, s.Q, s.Ca}
	dt := s.Dt
	k1 := s.derivatives(y, currentSoma, currentDend)
	k2 := s.derivatives(axpyPinskyRinzel(y, k1, dt/2.0), currentSoma, currentDend)
	k3 := s.derivatives(axpyPinskyRinzel(y, k2, dt/2.0), currentSoma, currentDend)
	k4 := s.derivatives(axpyPinskyRinzel(y, k3, dt), currentSoma, currentDend)
	var nxt [8]float64
	for i := 0; i < 8; i++ {
		nxt[i] = y[i] + (dt/6.0)*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])
		if !finitePinskyRinzel(nxt[i]) {
			return -1
		}
	}
	s.VS = nxt[0]
	s.VD = nxt[1]
	s.H = clampUnitPinskyRinzel(nxt[2])
	s.N = clampUnitPinskyRinzel(nxt[3])
	s.S = clampUnitPinskyRinzel(nxt[4])
	s.C = clampUnitPinskyRinzel(nxt[5])
	s.Q = clampUnitPinskyRinzel(nxt[6])
	s.Ca = math.Max(nxt[7], 0.0)
	if s.VS >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulatePinskyRinzelNeuron runs the neuron for n steps under constant drive.
func SimulatePinskyRinzelNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewPinskyRinzelNeuron()
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

func axpyPinskyRinzel(y [8]float64, k [8]float64, f float64) [8]float64 {
	var out [8]float64
	for i := 0; i < 8; i++ {
		out[i] = y[i] + f*k[i]
	}
	return out
}

// exprelMinusPinskyRinzel evaluates a*dv/(1-exp(-dv/k)) with removable limit a*k.
func exprelMinusPinskyRinzel(a float64, dv float64, k float64) float64 {
	if math.Abs(dv) < 1e-6 {
		return a * k
	}
	return a * dv / (1.0 - math.Exp(-dv/k))
}

// exprelPlusPinskyRinzel evaluates a*dv/(exp(dv/k)-1) with removable limit a*k.
func exprelPlusPinskyRinzel(a float64, dv float64, k float64) float64 {
	if math.Abs(dv) < 1e-6 {
		return a * k
	}
	return a * dv / (math.Exp(dv/k) - 1.0)
}

func clampUnitPinskyRinzel(value float64) float64 {
	return math.Min(math.Max(value, 0.0), 1.0)
}

func finitePinskyRinzel(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func gatePinskyRinzel(value float64) bool {
	return finitePinskyRinzel(value) && value >= 0.0 && value <= 1.0
}

func validatePinskyRinzelState(s *PinskyRinzelNeuronState) bool {
	if s == nil {
		return false
	}
	return finitePinskyRinzel(s.VS) && finitePinskyRinzel(s.VD) &&
		gatePinskyRinzel(s.H) && gatePinskyRinzel(s.N) && gatePinskyRinzel(s.S) &&
		gatePinskyRinzel(s.C) && gatePinskyRinzel(s.Q) &&
		finitePinskyRinzel(s.Ca) && s.Ca >= 0.0 &&
		finitePinskyRinzel(s.Cm) && s.Cm > 0.0 &&
		finitePinskyRinzel(s.Gc) && s.Gc > 0.0 &&
		finitePinskyRinzel(s.P) && s.P > 0.0 && s.P < 1.0 &&
		finitePinskyRinzel(s.GNa) && s.GNa > 0.0 &&
		finitePinskyRinzel(s.GKdr) && s.GKdr > 0.0 &&
		finitePinskyRinzel(s.GCa) && s.GCa > 0.0 &&
		finitePinskyRinzel(s.GKahp) && s.GKahp > 0.0 &&
		finitePinskyRinzel(s.GKc) && s.GKc > 0.0 &&
		finitePinskyRinzel(s.GL) && s.GL > 0.0 &&
		finitePinskyRinzel(s.ENa) && finitePinskyRinzel(s.EK) &&
		finitePinskyRinzel(s.ECa) && finitePinskyRinzel(s.EL) &&
		finitePinskyRinzel(s.Dt) && s.Dt > 0.0 &&
		finitePinskyRinzel(s.VThreshold)
}
