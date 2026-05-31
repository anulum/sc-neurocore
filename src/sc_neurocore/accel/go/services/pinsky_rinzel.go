// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for pinsky_rinzel

package services

import (
	"math"
)

// PinskyRinzelNeuronState holds the neuron state
type PinskyRinzelNeuronState struct {
	VS         float64
	VD         float64
	H          float64
	N          float64
	S          float64
	C          float64
	Q          float64
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

// NewPinskyRinzelNeuron creates a new PinskyRinzelNeuron neuron with default parameters
func NewPinskyRinzelNeuron() *PinskyRinzelNeuronState {
	return &PinskyRinzelNeuronState{
		VS:         -60.0,
		VD:         -60.0,
		H:          0.9,
		N:          0.1,
		S:          0.0,
		C:          0.0,
		Q:          0.0,
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

// Step advances the neuron by one timestep
func (s *PinskyRinzelNeuronState) Step(iExt float64) int {
	return s.StepDend(iExt, 0.0)
}

func (s *PinskyRinzelNeuronState) StepDend(currentSoma float64, currentDend float64) int {
	if !validatePinskyRinzelState(s) || !finitePinskyRinzel(currentSoma) || !finitePinskyRinzel(currentDend) {
		return -1
	}
	vPrev := s.VS
	am := alphaPinskyRinzel(0.32, s.VS+54.0, 4.0, 8.0, false)
	bm := alphaPinskyRinzel(0.28, s.VS+27.0, 5.0, 5.6, true)
	mInf := am / (am + bm)
	ah := 0.128 * math.Exp(-(s.VS+50.0)/18.0)
	bh := 4.0 * logisticPinskyRinzel((s.VS+27.0)/5.0)
	an := alphaPinskyRinzel(0.032, s.VS+52.0, 5.0, 0.32, false)
	bn := 0.5 * math.Exp(-(s.VS+57.0)/40.0)
	sInf := logisticPinskyRinzel((s.VD + 20.0) / 9.0)

	iNa := s.GNa * math.Pow(mInf, 2.0) * s.H * (s.VS - s.ENa)
	iKdr := s.GKdr * s.N * (s.VS - s.EK)
	iLs := s.GL * (s.VS - s.EL)
	iDs := (s.Gc / s.P) * (s.VS - s.VD)
	iCa := s.GCa * math.Pow(s.S, 2.0) * (s.VD - s.ECa)
	iKahp := s.GKahp * s.Q * (s.VD - s.EK)
	chi := 2.0
	if s.VD <= 50.0 {
		chi = math.Min(s.VD/250.0+0.5, 1.0)
	}
	iKc := s.GKc * s.C * chi * (s.VD - s.EK)
	iLd := s.GL * (s.VD - s.EL)
	iSd := (s.Gc / (1.0 - s.P)) * (s.VD - s.VS)

	next := *s
	next.VS += (-iNa - iKdr - iLs - iDs + currentSoma/s.P) * s.Dt
	next.VD += (-iCa - iKahp - iKc - iLd - iSd + currentDend/(1.0-s.P)) * s.Dt
	next.H += (ah*(1.0-s.H) - bh*s.H) * s.Dt
	next.N += (an*(1.0-s.N) - bn*s.N) * s.Dt
	next.S += ((sInf - s.S) / 5.0) * s.Dt
	next.C = math.Max(0.0, s.C+(-0.13*iCa-0.075*s.C)*s.Dt)
	qInf := math.Min(next.C/(next.C+2.0), 1.0)
	next.Q += ((qInf - s.Q) / 100.0) * s.Dt
	if !validatePinskyRinzelState(&next) {
		return -1
	}
	*s = next
	if s.VS >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulatePinskyRinzelNeuron runs the neuron for n steps
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

func alphaPinskyRinzel(scale float64, x float64, divisor float64, fallback float64, positiveExp bool) float64 {
	if math.Abs(x) <= 1e-6 {
		return fallback
	}
	if positiveExp {
		return scale * x / (math.Exp(x/divisor) - 1.0)
	}
	return scale * x / (1.0 - math.Exp(-x/divisor))
}

func logisticPinskyRinzel(value float64) float64 {
	if value >= 0.0 {
		return 1.0 / (1.0 + math.Exp(-value))
	}
	expValue := math.Exp(value)
	return expValue / (1.0 + expValue)
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
		finitePinskyRinzel(s.C) && s.C >= 0.0 && gatePinskyRinzel(s.Q) &&
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
