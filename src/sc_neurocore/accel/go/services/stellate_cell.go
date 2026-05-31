// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for stellate_cell

package services

import "math"

// StellateCellState holds the WB/Kv3.1 cerebellar stellate-cell state.
type StellateCellState struct {
	V          float64
	H          float64
	N          float64
	P          float64
	GNa        float64
	GK         float64
	GKv3       float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewStellateCell creates a new StellateCell neuron with default parameters.
func NewStellateCell() *StellateCellState {
	return &StellateCellState{
		V:          -65.0,
		H:          0.6,
		N:          0.32,
		P:          0.0,
		GNa:        35.0,
		GK:         9.0,
		GKv3:       3.0,
		GL:         0.1,
		ENa:        55.0,
		EK:         -90.0,
		EL:         -65.0,
		CM:         0.5,
		Phi:        5.0,
		Dt:         0.5,
		VThreshold: -20.0,
		Gain:       1.0,
		SubSteps:   50,
	}
}

func safeExpStellate(value float64) float64 {
	if value > 60.0 {
		value = 60.0
	} else if value < -60.0 {
		value = -60.0
	}
	return math.Exp(value)
}

func safeRateStellate(a float64, vhalf float64, v float64, k float64, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	z := -d / k
	if z > 60.0 {
		return 0.0
	}
	if z < -60.0 {
		return a * d
	}
	return a * d / (1.0 - math.Exp(z))
}

func boltzStellate(v float64, vh float64, k float64) float64 {
	z := -(v - vh) / k
	if z > 60.0 {
		return 0.0
	}
	if z < -60.0 {
		return 1.0
	}
	return 1.0 / (1.0 + math.Exp(z))
}

func clamp01Stellate(value float64) float64 {
	if value < 0.0 {
		return 0.0
	}
	if value > 1.0 {
		return 1.0
	}
	return value
}

func exactRelaxStellate(value, target, tau, dt float64) float64 {
	return target + (value-target)*math.Exp(-dt/tau)
}

func exactHHGateStellate(value, alpha, beta, phi, dt float64) float64 {
	rate := phi * (alpha + beta)
	target := alpha / (alpha + beta)
	return target + (value-target)*math.Exp(-rate*dt)
}

func exactVoltageStellate(v, inputCurrent, cM, dt float64, conductances [][2]float64) float64 {
	gTotal := 0.0
	reversalDrive := 0.0
	for _, pair := range conductances {
		gTotal += pair[0]
		reversalDrive += pair[0] * pair[1]
	}
	if gTotal <= 0.0 {
		return v + dt*inputCurrent/cM
	}
	vInf := (inputCurrent + reversalDrive) / gTotal
	return vInf + (v-vInf)*math.Exp(-dt*gTotal/cM)
}

func finiteStellate(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *StellateCellState) valid() bool {
	return finiteStellate(
		s.V, s.H, s.N, s.P, s.GNa, s.GK, s.GKv3, s.GL, s.ENa, s.EK, s.EL,
		s.CM, s.Phi, s.Dt, s.VThreshold, s.Gain,
	) &&
		s.H >= 0.0 && s.H <= 1.0 && s.N >= 0.0 && s.N <= 1.0 && s.P >= 0.0 && s.P <= 1.0 &&
		s.V >= -100.0 && s.V <= 60.0 &&
		s.GNa >= 0.0 && s.GK >= 0.0 && s.GKv3 >= 0.0 && s.GL >= 0.0 &&
		s.CM > 0.0 && s.Phi > 0.0 && s.Dt > 0.0 && s.SubSteps > 0 && s.Gain >= 0.0
}

// Step advances the neuron by one timestep. Invalid input or state leaves the
// old state untouched and returns no spike.
func (s *StellateCellState) Step(iExt float64) int {
	if !s.valid() || !finiteStellate(iExt) {
		return 0
	}

	inp := s.Gain * iExt
	subDt := s.Dt / float64(s.SubSteps)
	fired := 0
	v := s.V
	h := s.H
	n := s.N
	p := s.P

	for step := 0; step < s.SubSteps; step++ {
		alphaM := safeRateStellate(0.1, 35.0, v, 10.0, 1.0)
		betaM := 4.0 * safeExpStellate(-(v+60.0)/18.0)
		mInf := alphaM / (alphaM + betaM)
		alphaH := 0.07 * safeExpStellate(-(v+58.0)/20.0)
		betaH := boltzStellate(v, -28.0, 10.0)
		alphaN := safeRateStellate(0.01, 34.0, v, 10.0, 0.1)
		betaN := 0.125 * safeExpStellate(-(v+44.0)/80.0)
		pInf := boltzStellate(v, -10.0, 10.0)
		tauP := 1.0 + 4.0/(1.0+safeExpStellate((v+20.0)/15.0))

		h = clamp01Stellate(exactHHGateStellate(h, alphaH, betaH, s.Phi, subDt))
		n = clamp01Stellate(exactHHGateStellate(n, alphaN, betaN, s.Phi, subDt))
		p = clamp01Stellate(exactRelaxStellate(p, pInf, tauP, subDt))

		gNaEff := s.GNa * math.Pow(mInf, 3.0) * h
		gKEff := s.GK * math.Pow(n, 4.0)
		gKv3Eff := s.GKv3 * p * p
		v = math.Max(-100.0, math.Min(60.0, exactVoltageStellate(v, inp, s.CM, subDt, [][2]float64{
			{gNaEff, s.ENa},
			{gKEff, s.EK},
			{gKv3Eff, s.EK},
			{s.GL, s.EL},
		})))
		if !finiteStellate(v, h, n, p) {
			return 0
		}
		if v >= s.VThreshold {
			fired = 1
			v = -65.0
		}
	}

	s.V = v
	s.H = h
	s.N = n
	s.P = p
	return fired
}

// SimulateStellateCell runs the neuron for n steps.
func SimulateStellateCell(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return []float64{}, 0
	}
	s := NewStellateCell()
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
