// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for renshaw_cell

package services

import (
	"math"
)

// RenshawCellState holds the neuron state
type RenshawCellState struct {
	V          float64
	H          float64
	N          float64
	Adapt      float64
	GNa        float64
	GK         float64
	GAdapt     float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	TauAdapt   float64
	Dt         float64
	VThreshold float64
}

// NewRenshawCell creates a new RenshawCell neuron with default parameters
func NewRenshawCell() *RenshawCellState {
	return &RenshawCellState{
		V:          -65.0,
		H:          0.8,
		N:          0.1,
		Adapt:      0.0,
		GNa:        35.0,
		GK:         9.0,
		GAdapt:     5.0,
		GL:         0.12,
		ENa:        55.0,
		EK:         -90.0,
		EL:         -65.0,
		CM:         1.0,
		Phi:        5.0,
		TauAdapt:   50.0,
		Dt:         0.01,
		VThreshold: -20.0,
	}
}

func renshawCellFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func renshawCellClamp01(value float64) float64 {
	return math.Max(0.0, math.Min(1.0, value))
}

func renshawCellProbability(value float64) bool {
	return renshawCellFinite(value) && value >= 0.0 && value <= 1.0
}

func renshawCellVoltage(value float64) bool {
	return renshawCellFinite(value) && value >= -150.0 && value <= 100.0
}

func renshawCellSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1.0 - math.Exp(-d/k))
}

func renshawCellExactGate(previous, alpha, beta, phi, dt float64) (float64, bool) {
	total := phi * (alpha + beta)
	if !renshawCellFinite(previous, alpha, beta, total, dt) || total <= 0.0 {
		return previous, false
	}
	steady := alpha / (alpha + beta)
	return renshawCellClamp01(steady + (previous-steady)*math.Exp(-total*dt)), true
}

func renshawCellExactRelax(previous, steady, tau, dt float64) (float64, bool) {
	if !renshawCellFinite(previous, steady, tau, dt) || tau <= 0.0 {
		return previous, false
	}
	return renshawCellClamp01(steady + (previous-steady)*math.Exp(-dt/tau)), true
}

func (s *RenshawCellState) valid() bool {
	return renshawCellVoltage(s.V) &&
		renshawCellProbability(s.H) &&
		renshawCellProbability(s.N) &&
		renshawCellProbability(s.Adapt) &&
		renshawCellFinite(s.GNa, s.GK, s.GAdapt, s.GL, s.ENa, s.EK, s.EL, s.CM, s.Phi, s.TauAdapt, s.Dt, s.VThreshold) &&
		s.GNa >= 0.0 &&
		s.GK >= 0.0 &&
		s.GAdapt >= 0.0 &&
		s.GL >= 0.0 &&
		s.CM > 0.0 &&
		s.Phi > 0.0 &&
		s.TauAdapt > 0.0 &&
		s.Dt > 0.0
}

// Step advances the neuron by one timestep
func (s *RenshawCellState) Step(iExt float64) int {
	if !renshawCellFinite(iExt) || !s.valid() {
		return 0
	}

	vPrev := s.V
	v := s.V
	h := s.H
	n := s.N
	adapt := s.Adapt
	nSub := int(0.5 / math.Max(s.Dt, 0.001))
	if nSub < 1 {
		nSub = 1
	}
	for i := 0; i < nSub; i++ {
		am := renshawCellSafeRate(0.1, 35.0, v, 10.0, 1.0)
		bm := 4.0 * math.Exp(-(v+60.0)/18.0)
		mInf := am / (am + bm)
		ah := 0.07 * math.Exp(-(v+58.0)/20.0)
		bh := 1.0 / (1.0 + math.Exp(-(v+28.0)/10.0))
		an := renshawCellSafeRate(0.01, 34.0, v, 10.0, 0.1)
		bn := 0.125 * math.Exp(-(v+44.0)/80.0)

		hNext, ok := renshawCellExactGate(h, ah, bh, s.Phi, s.Dt)
		if !ok {
			return 0
		}
		nNext, ok := renshawCellExactGate(n, an, bn, s.Phi, s.Dt)
		if !ok {
			return 0
		}
		adaptInf := 1.0 / (1.0 + math.Exp(-(v+30.0)/5.0))
		adaptNext, ok := renshawCellExactRelax(adapt, adaptInf, s.TauAdapt, s.Dt)
		if !ok {
			return 0
		}

		gNa := s.GNa * math.Pow(mInf, 3.0) * hNext
		gK := s.GK * math.Pow(nNext, 4.0)
		gAdapt := s.GAdapt * adaptNext
		gTotal := gNa + gK + gAdapt + s.GL
		if !renshawCellFinite(gTotal) || gTotal <= 0.0 {
			return 0
		}
		steadyV := (iExt + gNa*s.ENa + gK*s.EK + gAdapt*s.EK + s.GL*s.EL) / gTotal
		vNext := steadyV + (v-steadyV)*math.Exp(-(gTotal/s.CM)*s.Dt)
		if !renshawCellVoltage(vNext) || !renshawCellProbability(hNext) || !renshawCellProbability(nNext) || !renshawCellProbability(adaptNext) {
			return 0
		}

		v = vNext
		h = hNext
		n = nNext
		adapt = adaptNext
	}

	s.V = v
	s.H = h
	s.N = n
	s.Adapt = adapt
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1
	}
	return 0
}

// SimulateRenshawCell runs the neuron for n steps
func SimulateRenshawCell(nSteps int, iExt float64) ([]float64, int) {
	s := NewRenshawCell()
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
