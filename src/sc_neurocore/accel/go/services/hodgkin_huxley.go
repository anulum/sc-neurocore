// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for hodgkin_huxley

package services

import (
	"errors"
	"math"
)

// HodgkinHuxleyNeuronState holds the four-state Hodgkin-Huxley (1952) conductance model.
type HodgkinHuxleyNeuronState struct {
	V          float64
	M          float64
	H          float64
	N          float64
	CM         float64
	GNa        float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	Dt         float64
	VThreshold float64
}

// NewHodgkinHuxleyNeuron creates a Hodgkin-Huxley neuron with default parameters.
func NewHodgkinHuxleyNeuron() *HodgkinHuxleyNeuronState {
	return &HodgkinHuxleyNeuronState{
		V: -65.0, M: 0.05, H: 0.6, N: 0.32,
		CM: 1.0, GNa: 120.0, GK: 36.0, GL: 0.3,
		ENa: 50.0, EK: -77.0, EL: -54.4,
		Dt: 0.01, VThreshold: 0.0,
	}
}

type hodgkinHuxleyState struct {
	v float64
	m float64
	h float64
	n float64
}

func finiteHodgkin(xs ...float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func safeExpHodgkin(x float64) (float64, bool) {
	if !finiteHodgkin(x) || x > 700.0 {
		return 0, false
	}
	return math.Exp(x), true
}

// openingRateHodgkin evaluates scale*d/(1-exp(-d/denom)) with d = v+shift, returning the
// analytic limit scale*denom when |d| < 1e-7 (bit-for-bit the guard in models/hodgkin_huxley.py).
func openingRateHodgkin(scale, shift, denom, limit, v float64) (float64, bool) {
	d := v + shift
	if math.Abs(d) < 1e-7 {
		return limit, true
	}
	e, ok := safeExpHodgkin(-d / denom)
	if !ok {
		return 0, false
	}
	value := scale * d / (1.0 - e)
	return value, finiteHodgkin(value)
}

func hodgkinAlphaM(v float64) (float64, bool) { return openingRateHodgkin(0.1, 40.0, 10.0, 1.0, v) }

func hodgkinBetaM(v float64) (float64, bool) {
	e, ok := safeExpHodgkin(-(v + 65.0) / 18.0)
	if !ok {
		return 0, false
	}
	return 4.0 * e, true
}

func hodgkinAlphaH(v float64) (float64, bool) {
	e, ok := safeExpHodgkin(-(v + 65.0) / 20.0)
	if !ok {
		return 0, false
	}
	return 0.07 * e, true
}

func hodgkinBetaH(v float64) (float64, bool) {
	e, ok := safeExpHodgkin(-(v + 35.0) / 10.0)
	if !ok {
		return 0, false
	}
	return 1.0 / (1.0 + e), true
}

func hodgkinAlphaN(v float64) (float64, bool) { return openingRateHodgkin(0.01, 55.0, 10.0, 0.1, v) }

func hodgkinBetaN(v float64) (float64, bool) {
	e, ok := safeExpHodgkin(-(v + 65.0) / 80.0)
	if !ok {
		return 0, false
	}
	return 0.125 * e, true
}

func (s *HodgkinHuxleyNeuronState) validStatic() bool {
	return finiteHodgkin(s.CM, s.GNa, s.GK, s.GL, s.ENa, s.EK, s.EL, s.Dt, s.VThreshold) &&
		s.GNa >= 0.0 && s.GK >= 0.0 && s.GL >= 0.0 && s.CM > 0.0 && s.Dt > 0.0
}

func (s *HodgkinHuxleyNeuronState) validState(v, m, h, n float64) bool {
	return finiteHodgkin(v, m, h, n) &&
		v >= -250.0 && v <= 250.0 &&
		m >= -0.05 && m <= 1.05 && h >= -0.05 && h <= 1.05 && n >= -0.05 && n <= 1.05
}

// eulerCandidate advances one macro step over round(1/dt) explicit-Euler sub-steps into a
// candidate that Step only commits on success. Gates update first, then the membrane voltage
// uses the freshly-updated gates (the models/hodgkin_huxley.py _step_baseline_euler order).
// Fail-closed: returns ok=false (state untouched) on any non-finite parameter/current/intermediate.
func (s *HodgkinHuxleyNeuronState) eulerCandidate(current float64) (hodgkinHuxleyState, bool) {
	if !s.validStatic() || !finiteHodgkin(current) || !s.validState(s.V, s.M, s.H, s.N) {
		return hodgkinHuxleyState{}, false
	}
	state := hodgkinHuxleyState{v: s.V, m: s.M, h: s.H, n: s.N}
	substeps := int(math.Round(1.0 / s.Dt))
	for i := 0; i < substeps; i++ {
		am, ok := hodgkinAlphaM(state.v)
		if !ok {
			return hodgkinHuxleyState{}, false
		}
		bm, ok := hodgkinBetaM(state.v)
		if !ok {
			return hodgkinHuxleyState{}, false
		}
		ah, ok := hodgkinAlphaH(state.v)
		if !ok {
			return hodgkinHuxleyState{}, false
		}
		bh, ok := hodgkinBetaH(state.v)
		if !ok {
			return hodgkinHuxleyState{}, false
		}
		an, ok := hodgkinAlphaN(state.v)
		if !ok {
			return hodgkinHuxleyState{}, false
		}
		bn, ok := hodgkinBetaN(state.v)
		if !ok {
			return hodgkinHuxleyState{}, false
		}
		state.m += (am*(1.0-state.m) - bm*state.m) * s.Dt
		state.h += (ah*(1.0-state.h) - bh*state.h) * s.Dt
		state.n += (an*(1.0-state.n) - bn*state.n) * s.Dt
		iNa := s.GNa * math.Pow(state.m, 3.0) * state.h * (state.v - s.ENa)
		iK := s.GK * math.Pow(state.n, 4.0) * (state.v - s.EK)
		iL := s.GL * (state.v - s.EL)
		state.v += (-iNa - iK - iL + current) / s.CM * s.Dt
		if !s.validState(state.v, state.m, state.h, state.n) {
			return hodgkinHuxleyState{}, false
		}
	}
	return state, true
}

// Step advances the neuron by one macro-step using candidate-first explicit-Euler sub-steps.
func (s *HodgkinHuxleyNeuronState) Step(iExt float64) (int, error) {
	vPrev := s.V
	candidate, ok := s.eulerCandidate(iExt)
	if !ok {
		return 0, errors.New("invalid Hodgkin-Huxley state, parameters, current, or candidate")
	}
	s.V, s.M, s.H, s.N = candidate.v, candidate.m, candidate.h, candidate.n
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateHodgkinHuxleyNeuron runs the neuron for n steps and records voltage.
func SimulateHodgkinHuxleyNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewHodgkinHuxleyNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		trace[t] = s.V
		if err != nil {
			continue
		}
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
