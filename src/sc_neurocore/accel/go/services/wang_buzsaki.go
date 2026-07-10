// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for wang_buzsaki

package services

import (
	"errors"
	"math"
)

// WangBuzsakiNeuronState holds the three-state Wang-Buzsáki (1996) fast-spiking
// interneuron: a simplified Hodgkin-Huxley (Na + delayed-rectifier K) with instantaneous
// sodium activation m = m_inf. It mirrors the Python golden
// sc_neurocore.neurons.models.wang_buzsaki.WangBuzsakiNeuron: a 0.5 ms macro step of 50
// inner dt=0.01 sub-steps advanced sequentially (Gauss-Seidel — gates h/n from the old
// voltage, then voltage v from the new gates), rising-edge v >= v_threshold crossing on the
// macro boundary, no reset. Reference DOI 10.1523/JNEUROSCI.16-20-06402.1996.
type WangBuzsakiNeuronState struct {
	V          float64
	H          float64
	N          float64
	GNa        float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewWangBuzsakiNeuron creates a new WangBuzsakiNeuron neuron with default parameters
func NewWangBuzsakiNeuron() *WangBuzsakiNeuronState {
	return &WangBuzsakiNeuronState{
		V:          -65.0,
		H:          0.8,
		N:          0.1,
		GNa:        35.0,
		GK:         9.0,
		GL:         0.1,
		ENa:        55.0,
		EK:         -90.0,
		EL:         -65.0,
		CM:         1.0,
		Phi:        5.0,
		Dt:         0.01,
		VThreshold: -20.0,
	}
}

// ValidateWangBuzsaki checks the finite physical state needed before stepping.
func ValidateWangBuzsaki(s *WangBuzsakiNeuronState) bool {
	if s == nil {
		return false
	}
	return wbFinite(s.V, s.H, s.N, s.ENa, s.EK, s.EL, s.VThreshold) &&
		s.GNa > 0 && wbFinite(s.GNa) &&
		s.GK > 0 && wbFinite(s.GK) &&
		s.GL > 0 && wbFinite(s.GL) &&
		s.CM > 0 && wbFinite(s.CM) &&
		s.Phi > 0 && wbFinite(s.Phi) &&
		s.Dt > 0 && wbFinite(s.Dt)
}

func wbFinite(xs ...float64) bool {
	for _, x := range xs {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func wbSafeExp(x float64) (float64, bool) {
	if !wbFinite(x) || x > 700.0 {
		return 0, false
	}
	return math.Exp(x), true
}

// wbGatingRates returns (m_inf, alpha_h, beta_h, alpha_n, beta_n) at voltage v, matching the
// Python golden's _gating_rates: instantaneous m_inf = alpha_m/(alpha_m+beta_m) with the
// removable singularities of alpha_m (at v=-35) and alpha_n (at v=-34) resolved to their
// limits, all transcendentals guarded against overflow.
func wbGatingRates(v float64) (mInf, alphaH, betaH, alphaN, betaN float64, ok bool) {
	var alphaM float64
	if math.Abs(v+35.0) > 1e-6 {
		e, good := wbSafeExp(-(v + 35.0) / 10.0)
		if !good {
			return 0, 0, 0, 0, 0, false
		}
		alphaM = 0.1 * (v + 35.0) / (1.0 - e)
	} else {
		alphaM = 1.0
	}
	betaMExp, ok := wbSafeExp(-(v + 60.0) / 18.0)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	betaM := 4.0 * betaMExp
	denomM := alphaM + betaM
	if denomM == 0.0 || !wbFinite(denomM) {
		return 0, 0, 0, 0, 0, false
	}
	mInf = alphaM / denomM
	ahExp, ok := wbSafeExp(-(v + 58.0) / 20.0)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	alphaH = 0.07 * ahExp
	bhExp, ok := wbSafeExp(-(v + 28.0) / 10.0)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	betaH = 1.0 / (1.0 + bhExp)
	if math.Abs(v+34.0) > 1e-6 {
		e, good := wbSafeExp(-(v + 34.0) / 10.0)
		if !good {
			return 0, 0, 0, 0, 0, false
		}
		alphaN = 0.01 * (v + 34.0) / (1.0 - e)
	} else {
		alphaN = 0.1
	}
	bnExp, ok := wbSafeExp(-(v + 44.0) / 80.0)
	if !ok {
		return 0, 0, 0, 0, 0, false
	}
	betaN = 0.125 * bnExp
	if !wbFinite(mInf, alphaH, betaH, alphaN, betaN) {
		return 0, 0, 0, 0, 0, false
	}
	return mInf, alphaH, betaH, alphaN, betaN, true
}

// Step advances the neuron by one 0.5 ms macro step (50 sequential sub-steps) and reports a
// spike on a rising-edge v >= v_threshold crossing (no reset). It fails closed: the state is
// committed only once the whole macro step stays finite, so an invalid input or a divergent
// integration leaves the neuron unchanged and returns an error.
func (s *WangBuzsakiNeuronState) Step(iExt float64) (int, error) {
	if !ValidateWangBuzsaki(s) || math.IsInf(iExt, 0) || math.IsNaN(iExt) {
		return 0, errors.New("invalid Wang-Buzsaki state or input")
	}
	vPrev := s.V
	v, h, n := s.V, s.H, s.N
	substeps := int(0.5 / math.Max(s.Dt, 0.001))
	for i := 0; i < substeps; i++ {
		// Gauss-Seidel order: gates from the old voltage, then voltage from the new gates.
		mInf, alphaH, betaH, alphaN, betaN, ok := wbGatingRates(v)
		if !ok {
			return 0, errors.New("non-finite Wang-Buzsaki gating rate")
		}
		nextH := h + s.Phi*(alphaH*(1.0-h)-betaH*h)*s.Dt
		nextN := n + s.Phi*(alphaN*(1.0-n)-betaN*n)*s.Dt
		iNa := s.GNa * math.Pow(mInf, 3.0) * nextH * (v - s.ENa)
		iK := s.GK * math.Pow(nextN, 4.0) * (v - s.EK)
		iL := s.GL * (v - s.EL)
		nextV := v + (-iNa-iK-iL+iExt)/s.CM*s.Dt
		if !wbFinite(nextV, nextH, nextN) {
			return 0, errors.New("non-finite Wang-Buzsaki update")
		}
		v, h, n = nextV, nextH, nextN
	}
	s.V, s.H, s.N = v, h, n
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateWangBuzsakiNeuron runs the neuron for n macro steps and records voltage.
func SimulateWangBuzsakiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewWangBuzsakiNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			trace[t] = math.NaN()
			continue
		}
		trace[t] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
