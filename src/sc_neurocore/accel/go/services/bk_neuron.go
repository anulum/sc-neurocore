// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for bk_neuron

package services

import (
	"errors"
	"math"
)

const bkExpMax = 709.0
const bkExpMin = -745.0

// BKNeuronState holds the neuron state
type BKNeuronState struct {
	V          float64
	H          float64
	N          float64
	Ca         float64
	GNa        float64
	GK         float64
	GBk        float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	TauCa      float64
	Dt         float64
	VThreshold float64
	Gain       float64
	SubSteps   int
}

// NewBKNeuron creates a new BKNeuron neuron with default parameters
func NewBKNeuron() *BKNeuronState {
	return &BKNeuronState{
		V:          -65.0,
		H:          0.6,
		N:          0.32,
		Ca:         0.0,
		GNa:        35.0,
		GK:         9.0,
		GBk:        3.0,
		GL:         0.1,
		ENa:        55.0,
		EK:         -90.0,
		EL:         -65.0,
		CM:         1.0,
		Phi:        5.0,
		TauCa:      50.0,
		Dt:         0.5,
		VThreshold: -20.0,
		Gain:       1.0,
		SubSteps:   50,
	}
}

func bkFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func bkProbability(x float64) bool {
	return bkFinite(x) && x >= 0.0 && x <= 1.0
}

func checkedBKExp(x float64) (float64, error) {
	if !bkFinite(x) || x > bkExpMax {
		return math.NaN(), errors.New("unstable BK exponential argument")
	}
	if x < bkExpMin {
		return 0.0, nil
	}
	return math.Exp(x), nil
}

func bkSafeRate(a, vhalf, v, k, fallback float64) (float64, error) {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback, nil
	}
	expTerm, err := checkedBKExp(-d / k)
	if err != nil {
		return math.NaN(), err
	}
	rate := a * d / (1.0 - expTerm)
	if !bkFinite(rate) {
		return math.NaN(), errors.New("non-finite BK rate candidate")
	}
	return rate, nil
}

// ValidateBKNeuron checks the model's physical state contract.
func ValidateBKNeuron(s *BKNeuronState) bool {
	if s == nil {
		return false
	}
	for _, value := range []float64{s.V, s.ENa, s.EK, s.EL, s.VThreshold, s.Gain} {
		if !bkFinite(value) {
			return false
		}
	}
	if !bkProbability(s.H) || !bkProbability(s.N) || !bkFinite(s.Ca) || s.Ca < 0.0 {
		return false
	}
	for _, value := range []float64{s.GNa, s.GK, s.GBk, s.GL} {
		if !bkFinite(value) || value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.CM, s.Phi, s.TauCa, s.Dt} {
		if !bkFinite(value) || value <= 0.0 {
			return false
		}
	}
	return s.SubSteps > 0
}

// Step advances the neuron by one timestep.
func (s *BKNeuronState) Step(iExt float64) (int, error) {
	if !ValidateBKNeuron(s) || !bkFinite(iExt) {
		return 0, errors.New("invalid BK neuron state or input")
	}
	inp := s.Gain * iExt
	if !bkFinite(inp) {
		return 0, errors.New("invalid BK input drive")
	}
	subDt := s.Dt / float64(s.SubSteps)
	if !bkFinite(subDt) || subDt <= 0.0 {
		return 0, errors.New("invalid BK substep")
	}
	v, h, n, ca := s.V, s.H, s.N, s.Ca
	fired := 0
	for i := 0; i < s.SubSteps; i++ {
		alphaM, err := bkSafeRate(0.1, 35.0, v, 10.0, 1.0)
		if err != nil {
			return 0, err
		}
		betaMExp, err := checkedBKExp(-(v + 60.0) / 18.0)
		if err != nil {
			return 0, err
		}
		mInf := alphaM / (alphaM + 4.0*betaMExp)
		alphaHExp, err := checkedBKExp(-(v + 58.0) / 20.0)
		if err != nil {
			return 0, err
		}
		alphaH := 0.07 * alphaHExp
		betaHExp, err := checkedBKExp(-(v + 28.0) / 10.0)
		if err != nil {
			return 0, err
		}
		betaH := 1.0 / (1.0 + betaHExp)
		alphaN, err := bkSafeRate(0.01, 34.0, v, 10.0, 0.1)
		if err != nil {
			return 0, err
		}
		betaNExp, err := checkedBKExp(-(v + 44.0) / 80.0)
		if err != nil {
			return 0, err
		}
		betaN := 0.125 * betaNExp
		caDecay := math.Max(ca+subDt*(-ca/s.TauCa), 0.0)
		denom := caDecay + 0.5
		if !bkFinite(caDecay) || !bkFinite(denom) || denom <= 0.0 {
			return 0, errors.New("invalid BK calcium candidate")
		}
		vHalfBK := 10.0 - 30.0*(caDecay/denom)
		bkExp, err := checkedBKExp(-(v - vHalfBK) / 15.0)
		if err != nil {
			return 0, err
		}
		bkInf := 1.0 / (1.0 + bkExp)
		hNext := h + subDt*s.Phi*(alphaH*(1.0-h)-betaH*h)
		nNext := n + subDt*s.Phi*(alphaN*(1.0-n)-betaN*n)
		iNa := s.GNa * math.Pow(mInf, 3.0) * hNext * (v - s.ENa)
		iK := s.GK * math.Pow(nNext, 4.0) * (v - s.EK)
		iBK := s.GBk * bkInf * (v - s.EK)
		iL := s.GL * (v - s.EL)
		dV := (-iNa - iK - iBK - iL + inp) / s.CM
		vNext := v + subDt*dV
		caNext := caDecay
		if vNext >= s.VThreshold {
			fired = 1
			vNext = -65.0
			caNext += 0.3
		}
		if !(bkFinite(vNext) && vNext >= -100.0 && vNext <= 60.0 && bkProbability(hNext) && bkProbability(nNext) && bkProbability(bkInf) && bkFinite(caNext) && caNext >= 0.0) {
			return 0, errors.New("invalid BK candidate state")
		}
		v, h, n, ca = vNext, hNext, nNext, caNext
	}
	s.V, s.H, s.N, s.Ca = v, h, n, ca
	return fired, nil
}

// SimulateBKNeuron runs the neuron for n steps.
func SimulateBKNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return nil, 0
	}
	s := NewBKNeuron()
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
