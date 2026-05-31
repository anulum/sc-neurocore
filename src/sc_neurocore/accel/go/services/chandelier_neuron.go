// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for chandelier_neuron

package services

import (
	"errors"
	"math"
)

const chandelierExpMax = 709.0
const chandelierExpMin = -745.0

// ChandelierNeuronState holds the neuron state
type ChandelierNeuronState struct {
	V          float64
	H          float64
	N          float64
	D          float64
	P          float64
	GNa        float64
	GK         float64
	GKv1       float64
	GKv3       float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewChandelierNeuron creates a new ChandelierNeuron neuron with default parameters
func NewChandelierNeuron() *ChandelierNeuronState {
	return &ChandelierNeuronState{
		V:          -65.0,
		H:          0.8,
		N:          0.1,
		D:          0.0,
		P:          0.0,
		GNa:        35.0,
		GK:         9.0,
		GKv1:       3.0,
		GKv3:       4.0,
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

func chandelierFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func chandelierProbability(x float64) bool {
	return chandelierFinite(x) && x >= 0.0 && x <= 1.0
}

func checkedChandelierExp(x float64) (float64, error) {
	if !chandelierFinite(x) || x > chandelierExpMax {
		return math.NaN(), errors.New("unstable chandelier exponential argument")
	}
	if x < chandelierExpMin {
		return 0.0, nil
	}
	return math.Exp(x), nil
}

func chandelierSafeRate(a, vhalf, v, k, fallback float64) (float64, error) {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback, nil
	}
	expTerm, err := checkedChandelierExp(-d / k)
	if err != nil {
		return math.NaN(), err
	}
	rate := a * d / (1.0 - expTerm)
	if !chandelierFinite(rate) {
		return math.NaN(), errors.New("non-finite chandelier rate candidate")
	}
	return rate, nil
}

// ValidateChandelierNeuron checks the model's physical state contract.
func ValidateChandelierNeuron(s *ChandelierNeuronState) bool {
	if s == nil {
		return false
	}
	for _, value := range []float64{s.V, s.ENa, s.EK, s.EL, s.VThreshold} {
		if !chandelierFinite(value) {
			return false
		}
	}
	for _, value := range []float64{s.H, s.N, s.D, s.P} {
		if !chandelierProbability(value) {
			return false
		}
	}
	for _, value := range []float64{s.GNa, s.GK, s.GKv1, s.GKv3, s.GL} {
		if !chandelierFinite(value) || value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.CM, s.Phi, s.Dt} {
		if !chandelierFinite(value) || value <= 0.0 {
			return false
		}
	}
	return true
}

// Step advances the neuron by one timestep.
func (s *ChandelierNeuronState) Step(iExt float64) (int, error) {
	if !ValidateChandelierNeuron(s) || !chandelierFinite(iExt) {
		return 0, errors.New("invalid chandelier state or input")
	}
	vPrev := s.V
	nSub := int(0.5 / math.Max(s.Dt, 0.001))
	if nSub < 1 {
		nSub = 1
	}
	v, h, n := s.V, s.H, s.N
	dGate, pGate := s.D, s.P
	for i := 0; i < nSub; i++ {
		am, err := chandelierSafeRate(0.1, 35.0, v, 10.0, 1.0)
		if err != nil {
			return 0, err
		}
		bmExp, err := checkedChandelierExp(-(v + 60.0) / 18.0)
		if err != nil {
			return 0, err
		}
		mInf := am / (am + 4.0*bmExp)
		ahExp, err := checkedChandelierExp(-(v + 58.0) / 20.0)
		if err != nil {
			return 0, err
		}
		ah := 0.07 * ahExp
		bhExp, err := checkedChandelierExp(-(v + 28.0) / 10.0)
		if err != nil {
			return 0, err
		}
		bh := 1.0 / (1.0 + bhExp)
		an, err := chandelierSafeRate(0.01, 34.0, v, 10.0, 0.1)
		if err != nil {
			return 0, err
		}
		bnExp, err := checkedChandelierExp(-(v + 44.0) / 80.0)
		if err != nil {
			return 0, err
		}
		bn := 0.125 * bnExp
		hNext := h + s.Phi*(ah*(1.0-h)-bh*h)*s.Dt
		nNext := n + s.Phi*(an*(1.0-n)-bn*n)*s.Dt
		dInfExp, err := checkedChandelierExp(-(v + 50.0) / 10.0)
		if err != nil {
			return 0, err
		}
		dInf := 1.0 / (1.0 + dInfExp)
		dNext := dGate + (dInf-dGate)/150.0*s.Dt
		pInfExp, err := checkedChandelierExp(-(v + 10.0) / 10.0)
		if err != nil {
			return 0, err
		}
		pInf := 1.0 / (1.0 + pInfExp)
		pNext := pGate + s.Phi*(pInf-pGate)*s.Dt
		iNa := s.GNa * math.Pow(mInf, 3.0) * hNext * (v - s.ENa)
		iK := s.GK * math.Pow(nNext, 4.0) * (v - s.EK)
		iKv1 := s.GKv1 * math.Pow(dNext, 4.0) * (v - s.EK)
		iKv3 := s.GKv3 * pNext * (v - s.EK)
		iL := s.GL * (v - s.EL)
		vNext := v + (-iNa-iK-iKv1-iKv3-iL+iExt)/s.CM*s.Dt
		if !(chandelierFinite(vNext) && vNext >= -100.0 && vNext <= 60.0 && chandelierProbability(hNext) && chandelierProbability(nNext) && chandelierProbability(dNext) && chandelierProbability(pNext)) {
			return 0, errors.New("invalid chandelier candidate state")
		}
		v, h, n = vNext, hNext, nNext
		dGate, pGate = dNext, pNext
	}
	s.V, s.H, s.N = v, h, n
	s.D, s.P = dGate, pGate
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateChandelierNeuron runs the neuron for n steps.
func SimulateChandelierNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return nil, 0
	}
	s := NewChandelierNeuron()
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
