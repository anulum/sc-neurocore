// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for cerebellar_basket_neuron

package services

import (
	"errors"
	"math"
)

const cerebellarBasketExpMax = 709.0
const cerebellarBasketExpMin = -745.0

// CerebellarBasketNeuronState holds the neuron state
type CerebellarBasketNeuronState struct {
	V          float64
	H          float64
	N          float64
	A          float64
	B          float64
	Ca         float64
	GNa        float64
	GK         float64
	GA         float64
	GKca       float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CM         float64
	Phi        float64
	Dt         float64
	VThreshold float64
}

// NewCerebellarBasketNeuron creates a new CerebellarBasketNeuron neuron with default parameters
func NewCerebellarBasketNeuron() *CerebellarBasketNeuronState {
	return &CerebellarBasketNeuronState{
		V:          -65.0,
		H:          0.8,
		N:          0.1,
		A:          0.0,
		B:          0.9,
		Ca:         0.05,
		GNa:        35.0,
		GK:         9.0,
		GA:         3.0,
		GKca:       2.0,
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

func cerebellarBasketFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func cerebellarBasketProbability(x float64) bool {
	return cerebellarBasketFinite(x) && x >= 0.0 && x <= 1.0
}

func checkedCerebellarBasketExp(x float64) (float64, error) {
	if !cerebellarBasketFinite(x) || x > cerebellarBasketExpMax {
		return math.NaN(), errors.New("unstable cerebellar basket exponential argument")
	}
	if x < cerebellarBasketExpMin {
		return 0.0, nil
	}
	return math.Exp(x), nil
}

func cerebellarBasketSafeRate(a, vhalf, v, k, fallback float64) (float64, error) {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback, nil
	}
	expTerm, err := checkedCerebellarBasketExp(-d / k)
	if err != nil {
		return math.NaN(), err
	}
	rate := a * d / (1.0 - expTerm)
	if !cerebellarBasketFinite(rate) {
		return math.NaN(), errors.New("non-finite cerebellar basket rate candidate")
	}
	return rate, nil
}

// ValidateCerebellarBasketNeuron checks the model's physical state contract.
func ValidateCerebellarBasketNeuron(s *CerebellarBasketNeuronState) bool {
	if s == nil {
		return false
	}
	for _, value := range []float64{s.V, s.ENa, s.EK, s.EL, s.VThreshold} {
		if !cerebellarBasketFinite(value) {
			return false
		}
	}
	for _, value := range []float64{s.H, s.N, s.A, s.B} {
		if !cerebellarBasketProbability(value) {
			return false
		}
	}
	if !cerebellarBasketFinite(s.Ca) || s.Ca < 0.0 {
		return false
	}
	for _, value := range []float64{s.GNa, s.GK, s.GA, s.GKca, s.GL} {
		if !cerebellarBasketFinite(value) || value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.CM, s.Phi, s.Dt} {
		if !cerebellarBasketFinite(value) || value <= 0.0 {
			return false
		}
	}
	return true
}

// Step advances the neuron by one timestep.
func (s *CerebellarBasketNeuronState) Step(iExt float64) (int, error) {
	if !ValidateCerebellarBasketNeuron(s) || !cerebellarBasketFinite(iExt) {
		return 0, errors.New("invalid cerebellar basket state or input")
	}
	vPrev := s.V
	nSub := int(0.5 / math.Max(s.Dt, 0.001))
	if nSub < 1 {
		nSub = 1
	}
	v, h, n := s.V, s.H, s.N
	aGate, bGate, ca := s.A, s.B, s.Ca
	for i := 0; i < nSub; i++ {
		am, err := cerebellarBasketSafeRate(0.1, 35.0, v, 10.0, 1.0)
		if err != nil {
			return 0, err
		}
		bmExp, err := checkedCerebellarBasketExp(-(v + 60.0) / 18.0)
		if err != nil {
			return 0, err
		}
		mInf := am / (am + 4.0*bmExp)
		ahExp, err := checkedCerebellarBasketExp(-(v + 58.0) / 20.0)
		if err != nil {
			return 0, err
		}
		ah := 0.07 * ahExp
		bhExp, err := checkedCerebellarBasketExp(-(v + 28.0) / 10.0)
		if err != nil {
			return 0, err
		}
		bh := 1.0 / (1.0 + bhExp)
		an, err := cerebellarBasketSafeRate(0.01, 34.0, v, 10.0, 0.1)
		if err != nil {
			return 0, err
		}
		bnExp, err := checkedCerebellarBasketExp(-(v + 44.0) / 80.0)
		if err != nil {
			return 0, err
		}
		bn := 0.125 * bnExp
		hNext := h + s.Phi*(ah*(1.0-h)-bh*h)*s.Dt
		nNext := n + s.Phi*(an*(1.0-n)-bn*n)*s.Dt
		aInfExp, err := checkedCerebellarBasketExp(-(v + 45.0) / 15.0)
		if err != nil {
			return 0, err
		}
		bInfExp, err := checkedCerebellarBasketExp((v + 75.0) / 8.0)
		if err != nil {
			return 0, err
		}
		aInf := 1.0 / (1.0 + aInfExp)
		bInf := 1.0 / (1.0 + bInfExp)
		aNext := aGate + s.Phi*(aInf-aGate)/5.0*s.Dt
		bNext := bGate + (bInf-bGate)/50.0*s.Dt
		denom := ca + 0.2
		if !cerebellarBasketFinite(denom) || denom <= 0.0 {
			return 0, errors.New("invalid KCa calcium denominator")
		}
		qInf := ca / denom
		iCaEntry := 0.0
		if v > -20.0 {
			iCaEntry = 0.01 * (v + 20.0)
		}
		caNext := math.Max(0.0, ca+(-ca/80.0+iCaEntry)*s.Dt)
		iNa := s.GNa * math.Pow(mInf, 3.0) * hNext * (v - s.ENa)
		iK := s.GK * math.Pow(nNext, 4.0) * (v - s.EK)
		iA := s.GA * math.Pow(aNext, 3.0) * bNext * (v - s.EK)
		iKca := s.GKca * qInf * (v - s.EK)
		iL := s.GL * (v - s.EL)
		vNext := v + (-iNa-iK-iA-iKca-iL+iExt)/s.CM*s.Dt
		if !(cerebellarBasketFinite(vNext) && vNext >= -100.0 && vNext <= 60.0 && cerebellarBasketProbability(hNext) && cerebellarBasketProbability(nNext) && cerebellarBasketProbability(aNext) && cerebellarBasketProbability(bNext) && cerebellarBasketProbability(qInf) && cerebellarBasketFinite(caNext) && caNext >= 0.0) {
			return 0, errors.New("invalid cerebellar basket candidate state")
		}
		v, h, n = vNext, hNext, nNext
		aGate, bGate, ca = aNext, bNext, caNext
	}
	s.V, s.H, s.N = v, h, n
	s.A, s.B, s.Ca = aGate, bGate, ca
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateCerebellarBasketNeuron runs the neuron for n steps.
func SimulateCerebellarBasketNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return nil, 0
	}
	s := NewCerebellarBasketNeuron()
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
