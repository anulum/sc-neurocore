// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for atype_k_neuron

package services

import (
	"errors"
	"math"
)

const aTypeKExpMax = 709.0
const aTypeKExpMin = -745.0

// ATypeKNeuronState holds the neuron state
type ATypeKNeuronState struct {
	V          float64
	H          float64
	N          float64
	A          float64
	B          float64
	GNa        float64
	GK         float64
	GA         float64
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

// NewATypeKNeuron creates a new ATypeKNeuron neuron with default parameters
func NewATypeKNeuron() *ATypeKNeuronState {
	return &ATypeKNeuronState{
		V:          -65.0,
		H:          0.6,
		N:          0.32,
		A:          0.1,
		B:          0.8,
		GNa:        35.0,
		GK:         9.0,
		GA:         8.0,
		GL:         0.1,
		ENa:        55.0,
		EK:         -90.0,
		EL:         -65.0,
		CM:         1.0,
		Phi:        5.0,
		Dt:         0.5,
		VThreshold: -20.0,
		Gain:       1.0,
		SubSteps:   50,
	}
}

func aTypeKFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func aTypeKProbability(x float64) bool {
	return aTypeKFinite(x) && x >= 0.0 && x <= 1.0
}

func checkedATypeKExp(x float64) (float64, error) {
	if !aTypeKFinite(x) || x > aTypeKExpMax {
		return math.NaN(), errors.New("unstable A-type K exponential argument")
	}
	if x < aTypeKExpMin {
		return 0.0, nil
	}
	return math.Exp(x), nil
}

func aTypeKSafeRate(a, vhalf, v, k, fallback float64) (float64, error) {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback, nil
	}
	expTerm, err := checkedATypeKExp(-d / k)
	if err != nil {
		return math.NaN(), err
	}
	rate := a * d / (1.0 - expTerm)
	if !aTypeKFinite(rate) {
		return math.NaN(), errors.New("non-finite A-type K rate candidate")
	}
	return rate, nil
}

// ValidateATypeKNeuron checks the model's physical state contract.
func ValidateATypeKNeuron(s *ATypeKNeuronState) bool {
	if s == nil {
		return false
	}
	for _, value := range []float64{s.V, s.ENa, s.EK, s.EL, s.VThreshold, s.Gain} {
		if !aTypeKFinite(value) {
			return false
		}
	}
	for _, value := range []float64{s.H, s.N, s.A, s.B} {
		if !aTypeKProbability(value) {
			return false
		}
	}
	for _, value := range []float64{s.GNa, s.GK, s.GA, s.GL} {
		if !aTypeKFinite(value) || value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.CM, s.Phi, s.Dt} {
		if !aTypeKFinite(value) || value <= 0.0 {
			return false
		}
	}
	return s.SubSteps > 0
}

// Step advances the neuron by one timestep.
func (s *ATypeKNeuronState) Step(iExt float64) (int, error) {
	if !ValidateATypeKNeuron(s) || !aTypeKFinite(iExt) {
		return 0, errors.New("invalid A-type K neuron state or input")
	}
	inp := s.Gain * iExt
	if !aTypeKFinite(inp) {
		return 0, errors.New("invalid A-type K input drive")
	}
	subDt := s.Dt / float64(s.SubSteps)
	if !aTypeKFinite(subDt) || subDt <= 0.0 {
		return 0, errors.New("invalid A-type K substep")
	}
	v, h, n := s.V, s.H, s.N
	aGate, bGate := s.A, s.B
	fired := 0
	for i := 0; i < s.SubSteps; i++ {
		alphaM, err := aTypeKSafeRate(0.1, 35.0, v, 10.0, 1.0)
		if err != nil {
			return 0, err
		}
		betaMExp, err := checkedATypeKExp(-(v + 60.0) / 18.0)
		if err != nil {
			return 0, err
		}
		mInf := alphaM / (alphaM + 4.0*betaMExp)
		alphaHExp, err := checkedATypeKExp(-(v + 58.0) / 20.0)
		if err != nil {
			return 0, err
		}
		alphaH := 0.07 * alphaHExp
		betaHExp, err := checkedATypeKExp(-(v + 28.0) / 10.0)
		if err != nil {
			return 0, err
		}
		betaH := 1.0 / (1.0 + betaHExp)
		alphaN, err := aTypeKSafeRate(0.01, 34.0, v, 10.0, 0.1)
		if err != nil {
			return 0, err
		}
		betaNExp, err := checkedATypeKExp(-(v + 44.0) / 80.0)
		if err != nil {
			return 0, err
		}
		betaN := 0.125 * betaNExp
		aInfExp, err := checkedATypeKExp(-(v + 50.0) / 20.0)
		if err != nil {
			return 0, err
		}
		aInf := 1.0 / (1.0 + aInfExp)
		bInfExp, err := checkedATypeKExp((v + 70.0) / 6.0)
		if err != nil {
			return 0, err
		}
		bInf := 1.0 / (1.0 + bInfExp)
		hNext := h + subDt*s.Phi*(alphaH*(1.0-h)-betaH*h)
		nNext := n + subDt*s.Phi*(alphaN*(1.0-n)-betaN*n)
		aNext := aGate + subDt*(aInf-aGate)/2.0
		bNext := bGate + subDt*(bInf-bGate)/50.0
		iNa := s.GNa * math.Pow(mInf, 3.0) * hNext * (v - s.ENa)
		iK := s.GK * math.Pow(nNext, 4.0) * (v - s.EK)
		iA := s.GA * math.Pow(aNext, 3.0) * bNext * (v - s.EK)
		iL := s.GL * (v - s.EL)
		dV := (-iNa - iK - iA - iL + inp) / s.CM
		vNext := v + subDt*dV
		if vNext >= s.VThreshold {
			fired = 1
			vNext = -65.0
		}
		if !(aTypeKFinite(vNext) && vNext >= -100.0 && vNext <= 60.0 && aTypeKProbability(hNext) && aTypeKProbability(nNext) && aTypeKProbability(aNext) && aTypeKProbability(bNext)) {
			return 0, errors.New("invalid A-type K candidate state")
		}
		v, h, n = vNext, hNext, nNext
		aGate, bGate = aNext, bNext
	}
	s.V, s.H, s.N = v, h, n
	s.A, s.B = aGate, bGate
	return fired, nil
}

// SimulateATypeKNeuron runs the neuron for n steps.
func SimulateATypeKNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return nil, 0
	}
	s := NewATypeKNeuron()
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
