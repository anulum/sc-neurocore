// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for alpha_motor_neuron

package services

import (
	"errors"
	"math"
)

const alphaMotorExpMax = 709.0
const alphaMotorExpMin = -745.0

// AlphaMotorNeuronState holds the neuron state
type AlphaMotorNeuronState struct {
	V          float64
	H          float64
	N          float64
	MPic       float64
	HPic       float64
	Ca         float64
	CaBuf      float64
	GNa        float64
	GK         float64
	GPic       float64
	GAhp       float64
	GL         float64
	ENa        float64
	EK         float64
	ECa        float64
	EL         float64
	CM         float64
	Phi        float64
	TauCa      float64
	BufRatio   float64
	Dt         float64
	VThreshold float64
}

// NewAlphaMotorNeuron creates a new AlphaMotorNeuron neuron with default parameters
func NewAlphaMotorNeuron() *AlphaMotorNeuronState {
	return &AlphaMotorNeuronState{
		V:          -65.0,
		H:          0.8,
		N:          0.1,
		MPic:       0.0,
		HPic:       1.0,
		Ca:         0.0,
		CaBuf:      0.0,
		GNa:        35.0,
		GK:         9.0,
		GPic:       0.15,
		GAhp:       3.0,
		GL:         0.3,
		ENa:        55.0,
		EK:         -90.0,
		ECa:        120.0,
		EL:         -65.0,
		CM:         1.5,
		Phi:        4.0,
		TauCa:      150.0,
		BufRatio:   0.003,
		Dt:         0.01,
		VThreshold: -20.0,
	}
}

func alphaMotorIsFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func alphaMotorProbability(x float64) bool {
	return alphaMotorIsFinite(x) && x >= 0.0 && x <= 1.0
}

func checkedAlphaMotorExp(x float64) (float64, error) {
	if !alphaMotorIsFinite(x) || x > alphaMotorExpMax {
		return math.NaN(), errors.New("unstable alpha motor exponential argument")
	}
	if x < alphaMotorExpMin {
		return 0.0, nil
	}
	return math.Exp(x), nil
}

func alphaMotorSafeRate(a, vhalf, v, k, fallback float64) (float64, error) {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback, nil
	}
	expTerm, err := checkedAlphaMotorExp(-d / k)
	if err != nil {
		return math.NaN(), err
	}
	rate := a * d / (1.0 - expTerm)
	if !alphaMotorIsFinite(rate) {
		return math.NaN(), errors.New("non-finite alpha motor rate candidate")
	}
	return rate, nil
}

// ValidateAlphaMotorNeuron checks state and physical parameter contracts.
func ValidateAlphaMotorNeuron(s *AlphaMotorNeuronState) bool {
	if s == nil {
		return false
	}
	finiteCore := []float64{s.V, s.ENa, s.EK, s.ECa, s.EL, s.VThreshold}
	for _, value := range finiteCore {
		if !alphaMotorIsFinite(value) {
			return false
		}
	}
	for _, value := range []float64{s.H, s.N, s.MPic, s.HPic} {
		if !alphaMotorProbability(value) {
			return false
		}
	}
	for _, value := range []float64{s.Ca, s.CaBuf} {
		if !alphaMotorIsFinite(value) || value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.GNa, s.GK, s.GPic, s.GAhp, s.GL} {
		if !alphaMotorIsFinite(value) || value < 0.0 {
			return false
		}
	}
	for _, value := range []float64{s.CM, s.Phi, s.TauCa, s.Dt} {
		if !alphaMotorIsFinite(value) || value <= 0.0 {
			return false
		}
	}
	return alphaMotorIsFinite(s.BufRatio) && s.BufRatio >= 0.0 && s.BufRatio <= 1.0
}

// Step advances the neuron by one timestep.
func (s *AlphaMotorNeuronState) Step(iExt float64) (int, error) {
	if !ValidateAlphaMotorNeuron(s) || !alphaMotorIsFinite(iExt) {
		return 0, errors.New("invalid alpha motor neuron state or input")
	}
	vPrev := s.V
	v, h, n := s.V, s.H, s.N
	mPic, hPic := s.MPic, s.HPic
	ca, caBuf := s.Ca, s.CaBuf
	nSub := int(0.5 / math.Max(s.Dt, 0.001))
	if nSub < 1 {
		nSub = 1
	}
	for i := 0; i < nSub; i++ {
		am, err := alphaMotorSafeRate(0.1, 35.0, v, 10.0, 1.0)
		if err != nil {
			return 0, err
		}
		bmExp, err := checkedAlphaMotorExp(-(v + 60.0) / 18.0)
		if err != nil {
			return 0, err
		}
		bm := 4.0 * bmExp
		mInf := am / (am + bm)
		ahExp, err := checkedAlphaMotorExp(-(v + 58.0) / 20.0)
		if err != nil {
			return 0, err
		}
		ah := 0.07 * ahExp
		bhExp, err := checkedAlphaMotorExp(-(v + 28.0) / 10.0)
		if err != nil {
			return 0, err
		}
		bh := 1.0 / (1.0 + bhExp)
		an, err := alphaMotorSafeRate(0.01, 34.0, v, 10.0, 0.1)
		if err != nil {
			return 0, err
		}
		bnExp, err := checkedAlphaMotorExp(-(v + 44.0) / 80.0)
		if err != nil {
			return 0, err
		}
		bn := 0.125 * bnExp
		hNext := h + s.Phi*(ah*(1.0-h)-bh*h)*s.Dt
		nNext := n + s.Phi*(an*(1.0-n)-bn*n)*s.Dt
		mPicExp, err := checkedAlphaMotorExp(-(v + 40.0) / 5.0)
		if err != nil {
			return 0, err
		}
		mPicInf := 1.0 / (1.0 + mPicExp)
		mPicNext := mPic + (mPicInf-mPic)/50.0*s.Dt
		hPicExp, err := checkedAlphaMotorExp((v + 40.0) / 8.0)
		if err != nil {
			return 0, err
		}
		hPicInf := 1.0 / (1.0 + hPicExp)
		tauHPic := 200.0 + 100.0/math.Max(0.01, 1.0+math.Pow((v+40.0)/10.0, 2.0))
		hPicNext := math.Min(1.0, math.Max(0.0, hPic+(hPicInf-hPic)/tauHPic*s.Dt))
		iCaEntry := s.GPic * mPicNext * hPicNext * (v - s.ECa)
		caInflux := 0.0
		if iCaEntry < 0.0 {
			caInflux = -iCaEntry * 0.001
		}
		caSpike := 0.0
		if v > -10.0 {
			caSpike = 0.02
		}
		caNext := math.Max(0.0, ca+(-ca/s.TauCa+(caInflux+caSpike)*s.BufRatio)*s.Dt)
		caBufNext := math.Max(0.0, caBuf+((caInflux+caSpike)*(1.0-s.BufRatio)-caBuf/(s.TauCa*5.0))*s.Dt)
		caTotal := caNext + caBufNext*0.01
		ahpInf := math.Pow(caTotal, 2.0) / (math.Pow(caTotal, 2.0) + 0.25)
		iNa := s.GNa * math.Pow(mInf, 3.0) * hNext * (v - s.ENa)
		iK := s.GK * math.Pow(nNext, 4.0) * (v - s.EK)
		iPic := s.GPic * mPicNext * hPicNext * (v - s.ECa)
		iAhp := s.GAhp * ahpInf * (v - s.EK)
		iL := s.GL * (v - s.EL)
		vNext := v + (-iNa-iK-iPic-iAhp-iL+iExt)/s.CM*s.Dt
		if !(alphaMotorIsFinite(vNext) && alphaMotorProbability(hNext) && alphaMotorProbability(nNext) && alphaMotorProbability(mPicNext) && alphaMotorProbability(hPicNext) && alphaMotorIsFinite(caNext) && caNext >= 0.0 && alphaMotorIsFinite(caBufNext) && caBufNext >= 0.0) {
			return 0, errors.New("invalid alpha motor neuron candidate state")
		}
		v, h, n = vNext, hNext, nNext
		mPic, hPic = mPicNext, hPicNext
		ca, caBuf = caNext, caBufNext
	}
	s.V, s.H, s.N = v, h, n
	s.MPic, s.HPic = mPic, hPic
	s.Ca, s.CaBuf = ca, caBuf
	if s.V >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// SimulateAlphaMotorNeuron runs the neuron for n steps.
func SimulateAlphaMotorNeuron(nSteps int, iExt float64) ([]float64, int) {
	if nSteps < 0 {
		return nil, 0
	}
	s := NewAlphaMotorNeuron()
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
