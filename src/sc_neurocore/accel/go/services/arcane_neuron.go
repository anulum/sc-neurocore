// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for arcane_neuron

package services

import (
	"errors"
	"math"
)

// ArcaneNeuronState holds the five-subsystem ArcaneNeuron state.
type ArcaneNeuronState struct {
	VFast            float64
	TauFast          float64
	VWork            float64
	TauWork          float64
	AlphaW           float64
	VDeep            float64
	TauDeep          float64
	AlphaD           float64
	Theta            float64
	Gamma            float64
	DeltaConf        float64
	WGate            [4]float64
	WPred            [3]float64
	Kappa            float64
	SurpriseBaseline float64
	LrBase           float64
	Eta              float64
	Prediction       float64
	Surprise         float64
	Novelty          float64
	Confidence       float64
	SpikeHistory     [50]float64
	NoveltyHistory   [20]float64
	HistIdx          int
	NovIdx           int
	TotalSteps       int
	IdentityDrift    float64
	WInh             float64
	Dt               float64
}

// NewArcaneNeuron creates a new ArcaneNeuron neuron with default parameters.
func NewArcaneNeuron() *ArcaneNeuronState {
	state := &ArcaneNeuronState{
		VFast:            0.0,
		TauFast:          5.0,
		VWork:            0.0,
		TauWork:          200.0,
		AlphaW:           0.3,
		VDeep:            0.0,
		TauDeep:          10000.0,
		AlphaD:           0.05,
		Theta:            1.0,
		Gamma:            0.2,
		DeltaConf:        0.3,
		WGate:            [4]float64{0.8, 0.1, 0.05, 0.05},
		WPred:            [3]float64{0.6, 0.3, 0.1},
		Kappa:            5.0,
		SurpriseBaseline: 0.1,
		LrBase:           0.01,
		Eta:              2.0,
		Prediction:       0.0,
		Surprise:         0.0,
		Novelty:          0.0,
		Confidence:       0.5,
		HistIdx:          0,
		NovIdx:           0,
		TotalSteps:       0,
		IdentityDrift:    0.0,
		WInh:             0.3,
		Dt:               1.0,
	}
	for i := range state.NoveltyHistory {
		state.NoveltyHistory[i] = 0.5
	}
	return state
}

func (s ArcaneNeuronState) Valid() bool {
	if !arcaneFinite(s.VFast) || !arcaneFinite(s.VWork) || !arcaneFinite(s.VDeep) || !arcaneFinite(s.Prediction) || !arcaneFinite(s.Surprise) || !arcaneFinite(s.Novelty) || !arcaneFinite(s.Confidence) || !arcaneFinite(s.IdentityDrift) {
		return false
	}
	if !arcaneFinite(s.TauFast) || s.TauFast <= 0.0 || !arcaneFinite(s.TauWork) || s.TauWork <= 0.0 || !arcaneFinite(s.TauDeep) || s.TauDeep <= 0.0 || !arcaneFinite(s.Dt) || s.Dt <= 0.0 {
		return false
	}
	if !arcaneFinite(s.AlphaW) || s.AlphaW < 0.0 || !arcaneFinite(s.AlphaD) || s.AlphaD < 0.0 || !arcaneFinite(s.Theta) || s.Theta <= 0.0 || !arcaneFinite(s.Gamma) || !arcaneFinite(s.DeltaConf) || !arcaneFinite(s.Kappa) || !arcaneFinite(s.SurpriseBaseline) || !arcaneFinite(s.LrBase) || s.LrBase < 0.0 || !arcaneFinite(s.Eta) || !arcaneFinite(s.WInh) || s.WInh < 0.0 {
		return false
	}
	for _, value := range s.WGate {
		if !arcaneFinite(value) {
			return false
		}
	}
	for _, value := range s.WPred {
		if !arcaneFinite(value) {
			return false
		}
	}
	for _, value := range s.SpikeHistory {
		if value != 0.0 && value != 1.0 {
			return false
		}
	}
	for _, value := range s.NoveltyHistory {
		if !arcaneFinite(value) {
			return false
		}
	}
	return s.HistIdx >= 0 && s.NovIdx >= 0 && s.TotalSteps >= 0
}

// Step advances the neuron by one timestep. Invalid inputs do not mutate state.
func (s *ArcaneNeuronState) Step(iExt float64) (int, error) {
	if !arcaneFinite(iExt) || !s.Valid() {
		return 0, ErrArcaneNeuronInvalidState
	}

	spikeRate := arcaneMean50(s.SpikeHistory)
	confidence := 1.0 - arcaneMean20(s.NoveltyHistory)
	gateInput := s.WGate[0]*iExt + s.WGate[1]*s.VFast + s.WGate[2]*s.VWork + s.WGate[3]*confidence
	gate := arcaneSigmoid(gateInput)
	iEff := gate * iExt
	fastDrive := iEff - s.WInh*spikeRate
	nextVFastContinuous := arcaneExactRelaxation(s.VFast, fastDrive, s.Dt, s.TauFast)
	if !arcaneFinite(nextVFastContinuous) {
		return 0, ErrArcaneNeuronNonFiniteUpdate
	}

	prediction := s.WPred[0]*nextVFastContinuous + s.WPred[1]*s.VWork + s.WPred[2]*s.VDeep
	if !arcaneFinite(prediction) {
		return 0, ErrArcaneNeuronNonFiniteUpdate
	}
	surprise := math.Abs(nextVFastContinuous - prediction)
	novelty := arcaneSigmoid(s.Kappa * (surprise - s.SurpriseBaseline))

	effThreshold := s.Theta * (1.0 + s.Gamma*s.VDeep) * (1.0 - s.DeltaConf*confidence)
	if !arcaneFinite(effThreshold) {
		return 0, ErrArcaneNeuronNonFiniteUpdate
	}
	if effThreshold < 0.1 {
		effThreshold = 0.1
	}

	spike := 0
	acceptedVFast := nextVFastContinuous
	if nextVFastContinuous >= effThreshold {
		spike = 1
		acceptedVFast = 0.0
	}

	workDrive := 0.0
	if spike == 1 {
		workDrive = s.AlphaW * nextVFastContinuous
	}
	nextVWork := arcaneExactRelaxation(s.VWork, workDrive, s.Dt, s.TauWork)
	if !arcaneFinite(nextVWork) {
		return 0, ErrArcaneNeuronNonFiniteUpdate
	}

	deepDrive := s.AlphaD * nextVWork * novelty
	nextVDeep := arcaneExactRelaxation(s.VDeep, deepDrive, s.Dt, s.TauDeep)
	if !arcaneFinite(nextVDeep) {
		return 0, ErrArcaneNeuronNonFiniteUpdate
	}

	metaLR := s.LrBase * (1.0 + s.Eta*novelty)
	errorTerm := acceptedVFast - prediction
	nextWPred := s.WPred
	nextWPred[0] += metaLR * errorTerm * acceptedVFast
	nextWPred[1] += metaLR * errorTerm * nextVWork
	nextWPred[2] += metaLR * errorTerm * nextVDeep
	norm := math.Sqrt(nextWPred[0]*nextWPred[0] + nextWPred[1]*nextWPred[1] + nextWPred[2]*nextWPred[2])
	if !arcaneFinite(norm) {
		return 0, ErrArcaneNeuronNonFiniteUpdate
	}
	if norm > 0.0 {
		nextWPred[0] /= norm
		nextWPred[1] /= norm
		nextWPred[2] /= norm
	}
	for _, value := range nextWPred {
		if !arcaneFinite(value) {
			return 0, ErrArcaneNeuronNonFiniteUpdate
		}
	}

	nextNoveltyHistory := s.NoveltyHistory
	nextNoveltyHistory[s.NovIdx%len(nextNoveltyHistory)] = novelty
	nextSpikeHistory := s.SpikeHistory
	nextSpikeHistory[s.HistIdx%len(nextSpikeHistory)] = float64(spike)

	oldVDeep := s.VDeep
	s.VFast = acceptedVFast
	s.VWork = nextVWork
	s.VDeep = nextVDeep
	s.Prediction = prediction
	s.Surprise = surprise
	s.Novelty = novelty
	s.Confidence = confidence
	s.NoveltyHistory = nextNoveltyHistory
	s.NovIdx++
	s.IdentityDrift += math.Abs(nextVDeep - oldVDeep)
	s.WPred = nextWPred
	s.SpikeHistory = nextSpikeHistory
	s.HistIdx++
	s.TotalSteps++
	return spike, nil
}

func (s *ArcaneNeuronState) Reset() {
	s.VFast = 0.0
	s.VWork = 0.0
	s.Prediction = 0.0
	s.Surprise = 0.0
	s.Novelty = 0.0
	s.SpikeHistory = [50]float64{}
	s.HistIdx = 0
	s.IdentityDrift = 0.0
}

func (s ArcaneNeuronState) IdentityState() float64    { return s.VDeep }
func (s ArcaneNeuronState) MetaLearningRate() float64 { return s.LrBase * (1.0 + s.Eta*s.Novelty) }

// SimulateArcaneNeuron runs the neuron for n steps.
func SimulateArcaneNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewArcaneNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result, err := s.Step(iExt)
		if err != nil {
			panic(err)
		}
		trace[t] = s.VFast
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}

var (
	ErrArcaneNeuronInvalidState    = errors.New("ArcaneNeuron state/current must be finite and physically valid")
	ErrArcaneNeuronNonFiniteUpdate = errors.New("ArcaneNeuron exact relaxation update became non-finite")
)

func arcaneExactRelaxation(state float64, steadyState float64, dt float64, tau float64) float64 {
	decay := math.Exp(-dt / tau)
	return decay*state + (1.0-decay)*steadyState
}

func arcaneSigmoid(x float64) float64 {
	if math.IsInf(x, 1) {
		return 1.0
	}
	if math.IsInf(x, -1) {
		return 0.0
	}
	if x >= 0.0 {
		z := math.Exp(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}

func arcaneMean50(values [50]float64) float64 {
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func arcaneMean20(values [20]float64) float64 {
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func arcaneFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
