// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for gamma_renewal

package services

import (
	"math"
)

// GammaRenewalNeuronState holds the neuron state
type GammaRenewalNeuronState struct {
	RateHz         float64
	ShapeK         float64
	DtMs           float64
	TimeSinceSpike float64
	Rng            float64
}

// NewGammaRenewalNeuron creates a new GammaRenewalNeuron neuron with default parameters
func NewGammaRenewalNeuron() *GammaRenewalNeuronState {
	return &GammaRenewalNeuronState{
		RateHz:         50.0,
		ShapeK:         3.0,
		DtMs:           1.0,
		TimeSinceSpike: 0.0,
		Rng:            0.0,
	}
}

// Step advances the neuron by one timestep
func (s *GammaRenewalNeuronState) Step(iExt float64) int {
	if !ValidateGammaRenewal(s) || !finite(iExt) {
		return 0
	}
	rateHz := s.RateHz
	if iExt >= 0.0 {
		rateHz = iExt
	}
	if !finite(rateHz) || rateHz < 0.0 {
		return 0
	}
	s.TimeSinceSpike += s.DtMs / 1000.0
	pSpike := s.SpikeProbabilityAt(s.TimeSinceSpike, rateHz)
	if pSpike >= 1.0 {
		s.TimeSinceSpike = 0.0
		return 1
	}
	return 0
}

// SpikeProbabilityAt returns the bounded interval probability from the renewal hazard.
func (s *GammaRenewalNeuronState) SpikeProbabilityAt(elapsedS float64, rateHz float64) float64 {
	if !finite(elapsedS) || elapsedS < 0.0 || !finite(rateHz) || rateHz < 0.0 {
		return 0.0
	}
	if elapsedS < 1.0e-12 || rateHz == 0.0 {
		return 0.0
	}
	k := int(s.ShapeK)
	lambda := float64(k) * rateHz
	x := lambda * elapsedS
	logF := float64(k)*math.Log(lambda) + (float64(k)-1.0)*math.Log(elapsedS) - x - logGammaInt(k)
	density := math.Exp(math.Max(-50.0, math.Min(50.0, logF)))
	survival := math.Max(gammaSurvival(k, x), 1.0e-15)
	hazard := density / survival
	return -math.Expm1(-(hazard * s.DtMs / 1000.0))
}

// ValidateGammaRenewal enforces finite, physically valid renewal parameters.
func ValidateGammaRenewal(s *GammaRenewalNeuronState) bool {
	if s == nil {
		return false
	}
	shape := math.Trunc(s.ShapeK)
	return finite(s.RateHz) && s.RateHz >= 0.0 &&
		finite(s.ShapeK) && s.ShapeK == shape && s.ShapeK > 0.0 &&
		finite(s.DtMs) && s.DtMs > 0.0 &&
		finite(s.TimeSinceSpike) && s.TimeSinceSpike >= 0.0
}

func logGammaInt(k int) float64 {
	if k <= 1 {
		return 0.0
	}
	total := 0.0
	for i := 1; i < k; i++ {
		total += math.Log(float64(i))
	}
	return total
}

func gammaSurvival(k int, x float64) float64 {
	if k <= 0 || !finite(x) {
		return 0.0
	}
	if x < 0.0 {
		return 1.0
	}
	series := 1.0
	term := 1.0
	for i := 1; i < k; i++ {
		term *= x / float64(i)
		series += term
	}
	return math.Exp(-x) * series
}

// SimulateGammaRenewalNeuron runs the neuron for n steps
func SimulateGammaRenewalNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGammaRenewalNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		result := s.Step(iExt)
		trace[t] = s.RateHz
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
