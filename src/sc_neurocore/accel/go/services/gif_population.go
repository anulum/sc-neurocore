// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for gif_population

package services

import "math"

// GIFPopulationNeuronState holds the Mensi et al. GIF state and parameters.
type GIFPopulationNeuronState struct {
	V            float64
	Theta        float64
	Eta          float64
	TauM         float64
	TauEta       float64
	DeltaV       float64
	Lambda0      float64
	EtaIncrement float64
	VRest        float64
	VReset       float64
	Dt           float64
	Seed         uint64
	rng          uint64
}

// NewGIFPopulationNeuron creates a new GIFPopulationNeuron neuron with default parameters.
func NewGIFPopulationNeuron() *GIFPopulationNeuronState {
	return NewGIFPopulationNeuronWithSeed(42)
}

// NewGIFPopulationNeuronWithSeed creates a deterministic seeded GIF neuron.
func NewGIFPopulationNeuronWithSeed(seed uint64) *GIFPopulationNeuronState {
	if seed == 0 {
		seed = 1
	}
	return &GIFPopulationNeuronState{
		V:            -65.0,
		Theta:        -50.0,
		Eta:          0.0,
		TauM:         20.0,
		TauEta:       100.0,
		DeltaV:       2.0,
		Lambda0:      0.001,
		EtaIncrement: 5.0,
		VRest:        -65.0,
		VReset:       -65.0,
		Dt:           0.5,
		Seed:         seed,
		rng:          seed,
	}
}

func finiteGIFValues(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func (s *GIFPopulationNeuronState) validRuntime() bool {
	return finiteGIFValues(s.V, s.Theta, s.Eta, s.TauM, s.TauEta, s.DeltaV, s.Lambda0, s.EtaIncrement, s.VRest, s.VReset, s.Dt) &&
		s.TauM > 0.0 && s.TauEta > 0.0 && s.DeltaV > 0.0 && s.Lambda0 >= 0.0 && s.Dt > 0.0
}

func (s *GIFPopulationNeuronState) uniform() float64 {
	x := s.rng
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	s.rng = x
	return float64((x*2685821657736338717)>>11) * (1.0 / 9007199254740992.0)
}

func (s *GIFPopulationNeuronState) advanceSubthreshold(iExt float64) (float64, float64, bool) {
	etaDecay := math.Exp(-s.Dt / s.TauEta)
	membraneDecay := math.Exp(-s.Dt / s.TauM)
	x0 := s.V - s.VRest - iExt
	etaNew := s.Eta * etaDecay
	var xNew float64
	if math.Abs(s.TauM-s.TauEta) <= 1e-12 {
		xNew = membraneDecay * (x0 - s.Eta*s.Dt/s.TauM)
	} else {
		coupling := s.TauEta / (s.TauEta - s.TauM)
		xNew = x0*membraneDecay - s.Eta*coupling*(etaDecay-membraneDecay)
	}
	vNew := s.VRest + iExt + xNew
	return vNew, etaNew, finiteGIFValues(vNew, etaNew)
}

func (s *GIFPopulationNeuronState) spikeProbability(voltage float64) float64 {
	if s.Lambda0 == 0.0 {
		return 0.0
	}
	exponent := math.Max(math.Min((voltage-s.Theta)/s.DeltaV, 20.0), -745.0)
	hazard := s.Lambda0 * math.Exp(exponent)
	pSpike := 1.0 - math.Exp(-hazard*s.Dt)
	return math.Max(math.Min(pSpike, 1.0), 0.0)
}

// Step advances the neuron by one timestep.
func (s *GIFPopulationNeuronState) Step(iExt float64) int {
	if math.IsNaN(iExt) || math.IsInf(iExt, 0) || !s.validRuntime() {
		return 0
	}
	vCandidate, etaCandidate, ok := s.advanceSubthreshold(iExt)
	if !ok {
		return 0
	}
	s.V = vCandidate
	s.Eta = etaCandidate
	if s.uniform() < s.spikeProbability(s.V) {
		s.V = s.VReset
		s.Eta += s.EtaIncrement
		return 1
	}
	return 0
}

// Reset restores the dynamic state and deterministic random stream.
func (s *GIFPopulationNeuronState) Reset() {
	s.V = s.VRest
	s.Eta = 0.0
	s.rng = s.Seed
}

// SimulateGIFPopulationNeuron runs the neuron for n steps.
func SimulateGIFPopulationNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewGIFPopulationNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for step := 0; step < nSteps; step++ {
		result := s.Step(iExt)
		trace[step] = s.V
		if result > 0 {
			spikes++
		}
	}
	return trace, spikes
}
