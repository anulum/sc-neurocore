// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go service for MainenSejnowskiNeuron

package services

import (
	"errors"
	"math"
)

// MainenSejnowskiNeuronState holds the complete two-compartment state.
type MainenSejnowskiNeuronState struct {
	Vs         float64
	Va         float64
	M          float64
	H          float64
	N          float64
	Kappa      float64
	GNa        float64
	GK         float64
	GL         float64
	ENa        float64
	EK         float64
	EL         float64
	CS         float64
	CA         float64
	Dt         float64
	VThreshold float64
}

// NewMainenSejnowskiNeuron creates the neuron with the canonical defaults.
func NewMainenSejnowskiNeuron() *MainenSejnowskiNeuronState {
	return &MainenSejnowskiNeuronState{
		Vs: -65.0, Va: -65.0, M: 0.05, H: 0.6, N: 0.3,
		Kappa: 10.0, GNa: 3000.0, GK: 1500.0, GL: 1.0,
		ENa: 50.0, EK: -90.0, EL: -70.0,
		CS: 1.0, CA: 0.1, Dt: 0.005, VThreshold: -20.0,
	}
}

func mainenFinite(values ...float64) bool {
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

func mainenBetween(value, lower, upper float64) bool {
	return value >= lower && value <= upper
}

// ValidMainenSejnowski enforces the public descriptor and runtime bounds.
func ValidMainenSejnowski(s *MainenSejnowskiNeuronState) bool {
	if s == nil || !mainenFinite(
		s.Vs, s.Va, s.M, s.H, s.N, s.Kappa, s.GNa, s.GK, s.GL,
		s.ENa, s.EK, s.EL, s.CS, s.CA, s.Dt, s.VThreshold,
	) {
		return false
	}
	return mainenBetween(s.Vs, -200.0, 200.0) && mainenBetween(s.Va, -200.0, 200.0) &&
		mainenBetween(s.M, 0.0, 1.0) && mainenBetween(s.H, 0.0, 1.0) &&
		mainenBetween(s.N, 0.0, 1.0) &&
		mainenBetween(s.Kappa, 0.0, 100.0) &&
		mainenBetween(s.GNa, 0.0, 5000.0) && mainenBetween(s.GK, 0.0, 3000.0) &&
		mainenBetween(s.GL, 0.0, 5.0) &&
		mainenBetween(s.ENa, 30.0, 70.0) && mainenBetween(s.EK, -100.0, -70.0) &&
		mainenBetween(s.EL, -90.0, -50.0) &&
		mainenBetween(s.CS, 0.5, 2.0) && mainenBetween(s.CA, 0.05, 1.0) &&
		s.Dt > 0.0 && s.Dt <= 0.1 && mainenBetween(s.VThreshold, -40.0, 20.0)
}

func mainenLinoid(x, k float64) float64 {
	if x == 0.0 {
		return k
	}
	return x / -math.Expm1(-x/k)
}

func mainenClamp(value, lower, upper float64) float64 {
	if value < lower {
		return lower
	}
	if value > upper {
		return upper
	}
	return value
}

// TryStep advances the complete recurrence atomically or returns an error.
func (s *MainenSejnowskiNeuronState) TryStep(current float64) (int, error) {
	if !mainenFinite(current) {
		return 0, errors.New("current must be finite")
	}
	if !ValidMainenSejnowski(s) {
		return 0, errors.New("Mainen-Sejnowski state and parameters must satisfy the public bounds")
	}

	next := *s
	vPrev := next.Vs
	for index := 0; index < 20; index++ {
		va := next.Va
		am := 0.182 * mainenLinoid(va+25.0, 9.0)
		bm := 0.124 * mainenLinoid(-(va + 25.0), 9.0)
		ah := 0.024 * mainenLinoid(va+40.0, 5.0)
		bh := 0.0091 * mainenLinoid(-(va + 65.0), 5.0)
		an := 0.02 * mainenLinoid(va-20.0, 9.0)
		bn := 0.002 * mainenLinoid(-(va - 20.0), 9.0)

		next.M = mainenClamp(next.M+(am*(1.0-next.M)-bm*next.M)*next.Dt, 0.0, 1.0)
		next.H = mainenClamp(next.H+(ah*(1.0-next.H)-bh*next.H)*next.Dt, 0.0, 1.0)
		next.N = mainenClamp(next.N+(an*(1.0-next.N)-bn*next.N)*next.Dt, 0.0, 1.0)

		iNa := next.GNa * next.M * next.M * next.M * next.H * (va - next.ENa)
		iK := next.GK * next.N * (va - next.EK)
		iL := next.GL * (next.Vs - next.EL)

		dvs := (-iL + next.Kappa*(va-next.Vs) + current) / next.CS * next.Dt
		dva := (-iNa - iK + next.Kappa*(next.Vs-va)) / next.CA * next.Dt
		next.Vs = mainenClamp(next.Vs+dvs, -200.0, 200.0)
		next.Va = mainenClamp(va+dva, -200.0, 200.0)

		if !mainenFinite(next.Vs, next.Va, next.M, next.H, next.N) {
			return 0, errors.New("Mainen-Sejnowski candidate state became non-finite")
		}
	}

	*s = next
	if s.Vs >= s.VThreshold && vPrev < s.VThreshold {
		return 1, nil
	}
	return 0, nil
}

// Step advances the neuron and fails closed for legacy direct callers.
func (s *MainenSejnowskiNeuronState) Step(iExt float64) int {
	spike, err := s.TryStep(iExt)
	if err != nil {
		return 0
	}
	return spike
}

// Reset restores dynamic state while preserving configuration.
func (s *MainenSejnowskiNeuronState) Reset() {
	s.Vs, s.Va, s.M, s.H, s.N = -65.0, -65.0, 0.05, 0.6, 0.3
}

// SimulateMainenSejnowskiNeuron runs the neuron for n steps.
func SimulateMainenSejnowskiNeuron(nSteps int, iExt float64) ([]float64, int) {
	s := NewMainenSejnowskiNeuron()
	trace := make([]float64, nSteps)
	spikes := 0
	for t := 0; t < nSteps; t++ {
		spikes += s.Step(iExt)
		trace[t] = s.Vs
	}
	return trace, spikes
}
