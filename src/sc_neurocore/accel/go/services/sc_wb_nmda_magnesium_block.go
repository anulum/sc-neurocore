// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go retained SC WB plus NMDA recurrence

package services

import (
	"errors"
	"math"
)

// SCWBNMDAMagnesiumBlockNeuronState retains the historical project state.
type SCWBNMDAMagnesiumBlockNeuronState struct {
	V, H, N, SNmda, GNa, GK, GNmda, GL, ENa, EK, ENmda, EL, CM, Phi, MgConc, TauRise, TauDecay, Dt, VThreshold, Gain float64
	SubSteps                                                                                                         int
}

func NewSCWBNMDAMagnesiumBlockNeuron() *SCWBNMDAMagnesiumBlockNeuronState {
	return &SCWBNMDAMagnesiumBlockNeuronState{
		V: -65, H: 0.6, N: 0.32, GNa: 35, GK: 9, GNmda: 0.5, GL: 0.1, ENa: 55, EK: -90, EL: -65, CM: 1, Phi: 5,
		MgConc: 1, TauRise: 10, TauDecay: 100, Dt: 0.5, VThreshold: -20, Gain: 1, SubSteps: 50}
}
func scNmdaSafeRate(a, vhalf, v, k, fallback float64) float64 {
	d := v + vhalf
	if math.Abs(d) < 1e-7 {
		return fallback
	}
	return a * d / (1 - math.Exp(-d/k))
}
func ValidSCWBNMDAMagnesiumBlockNeuron(s *SCWBNMDAMagnesiumBlockNeuronState) bool {
	return s != nil && nmdaFinite(s.V, s.H, s.N, s.SNmda, s.GNa, s.GK, s.GNmda, s.GL, s.ENa, s.EK, s.ENmda, s.EL, s.CM, s.Phi, s.MgConc, s.TauRise, s.TauDecay, s.Dt, s.VThreshold, s.Gain) &&
		nmdaBetween(s.V, -100, 60) && nmdaBetween(s.H, 0, 1) && nmdaBetween(s.N, 0, 1) && nmdaBetween(s.SNmda, 0, 1) &&
		nmdaBetween(s.GNa, 0, 200) && nmdaBetween(s.GK, 0, 100) && nmdaBetween(s.GNmda, 0, 20) && nmdaBetween(s.GL, 0, 5) &&
		nmdaBetween(s.ENa, 30, 70) && nmdaBetween(s.EK, -100, -70) && nmdaBetween(s.ENmda, -10, 10) && nmdaBetween(s.EL, -80, -40) &&
		nmdaBetween(s.CM, 0.5, 2) && nmdaBetween(s.Phi, 0.5, 10) && nmdaBetween(s.MgConc, 0, 5) && nmdaBetween(s.TauRise, 0.1, 20) &&
		nmdaBetween(s.TauDecay, 10, 500) && s.Dt > 0 && s.Dt <= 1 && nmdaBetween(s.VThreshold, -20, 20) && nmdaBetween(s.Gain, 0, 10) && s.SubSteps >= 1 && s.SubSteps <= 10000
}
func (s *SCWBNMDAMagnesiumBlockNeuronState) TryStep(current float64) (int, error) {
	if !nmdaFinite(current) {
		return 0, errors.New("current must be finite")
	}
	if !ValidSCWBNMDAMagnesiumBlockNeuron(s) {
		return 0, errors.New("SC NMDA state and parameters must satisfy the public bounds")
	}
	next := *s
	input := next.Gain * current
	subDt := next.Dt / float64(next.SubSteps)
	drive := 0.0
	if input > 0 {
		drive = input / (input + 5)
	}
	tau := next.TauDecay
	if drive > next.SNmda {
		tau = next.TauRise
	}
	next.SNmda = math.Max(0, math.Min(1, next.SNmda+next.Dt*(drive-next.SNmda)/tau))
	event := 0
	for i := 0; i < next.SubSteps; i++ {
		v := next.V
		am := scNmdaSafeRate(0.1, 35, v, 10, 1)
		bm := 4 * math.Exp(-(v+60)/18)
		mi := am / (am + bm)
		ah := 0.07 * math.Exp(-(v+58)/20)
		bh := 1 / (1 + math.Exp(-(v+28)/10))
		an := scNmdaSafeRate(0.01, 34, v, 10, 0.1)
		bn := 0.125 * math.Exp(-(v+44)/80)
		block := 1 / (1 + (next.MgConc/3.57)*math.Exp(-0.062*v))
		next.H += subDt * next.Phi * (ah*(1-next.H) - bh*next.H)
		next.N += subDt * next.Phi * (an*(1-next.N) - bn*next.N)
		ina := next.GNa * math.Pow(mi, 3) * next.H * (v - next.ENa)
		ik := next.GK * math.Pow(next.N, 4) * (v - next.EK)
		inmda := next.GNmda * next.SNmda * block * (v - next.ENmda)
		il := next.GL * (v - next.EL)
		next.V += subDt * (-ina - ik - inmda - il + input) / next.CM
		if !nmdaFinite(next.V, next.H, next.N) {
			return 0, errors.New("SC NMDA candidate state became non-finite")
		}
		if next.V >= next.VThreshold {
			event = 1
			next.V = -65
		}
	}
	next.V = math.Max(-100, math.Min(60, next.V))
	next.H = math.Max(0, math.Min(1, next.H))
	next.N = math.Max(0, math.Min(1, next.N))
	*s = next
	return event, nil
}
func (s *SCWBNMDAMagnesiumBlockNeuronState) Step(current float64) int {
	e, err := s.TryStep(current)
	if err != nil {
		return 0
	}
	return e
}
func (s *SCWBNMDAMagnesiumBlockNeuronState) Reset() { s.V = -65; s.H = 0.6; s.N = 0.32; s.SNmda = 0 }
