// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// SPDX-FileCopyrightText: 2026 Arcane Sapience
//
// This file is part of SC-NeuroCore.
// Licensed under the GNU Affero General Public License v3.0 or later.
// See <https://www.gnu.org/licenses/>.

package services

import (
	"math"
	"testing"
)

type buteraState struct {
	v    float64
	n    float64
	hNap float64
}

func buteraReferenceRates(v float64) (float64, float64, float64, float64, float64, float64) {
	mNa := 1.0 / (1.0 + math.Exp(-(v+34.0)/5.0))
	mNap := 1.0 / (1.0 + math.Exp(-(v+40.0)/6.0))
	hNapInf := 1.0 / (1.0 + math.Exp((v+48.0)/6.0))
	nInf := 1.0 / (1.0 + math.Exp(-(v+29.0)/4.0))
	tauN := math.Max(10.0/math.Max(math.Cosh((v+29.0)/8.0), 1e-12), 0.01)
	tauH := math.Max(10000.0/math.Max(math.Cosh((v+48.0)/12.0), 1e-12), 0.1)
	return mNa, mNap, hNapInf, nInf, tauN, tauH
}

func buteraDerivatives(s buteraState, cell ButeraRespiratoryNeuronState, current float64) buteraState {
	mNa, mNap, hNapInf, nInf, tauN, tauH := buteraReferenceRates(s.v)
	iNa := cell.GNa * math.Pow(mNa, 3) * (1.0 - s.n) * (s.v - cell.ENa)
	iNap := cell.GNap * mNap * s.hNap * (s.v - cell.ENa)
	iK := cell.GK * math.Pow(s.n, 4) * (s.v - cell.EK)
	iL := cell.GL * (s.v - cell.EL)
	return buteraState{
		v:    -iNa - iNap - iK - iL + current,
		n:    (nInf - s.n) / tauN,
		hNap: (hNapInf - s.hNap) / tauH,
	}
}

func buteraReferenceRK4(cell ButeraRespiratoryNeuronState, current float64) buteraState {
	state := buteraState{v: cell.V, n: cell.N, hNap: cell.HNap}
	k1 := buteraDerivatives(state, cell, current)
	k2 := buteraDerivatives(buteraState{v: state.v + 0.5*cell.Dt*k1.v, n: state.n + 0.5*cell.Dt*k1.n, hNap: state.hNap + 0.5*cell.Dt*k1.hNap}, cell, current)
	k3 := buteraDerivatives(buteraState{v: state.v + 0.5*cell.Dt*k2.v, n: state.n + 0.5*cell.Dt*k2.n, hNap: state.hNap + 0.5*cell.Dt*k2.hNap}, cell, current)
	k4 := buteraDerivatives(buteraState{v: state.v + cell.Dt*k3.v, n: state.n + cell.Dt*k3.n, hNap: state.hNap + cell.Dt*k3.hNap}, cell, current)
	return buteraState{
		v:    state.v + cell.Dt*(k1.v+2.0*k2.v+2.0*k3.v+k4.v)/6.0,
		n:    state.n + cell.Dt*(k1.n+2.0*k2.n+2.0*k3.n+k4.n)/6.0,
		hNap: state.hNap + cell.Dt*(k1.hNap+2.0*k2.hNap+2.0*k3.hNap+k4.hNap)/6.0,
	}
}

func TestButeraRespiratoryMatchesIndependentRK4(t *testing.T) {
	cell := NewButeraRespiratoryNeuron()
	cell.V = -48.0
	cell.N = 0.08
	cell.HNap = 0.62
	cell.Dt = 0.025
	expected := buteraReferenceRK4(*cell, 18.0)

	spike, err := cell.Step(18.0)
	if err != nil {
		t.Fatalf("step returned error: %v", err)
	}
	if spike != 0 && spike != 1 {
		t.Fatalf("invalid spike value %d", spike)
	}
	if math.Abs(cell.V-expected.v) > 1e-10 || math.Abs(cell.N-expected.n) > 1e-10 || math.Abs(cell.HNap-expected.hNap) > 1e-10 {
		t.Fatalf("state mismatch got (%g,%g,%g) expected (%g,%g,%g)", cell.V, cell.N, cell.HNap, expected.v, expected.n, expected.hNap)
	}
}

func TestButeraRespiratoryInvalidCurrentPreservesState(t *testing.T) {
	cell := NewButeraRespiratoryNeuron()
	before := *cell
	if _, err := cell.Step(math.NaN()); err == nil {
		t.Fatal("expected invalid current error")
	}
	if *cell != before {
		t.Fatalf("state mutated after invalid current")
	}
}
