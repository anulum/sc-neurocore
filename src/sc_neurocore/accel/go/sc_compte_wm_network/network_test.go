// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.

package sccomptewmnetwork

import (
	"math"
	"testing"

	"github.com/anulum/sc-neurocore/accel/services"
)

func newTestNetwork(t *testing.T, spec Spec, state *State) *Network {
	t.Helper()
	network, err := NewNetwork(spec, state)
	if err != nil {
		t.Fatal(err)
	}
	return network
}

func zeroInputs() ([]float64, []uint64, []uint64) {
	return make([]float64, NExcitatory), make([]uint64, NExcitatory), make([]uint64, NInhibitory)
}

func TestFixedPopulationsAndCounterFixture(t *testing.T) {
	network := newTestNetwork(t, DefaultSpec(), nil)
	if len(network.State.VExcMV) != NExcitatory || len(network.State.VInhMV) != NInhibitory {
		t.Fatal("fixed populations are incomplete")
	}
	counts, err := CounterPoissonCounts(64, 1800, DtMS, 42, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	active := make([]int, 0)
	for index, count := range counts {
		if count != 0 {
			active = append(active, index)
		}
	}
	if len(active) != 2 || active[0] != 49 || active[1] != 61 || sumUint64(counts) != 2 {
		t.Fatalf("counter fixture mismatch: active=%v total=%d", active, sumUint64(counts))
	}
}

func TestIsolatedImpulseMatchesPreservedScalarCell(t *testing.T) {
	network := newTestNetwork(t, DefaultSpec(), nil)
	current, exc, inh := zeroInputs()
	exc[17] = 1
	receipt, err := network.StepWithEvents(current, exc, inh)
	if err != nil {
		t.Fatal(err)
	}
	original := services.NewCompteWMNeuron()
	if _, err = original.StepWithEvents(0, false, true, false); err != nil {
		t.Fatal(err)
	}
	if math.Abs(network.State.VExcMV[17]-original.V) > 2e-14 ||
		math.Abs(network.State.ExternalAMPAExc[17]-original.SAmpa) > 2e-14 {
		t.Fatal("network impulse does not match preserved scalar cell")
	}
	if receipt.ExcitatoryInputEvents != 1 {
		t.Fatal("event receipt lost isolated impulse")
	}
}

func TestRecurrentFFTParityAnchor(t *testing.T) {
	state := NewState()
	for index := range state.VExcMV {
		state.VExcMV[index] = -60
	}
	indices := []int{0, 37, 1024, 1901}
	values := []float64{0.2, 0.4, 0.1, 0.3}
	for index := range indices {
		state.RecurrentNMDA[indices[index]] = values[index]
	}
	network := newTestNetwork(t, DefaultSpec(), state)
	current, exc, inh := zeroInputs()
	if _, err := network.StepWithEvents(current, exc, inh); err != nil {
		t.Fatal(err)
	}
	if math.Abs(network.State.VExcMV[113]-(-60.0099068230443)) > 3e-13 {
		t.Fatalf("recurrent parity mismatch: %.17g", network.State.VExcMV[113])
	}
	if network.State.RecurrentNMDA[37] != 0.39992000800000005 {
		t.Fatal("NMDA midpoint fixture mismatch")
	}
}

func TestDeterministicReceiptsAndSeedSeparation(t *testing.T) {
	first, err := newTestNetwork(t, DefaultSpec(), nil).Run(0.1, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	second, err := newTestNetwork(t, DefaultSpec(), nil).Run(0.1, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	thirdSpec := DefaultSpec()
	thirdSpec.Seed = 43
	third, err := newTestNetwork(t, thirdSpec, nil).Run(0.1, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if first.InputSHA256 != second.InputSHA256 || first.SpikeSHA256 != second.SpikeSHA256 ||
		first.FinalStateSHA256 != second.FinalStateSHA256 {
		t.Fatal("same seed is not deterministic")
	}
	if first.InputSHA256 == third.InputSHA256 || first.FinalStateSHA256 == third.FinalStateSHA256 {
		t.Fatal("different seed did not separate receipts")
	}
}

func TestFullPopulationStimulusAndRefractory(t *testing.T) {
	network := newTestNetwork(t, DefaultSpec(), nil)
	stimulus := Stimulus{StartMS: 0, DurationMS: DtMS, CurrentPA: 600000, Kind: "global_current"}
	receipt, err := network.Run(DtMS, []Stimulus{stimulus}, DtMS)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.ExcitatorySpikes != NExcitatory || receipt.Windows[0].Statistics == nil {
		t.Fatal("global current did not recruit complete excitatory population")
	}
	for index := range network.State.VExcMV {
		if network.State.VExcMV[index] != -60 || network.State.RefractoryExcMS[index] != 2 {
			t.Fatal("spike reset/refractory mismatch")
		}
	}
	current, exc, inh := zeroInputs()
	if _, err = network.StepWithEvents(current, exc, inh); err != nil {
		t.Fatal(err)
	}
	for _, voltage := range network.State.VExcMV {
		if voltage != -60 {
			t.Fatal("refractory voltage was not clamped")
		}
	}
}

func TestInvalidInputIsAtomic(t *testing.T) {
	network := newTestNetwork(t, DefaultSpec(), nil)
	before, _ := StateSHA256(network.State)
	current, exc, inh := zeroInputs()
	current[4] = math.NaN()
	if _, err := network.StepWithEvents(current, exc, inh); err == nil {
		t.Fatal("non-finite current did not fail")
	}
	after, _ := StateSHA256(network.State)
	if before != after {
		t.Fatal("invalid current mutated state")
	}
	current[4] = 0
	if _, err := network.StepWithEvents(current, exc[:len(exc)-1], inh); err == nil {
		t.Fatal("invalid shape did not fail")
	}
	after, _ = StateSHA256(network.State)
	if before != after {
		t.Fatal("invalid shape mutated state")
	}
}
