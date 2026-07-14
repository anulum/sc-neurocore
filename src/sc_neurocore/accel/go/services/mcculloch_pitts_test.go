// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go tests for the source-faithful McCulloch-Pitts service

package services

import (
	"reflect"
	"testing"
)

func TestMcCullochPittsDefaultAndResetlessState(t *testing.T) {
	state := NewMcCullochPittsNeuron()
	if !ValidateMcCullochPitts(state) || state.Theta != 1 {
		t.Fatalf("invalid default state: %+v", state)
	}
}

func TestMcCullochPittsThetaOneIsOR(t *testing.T) {
	state := NewMcCullochPittsNeuron()
	for count, want := range []int{0, 1, 1} {
		got, err := state.Step(int64(count), false)
		if err != nil || got != want {
			t.Fatalf("count %d: got (%d, %v), want %d", count, got, err, want)
		}
	}
}

func TestMcCullochPittsThetaTwoIsAND(t *testing.T) {
	state, err := NewMcCullochPittsNeuronWithTheta(2)
	if err != nil {
		t.Fatal(err)
	}
	for count, want := range []int{0, 0, 1} {
		got, stepErr := state.Step(int64(count), false)
		if stepErr != nil || got != want {
			t.Fatalf("count %d: got (%d, %v), want %d", count, got, stepErr, want)
		}
	}
}

func TestMcCullochPittsAbsoluteInhibition(t *testing.T) {
	got, err := NewMcCullochPittsNeuron().Step(maxMcCullochPittsCount, true)
	if err != nil || got != 0 {
		t.Fatalf("absolute inhibition returned (%d, %v)", got, err)
	}
}

func TestMcCullochPittsRejectsInvalidThresholdAndCount(t *testing.T) {
	for _, theta := range []int64{0, -1, maxMcCullochPittsCount + 1} {
		if _, err := NewMcCullochPittsNeuronWithTheta(theta); err == nil {
			t.Fatalf("theta %d did not fail", theta)
		}
	}
	for _, count := range []int64{-1, maxMcCullochPittsCount + 1} {
		if _, err := NewMcCullochPittsNeuron().Step(count, false); err == nil {
			t.Fatalf("count %d did not fail", count)
		}
	}
}

func TestMcCullochPittsBatchIsExact(t *testing.T) {
	events, count, err := EvaluateMcCullochPittsBatch(
		2,
		[]int64{0, 1, 2, maxMcCullochPittsCount},
		[]uint8{0, 0, 0, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(events, []uint8{0, 0, 1, 0}) || count != 1 {
		t.Fatalf("got (%v, %d)", events, count)
	}
}

func TestMcCullochPittsEmptyBatch(t *testing.T) {
	events, count, err := EvaluateMcCullochPittsBatch(1, nil, nil)
	if err != nil || len(events) != 0 || count != 0 {
		t.Fatalf("got (%v, %d, %v)", events, count, err)
	}
}

func TestMcCullochPittsMalformedBatchFails(t *testing.T) {
	cases := []struct {
		counts []int64
		flags  []uint8
	}{
		{[]int64{1}, nil},
		{[]int64{-1}, []uint8{0}},
		{[]int64{maxMcCullochPittsCount + 1}, []uint8{0}},
		{[]int64{1}, []uint8{2}},
	}
	for _, testCase := range cases {
		if _, _, err := EvaluateMcCullochPittsBatch(1, testCase.counts, testCase.flags); err == nil {
			t.Fatalf("malformed batch did not fail: %+v", testCase)
		}
	}
}
