// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go public dense IF ownership and admission

package conversion_test

import (
	c "github.com/anulum/sc-neurocore/accel/conversion"
	"math"
	"reflect"
	"testing"
)

// TestPublicReplayOwnershipAndContinuation checks that a model owns copies of its
// coefficients and of the initial state, that the trace does not alias the final state,
// and that a replay continues from a returned final state.
func TestPublicReplayOwnershipAndContinuation(t *testing.T) {
	w, b := []float64{1}, []float64{0.25}
	layer, err := c.NewDenseLayer(1, 1, w, b, 1, 0.5, 1024)
	if err != nil {
		t.Fatal(err)
	}
	model, err := c.NewConvertedSNN([]c.DenseLayer{layer}, c.Spikes, 1024)
	if err != nil {
		t.Fatal(err)
	}
	w[0] = 100
	b[0] = 100
	state := [][]float64{{0.5}}
	options := c.ReplayOptions{InitialState: state, Trace: true, MaxWorkingBytes: 1024}
	result, err := model.Replay([]float64{0.5}, 1, 1, options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Output, []float64{1}) || result.FinalState[0][0] != 0.25 || state[0][0] != 0.5 {
		t.Fatal(result, state)
	}
	result.FinalState[0][0] = 0.5
	if result.StateTrace[0][0] != 0.25 {
		t.Fatal("trace aliases final state")
	}
	next, err := model.Replay([]float64{0.5}, 1, 1, c.ReplayOptions{InitialState: result.FinalState, MaxWorkingBytes: 1024})
	if err != nil {
		t.Fatal(err)
	}
	if next.Output[0] != 1 || result.FinalState[0][0] != 0.5 {
		t.Fatal(next, result)
	}
}

// TestPublicReplayOverflowAtomicityAndSignedClassification checks that an overflowing
// membrane update is refused with ErrOverflow and no partial result.
func TestPublicReplayOverflowAtomicityAndSignedClassification(t *testing.T) {
	maximum := math.MaxFloat64
	for _, pair := range []struct{ weight, bias, theta, fraction float64 }{{maximum, maximum, 1, 0}, {1, 0, maximum, maximum}} {
		layer, _ := c.NewDenseLayer(1, 1, []float64{pair.weight}, []float64{pair.bias}, pair.theta, pair.fraction, 1024)
		model, _ := c.NewConvertedSNN([]c.DenseLayer{layer}, c.Spikes, 1024)
		if r, err := model.Replay([]float64{1}, 1, 1, c.ReplayOptions{MaxWorkingBytes: 1024}); err != c.ErrOverflow || r != nil {
			t.Fatal(r, err)
		}
	}
	layer, _ := c.NewDenseLayer(1, 2, []float64{-1, -1}, nil, 1, 0, 1024)
	model, _ := c.NewConvertedSNN([]c.DenseLayer{layer}, c.Linear, 1024)
	result, err := model.Replay([]float64{1}, 1, 1, c.ReplayOptions{MaxWorkingBytes: 1024})
	if err != nil {
		t.Fatal(err)
	}
	labels, err := model.Classify(result, 1)
	if err != nil || !reflect.DeepEqual(labels, []int{0}) {
		t.Fatal(labels, err)
	}
	for _, entry := range []struct {
		result *c.ReplayResult
		batch  int
	}{{nil, 1}, {result, -1}, {result, 2}, {&c.ReplayResult{Output: []float64{math.NaN(), 0}}, 1}} {
		if _, err := model.Classify(entry.result, entry.batch); err != c.ErrInvalidInput {
			t.Fatal(err)
		}
	}
}

// TestPublicSignedTraceAndNonFirstClassification checks a linear readout's signed state
// trace and that classification picks the largest output even when it is not the first.
func TestPublicSignedTraceAndNonFirstClassification(t *testing.T) {
	layer, err := c.NewDenseLayer(1, 2, []float64{-2, -1}, nil, 1, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	model, err := c.NewConvertedSNN([]c.DenseLayer{layer}, c.Linear, 1024)
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.Replay([]float64{1}, 1, 1, c.ReplayOptions{Trace: true, MaxWorkingBytes: 216})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.StateTrace, [][]float64{{-2, -1}}) || len(result.SpikeTrace) != 0 {
		t.Fatal(result)
	}
	labels, err := model.Classify(result, 1)
	if err != nil || !reflect.DeepEqual(labels, []int{1}) {
		t.Fatal(labels, err)
	}
}

// TestPublicConcurrentReplayIsolatesOwnedStates checks that concurrent replays of one
// model return identical, independently owned results.
func TestPublicConcurrentReplayIsolatesOwnedStates(t *testing.T) {
	layer, _ := c.NewDenseLayer(1, 1, []float64{1}, nil, 1, 0, 1024)
	model, _ := c.NewConvertedSNN([]c.DenseLayer{layer}, c.Spikes, 1024)
	results := make(chan *c.ReplayResult, 4)
	errs := make(chan error, 4)
	for i := 0; i < 4; i++ {
		go func() {
			r, err := model.Replay([]float64{1, 1}, 2, 1, c.ReplayOptions{Trace: true, BinaryInputs: true, MaxWorkingBytes: 1024})
			results <- r
			errs <- err
		}()
	}
	for i := 0; i < 4; i++ {
		r := <-results
		err := <-errs
		if err != nil || r.Output[0] != 2 {
			t.Fatal(r, err)
		}
		r.FinalState[0][0] = 100
	}
	r, err := model.Replay([]float64{1}, 1, 1, c.ReplayOptions{MaxWorkingBytes: 1024})
	if err != nil || r.FinalState[0][0] != 0 {
		t.Fatal(r, err)
	}
}
