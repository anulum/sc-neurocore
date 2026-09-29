// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go complete deterministic dense IF replay

package conversion

import "math/big"

// ReplayOptions declares owned-state continuation, complete traces and limits.
type ReplayOptions struct {
	// InitialState contains batch/output row-major state per layer; nil selects preloads.
	InitialState [][]float64
	// Trace retains every post-reset state and every IF spike event.
	Trace bool
	// BinaryInputs requires exact zero/one events; false admits finite currents in [0,1].
	BinaryInputs bool
	// MaxWorkingBytes is a positive numeric reservation limit, excluding caller/runtime storage.
	MaxWorkingBytes int
}

// ReplayResult owns responses, final states and optional layer-major full traces.
type ReplayResult struct {
	// Output is batch-major IF event counts or cumulative signed linear integration.
	Output []float64
	// FinalState contains batch/output row-major state per weighted layer.
	FinalState [][]float64
	// StateTrace contains time/batch/output row-major post-step state per layer.
	StateTrace [][]float64
	// SpikeTrace contains time/batch/output row-major events per IF layer.
	SpikeTrace [][]float64
}

// Replay consumes time/batch/input row-major frames. Multiply and add are rounded
// separately to float64 in ascending input-column order; bias applies each step.
// Layers consume same-step preceding events. IF thresholds are inclusive, reset
// subtractive, and each neuron emits at most one event per step. Empty time/batch
// preserves initial state. Refusal returns no partial output and never edits caller
// states. Buffers are admitted before copying frames, states or allocating traces.
func (s *ConvertedSNN) Replay(frames []float64, steps, batch int, options ReplayOptions) (*ReplayResult, error) {
	if s == nil || len(s.layers) == 0 || steps < 0 || batch < 0 {
		return nil, ErrInvalidInput
	}
	expected := new(big.Int).Mul(big.NewInt(int64(steps)), big.NewInt(int64(batch)))
	expected.Mul(expected, big.NewInt(int64(s.layers[0].inputs)))
	if expected.Cmp(big.NewInt(int64(len(frames)))) != 0 ||
		(options.InitialState != nil && len(options.InitialState) != len(s.layers)) {
		return nil, ErrInvalidInput
	}
	if err := admitReplay(s.layers, steps, batch, options.Trace, s.mode == Linear, options.MaxWorkingBytes); err != nil {
		return nil, err
	}
	for _, v := range frames {
		if !finite(v) || v < 0 || v > 1 || (options.BinaryInputs && v != 0 && v != 1) {
			return nil, ErrInvalidInput
		}
	}
	ownedFrames := append([]float64(nil), frames...)
	spiking := len(s.layers)
	if s.mode == Linear {
		spiking--
	}
	result := &ReplayResult{Output: make([]float64, batch*s.layers[len(s.layers)-1].outputs)}
	for index, layer := range s.layers {
		width := batch * layer.outputs
		state := make([]float64, width)
		if options.InitialState == nil {
			shift := 0.0
			if index < spiking {
				shift = float64(layer.fraction * layer.threshold)
			}
			if !finite(shift) {
				return nil, ErrOverflow
			}
			for i := range state {
				state[i] = shift
			}
		} else {
			if len(options.InitialState[index]) != width {
				return nil, ErrInvalidInput
			}
			for _, v := range options.InitialState[index] {
				if !finite(v) {
					return nil, ErrInvalidInput
				}
			}
			copy(state, options.InitialState[index])
		}
		result.FinalState = append(result.FinalState, state)
		if options.Trace {
			result.StateTrace = append(result.StateTrace, make([]float64, steps*width))
			if index < spiking {
				result.SpikeTrace = append(result.SpikeTrace, make([]float64, steps*width))
			}
		}
	}
	if batch != 0 {
		for step := 0; step < steps; step++ {
			width := batch * s.layers[0].inputs
			drive := ownedFrames[step*width : (step+1)*width]
			for index, layer := range s.layers {
				events := make([]float64, batch*layer.outputs)
				for row := 0; row < batch; row++ {
					for node := 0; node < layer.outputs; node++ {
						current := 0.0
						for column := 0; column < layer.inputs; column++ {
							product := float64(drive[row*layer.inputs+column] * layer.weights[node*layer.inputs+column])
							current = float64(current + product)
						}
						if layer.bias != nil {
							current = float64(current + layer.bias[node])
						}
						slot := row*layer.outputs + node
						state := float64(result.FinalState[index][slot] + current)
						if !finite(current) || !finite(state) {
							return nil, ErrOverflow
						}
						if index < spiking {
							event := 0.0
							if state >= layer.threshold {
								event = 1.0
							}
							reset := float64(event * layer.threshold)
							state = float64(state - reset)
							events[slot] = event
							if options.Trace {
								result.SpikeTrace[index][step*batch*layer.outputs+slot] = event
							}
							if index == len(s.layers)-1 {
								result.Output[slot] = float64(result.Output[slot] + event)
							}
						}
						result.FinalState[index][slot] = state
						if options.Trace {
							result.StateTrace[index][step*batch*layer.outputs+slot] = state
						}
					}
				}
				drive = events
			}
		}
	}
	if s.mode == Linear {
		result.Output = append([]float64(nil), result.FinalState[len(s.layers)-1]...)
	}
	return result, nil
}

// Classify selects the first maximal finite response in each row, using zero-based labels.
func (s *ConvertedSNN) Classify(result *ReplayResult, batch int) ([]int, error) {
	if s == nil || len(s.layers) == 0 || result == nil || batch < 0 {
		return nil, ErrInvalidInput
	}
	width := s.layers[len(s.layers)-1].outputs
	if new(big.Int).Mul(big.NewInt(int64(batch)), big.NewInt(int64(width))).Cmp(big.NewInt(int64(len(result.Output)))) != 0 {
		return nil, ErrInvalidInput
	}
	for _, v := range result.Output {
		if !finite(v) {
			return nil, ErrInvalidInput
		}
	}
	labels := make([]int, batch)
	for row := range labels {
		best := 0
		for node := 1; node < width; node++ {
			if result.Output[row*width+node] > result.Output[row*width+best] {
				best = node
			}
		}
		labels[row] = best
	}
	return labels, nil
}
