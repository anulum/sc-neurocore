// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go dense IF numeric reservation

package conversion

import "math/big"

// admitReplay checks 8*(2P+2F+2S+2O+H+5M+I) without machine integer overflow.
func admitReplay(layers []DenseLayer, steps, batch int, trace, linear bool, maximum int) error {
	if maximum <= 0 {
		return ErrInvalidInput
	}
	coefficients, nodes := new(big.Int), new(big.Int)
	largest := 0
	for _, layer := range layers {
		coefficients.Add(coefficients, big.NewInt(int64(len(layer.weights))))
		coefficients.Add(coefficients, big.NewInt(int64(len(layer.bias))))
		nodes.Add(nodes, big.NewInt(int64(layer.outputs)))
		if layer.outputs > largest {
			largest = layer.outputs
		}
	}
	b, t := big.NewInt(int64(batch)), big.NewInt(int64(steps))
	last := big.NewInt(int64(layers[len(layers)-1].outputs))
	frame := new(big.Int).Mul(b, big.NewInt(int64(layers[0].inputs)))
	frames := new(big.Int).Mul(t, frame)
	states := new(big.Int).Mul(b, nodes)
	output := new(big.Int).Mul(b, last)
	maxLayer := new(big.Int).Mul(b, big.NewInt(int64(largest)))
	total := new(big.Int)
	for _, v := range []*big.Int{coefficients, frames, states, output} {
		total.Add(total, new(big.Int).Mul(big.NewInt(2), v))
	}
	if trace {
		traceNodes := new(big.Int).Mul(big.NewInt(2), nodes)
		if linear {
			traceNodes.Sub(traceNodes, last)
		}
		total.Add(total, new(big.Int).Mul(new(big.Int).Mul(t, b), traceNodes))
	}
	total.Add(total, new(big.Int).Mul(big.NewInt(5), maxLayer))
	total.Add(total, frame)
	total.Mul(total, big.NewInt(8))
	if total.Cmp(big.NewInt(int64(maximum))) > 0 {
		return ErrResourceLimit
	}
	return nil
}
