// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go KL refine for HierarchicalPartitioner (parity with Rust)
//
// Build:
//   PATH=/usr/local/go/bin:$PATH GOTOOLCHAIN=local \
//     go build -buildmode=c-shared -o libpartition.so partition.go
//
// Bit-exact parity contract with the Python _refine and Rust
// kl_refine: same iteration order
// (`for i in 0..P: for v in snapshot(parts[i])`), same gain
// comparison, same move semantics. Returns total moves performed.

package main

/*
#include <stdint.h>
*/
import "C"

import "unsafe"

func wrapI64(p *C.int64_t, n int) []int64 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int64)(unsafe.Pointer(p)), n)
}

func wrapI32(p *C.int32_t, n int) []int32 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*int32)(unsafe.Pointer(p)), n)
}

func wrapF64(p *C.double, n int) []float64 {
	if n == 0 {
		return nil
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(p)), n)
}

func klRefine(
	adjOffsets []int64, adjNeighbours []int32, adjScc []float64,
	vertexWeights []float64, partMap []int32,
	nParts int32, klIter int32, corrPenalty float64,
) uint64 {
	nP := int(nParts)
	V := len(vertexWeights)

	// Build per-partition vertex lists.
	parts := make([][]int32, nP)
	for v := 0; v < V; v++ {
		p := int(partMap[v])
		if p >= 0 && p < nP {
			parts[p] = append(parts[p], int32(v))
		}
	}

	weightTo := make([]float64, nP)
	var totalMoves uint64

	for iter := int32(0); iter < klIter; iter++ {
		improved := false
		for i := 0; i < nP; i++ {
			// Snapshot at entry — matches Python `for v in list(part)`.
			snapshot := make([]int32, len(parts[i]))
			copy(snapshot, parts[i])
			for _, v32 := range snapshot {
				v := int(v32)
				if len(parts[i]) <= 1 {
					continue
				}
				if int(partMap[v]) != i {
					continue
				}
				vw := vertexWeights[v]

				for w := range weightTo {
					weightTo[w] = 0.0
				}
				totalWeight := 0.0
				lo := int(adjOffsets[v])
				hi := int(adjOffsets[v+1])
				for k := lo; k < hi; k++ {
					n := int(adjNeighbours[k])
					contrib := vw * (1.0 + adjScc[k]*corrPenalty)
					totalWeight += contrib
					tgt := int(partMap[n])
					if tgt >= 0 && tgt < nP {
						weightTo[tgt] += contrib
					}
				}

				currentCost := totalWeight - weightTo[i]
				bestTarget := int32(i)
				bestGain := 0.0
				for j := 0; j < nP; j++ {
					if j == i {
						continue
					}
					gain := currentCost - (totalWeight - weightTo[j])
					if gain > bestGain {
						bestGain = gain
						bestTarget = int32(j)
					}
				}

				if bestTarget != int32(i) && bestGain > 0.0 {
					// Remove v from parts[i] (linear scan, mirrors Python).
					for idx, x := range parts[i] {
						if x == v32 {
							parts[i] = append(parts[i][:idx], parts[i][idx+1:]...)
							break
						}
					}
					parts[bestTarget] = append(parts[bestTarget], v32)
					partMap[v] = bestTarget
					totalMoves++
					improved = true
				}
			}
		}
		if !improved {
			break
		}
	}

	return totalMoves
}

//export kl_refine_c
func kl_refine_c(
	adjOffsetsPtr *C.int64_t,
	adjNeighboursPtr *C.int32_t,
	adjSccPtr *C.double,
	vertexWeightsPtr *C.double,
	partMapPtr *C.int32_t,
	vTotal C.int64_t,
	eTotal C.int64_t,
	nParts C.int32_t,
	klIterations C.int32_t,
	correlationPenalty C.double,
) C.uint64_t {
	V := int(vTotal)
	E := int(eTotal)
	moves := klRefine(
		wrapI64(adjOffsetsPtr, V+1),
		wrapI32(adjNeighboursPtr, E),
		wrapF64(adjSccPtr, E),
		wrapF64(vertexWeightsPtr, V),
		wrapI32(partMapPtr, V),
		int32(nParts), int32(klIterations), float64(correlationPenalty),
	)
	return C.uint64_t(moves)
}

func main() {} // required for c-shared
