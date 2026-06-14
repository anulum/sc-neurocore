// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Ollivier-Ricci curvature (parity with math/topology.py)

// Package main exposes a C-ABI shared library
// (`go build -buildmode=c-shared -o libtopology.so topology.go`) that the
// Python dispatcher loads via ctypes.
//
// Parity contract: `ollivier_ricci_curvature_c` reproduces the value of
// `sc_neurocore.math.topology.ollivier_ricci_curvature` within float64
// round-off for the same coupling matrix and node pair. The min-cost-flow
// loop follows the same Bellman-Ford ascending-node iteration order as the
// Python and Rust references so the chosen augmenting paths — and therefore
// the floating-point accumulation of the transport cost — match.
//
// Reference: Ollivier (2009), J. Functional Analysis 256(3): 810-864.
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

const idleness = 0.5
const tolerance = 1e-12

// shortestPathDistances returns the unweighted (hop-count) all-pairs
// distances via BFS. Row-major n*n; unreachable pairs stay +Inf.
func shortestPathDistances(graph []float64, n int) []float64 {
	distances := make([]float64, n*n)
	for k := range distances {
		distances[k] = math.Inf(1)
	}
	queue := make([]int, 0, n)
	for source := 0; source < n; source++ {
		distances[source*n+source] = 0.0
		queue = queue[:0]
		queue = append(queue, source)
		head := 0
		for head < len(queue) {
			current := queue[head]
			head++
			nextDistance := distances[source*n+current] + 1.0
			for target := 0; target < n; target++ {
				if target == current || graph[current*n+target] <= 0.0 {
					continue
				}
				if nextDistance < distances[source*n+target] {
					distances[source*n+target] = nextDistance
					queue = append(queue, target)
				}
			}
		}
	}
	return distances
}

// lazyRandomWalk returns the lazy random-walk distribution from node.
func lazyRandomWalk(graph []float64, n, node int) []float64 {
	distribution := make([]float64, n)
	distribution[node] = idleness
	rowSum := 0.0
	for k := 0; k < n; k++ {
		if k != node {
			rowSum += graph[node*n+k]
		}
	}
	if rowSum == 0.0 {
		distribution[node] = 1.0
		return distribution
	}
	for k := 0; k < n; k++ {
		if k != node {
			distribution[k] += (1.0 - idleness) * graph[node*n+k] / rowSum
		}
	}
	return distribution
}

// minimumTransportCost is the exact Wasserstein-1 cost under the hop
// metric, via a successive-shortest-path min-cost flow. Returns +Inf
// when a required transport edge is unreachable. The second return is
// false when the transport sub-problem is infeasible.
func minimumTransportCost(source, target, distances []float64, n int) (float64, bool) {
	sourceNodes := make([]int, 0, n)
	targetNodes := make([]int, 0, n)
	for k := 0; k < n; k++ {
		if source[k] > 0.0 {
			sourceNodes = append(sourceNodes, k)
		}
		if target[k] > 0.0 {
			targetNodes = append(targetNodes, k)
		}
	}
	if len(sourceNodes) == 0 || len(targetNodes) == 0 {
		return 0.0, true
	}

	totalSupply := len(sourceNodes)
	totalDemand := len(targetNodes)
	costs := make([]float64, totalSupply*totalDemand)
	for sIdx, sNode := range sourceNodes {
		for dIdx, dNode := range targetNodes {
			cost := distances[sNode*n+dNode]
			if math.IsInf(cost, 1) {
				return math.Inf(1), true
			}
			costs[sIdx*totalDemand+dIdx] = cost
		}
	}

	sourceID := totalSupply + totalDemand
	sinkID := sourceID + 1
	nodeCount := sinkID + 1
	residual := make([]float64, nodeCount*nodeCount)
	edgeCost := make([]float64, nodeCount*nodeCount)

	for idx, sNode := range sourceNodes {
		residual[sourceID*nodeCount+idx] = source[sNode]
	}
	for idx, dNode := range targetNodes {
		residual[(totalSupply+idx)*nodeCount+sinkID] = target[dNode]
	}
	for sIdx := 0; sIdx < totalSupply; sIdx++ {
		for dIdx := 0; dIdx < totalDemand; dIdx++ {
			u := sIdx
			v := totalSupply + dIdx
			cost := costs[sIdx*totalDemand+dIdx]
			residual[u*nodeCount+v] = math.Inf(1)
			edgeCost[u*nodeCount+v] = cost
			edgeCost[v*nodeCount+u] = -cost
		}
	}

	required := 0.0
	for _, value := range source {
		required += value
	}
	transported := 0.0
	totalCost := 0.0

	for transported+tolerance < required {
		dist := make([]float64, nodeCount)
		parent := make([]int, nodeCount)
		for k := 0; k < nodeCount; k++ {
			dist[k] = math.Inf(1)
			parent[k] = -1
		}
		dist[sourceID] = 0.0
		for iter := 0; iter < nodeCount-1; iter++ {
			updated := false
			for u := 0; u < nodeCount; u++ {
				if math.IsInf(dist[u], 0) {
					continue
				}
				for v := 0; v < nodeCount; v++ {
					if residual[u*nodeCount+v] <= tolerance {
						continue
					}
					candidate := dist[u] + edgeCost[u*nodeCount+v]
					if candidate < dist[v]-tolerance {
						dist[v] = candidate
						parent[v] = u
						updated = true
					}
				}
			}
			if !updated {
				break
			}
		}
		if parent[sinkID] == -1 {
			return 0.0, false
		}

		increment := required - transported
		for v := sinkID; v != sourceID; {
			u := parent[v]
			if residual[u*nodeCount+v] < increment {
				increment = residual[u*nodeCount+v]
			}
			v = u
		}
		for v := sinkID; v != sourceID; {
			u := parent[v]
			residual[u*nodeCount+v] -= increment
			residual[v*nodeCount+u] += increment
			totalCost += increment * edgeCost[u*nodeCount+v]
			v = u
		}
		transported += increment
	}
	return totalCost, true
}

// ollivierRicciCurvature computes kappa(i, j) on a validated coupling
// graph. Returns NaN on an infeasible transport sub-problem (the Python
// path raises in that case; the caller treats NaN as an error signal).
func ollivierRicciCurvature(graph []float64, n, i, j int) float64 {
	if i == j {
		return 0.0
	}
	distances := shortestPathDistances(graph, n)
	graphDistance := distances[i*n+j]
	if math.IsInf(graphDistance, 1) || graphDistance <= 0.0 {
		return 0.0
	}
	muI := lazyRandomWalk(graph, n, i)
	muJ := lazyRandomWalk(graph, n, j)
	w1, ok := minimumTransportCost(muI, muJ, distances, n)
	if !ok {
		return math.NaN()
	}
	return 1.0 - w1/graphDistance
}

//export ollivier_ricci_curvature_c
func ollivier_ricci_curvature_c(knmPtr *C.double, n, i, j C.int) C.double {
	count := int(n) * int(n)
	graph := make([]float64, count)
	src := unsafe.Slice((*float64)(unsafe.Pointer(knmPtr)), count)
	copy(graph, src)
	return C.double(ollivierRicciCurvature(graph, int(n), int(i), int(j)))
}

func main() {} // required for c-shared
