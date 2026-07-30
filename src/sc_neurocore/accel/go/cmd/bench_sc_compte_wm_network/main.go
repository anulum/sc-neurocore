// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.

// Command bench_sc_compte_wm_network measures the complete Go SC network
// transition and writes a source-bound local-regression receipt.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"

	network "github.com/anulum/sc-neurocore/accel/sc_compte_wm_network"
)

const (
	steps   = 1000
	repeats = 3
	output  = "../../../../benchmarks/results/bench_sc_compte_wm_network_go.json"
)

var sourcePaths = []string{
	"src/sc_neurocore/accel/go/go.mod",
	"src/sc_neurocore/accel/go/sc_compte_wm_network/network.go",
	"src/sc_neurocore/accel/go/cmd/bench_sc_compte_wm_network/main.go",
}

func sourceSHA256(repository, path string) (string, error) {
	data, err := os.ReadFile(repository + "/" + path)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func median(values []int64) int64 {
	ordered := append([]int64(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	return ordered[len(ordered)/2]
}

func main() {
	// The command runs from src/sc_neurocore/accel/go.
	repository := "../../../.."
	warmup, err := network.NewNetwork(network.DefaultSpec(), nil)
	if err != nil {
		panic(err)
	}
	if _, err = warmup.Run(16*network.DtMS, nil, 16*network.DtMS); err != nil {
		panic(err)
	}
	samples := make([]int64, 0, repeats)
	inputDigests, spikeDigests, stateDigests := make([]string, 0, repeats), make([]string, 0, repeats), make([]string, 0, repeats)
	spikeCounts := make([][2]int, 0, repeats)
	for range repeats {
		executor, createErr := network.NewNetwork(network.DefaultSpec(), nil)
		if createErr != nil {
			panic(createErr)
		}
		started := time.Now()
		receipt, runErr := executor.Run(steps*network.DtMS, nil, 500)
		if runErr != nil {
			panic(runErr)
		}
		samples = append(samples, time.Since(started).Nanoseconds())
		inputDigests = append(inputDigests, receipt.InputSHA256)
		spikeDigests = append(spikeDigests, receipt.SpikeSHA256)
		stateDigests = append(stateDigests, receipt.FinalStateSHA256)
		spikeCounts = append(spikeCounts, [2]int{receipt.ExcitatorySpikes, receipt.InhibitorySpikes})
	}
	deterministic := true
	for index := 1; index < repeats; index++ {
		deterministic = deterministic && inputDigests[index] == inputDigests[0] &&
			spikeDigests[index] == spikeDigests[0] && stateDigests[index] == stateDigests[0] &&
			spikeCounts[index] == spikeCounts[0]
	}
	medianNS := median(samples)
	sourceHashes := make(map[string]string, len(sourcePaths))
	for _, path := range sourcePaths {
		digest, hashErr := sourceSHA256(repository, path)
		if hashErr != nil {
			panic(hashErr)
		}
		sourceHashes[path] = digest
	}
	payload := map[string]any{
		"schema_version":                "sc-neurocore.sc-compte-wm-network-benchmark.v1",
		"generated_at":                  time.Now().UTC().Format(time.RFC3339Nano),
		"model":                         "SC-COMPTE-WM-NETWORK",
		"execution_path":                "go-midpoint-rk2-radix2-fft",
		"evidence_class":                "local_regression",
		"production_speed_claimed":      false,
		"hardware_measurement_claimed":  false,
		"persistent_bump_claimed":       false,
		"distractor_resistance_claimed": false,
		"configuration": map[string]any{
			"cells": 2560, "excitatory_cells": network.NExcitatory,
			"inhibitory_cells": network.NInhibitory, "dt_ms": network.DtMS,
			"steps": steps, "duration_ms": steps * network.DtMS,
			"repeats": repeats, "seed": 42,
		},
		"environment": map[string]any{
			"go": runtime.Version(), "os": runtime.GOOS, "architecture": runtime.GOARCH,
			"cpu_threads": runtime.NumCPU(), "gomaxprocs": runtime.GOMAXPROCS(0),
		},
		"source_sha256":                  sourceHashes,
		"samples_ns":                     samples,
		"median_ns":                      medianNS,
		"median_ns_per_network_step":     float64(medianNS) / steps,
		"median_cell_updates_per_second": 2560 * steps / (float64(medianNS) / 1e9),
		"input_sha256":                   inputDigests[0], "spike_sha256": spikeDigests[0],
		"final_state_sha256":    stateDigests[0],
		"spike_counts":          map[string]int{"excitatory": spikeCounts[0][0], "inhibitory": spikeCounts[0][1]},
		"repeat_receipts_exact": deterministic, "passed": deterministic,
	}
	encoded, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		panic(err)
	}
	encoded = append(encoded, '\n')
	if err = os.WriteFile(output, encoded, 0o644); err != nil {
		panic(err)
	}
	fmt.Print(string(encoded))
	if !deterministic {
		os.Exit(1)
	}
}
