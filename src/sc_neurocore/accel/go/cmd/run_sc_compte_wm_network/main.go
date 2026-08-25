// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

// Command run_sc_compte_wm_network is the JSON adapter for public Go dispatch.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	network "github.com/anulum/sc-neurocore/accel/sc_compte_wm_network"
)

type stimulusFlags []network.Stimulus

func (values *stimulusFlags) String() string { return fmt.Sprintf("%v", *values) }

func (values *stimulusFlags) Set(encoded string) error {
	fields := strings.Split(encoded, ",")
	if len(fields) != 5 {
		return errors.New("stimulus must have start,duration,current,kind,center")
	}
	numbers := make([]float64, 3)
	for index := range numbers {
		value, err := strconv.ParseFloat(fields[index], 64)
		if err != nil {
			return err
		}
		numbers[index] = value
	}
	var center *float64
	if fields[4] != "none" {
		value, err := strconv.ParseFloat(fields[4], 64)
		if err != nil {
			return err
		}
		center = &value
	}
	*values = append(*values, network.Stimulus{StartMS: numbers[0], DurationMS: numbers[1],
		CurrentPA: numbers[2], Kind: fields[3], CenterDeg: center})
	return nil
}

type statisticsOutput struct {
	ExcitatoryRateHz float64  `json:"excitatory_rate_hz"`
	InhibitoryRateHz float64  `json:"inhibitory_rate_hz"`
	BumpAngleDeg     float64  `json:"bump_angle_deg"`
	ResultantLength  float64  `json:"resultant_length"`
	CircularWidthDeg *float64 `json:"circular_width_deg"`
}

type windowOutput struct {
	StartMS          float64           `json:"start_ms"`
	EndMS            float64           `json:"end_ms"`
	ExcitatorySpikes int               `json:"excitatory_spikes"`
	InhibitorySpikes int               `json:"inhibitory_spikes"`
	Statistics       *statisticsOutput `json:"statistics"`
}

type runOutput struct {
	Runtime              string         `json:"runtime"`
	ExecutionNS          int64          `json:"execution_ns"`
	SpecificationVersion string         `json:"specification_version"`
	Seed                 uint64         `json:"seed"`
	DurationMS           float64        `json:"duration_ms"`
	Steps                int            `json:"steps"`
	ExcitatorySpikes     int            `json:"excitatory_spikes"`
	InhibitorySpikes     int            `json:"inhibitory_spikes"`
	Windows              []windowOutput `json:"windows"`
	InputSHA256          string         `json:"input_sha256"`
	SpikeSHA256          string         `json:"spike_sha256"`
	FinalStateSHA256     string         `json:"final_state_sha256"`
}

func convert(receipt *network.RunReceipt, elapsed int64) runOutput {
	windows := make([]windowOutput, len(receipt.Windows))
	for index, window := range receipt.Windows {
		var statistics *statisticsOutput
		if window.Statistics != nil {
			value := window.Statistics
			statistics = &statisticsOutput{value.ExcitatoryRateHz, value.InhibitoryRateHz,
				value.BumpAngleDeg, value.ResultantLength, value.CircularWidthDeg}
		}
		windows[index] = windowOutput{window.StartMS, window.EndMS, window.ExcitatorySpikes,
			window.InhibitorySpikes, statistics}
	}
	return runOutput{"go", elapsed, receipt.SpecificationVersion, receipt.Seed,
		receipt.DurationMS, receipt.Steps, receipt.ExcitatorySpikes, receipt.InhibitorySpikes,
		windows, receipt.InputSHA256, receipt.SpikeSHA256, receipt.FinalStateSHA256}
}

func main() {
	duration := flag.Float64("duration-ms", 0, "run duration in milliseconds")
	window := flag.Float64("statistics-window-ms", 0, "statistics window in milliseconds")
	seed := flag.Uint64("seed", 42, "counter stream seed")
	structuredEI := flag.Bool("structured-ei", false, "enable tuned E-to-I coupling")
	modulated := flag.Bool("modulated", false, "select modulated conductances")
	autapses := flag.Bool("allow-recurrent-autapses", false, "retain recurrent autapses")
	var stimuli stimulusFlags
	flag.Var(&stimuli, "stimulus", "start,duration,current,kind,center (repeatable)")
	flag.Parse()
	spec := network.DefaultSpec()
	spec.Seed, spec.StructuredEI, spec.Modulated = *seed, *structuredEI, *modulated
	spec.AllowRecurrentAutapses = *autapses
	runtime, err := network.NewNetwork(spec, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	started := time.Now()
	receipt, err := runtime.Run(*duration, stimuli, *window)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if err := json.NewEncoder(os.Stdout).Encode(convert(receipt, time.Since(started).Nanoseconds())); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}
