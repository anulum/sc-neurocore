// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public MAT simulation helpers against independent Python traces

//go:build coverage

package services

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"
)

// simulationOracle holds one complete trace produced by the Python model.
type simulationOracle struct {
	Steps    int       `json:"steps"`
	Current  float64   `json:"current"`
	Voltages []float64 `json:"voltages"`
	Events   int       `json:"events"`
}

// checkMATSimulation compares every returned sample and the exact event count.
func checkMATSimulation(t *testing.T, name string, simulate func(int, float64) ([]float64, int)) {
	t.Helper()
	data, err := os.ReadFile(os.Getenv("NEUROCORE_MAT_ORACLE"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus map[string]simulationOracle
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&corpus); err != nil {
		t.Fatal(err)
	}
	want, present := corpus[name]
	if !present || want.Steps <= 0 || len(want.Voltages) != want.Steps || want.Events <= 0 {
		t.Fatal("Independent Python trace is incomplete or contains no event")
	}
	actual, events := simulate(want.Steps, want.Current)
	if len(actual) != want.Steps || events != want.Events {
		t.Fatalf("%s shape or event mismatch: %d samples, %d events", name, len(actual), events)
	}
	for index, value := range actual {
		if math.IsNaN(value) || math.IsInf(value, 0) || math.Abs(value-want.Voltages[index]) > 2e-12 {
			t.Fatalf("%s sample[%d] = %.17g, Python reference %.17g", name, index, value, want.Voltages[index])
		}
	}
}

// TestMATSimulationPythonTrace checks the full public source-MAT simulation.
func TestMATSimulationPythonTrace(t *testing.T) {
	checkMATSimulation(t, "mat", SimulateMATNeuron)
}

// TestSCResettingMATSimulationPythonTrace checks the full public SC simulation.
func TestSCResettingMATSimulationPythonTrace(t *testing.T) {
	checkMATSimulation(t, "sc_resetting_mat", SimulateSCResettingMATNeuron)
}

// TestMATSimulationEmpty retains the public helper's empty result.
func TestMATSimulationEmpty(t *testing.T) {
	trace, events := SimulateMATNeuron(0, 0.5)
	if len(trace) != 0 || events != 0 {
		t.Fatal("Empty source MAT simulation changed its result")
	}
}

// TestSCResettingMATSimulationEmpty retains the public helper's empty result.
func TestSCResettingMATSimulationEmpty(t *testing.T) {
	trace, events := SimulateSCResettingMATNeuron(0, 50.0)
	if len(trace) != 0 || events != 0 {
		t.Fatal("Empty SC MAT simulation changed its result")
	}
}
