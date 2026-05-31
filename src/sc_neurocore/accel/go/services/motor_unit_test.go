// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
// (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore motor unit service tests

package services

import (
	"math"
	"testing"
)

func motorUnitRelax(previous, steady, tau, dt float64) float64 {
	return steady + (previous-steady)*math.Exp(-dt/tau)
}

func motorUnitReferenceStep(s MotorUnitState, drive float64) MotorUnitState {
	force := s.Force * math.Exp(-s.Dt/s.TauTwitch)
	inputDrive := s.Gain*math.Max(0.0, drive) - s.Adapt
	vTarget := s.VRest + inputDrive
	vCandidate := motorUnitRelax(s.V, vTarget, s.TauM, s.Dt)
	adaptTarget := s.AAdapt * (vCandidate - s.VRest)
	adapt := motorUnitRelax(s.Adapt, adaptTarget, s.TauAdapt, s.Dt)
	if vCandidate >= s.VThreshold {
		vCandidate = s.VReset
		force = math.Min(1.0, force+s.TwitchAmp)
	}
	s.V = vCandidate
	s.Adapt = adapt
	s.Force = force
	return s
}

func requireMotorUnitClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s mismatch: got %.17g want %.17g", name, got, want)
	}
}

func TestMotorUnitExactLIFAdaptationAndForceDecayStep(t *testing.T) {
	unit := NewMotorUnit()
	expected := motorUnitReferenceStep(*NewMotorUnit(), 20.0)

	if spike := unit.Step(20.0); spike != 0 {
		t.Fatalf("first exact motor-unit step should not spike, got %d", spike)
	}

	requireMotorUnitClose(t, "V", unit.V, expected.V)
	requireMotorUnitClose(t, "Adapt", unit.Adapt, expected.Adapt)
	requireMotorUnitClose(t, "Force", unit.Force, expected.Force)
}

func TestMotorUnitInvalidDrivePreservesState(t *testing.T) {
	unit := NewMotorUnit()
	for i := 0; i < 20; i++ {
		unit.Step(20.0)
	}
	before := *unit

	if spike := unit.Step(math.NaN()); spike != 0 {
		t.Fatalf("invalid drive must not spike, got %d", spike)
	}
	if *unit != before {
		t.Fatalf("invalid drive mutated state: got %#v want %#v", *unit, before)
	}
	if spike := unit.Step(math.Inf(1)); spike != 0 {
		t.Fatalf("infinite drive must not spike, got %d", spike)
	}
	if *unit != before {
		t.Fatalf("infinite drive mutated state: got %#v want %#v", *unit, before)
	}
}

func TestMotorUnitExcessDrivePreservesState(t *testing.T) {
	unit := NewMotorUnit()
	before := *unit

	if spike := unit.Step(1.0e8); spike != 0 {
		t.Fatalf("excess drive must not spike, got %d", spike)
	}
	if *unit != before {
		t.Fatalf("excess drive mutated state: got %#v want %#v", *unit, before)
	}
}

func TestMotorUnitSpikeAddsTwitchAndForceStaysBounded(t *testing.T) {
	unit := NewFastMotorUnit()
	spikes := 0
	for i := 0; i < 1000; i++ {
		spikes += unit.Step(50.0)
	}

	if spikes <= 0 {
		t.Fatalf("sustained drive should elicit spikes")
	}
	if unit.Force < 0.0 || unit.Force > 1.0 {
		t.Fatalf("force must remain bounded: %.17g", unit.Force)
	}
	forceAfterDrive := unit.Force
	for i := 0; i < 200; i++ {
		unit.Step(0.0)
	}
	if unit.Force < 0.0 || unit.Force > forceAfterDrive {
		t.Fatalf("force should decay without drive: before %.17g after %.17g", forceAfterDrive, unit.Force)
	}
}
