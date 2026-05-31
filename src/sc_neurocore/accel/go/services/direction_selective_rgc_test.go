package services

import (
	"math"
	"testing"
)

func exactRGCVoltage(v, drive, tau, dt float64) float64 {
	return drive + (v-drive)*math.Exp(-dt/tau)
}

func TestDirectionSelectiveRGCExactMembraneRelaxation(t *testing.T) {
	cell := NewDirectionSelectiveRGC()
	cell.Tau = 7.0
	cell.Theta = 100.0
	cell.Dt = 1.25
	cell.WCentre = 1.4
	cell.WSurround = 0.2
	cell.V = 0.35
	expectedSurround := 0.9*cell.Surround + 0.1*0.5
	expectedDrive := cell.WCentre*(2.0-cell.PrevIntensity) - cell.WSurround*expectedSurround
	expectedV := exactRGCVoltage(cell.V, expectedDrive, cell.Tau, cell.Dt)

	spike, err := cell.StepRF(2.0, 0.5)
	if err != nil {
		t.Fatalf("StepRF returned error: %v", err)
	}
	if spike != 0 {
		t.Fatalf("unexpected spike %d", spike)
	}
	if math.Abs(cell.V-expectedV) > 1e-12 || math.Abs(cell.Surround-expectedSurround) > 1e-12 || cell.PrevIntensity != 2.0 {
		t.Fatalf("unexpected state: %+v expected v=%g surround=%g", cell, expectedV, expectedSurround)
	}
}

func TestDirectionSelectiveRGCInvalidDrivePreservesState(t *testing.T) {
	cell := NewDirectionSelectiveRGC()
	before := *cell
	if _, err := cell.StepRF(math.NaN(), 0.0); err == nil {
		t.Fatal("expected invalid intensity error")
	}
	if *cell != before {
		t.Fatalf("state mutated after invalid drive")
	}
}

func TestDirectionSelectiveRGCCorruptStatePreservesState(t *testing.T) {
	cell := NewDirectionSelectiveRGC()
	cell.Surround = math.Inf(1)
	before := *cell
	if _, err := cell.StepRF(1.0, 0.0); err == nil {
		t.Fatal("expected corrupt state error")
	}
	if *cell != before {
		t.Fatalf("state mutated after corrupt state")
	}
}
