// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go N-MNIST recording tests

package loaders

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestDecodeNMNISTPreservesPublishedFields(t *testing.T) {
	raw := []byte{33, 32, 0x80, 3, 234, 1, 0, 0, 7, 212, 0, 33, 0x7f, 0xff, 0xff}
	got, err := DecodeNMNIST(raw)
	want := []float64{33, 32, 1, 1.002, 1, 0, 0, 2.004, 0, 33, 0, 8388.607}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("decoded %v, %v; want %v", got, err, want)
	}
}

func TestDecodeNMNISTRefusalPreservesDestination(t *testing.T) {
	for _, raw := range [][]byte{{1}, {1, 2, 3, 4}, {0, 0, 0, 0, 0, 1}} {
		output := []float64{9, 9, 9, 9}
		if !errors.Is(DecodeNMNISTInto(raw, output), ErrIncompleteNMNIST) {
			t.Fatal("truncated record was accepted")
		}
		if !reflect.DeepEqual(output, []float64{9, 9, 9, 9}) {
			t.Fatal("refusal changed destination")
		}
	}
	if err := DecodeNMNISTInto(make([]byte, 5), []float64{9}); err == nil {
		t.Fatal("insufficient destination was accepted")
	}
	got, err := DecodeNMNIST(nil)
	if err != nil || len(got) != 0 {
		t.Fatalf("empty recording: %v, %v", got, err)
	}
}

func TestLoadNmnistReadsSortedRecordedFiles(t *testing.T) {
	root := t.TempDir()
	for _, label := range []string{"2", "0"} {
		directory := filepath.Join(root, "Train", label)
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"b.bin", "a.bin"} {
			raw := []byte{33, 32, 0x80, 3, 234}
			if name == "b.bin" {
				raw[3], raw[4] = 7, 212
			}
			if err := os.WriteFile(filepath.Join(directory, name), raw, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(directory, "README.txt"), []byte("not an event"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	samples, labels, err := LoadNmnist(root, true)
	if err != nil || len(samples) != 4 || !reflect.DeepEqual(labels, []int64{0, 0, 2, 2}) {
		t.Fatalf("loaded %v, %v, %v", samples, labels, err)
	}
	for index, sample := range samples {
		timeMS := 1.002
		if index%2 == 1 {
			timeMS = 2.004
		}
		if !reflect.DeepEqual(sample, []float64{33, 32, 1, timeMS}) {
			t.Fatalf("changed recording: %v", sample)
		}
	}
	if _, _, err := LoadNmnist(root, false); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing Test split: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "Train", "0", "a.bin"), []byte{1}, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadNmnist(root, true); !errors.Is(err, ErrIncompleteNMNIST) {
		t.Fatalf("damaged recording: %v", err)
	}
}
