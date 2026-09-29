// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — N-MNIST binary recordings

package loaders

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ErrIncompleteNMNIST indicates a truncated published 40-bit event record.
var ErrIncompleteNMNIST = errors.New("N-MNIST file contains an incomplete 40-bit event")

// DecodeNMNISTInto writes row-major x, y, polarity and float64 millisecond times.
// Orchard et al. (2015) encode whole-byte addresses and a 23-bit microsecond time.
// Both input length and destination size are checked before any output mutation.
func DecodeNMNISTInto(raw []byte, output []float64) error {
	if len(raw)%5 != 0 {
		return ErrIncompleteNMNIST
	}
	if len(output)%4 != 0 || len(output)/4 != len(raw)/5 {
		return errors.New("N-MNIST output must have four columns per event")
	}
	for index := 0; index < len(raw)/5; index++ {
		r := raw[index*5 : index*5+5]
		row := output[index*4 : index*4+4]
		row[0] = float64(r[0])
		row[1] = float64(r[1])
		row[2] = float64(r[2] >> 7)
		timeUS := uint32(r[2]&0x7f)<<16 | uint32(r[3])<<8 | uint32(r[4])
		row[3] = float64(timeUS) / 1000.0
	}
	return nil
}

// DecodeNMNIST returns row-major float64 events without rescaling by an encoder dt.
func DecodeNMNIST(raw []byte) ([]float64, error) {
	if len(raw)%5 != 0 {
		return nil, ErrIncompleteNMNIST
	}
	output := make([]float64, len(raw)/5*4)
	if err := DecodeNMNISTInto(raw, output); err != nil {
		return nil, err
	}
	return output, nil
}

// ReadNMNIST reads one recording and refuses incomplete final events.
func ReadNMNIST(path string) ([]float64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return DecodeNMNIST(raw)
}

// LoadNmnist reads the Train or Test split, sorted by class folder and file name.
// Each sample is a row-major four-column array; labels follow that same ordering.
// No files are downloaded and missing data never triggers synthetic substitution.
func LoadNmnist(root string, train bool) ([][]float64, []int64, error) {
	split := "Test"
	if train {
		split = "Train"
	}
	entries, err := os.ReadDir(filepath.Join(root, split))
	if err != nil {
		return nil, nil, err
	}
	samples := make([][]float64, 0)
	labels := make([]int64, 0)
	for _, entry := range entries {
		directory := filepath.Join(root, split, entry.Name())
		info, err := os.Stat(directory)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, nil, err
		}
		if !info.IsDir() {
			continue
		}
		label, err := strconv.ParseInt(entry.Name(), 10, 64)
		if err != nil {
			return nil, nil, fmt.Errorf("N-MNIST class label: %w", err)
		}
		files, err := os.ReadDir(directory)
		if err != nil {
			return nil, nil, err
		}
		for _, file := range files {
			if !strings.HasSuffix(file.Name(), ".bin") {
				continue
			}
			events, err := ReadNMNIST(filepath.Join(directory, file.Name()))
			if err != nil {
				return nil, nil, err
			}
			samples = append(samples, events)
			labels = append(labels, label)
		}
	}
	return samples, labels, nil
}
