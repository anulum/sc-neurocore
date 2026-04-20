// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go Extended Services (bioware, datasets, model zoo)

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"time"
)

// ============================================================
// §11  Bioware Real-Time Bridge
// ============================================================

type BioSensorReading struct {
	Timestamp uint64
	ChannelID uint16
	Value     float32
	Quality   uint8
}

type BiowareStream struct {
	mu       sync.RWMutex
	buffer   []BioSensorReading
	maxSize  int
	dropped  uint64
	received uint64
}

func NewBiowareStream(bufferSize int) *BiowareStream {
	return &BiowareStream{
		buffer:  make([]BioSensorReading, 0, bufferSize),
		maxSize: bufferSize,
	}
}

func (bs *BiowareStream) Ingest(reading BioSensorReading) bool {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bs.received++
	if len(bs.buffer) >= bs.maxSize {
		bs.dropped++
		bs.buffer = bs.buffer[1:]
	}
	bs.buffer = append(bs.buffer, reading)
	return true
}

func (bs *BiowareStream) LFPPower(channelID uint16, windowSize int) float64 {
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	power := 0.0
	count := 0
	for i := len(bs.buffer) - 1; i >= 0 && count < windowSize; i-- {
		if bs.buffer[i].ChannelID == channelID {
			power += float64(bs.buffer[i].Value) * float64(bs.buffer[i].Value)
			count++
		}
	}
	if count > 0 {
		power /= float64(count)
	}
	return math.Sqrt(power)
}

func (bs *BiowareStream) BurstDetect(channelID uint16, threshold float32, minCount int) bool {
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	count := 0
	for i := len(bs.buffer) - 1; i >= 0; i-- {
		if bs.buffer[i].ChannelID == channelID {
			if bs.buffer[i].Value > threshold {
				count++
			} else {
				break
			}
		}
	}
	return count >= minCount
}

func (bs *BiowareStream) Stats() (received, dropped uint64) {
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	return bs.received, bs.dropped
}

// ============================================================
// §12  Dataset Loader (concurrent spike file parser)
// ============================================================

type SpikeEvent struct {
	NeuronID  uint32
	Timestamp float64
	Polarity  int8
}

type DatasetLoader struct {
	mu     sync.RWMutex
	events []SpikeEvent
}

func NewDatasetLoader() *DatasetLoader {
	return &DatasetLoader{}
}

func (dl *DatasetLoader) ParseBinaryChunk(data []byte) int {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	recordSize := 13 // 4 + 8 + 1
	count := len(data) / recordSize
	for i := 0; i < count; i++ {
		offset := i * recordSize
		ev := SpikeEvent{
			NeuronID:  binary.BigEndian.Uint32(data[offset : offset+4]),
			Timestamp: math.Float64frombits(binary.BigEndian.Uint64(data[offset+4 : offset+12])),
			Polarity:  int8(data[offset+12]),
		}
		dl.events = append(dl.events, ev)
	}
	return count
}

func (dl *DatasetLoader) Count() int {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	return len(dl.events)
}

func (dl *DatasetLoader) FilterByNeuron(neuronID uint32) []SpikeEvent {
	dl.mu.RLock()
	defer dl.mu.RUnlock()
	var result []SpikeEvent
	for _, ev := range dl.events {
		if ev.NeuronID == neuronID {
			result = append(result, ev)
		}
	}
	return result
}

// ============================================================
// §13  Model Zoo Registry
// ============================================================

type ModelEntry struct {
	Name       string  `json:"name"`
	Version    string  `json:"version"`
	Accuracy   float64 `json:"accuracy"`
	Parameters int     `json:"parameters"`
}

type ModelZoo struct {
	mu     sync.RWMutex
	models map[string]ModelEntry
}

func NewModelZoo() *ModelZoo {
	return &ModelZoo{models: make(map[string]ModelEntry)}
}

func (mz *ModelZoo) Register(entry ModelEntry) {
	mz.mu.Lock()
	defer mz.mu.Unlock()
	mz.models[entry.Name] = entry
}

func (mz *ModelZoo) Get(name string) (ModelEntry, bool) {
	mz.mu.RLock()
	defer mz.mu.RUnlock()
	m, ok := mz.models[name]
	return m, ok
}

func (mz *ModelZoo) List() []ModelEntry {
	mz.mu.RLock()
	defer mz.mu.RUnlock()
	result := make([]ModelEntry, 0, len(mz.models))
	for _, m := range mz.models {
		result = append(result, m)
	}
	return result
}

func (mz *ModelZoo) BestModel() ModelEntry {
	mz.mu.RLock()
	defer mz.mu.RUnlock()
	var best ModelEntry
	for _, m := range mz.models {
		if m.Accuracy > best.Accuracy {
			best = m
		}
	}
	return best
}

func main() {
	fmt.Println("=======================================================")
	fmt.Println("SC-NeuroCore Go Extended Services — Benchmark")
	fmt.Println("=======================================================")

	// §11 Bioware stream
	bio := NewBiowareStream(65536)
	t0 := time.Now()
	for i := 0; i < 1_000_000; i++ {
		bio.Ingest(BioSensorReading{
			Timestamp: uint64(i),
			ChannelID: uint16(i % 32),
			Value:     float32(i%100) / 100.0,
			Quality:   255,
		})
	}
	fmt.Printf("§11  Bioware ingest 1M:            %v\n", time.Since(t0))
	rec, drop := bio.Stats()
	fmt.Printf("      received=%d dropped=%d\n", rec, drop)

	t0 = time.Now()
	for i := 0; i < 10_000; i++ {
		bio.LFPPower(uint16(i%32), 100)
	}
	fmt.Printf("§11  LFP power 10k:               %v\n", time.Since(t0))

	// §12 Dataset loader
	dl := NewDatasetLoader()
	chunk := make([]byte, 13*10000) // 10k events
	for i := 0; i < 10000; i++ {
		binary.BigEndian.PutUint32(chunk[i*13:], uint32(i%256))
		binary.BigEndian.PutUint64(chunk[i*13+4:], math.Float64bits(float64(i)*0.001))
		chunk[i*13+12] = 1
	}
	t0 = time.Now()
	for i := 0; i < 1000; i++ {
		dl2 := NewDatasetLoader()
		dl2.ParseBinaryChunk(chunk)
	}
	_ = dl
	fmt.Printf("§12  Parse 10k events × 1k:       %v\n", time.Since(t0))

	// §13 Model zoo
	zoo := NewModelZoo()
	t0 = time.Now()
	for i := 0; i < 100_000; i++ {
		zoo.Register(ModelEntry{
			Name:       fmt.Sprintf("model_%d", i%1000),
			Version:    "v1.0",
			Accuracy:   float64(i%100) / 100.0,
			Parameters: i * 1000,
		})
	}
	fmt.Printf("§13  Zoo register 100k:            %v\n", time.Since(t0))
	fmt.Printf("      best=%s acc=%.2f\n", zoo.BestModel().Name, zoo.BestModel().Accuracy)

	fmt.Println("=======================================================")
	fmt.Println("3 new service groups, 15 functions total")
}
