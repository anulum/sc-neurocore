// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Package sccomptewmnetwork executes the separately named 2,560-cell
// SC-COMPTE-WM-NETWORK model. It preserves, rather than replaces, the scalar
// source-bounded Compte cell in package services.
package sccomptewmnetwork

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"math/bits"
)

const (
	// NExcitatory is the fixed pyramidal population size.
	NExcitatory = 2048
	// NInhibitory is the fixed interneuron population size.
	NInhibitory = 512
	// DtMS is the fixed source timestep in milliseconds.
	DtMS = 0.02

	golden    uint64 = 0x9e3779b97f4a7c15
	stepMix   uint64 = 0xd1b54a32d192ed03
	streamMix uint64 = 0x94d049bb133111eb
	gateMax          = 1.0e6
)

// Spec freezes the Go v1 runtime choices corresponding to the Python contract.
type Spec struct {
	Seed                   uint64
	StructuredEI           bool
	Modulated              bool
	AllowRecurrentAutapses bool
}

// DefaultSpec returns the control set, seed 42, uniform E-to-I projection, and
// no recurrent E-to-E or I-to-I autapses.
func DefaultSpec() Spec { return Spec{Seed: 42} }

// State owns every mutable scalar and array in the fixed network.
type State struct {
	StepIndex         uint64
	VExcMV            []float64
	VInhMV            []float64
	RefractoryExcMS   []float64
	RefractoryInhMS   []float64
	ExternalAMPAExc   []float64
	ExternalAMPAInh   []float64
	RecurrentNMDA     []float64
	RecurrentNMDARise []float64
	RecurrentGABAA    []float64
}

// NewState returns the leak-equilibrium, zero-gate state.
func NewState() *State {
	return &State{
		VExcMV:            filled(NExcitatory, -70),
		VInhMV:            filled(NInhibitory, -70),
		RefractoryExcMS:   make([]float64, NExcitatory),
		RefractoryInhMS:   make([]float64, NInhibitory),
		ExternalAMPAExc:   make([]float64, NExcitatory),
		ExternalAMPAInh:   make([]float64, NInhibitory),
		RecurrentNMDA:     make([]float64, NExcitatory),
		RecurrentNMDARise: make([]float64, NExcitatory),
		RecurrentGABAA:    make([]float64, NInhibitory),
	}
}

// Clone returns a deep checkpoint copy.
func (state *State) Clone() *State {
	if state == nil {
		return nil
	}
	return &State{
		StepIndex: state.StepIndex,
		VExcMV:    clone(state.VExcMV), VInhMV: clone(state.VInhMV),
		RefractoryExcMS:   clone(state.RefractoryExcMS),
		RefractoryInhMS:   clone(state.RefractoryInhMS),
		ExternalAMPAExc:   clone(state.ExternalAMPAExc),
		ExternalAMPAInh:   clone(state.ExternalAMPAInh),
		RecurrentNMDA:     clone(state.RecurrentNMDA),
		RecurrentNMDARise: clone(state.RecurrentNMDARise),
		RecurrentGABAA:    clone(state.RecurrentGABAA),
	}
}

// StepReceipt reports all output events, aggregate inputs, and custody hashes
// from one successful atomic transition.
type StepReceipt struct {
	StepIndex             uint64
	ExcitatorySpikes      []bool
	InhibitorySpikes      []bool
	ExcitatoryInputEvents uint64
	InhibitoryInputEvents uint64
	InputSHA256           string
	StateSHA256           string
}

// ActivityStatistics contains population rates and circular bump observables
// for one explicitly bounded non-empty window.
type ActivityStatistics struct {
	ExcitatoryRateHz float64
	InhibitoryRateHz float64
	BumpAngleDeg     float64
	ResultantLength  float64
	CircularWidthDeg *float64
}

// WindowReceipt contains spike totals and optional statistics for one run window.
type WindowReceipt struct {
	StartMS, EndMS                     float64
	ExcitatorySpikes, InhibitorySpikes int
	Statistics                         *ActivityStatistics
}

// RunReceipt is deterministic bounded evidence from a complete invocation.
type RunReceipt struct {
	SpecificationVersion string
	Seed                 uint64
	DurationMS           float64
	Steps                int
	ExcitatorySpikes     int
	InhibitorySpikes     int
	Windows              []WindowReceipt
	InputSHA256          string
	SpikeSHA256          string
	FinalStateSHA256     string
}

// Stimulus describes one localized-cue or global-current epoch in source pA.
type Stimulus struct {
	StartMS, DurationMS, CurrentPA float64
	Kind                           string
	CenterDeg                      *float64
}

// Validate rejects malformed stimulus epochs before execution.
func (stimulus Stimulus) Validate() error {
	if !finite(stimulus.StartMS) || stimulus.StartMS < 0 ||
		!finite(stimulus.DurationMS) || stimulus.DurationMS <= 0 ||
		!finite(stimulus.CurrentPA) || stimulus.CurrentPA <= 0 {
		return errors.New("invalid SC Compte stimulus bounds")
	}
	switch stimulus.Kind {
	case "localized_cue":
		if stimulus.CenterDeg == nil || !finite(*stimulus.CenterDeg) {
			return errors.New("localized cue requires finite center")
		}
	case "global_current":
		if stimulus.CenterDeg != nil {
			return errors.New("global current requires nil center")
		}
	default:
		return errors.New("unknown SC Compte stimulus kind")
	}
	return nil
}

// Network owns the complete state and cached spectra for one executor.
type Network struct {
	Spec       Spec
	State      *State
	eeKernel   []float64
	eeSpectrum []complex128
	eiSpectrum []complex128
}

// NewNetwork constructs and validates a network from an optional checkpoint.
func NewNetwork(spec Spec, state *State) (*Network, error) {
	if state == nil {
		state = NewState()
	}
	eeKernel := footprint(1.62, 18)
	network := &Network{Spec: spec, State: state.Clone(), eeKernel: eeKernel, eeSpectrum: fftReal(eeKernel)}
	if spec.StructuredEI {
		network.eiSpectrum = fftReal(footprint(1.25, 18))
	}
	if err := ValidateState(network.State); err != nil {
		return nil, err
	}
	return network, nil
}

// Reset restores zero-gate leak equilibrium without changing the specification.
func (network *Network) Reset() { network.State = NewState() }

// ValidateState fails closed unless every state array has its fixed shape and
// every scalar lies within the v1 safety envelope.
func ValidateState(state *State) error {
	if state == nil || len(state.VExcMV) != NExcitatory || len(state.VInhMV) != NInhibitory ||
		len(state.RefractoryExcMS) != NExcitatory || len(state.RefractoryInhMS) != NInhibitory ||
		len(state.ExternalAMPAExc) != NExcitatory || len(state.ExternalAMPAInh) != NInhibitory ||
		len(state.RecurrentNMDA) != NExcitatory || len(state.RecurrentNMDARise) != NExcitatory ||
		len(state.RecurrentGABAA) != NInhibitory {
		return errors.New("invalid SC Compte state shape")
	}
	for _, values := range [][]float64{state.VExcMV, state.VInhMV} {
		for _, value := range values {
			if !finite(value) || value < -200 || value > 100 {
				return errors.New("invalid SC Compte voltage state")
			}
		}
	}
	for _, values := range [][]float64{state.RefractoryExcMS, state.RefractoryInhMS,
		state.ExternalAMPAExc, state.ExternalAMPAInh, state.RecurrentNMDARise, state.RecurrentGABAA} {
		for _, value := range values {
			if !finite(value) || value < 0 || value > gateMax {
				return errors.New("invalid SC Compte gate state")
			}
		}
	}
	for _, value := range state.RecurrentNMDA {
		if !finite(value) || value < 0 || value > 1 {
			return errors.New("invalid SC Compte NMDA state")
		}
	}
	return nil
}

// StateSHA256 returns the canonical little-endian digest of the complete state.
func StateSHA256(state *State) (string, error) {
	if err := ValidateState(state); err != nil {
		return "", err
	}
	hash := sha256.New()
	writeUint64(hash.Write, state.StepIndex)
	for _, values := range [][]float64{state.VExcMV, state.VInhMV, state.RefractoryExcMS,
		state.RefractoryInhMS, state.ExternalAMPAExc, state.ExternalAMPAInh,
		state.RecurrentNMDA, state.RecurrentNMDARise, state.RecurrentGABAA} {
		for _, value := range values {
			writeUint64(hash.Write, math.Float64bits(value))
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// CounterPoissonCounts returns one portable aggregate count per cell for a
// seed/stream/step address using the fixed SplitMix64 inverse-CDF contract.
func CounterPoissonCounts(populationSize int, rateHz, dtMS float64, seed, stream, stepIndex uint64) ([]uint64, error) {
	if populationSize <= 0 || !finite(rateHz) || rateHz < 0 || !finite(dtMS) || dtMS <= 0 {
		return nil, errors.New("invalid counter-Poisson configuration")
	}
	mean := rateHz * dtMS / 1000
	if mean > 32 {
		return nil, errors.New("counter-Poisson mean exceeds safety envelope")
	}
	probability := math.Exp(-mean)
	cumulative := probability
	cdf := []float64{cumulative}
	for count := 1; cumulative < 1-1e-15; count++ {
		if count > 255 {
			return nil, errors.New("counter-Poisson inverse CDF exceeded event range")
		}
		probability *= mean / float64(count)
		cumulative += probability
		cdf = append(cdf, math.Min(1, cumulative))
	}
	cdf[len(cdf)-1] = 1
	counts := make([]uint64, populationSize)
	for cell := range counts {
		counter := seed + stepIndex*stepMix + stream*streamMix + uint64(cell)*golden
		uniform := (float64(splitmix64(counter)>>11) + 0.5) * math.Pow(2, -53)
		low, high := 0, len(cdf)
		for low < high {
			middle := low + (high-low)/2
			if cdf[middle] < uniform {
				low = middle + 1
			} else {
				high = middle
			}
		}
		counts[cell] = uint64(low)
	}
	return counts, nil
}

// Step advances with the canonical counter-addressed external input streams.
func (network *Network) Step(directExcCurrentPA []float64) (*StepReceipt, error) {
	exc, err := CounterPoissonCounts(NExcitatory, 1800, DtMS, network.Spec.Seed, 0, network.State.StepIndex)
	if err != nil {
		return nil, err
	}
	inh, err := CounterPoissonCounts(NInhibitory, 1800, DtMS, network.Spec.Seed, 1, network.State.StepIndex)
	if err != nil {
		return nil, err
	}
	return network.StepWithEvents(directExcCurrentPA, exc, inh)
}

// StepWithEvents advances atomically with explicit per-cell counts for
// independent-oracle and co-simulation use.
func (network *Network) StepWithEvents(directExcCurrentPA []float64, externalExcEvents, externalInhEvents []uint64) (*StepReceipt, error) {
	if err := ValidateState(network.State); err != nil {
		return nil, err
	}
	if len(directExcCurrentPA) != NExcitatory || len(externalExcEvents) != NExcitatory || len(externalInhEvents) != NInhibitory {
		return nil, errors.New("invalid SC Compte step input shape")
	}
	for _, value := range directExcCurrentPA {
		if !finite(value) {
			return nil, errors.New("SC Compte current must be finite")
		}
	}
	state := network.State
	start := stage{clone(state.VExcMV), clone(state.VInhMV), clone(state.ExternalAMPAExc),
		clone(state.ExternalAMPAInh), clone(state.RecurrentNMDA), clone(state.RecurrentNMDARise), clone(state.RecurrentGABAA)}
	for index, event := range externalExcEvents {
		start.extExc[index] += float64(event)
		if !finite(start.extExc[index]) || start.extExc[index] > gateMax {
			return nil, errors.New("excitatory event gate exceeds safety envelope")
		}
	}
	for index, event := range externalInhEvents {
		start.extInh[index] += float64(event)
		if !finite(start.extInh[index]) || start.extInh[index] > gateMax {
			return nil, errors.New("inhibitory event gate exceeds safety envelope")
		}
	}
	activeExc, activeInh := make([]bool, NExcitatory), make([]bool, NInhibitory)
	currentsNA := make([]float64, NExcitatory)
	for index := range activeExc {
		activeExc[index] = state.RefractoryExcMS[index] <= 0
		currentsNA[index] = directExcCurrentPA[index] / 1000
	}
	for index := range activeInh {
		activeInh[index] = state.RefractoryInhMS[index] <= 0
	}
	k1 := network.derivatives(start, currentsNA, activeExc, activeInh)
	midpoint := start.addScaled(k1, 0.5*DtMS)
	k2 := network.derivatives(midpoint, currentsNA, activeExc, activeInh)
	candidate := start.addScaled(k2, DtMS)
	refExc, refInh := make([]float64, NExcitatory), make([]float64, NInhibitory)
	excSpikes, inhSpikes := make([]bool, NExcitatory), make([]bool, NInhibitory)
	for index := range refExc {
		refExc[index] = math.Max(0, state.RefractoryExcMS[index]-DtMS)
		if !activeExc[index] {
			candidate.vExc[index] = -60
		} else if candidate.vExc[index] >= -50 {
			candidate.vExc[index], refExc[index], excSpikes[index] = -60, 2, true
			candidate.nmdaRise[index]++
		}
	}
	for index := range refInh {
		refInh[index] = math.Max(0, state.RefractoryInhMS[index]-DtMS)
		if !activeInh[index] {
			candidate.vInh[index] = -60
		} else if candidate.vInh[index] >= -50 {
			candidate.vInh[index], refInh[index], inhSpikes[index] = -60, 1, true
			candidate.gabaa[index]++
		}
	}
	if state.StepIndex == math.MaxUint64 {
		return nil, errors.New("SC Compte step counter overflow")
	}
	next := &State{state.StepIndex + 1, candidate.vExc, candidate.vInh, refExc, refInh,
		candidate.extExc, candidate.extInh, candidate.nmda, candidate.nmdaRise, candidate.gabaa}
	if err := ValidateState(next); err != nil {
		return nil, err
	}
	stateDigest, _ := StateSHA256(next)
	receipt := &StepReceipt{StepIndex: state.StepIndex, ExcitatorySpikes: excSpikes,
		InhibitorySpikes: inhSpikes, ExcitatoryInputEvents: sumUint64(externalExcEvents),
		InhibitoryInputEvents: sumUint64(externalInhEvents),
		InputSHA256:           inputSHA256(externalExcEvents, externalInhEvents, directExcCurrentPA),
		StateSHA256:           stateDigest}
	network.State = next
	return receipt, nil
}

// CueCurrentPA returns the SC compact raised-cosine cue over the excitatory ring.
func CueCurrentPA(centerDeg, peakPA float64) ([]float64, error) {
	if !finite(centerDeg) || !finite(peakPA) || peakPA <= 0 {
		return nil, errors.New("cue center and peak must be finite and positive")
	}
	current := make([]float64, NExcitatory)
	for index := range current {
		angle := float64(index) * 360 / NExcitatory
		distance := math.Abs(math.Mod(angle-centerDeg+540, 360) - 180)
		if distance < 18 {
			current[index] = 0.5 * peakPA * (1 + math.Cos(math.Pi*distance/18))
		}
	}
	return current, nil
}

// SummarizeActivity computes rates and circular observables for one non-empty window.
func SummarizeActivity(excCounts, inhCounts []int, windowMS float64) (*ActivityStatistics, error) {
	if len(excCounts) != NExcitatory || len(inhCounts) != NInhibitory || !finite(windowMS) || windowMS <= 0 {
		return nil, errors.New("invalid SC Compte statistics input")
	}
	totalExc, totalInh := 0, 0
	realPart, imagPart := 0.0, 0.0
	for index, count := range excCounts {
		if count < 0 {
			return nil, errors.New("spike counts must be non-negative")
		}
		totalExc += count
		angle := 2 * math.Pi * float64(index) / NExcitatory
		realPart += float64(count) * math.Cos(angle)
		imagPart += float64(count) * math.Sin(angle)
	}
	for _, count := range inhCounts {
		if count < 0 {
			return nil, errors.New("spike counts must be non-negative")
		}
		totalInh += count
	}
	if totalExc == 0 {
		return nil, errors.New("bump statistics require an excitatory spike")
	}
	resultant := math.Min(1, math.Hypot(realPart, imagPart)/float64(totalExc))
	angleDeg := math.Mod(math.Atan2(imagPart, realPart)*180/math.Pi+360, 360)
	var width *float64
	if resultant > 0 {
		value := math.Sqrt(-2*math.Log(resultant)) * 180 / math.Pi
		width = &value
	}
	seconds := windowMS / 1000
	return &ActivityStatistics{float64(totalExc) / (NExcitatory * seconds),
		float64(totalInh) / (NInhibitory * seconds), angleDeg, resultant, width}, nil
}

// Run executes integral timesteps and returns bounded input, spike, state, and
// activity-window receipts.
func (network *Network) Run(durationMS float64, stimuli []Stimulus, statisticsWindowMS float64) (*RunReceipt, error) {
	if !finite(durationMS) || durationMS <= 0 || !finite(statisticsWindowMS) || statisticsWindowMS <= 0 {
		return nil, errors.New("duration and statistics window must be finite and positive")
	}
	rawSteps, rawWindow := durationMS/DtMS, statisticsWindowMS/DtMS
	steps, windowSteps := int(math.Round(rawSteps)), int(math.Round(rawWindow))
	if math.Abs(rawSteps-float64(steps)) > 1e-10 || math.Abs(rawWindow-float64(windowSteps)) > 1e-10 {
		return nil, errors.New("run bounds must contain integral timesteps")
	}
	for _, stimulus := range stimuli {
		if err := stimulus.Validate(); err != nil || stimulus.StartMS+stimulus.DurationMS > durationMS+1e-12 {
			return nil, errors.New("stimulus epoch lies outside valid run")
		}
	}
	inputHash, spikeHash := sha256.New(), sha256.New()
	excWindow, inhWindow := make([]int, NExcitatory), make([]int, NInhibitory)
	windows := make([]WindowReceipt, 0)
	totalExc, totalInh, windowStart := 0, 0, 0
	for offset := 0; offset < steps; offset++ {
		current, err := stimulusCurrent(float64(offset)*DtMS, stimuli)
		if err != nil {
			return nil, err
		}
		receipt, err := network.Step(current)
		if err != nil {
			return nil, err
		}
		digest, _ := hex.DecodeString(receipt.InputSHA256)
		_, _ = inputHash.Write(digest)
		for index, event := range receipt.ExcitatorySpikes {
			if event {
				_, _ = spikeHash.Write([]byte{1})
				excWindow[index]++
				totalExc++
			} else {
				_, _ = spikeHash.Write([]byte{0})
			}
		}
		for index, event := range receipt.InhibitorySpikes {
			if event {
				_, _ = spikeHash.Write([]byte{1})
				inhWindow[index]++
				totalInh++
			} else {
				_, _ = spikeHash.Write([]byte{0})
			}
		}
		if (offset+1)%windowSteps == 0 || offset+1 == steps {
			excTotal, inhTotal := sumInt(excWindow), sumInt(inhWindow)
			var statistics *ActivityStatistics
			if excTotal > 0 {
				statistics, err = SummarizeActivity(excWindow, inhWindow, float64(offset+1-windowStart)*DtMS)
				if err != nil {
					return nil, err
				}
			}
			windows = append(windows, WindowReceipt{float64(windowStart) * DtMS,
				float64(offset+1) * DtMS, excTotal, inhTotal, statistics})
			clear(excWindow)
			clear(inhWindow)
			windowStart = offset + 1
		}
	}
	stateDigest, _ := StateSHA256(network.State)
	return &RunReceipt{"sc-neurocore.sc-compte-wm-network.v1", network.Spec.Seed,
		durationMS, steps, totalExc, totalInh, windows, hex.EncodeToString(inputHash.Sum(nil)),
		hex.EncodeToString(spikeHash.Sum(nil)), stateDigest}, nil
}

type stage struct{ vExc, vInh, extExc, extInh, nmda, nmdaRise, gabaa []float64 }

func (value stage) addScaled(slope stage, scale float64) stage {
	result := stage{make([]float64, NExcitatory), make([]float64, NInhibitory),
		make([]float64, NExcitatory), make([]float64, NInhibitory), make([]float64, NExcitatory),
		make([]float64, NExcitatory), make([]float64, NInhibitory)}
	pairs := [][2][]float64{{value.vExc, slope.vExc}, {value.vInh, slope.vInh},
		{value.extExc, slope.extExc}, {value.extInh, slope.extInh}, {value.nmda, slope.nmda},
		{value.nmdaRise, slope.nmdaRise}, {value.gabaa, slope.gabaa}}
	outputs := [][]float64{result.vExc, result.vInh, result.extExc, result.extInh,
		result.nmda, result.nmdaRise, result.gabaa}
	for group, pair := range pairs {
		for index := range outputs[group] {
			outputs[group][index] = pair[0][index] + scale*pair[1][index]
		}
	}
	return result
}

func (network *Network) derivatives(value stage, currents []float64, activeExc, activeInh []bool) stage {
	ee, ei, ie, ii := network.aggregates(value.nmda, value.gabaa)
	result := stage{make([]float64, NExcitatory), make([]float64, NInhibitory),
		make([]float64, NExcitatory), make([]float64, NInhibitory), make([]float64, NExcitatory),
		make([]float64, NExcitatory), make([]float64, NInhibitory)}
	nmdaScale, gabaScale := 1.0, 1.0
	if network.Spec.Modulated {
		nmdaScale, gabaScale = 1.2, 1.4
	}
	for index, voltage := range value.vExc {
		if activeExc[index] {
			result.vExc[index] = (-0.025*(voltage+70) - 0.0031*value.extExc[index]*voltage -
				0.000381*nmdaScale*ee[index]*mgBlock(voltage)*voltage -
				0.001336*gabaScale*ie[index]*(voltage+70) + currents[index]) / 0.5
		}
		result.extExc[index] = -value.extExc[index] / 2
		result.nmda[index] = -value.nmda[index]/100 + 0.5*value.nmdaRise[index]*(1-value.nmda[index])
		result.nmdaRise[index] = -value.nmdaRise[index] / 2
	}
	for index, voltage := range value.vInh {
		if activeInh[index] {
			result.vInh[index] = (-0.020*(voltage+70) - 0.00238*value.extInh[index]*voltage -
				0.000292*nmdaScale*ei[index]*mgBlock(voltage)*voltage -
				0.001024*gabaScale*ii[index]*(voltage+70)) / 0.2
		}
		result.extInh[index] = -value.extInh[index] / 2
		result.gabaa[index] = -value.gabaa[index] / 10
	}
	return result
}

func (network *Network) aggregates(nmda, gabaa []float64) ([]float64, []float64, []float64, []float64) {
	ee := circularSum(nmda, network.eeSpectrum)
	if !network.Spec.AllowRecurrentAutapses {
		for index := range ee {
			ee[index] -= network.eeKernel[0] * nmda[index]
		}
	}
	ei := make([]float64, NInhibitory)
	if network.eiSpectrum == nil {
		for index := range ei {
			ei[index] = sumFloat(nmda)
		}
	} else {
		values := circularSum(nmda, network.eiSpectrum)
		for index := range ei {
			ei[index] = values[index*4]
		}
	}
	total := sumFloat(gabaa)
	ie, ii := filled(NExcitatory, total), filled(NInhibitory, total)
	if !network.Spec.AllowRecurrentAutapses {
		for index := range ii {
			ii[index] -= gabaa[index]
		}
	}
	return ee, ei, ie, ii
}

func fft(values []complex128, inverse bool) []complex128 {
	n := len(values)
	result := append([]complex128(nil), values...)
	shift := uint(bits.UintSize - bits.Len(uint(n-1)))
	for index := 0; index < n; index++ {
		reversed := int(bits.Reverse(uint(index)) >> shift)
		if reversed > index {
			result[index], result[reversed] = result[reversed], result[index]
		}
	}
	sign := -1.0
	if inverse {
		sign = 1
	}
	for length := 2; length <= n; length <<= 1 {
		angle := sign * 2 * math.Pi / float64(length)
		root := complex(math.Cos(angle), math.Sin(angle))
		for start := 0; start < n; start += length {
			factor := complex(1.0, 0)
			for offset := 0; offset < length/2; offset++ {
				even := result[start+offset]
				odd := factor * result[start+offset+length/2]
				result[start+offset], result[start+offset+length/2] = even+odd, even-odd
				factor *= root
			}
		}
	}
	if inverse {
		for index := range result {
			result[index] /= complex(float64(n), 0)
		}
	}
	return result
}

func fftReal(values []float64) []complex128 {
	complexValues := make([]complex128, len(values))
	for index, value := range values {
		complexValues[index] = complex(value, 0)
	}
	return fft(complexValues, false)
}

func circularSum(source []float64, spectrum []complex128) []float64 {
	values := fftReal(source)
	for index := range values {
		values[index] *= spectrum[index]
	}
	values = fft(values, true)
	result := make([]float64, len(values))
	for index := range result {
		result[index] = real(values[index])
	}
	return result
}

func footprint(jPlus, sigmaDeg float64) []float64 {
	gaussian := make([]float64, NExcitatory)
	for index := range gaussian {
		angle := float64(index) * 360 / NExcitatory
		distance := math.Mod(angle+540, 360) - 180
		gaussian[index] = math.Exp(-0.5 * math.Pow(distance/sigmaDeg, 2))
	}
	mean := sumFloat(gaussian) / NExcitatory
	jMinus := (1 - jPlus*mean) / (1 - mean)
	weights := make([]float64, NExcitatory)
	for index, value := range gaussian {
		weights[index] = jMinus + (jPlus-jMinus)*value
	}
	weightMean := sumFloat(weights) / NExcitatory
	for index := range weights {
		weights[index] /= weightMean
	}
	return weights
}

func stimulusCurrent(timeMS float64, stimuli []Stimulus) ([]float64, error) {
	current := make([]float64, NExcitatory)
	for _, stimulus := range stimuli {
		if stimulus.StartMS <= timeMS && timeMS < stimulus.StartMS+stimulus.DurationMS {
			if stimulus.Kind == "global_current" {
				for index := range current {
					current[index] += stimulus.CurrentPA
				}
			} else {
				cue, err := CueCurrentPA(*stimulus.CenterDeg, stimulus.CurrentPA)
				if err != nil {
					return nil, err
				}
				for index := range current {
					current[index] += cue[index]
				}
			}
		}
	}
	return current, nil
}

func splitmix64(value uint64) uint64 {
	z := value + golden
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func inputSHA256(exc, inh []uint64, current []float64) string {
	hash := sha256.New()
	for _, value := range exc {
		writeUint64(hash.Write, value)
	}
	for _, value := range inh {
		writeUint64(hash.Write, value)
	}
	for _, value := range current {
		// Canonical 1e-9 pA custody ignores platform-libm rounding in cue cosines.
		quantized := int64(math.Floor(value*1_000_000_000.0 + 0.5))
		writeUint64(hash.Write, uint64(quantized))
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func writeUint64(write func([]byte) (int, error), value uint64) {
	var buffer [8]byte
	binary.LittleEndian.PutUint64(buffer[:], value)
	_, _ = write(buffer[:])
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
func filled(size int, value float64) []float64 {
	result := make([]float64, size)
	for i := range result {
		result[i] = value
	}
	return result
}
func clone(values []float64) []float64 { return append([]float64(nil), values...) }
func sumFloat(values []float64) float64 {
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total
}
func sumUint64(values []uint64) uint64 {
	var total uint64
	for _, value := range values {
		total += value
	}
	return total
}
func sumInt(values []int) int {
	total := 0
	for _, value := range values {
		total += value
	}
	return total
}
func mgBlock(voltage float64) float64 {
	return 1 / (1 + math.Exp(math.Max(-700, math.Min(700, -0.062*voltage)))/3.57)
}
