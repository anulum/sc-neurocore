// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for dna_mapper

#![allow(non_snake_case)]

const TOEHOLD_LENGTH: usize = 6;
const RECOGNITION_LENGTH: usize = 15;
const GC_TARGET_LOW: f64 = 0.40;
const GC_TARGET_HIGH: f64 = 0.60;
const MAX_HOMOPOLYMER: usize = 3;
const DEFAULT_TEMPERATURE_C: f64 = 37.0;
const R_GAS: f64 = 1.987e-3;
const NN_INIT_DG: f64 = 1.96;
const NN_INIT_DH: f64 = 0.2;
const NN_INIT_DS: f64 = -5.7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateType {
    And,
    Or,
    Not,
    Nand,
    Xor,
    Mux,
    Threshold,
    Catalytic,
    Amplifier,
    Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationMethod {
    Displacement,
    Enzymatic,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrandRole {
    Signal,
    Fuel,
    Output,
    Waste,
    Toehold,
    Translator,
    Threshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integrator {
    Euler,
    Rk4,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DNAStrand {
    pub name: String,
    pub sequence: String,
    pub role: StrandRole,
    pub concentration_nM: f64,
}

impl DNAStrand {
    pub fn new(
        name: impl Into<String>,
        sequence: impl Into<String>,
        role: StrandRole,
        concentration_nM: f64,
    ) -> Result<Self, String> {
        let name = name.into();
        let sequence = sequence.into().to_ascii_uppercase();
        if name.trim().is_empty() {
            return Err("strand name must not be empty".to_string());
        }
        if !concentration_nM.is_finite() || concentration_nM < 0.0 {
            return Err("strand concentration_nM must be finite and non-negative".to_string());
        }
        if !sequence
            .bytes()
            .all(|base| matches!(base, b'A' | b'C' | b'G' | b'T'))
        {
            return Err("strand sequence must contain only A, C, G, T".to_string());
        }
        Ok(Self {
            name,
            sequence,
            role,
            concentration_nM,
        })
    }

    pub fn length(&self) -> usize {
        self.sequence.len()
    }

    pub fn gc_content(&self) -> f64 {
        if self.sequence.is_empty() {
            return 0.0;
        }
        let gc = self
            .sequence
            .bytes()
            .filter(|base| matches!(base, b'G' | b'C'))
            .count();
        gc as f64 / self.sequence.len() as f64
    }

    pub fn complement(&self) -> String {
        reverse_complement(&self.sequence)
    }

    pub fn max_homopolymer_run(&self) -> usize {
        max_homopolymer_run(&self.sequence)
    }

    pub fn delta_g_37(&self) -> f64 {
        if self.sequence.len() < 2 {
            return 0.0;
        }
        let mut dg = NN_INIT_DG;
        for idx in 0..(self.sequence.len() - 1) {
            dg += nn_dg(&self.sequence[idx..idx + 2]);
        }
        dg
    }

    pub fn melting_temperature(&self, na_conc_M: f64, strand_conc_M: f64) -> Result<f64, String> {
        if !na_conc_M.is_finite() || na_conc_M <= 0.0 {
            return Err("na_conc_M must be finite and positive".to_string());
        }
        if !strand_conc_M.is_finite() || strand_conc_M <= 0.0 {
            return Err("strand_conc_M must be finite and positive".to_string());
        }
        if self.sequence.len() < 2 {
            return Err("melting_temperature requires at least two nucleotides".to_string());
        }

        let mut delta_h = NN_INIT_DH;
        let mut delta_s = NN_INIT_DS;
        for terminal in [
            self.sequence.as_bytes()[0],
            self.sequence.as_bytes()[self.sequence.len() - 1],
        ] {
            if matches!(terminal, b'A' | b'T') {
                delta_h += 2.2;
                delta_s += 6.9;
            } else {
                delta_h += 0.1;
                delta_s -= 2.8;
            }
        }
        for idx in 0..(self.sequence.len() - 1) {
            let pair = &self.sequence[idx..idx + 2];
            delta_h += nn_dh(pair);
            delta_s += nn_ds(pair);
        }

        let tm_kelvin =
            (1000.0 * delta_h) / (delta_s + (1000.0 * R_GAS) * (strand_conc_M / 4.0).ln());
        Ok(tm_kelvin - 273.15 + 16.6 * na_conc_M.log10())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DNAGate {
    pub gate_id: usize,
    pub gate_type: GateType,
    pub input_names: Vec<String>,
    pub output_name: String,
    pub strands: Vec<DNAStrand>,
    pub threshold: f64,
    pub leak_rate: f64,
}

impl DNAGate {
    pub fn strand_count(&self) -> usize {
        self.strands.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DNACircuitDesign {
    pub name: String,
    pub gates: Vec<DNAGate>,
    pub input_strands: Vec<DNAStrand>,
    pub output_strands: Vec<DNAStrand>,
    pub fuel_strands: Vec<DNAStrand>,
    pub method: CompilationMethod,
    pub temperature_c: f64,
    pub na_concentration_M: f64,
}

impl DNACircuitDesign {
    pub fn new(name: impl Into<String>, method: CompilationMethod, temperature_c: f64) -> Self {
        Self {
            name: name.into(),
            gates: Vec::new(),
            input_strands: Vec::new(),
            output_strands: Vec::new(),
            fuel_strands: Vec::new(),
            method,
            temperature_c,
            na_concentration_M: 1.0,
        }
    }

    pub fn total_strands(&self) -> usize {
        self.input_strands.len()
            + self.output_strands.len()
            + self.fuel_strands.len()
            + self.gates.iter().map(DNAGate::strand_count).sum::<usize>()
    }

    pub fn total_gates(&self) -> usize {
        self.gates.len()
    }

    pub fn total_nucleotides(&self) -> usize {
        self.all_strands()
            .iter()
            .map(|strand| strand.length())
            .sum()
    }

    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        for strand in self.all_strands() {
            let gc = strand.gc_content();
            if !(GC_TARGET_LOW..=GC_TARGET_HIGH).contains(&gc) {
                warnings.push(format!(
                    "{}: GC content {:.2} outside [{:.2}, {:.2}]",
                    strand.name, gc, GC_TARGET_LOW, GC_TARGET_HIGH
                ));
            }
            let run = strand.max_homopolymer_run();
            if run > MAX_HOMOPOLYMER {
                warnings.push(format!(
                    "{}: homopolymer run {} exceeds max {}",
                    strand.name, run, MAX_HOMOPOLYMER
                ));
            }
        }
        warnings
    }

    fn all_strands(&self) -> Vec<&DNAStrand> {
        let mut strands = Vec::new();
        strands.extend(self.input_strands.iter());
        strands.extend(self.output_strands.iter());
        strands.extend(self.fuel_strands.iter());
        for gate in &self.gates {
            strands.extend(gate.strands.iter());
        }
        strands
    }
}

pub type PlateLayout = DNACircuitDesign;

#[derive(Debug, Clone)]
pub struct SequenceDesigner {
    rng_state: u64,
    gc_target: (f64, f64),
    max_homopolymer: usize,
    used_sequences: Vec<String>,
}

impl Default for SequenceDesigner {
    fn default() -> Self {
        Self::new(42, (GC_TARGET_LOW, GC_TARGET_HIGH), MAX_HOMOPOLYMER)
            .expect("default DNA sequence design constraints are valid")
    }
}

impl SequenceDesigner {
    pub fn new(seed: u64, gc_target: (f64, f64), max_homopolymer: usize) -> Result<Self, String> {
        if !gc_target.0.is_finite()
            || !gc_target.1.is_finite()
            || !(0.0..=1.0).contains(&gc_target.0)
            || !(0.0..=1.0).contains(&gc_target.1)
            || gc_target.0 > gc_target.1
        {
            return Err("gc_target must be finite ordered fractions".to_string());
        }
        if max_homopolymer == 0 {
            return Err("max_homopolymer must be positive".to_string());
        }
        Ok(Self {
            rng_state: seed ^ 0x444e_415f_5343_504e,
            gc_target,
            max_homopolymer,
            used_sequences: Vec::new(),
        })
    }

    pub fn generate(&mut self, length: usize, name: &str) -> Result<String, String> {
        if length == 0 {
            return Err("sequence length must be positive".to_string());
        }
        let mut candidate_rng = self.rng_state ^ stable_hash(name);
        self.rng_state = splitmix64(self.rng_state);
        let mut best_seq = String::new();
        let mut best_score = f64::INFINITY;

        for _ in 0..200 {
            let mut seq = Vec::with_capacity(length);
            let mut gc_count = 0usize;
            for pos in 0..length {
                let mut weights = [1.0_f64; 4];
                if pos > 0 {
                    let current_gc = gc_count as f64 / pos as f64;
                    weights = if current_gc < self.gc_target.0 {
                        [0.15, 0.35, 0.35, 0.15]
                    } else if current_gc > self.gc_target.1 {
                        [0.35, 0.15, 0.15, 0.35]
                    } else {
                        [0.25, 0.25, 0.25, 0.25]
                    };
                }
                if seq.len() >= self.max_homopolymer {
                    let recent = &seq[seq.len() - self.max_homopolymer..];
                    if recent.iter().all(|base| *base == recent[0]) {
                        weights[nuc_index(recent[0])] = 0.0;
                    }
                }
                let nuc = weighted_base(&mut candidate_rng, weights);
                if matches!(nuc, b'G' | b'C') {
                    gc_count += 1;
                }
                seq.push(nuc);
            }
            let candidate = String::from_utf8(seq).expect("DNA alphabet is ASCII");
            let score = self.sequence_score(&candidate);
            if score < best_score {
                best_score = score;
                best_seq = candidate;
            }
            if best_score < 0.5 {
                break;
            }
            candidate_rng = splitmix64(candidate_rng);
        }
        self.used_sequences.push(best_seq.clone());
        Ok(best_seq)
    }

    pub fn generate_complement(&self, sequence: &str) -> String {
        reverse_complement(sequence)
    }

    pub fn generate_toehold(&mut self, name: &str) -> Result<String, String> {
        self.generate(TOEHOLD_LENGTH, name)
    }

    pub fn generate_recognition(&mut self, name: &str) -> Result<String, String> {
        self.generate(RECOGNITION_LENGTH, name)
    }

    fn sequence_score(&self, sequence: &str) -> f64 {
        let gc = DNAStrand::new("candidate", sequence, StrandRole::Signal, 1.0)
            .map(|strand| strand.gc_content())
            .unwrap_or(0.0);
        let mut score = (gc - 0.5).abs() * 10.0;
        let run = max_homopolymer_run(sequence);
        if run > self.max_homopolymer {
            score += (run - self.max_homopolymer) as f64 * 5.0;
        }
        for existing in &self.used_sequences {
            let overlap = sequence
                .bytes()
                .zip(existing.bytes())
                .filter(|(left, right)| left == right)
                .count();
            let denom = sequence.len().max(existing.len()).max(1) as f64;
            let similarity = overlap as f64 / denom;
            if similarity > 0.7 {
                score += similarity * 10.0;
            }
        }
        score
    }
}

#[derive(Debug, Clone)]
pub struct StrandDisplacementCompiler {
    designer: SequenceDesigner,
    temperature_c: f64,
    gate_counter: usize,
}

impl StrandDisplacementCompiler {
    pub fn new(designer: SequenceDesigner, temperature_c: f64) -> Result<Self, String> {
        if !temperature_c.is_finite() || temperature_c <= -273.15 {
            return Err("temperature_c must be finite and above absolute zero".to_string());
        }
        Ok(Self {
            designer,
            temperature_c,
            gate_counter: 0,
        })
    }

    pub fn compile_and(
        &mut self,
        input_a: &str,
        input_b: &str,
        output: &str,
    ) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let th_a = self.designer.generate_toehold(&format!("g{gid}_th_a"))?;
        let th_b = self.designer.generate_toehold(&format!("g{gid}_th_b"))?;
        let recog_a = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_a"))?;
        let recog_b = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_b"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let th_out = self.designer.generate_toehold(&format!("g{gid}_th_out"))?;
        let blocker = self
            .designer
            .generate_complement(&(th_a.clone() + &recog_a[..8]));
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_translator_top"),
                th_a.clone() + &recog_a + &recog_b + &th_b,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_translator_bot"),
                self.designer
                    .generate_complement(&(recog_a.clone() + &recog_b)),
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_output"),
                th_out + &recog_out,
                StrandRole::Output,
                0.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_fuel"),
                self.designer.generate_complement(&recog_a) + &th_a,
                StrandRole::Fuel,
                500.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_threshold"),
                blocker,
                StrandRole::Threshold,
                50.0,
            )?,
        ];
        let leak_rate = self.estimate_leak_rate(&strands[0], &strands[4]);
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::And,
            input_names: vec![input_a.to_string(), input_b.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate,
        })
    }

    pub fn compile_or(
        &mut self,
        input_a: &str,
        input_b: &str,
        output: &str,
    ) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let th_a = self.designer.generate_toehold(&format!("g{gid}_th_a"))?;
        let th_b = self.designer.generate_toehold(&format!("g{gid}_th_b"))?;
        let stem = self
            .designer
            .generate_recognition(&format!("g{gid}_stem"))?;
        let loop_seq = self.designer.generate(8, &format!("g{gid}_loop"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let hairpin = th_a.clone() + &stem + &loop_seq + &self.designer.generate_complement(&stem);
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_hairpin_a"),
                hairpin,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_hairpin_b"),
                th_b + &stem + &loop_seq + &self.designer.generate_complement(&stem),
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), recog_out, StrandRole::Output, 0.0)?,
            DNAStrand::new(
                format!("g{gid}_fuel"),
                self.designer.generate_complement(&stem) + &th_a,
                StrandRole::Fuel,
                500.0,
            )?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Or,
            input_names: vec![input_a.to_string(), input_b.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 1e-9,
        })
    }

    pub fn compile_not(&mut self, input_name: &str, output: &str) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let th = self.designer.generate_toehold(&format!("g{gid}_th"))?;
        let recog = self.designer.generate_recognition(&format!("g{gid}_rec"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_blocker"),
                th + &recog,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_output_complex"),
                self.designer.generate_complement(&recog) + &recog_out,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), recog_out, StrandRole::Output, 0.0)?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Not,
            input_names: vec![input_name.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 5e-10,
        })
    }

    pub fn compile_threshold(
        &mut self,
        input_name: &str,
        output: &str,
        threshold: f64,
    ) -> Result<DNAGate, String> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("threshold must be finite in [0, 1]".to_string());
        }
        let gid = self.next_gate_id();
        let th = self.designer.generate_toehold(&format!("g{gid}_th"))?;
        let recog = self.designer.generate_recognition(&format!("g{gid}_rec"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_absorber"),
                self.designer.generate_complement(&(th.clone() + &recog)),
                StrandRole::Threshold,
                threshold * 200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_translator"),
                th + &recog + &recog_out,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), recog_out, StrandRole::Output, 0.0)?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Threshold,
            input_names: vec![input_name.to_string()],
            output_name: output.to_string(),
            strands,
            threshold,
            leak_rate: 2e-9,
        })
    }

    pub fn compile_mux(
        &mut self,
        select: &str,
        input_a: &str,
        input_b: &str,
        output: &str,
    ) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let th_s = self.designer.generate_toehold(&format!("g{gid}_th_s"))?;
        let th_a = self.designer.generate_toehold(&format!("g{gid}_th_a"))?;
        let th_b = self.designer.generate_toehold(&format!("g{gid}_th_b"))?;
        let recog_a = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_a"))?;
        let recog_b = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_b"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_path_a"),
                th_s.clone() + &recog_a + &th_a,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_path_b"),
                self.designer.generate_complement(&th_s) + &recog_b + &th_b,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_combiner"),
                recog_out.clone(),
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), recog_out, StrandRole::Output, 0.0)?,
            DNAStrand::new(
                format!("g{gid}_fuel"),
                self.designer.generate_complement(&recog_a) + &th_s,
                StrandRole::Fuel,
                500.0,
            )?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Mux,
            input_names: vec![select.to_string(), input_a.to_string(), input_b.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 2e-9,
        })
    }

    pub fn compile_amplifier(&mut self, input_name: &str, output: &str) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let th = self.designer.generate_toehold(&format!("g{gid}_th"))?;
        let recog = self.designer.generate_recognition(&format!("g{gid}_rec"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let th_cat = self.designer.generate_toehold(&format!("g{gid}_th_cat"))?;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_catalyst_complex"),
                th.clone() + &recog + &th_cat,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_substrate"),
                self.designer.generate_complement(&recog) + &recog_out,
                StrandRole::Translator,
                500.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_fuel"),
                self.designer.generate_complement(&(th + &recog)),
                StrandRole::Fuel,
                1000.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), recog_out, StrandRole::Output, 0.0)?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Amplifier,
            input_names: vec![input_name.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 1e-9,
        })
    }

    pub fn compile_buffer(&mut self, input_name: &str, output: &str) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let th = self.designer.generate_toehold(&format!("g{gid}_th"))?;
        let recog = self.designer.generate_recognition(&format!("g{gid}_rec"))?;
        let recog_out = self
            .designer
            .generate_recognition(&format!("g{gid}_rec_out"))?;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_threshold"),
                self.designer
                    .generate_complement(&(th.clone() + &recog[..8])),
                StrandRole::Threshold,
                80.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_translator"),
                th + &recog + &recog_out,
                StrandRole::Translator,
                200.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), recog_out, StrandRole::Output, 0.0)?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Buffer,
            input_names: vec![input_name.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 5e-10,
        })
    }

    pub fn estimate_leak_rate(&self, strand: &DNAStrand, blocker: &DNAStrand) -> f64 {
        let dg = Self::strongest_blocker_delta_g(strand, blocker);
        let temp_k = self.temperature_c + 273.15;
        (1e-6 * (dg / (R_GAS * temp_k)).exp()).min(1e-6)
    }

    pub fn strongest_blocker_delta_g(strand: &DNAStrand, blocker: &DNAStrand) -> f64 {
        let query = strand.sequence.as_bytes();
        let target = blocker.complement();
        let target = target.as_bytes();
        let mut best_dg = 0.0;
        let start = -(target.len() as isize) + 1;
        let end = query.len() as isize;
        for offset in start..end {
            let mut run = Vec::new();
            for (idx, base) in query.iter().enumerate() {
                let j = idx as isize - offset;
                if j >= 0 && (j as usize) < target.len() && *base == target[j as usize] {
                    run.push(*base);
                    continue;
                }
                if run.len() >= 2 {
                    let seq = String::from_utf8(run.clone()).expect("DNA run is ASCII");
                    if let Ok(strand) = DNAStrand::new("blocker_run", seq, StrandRole::Signal, 1.0)
                    {
                        best_dg = f64::min(best_dg, strand.delta_g_37());
                    }
                }
                run.clear();
            }
            if run.len() >= 2 {
                let seq = String::from_utf8(run).expect("DNA run is ASCII");
                if let Ok(strand) = DNAStrand::new("blocker_run", seq, StrandRole::Signal, 1.0) {
                    best_dg = f64::min(best_dg, strand.delta_g_37());
                }
            }
        }
        best_dg
    }

    fn next_gate_id(&mut self) -> usize {
        let gid = self.gate_counter;
        self.gate_counter += 1;
        gid
    }
}

#[derive(Debug, Clone)]
pub struct EnzymaticGateCompiler {
    designer: SequenceDesigner,
    gate_counter: usize,
}

impl EnzymaticGateCompiler {
    pub fn new(designer: SequenceDesigner) -> Self {
        Self {
            designer,
            gate_counter: 0,
        }
    }

    pub fn compile_nand(
        &mut self,
        input_a: &str,
        input_b: &str,
        output: &str,
    ) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let flank_5 = self.designer.generate(20, &format!("g{gid}_flank5"))?;
        let flank_3 = self.designer.generate(20, &format!("g{gid}_flank3"))?;
        let spacer = self.designer.generate(10, &format!("g{gid}_spacer"))?;
        let out_seq = self.designer.generate_recognition(&format!("g{gid}_out"))?;
        let substrate = flank_5 + "GAATTC" + &spacer + &out_seq + &spacer + "GGATCC" + &flank_3;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_substrate"),
                substrate,
                StrandRole::Translator,
                100.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), out_seq, StrandRole::Output, 0.0)?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Nand,
            input_names: vec![input_a.to_string(), input_b.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 1e-9,
        })
    }

    pub fn compile_xor(
        &mut self,
        input_a: &str,
        input_b: &str,
        output: &str,
    ) -> Result<DNAGate, String> {
        let gid = self.next_gate_id();
        let left = self.designer.generate(20, &format!("g{gid}_left"))?;
        let right = self.designer.generate(20, &format!("g{gid}_right"))?;
        let out_seq = self.designer.generate_recognition(&format!("g{gid}_out"))?;
        let strands = vec![
            DNAStrand::new(
                format!("g{gid}_nick_a"),
                left.clone() + &out_seq[..7],
                StrandRole::Translator,
                100.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_nick_b"),
                out_seq[7..].to_string() + &right,
                StrandRole::Translator,
                100.0,
            )?,
            DNAStrand::new(
                format!("g{gid}_template"),
                self.designer
                    .generate_complement(&(left + &out_seq + &right)),
                StrandRole::Translator,
                100.0,
            )?,
            DNAStrand::new(format!("g{gid}_output"), out_seq, StrandRole::Output, 0.0)?,
        ];
        Ok(DNAGate {
            gate_id: gid,
            gate_type: GateType::Xor,
            input_names: vec![input_a.to_string(), input_b.to_string()],
            output_name: output.to_string(),
            strands,
            threshold: 0.0,
            leak_rate: 1e-9,
        })
    }

    fn next_gate_id(&mut self) -> usize {
        let gid = self.gate_counter;
        self.gate_counter += 1;
        gid
    }
}

#[derive(Debug, Clone)]
pub struct NupackInterface {
    temperature_c: f64,
    na_concentration_M: f64,
}

impl NupackInterface {
    pub fn new(temperature_c: f64, na_concentration_M: f64) -> Result<Self, String> {
        if !temperature_c.is_finite() || temperature_c <= -273.15 {
            return Err("temperature_c must be finite and above absolute zero".to_string());
        }
        if !na_concentration_M.is_finite() || na_concentration_M <= 0.0 {
            return Err("na_concentration_M must be finite and positive".to_string());
        }
        Ok(Self {
            temperature_c,
            na_concentration_M,
        })
    }

    pub fn compute_mfe(&self, sequence: &str) -> Result<(f64, String), String> {
        let strand = DNAStrand::new("query", sequence, StrandRole::Signal, 1.0)?;
        let salt_adjustment = 0.114 * self.na_concentration_M.log10();
        let temp_adjustment = 0.001 * (self.temperature_c - DEFAULT_TEMPERATURE_C);
        Ok((
            strand.delta_g_37() + salt_adjustment + temp_adjustment,
            ".".repeat(strand.length()),
        ))
    }

    pub fn compute_pair_probabilities(&self, sequence: &str) -> Result<Vec<Vec<f64>>, String> {
        DNAStrand::new("query", sequence, StrandRole::Signal, 1.0)?;
        let bases = sequence.as_bytes();
        let mut matrix = vec![vec![0.0; bases.len()]; bases.len()];
        for i in 0..bases.len() {
            for j in (i + 3)..bases.len() {
                if is_watson_crick(bases[i], bases[j]) {
                    let span = (j - i) as f64;
                    let p = (0.02 + 0.18 / span).min(0.25);
                    matrix[i][j] = p;
                    matrix[j][i] = p;
                }
            }
        }
        Ok(matrix)
    }

    pub fn validate_design(&self, design: &DNACircuitDesign) -> Vec<String> {
        let mut warnings = design.validate();
        for strand in design.all_strands() {
            if let Ok((energy, _)) = self.compute_mfe(&strand.sequence) {
                if energy < -2.0 {
                    warnings.push(format!(
                        "{}: self-structure energy {:.2} kcal/mol",
                        strand.name, energy
                    ));
                }
            }
        }
        warnings
    }
}

#[derive(Debug, Clone)]
pub struct KineticSimulator {
    rate_hybridization: f64,
    rate_displacement: f64,
    temperature_c: f64,
    integrator: Integrator,
}

impl KineticSimulator {
    pub fn new(
        rate_hybridization: f64,
        rate_displacement: f64,
        temperature_c: f64,
        integrator: Integrator,
    ) -> Result<Self, String> {
        if !rate_hybridization.is_finite() || rate_hybridization <= 0.0 {
            return Err("rate_hybridization must be finite and positive".to_string());
        }
        if !rate_displacement.is_finite() || rate_displacement <= 0.0 {
            return Err("rate_displacement must be finite and positive".to_string());
        }
        if !temperature_c.is_finite() || temperature_c <= -273.15 {
            return Err("temperature_c must be finite and above absolute zero".to_string());
        }
        Ok(Self {
            rate_hybridization,
            rate_displacement,
            temperature_c,
            integrator,
        })
    }

    pub fn simulate(
        &mut self,
        design: &DNACircuitDesign,
        input_concentrations: &[(&str, f64)],
        duration_s: f64,
        dt: f64,
    ) -> Result<SimulationResult, String> {
        if !duration_s.is_finite() || duration_s <= 0.0 || !dt.is_finite() || dt <= 0.0 {
            return Err("duration_s and dt must be finite and positive".to_string());
        }
        let n_steps = (duration_s / dt).floor() as usize;
        if n_steps < 2 {
            return Err("simulation requires at least two steps".to_string());
        }
        for (_, value) in input_concentrations {
            if !value.is_finite() || *value < 0.0 {
                return Err("input concentrations must be finite and non-negative".to_string());
            }
        }

        let time = (0..n_steps).map(|idx| idx as f64 * dt).collect::<Vec<_>>();
        let mut traces = Vec::new();
        for gate in &design.gates {
            let k_eff = self.compute_k_eff(gate, input_concentrations);
            let mut conc = vec![0.0; n_steps];
            let max_conc = 200.0;
            for step in 1..n_steps {
                let c = conc[step - 1];
                conc[step] = match self.integrator {
                    Integrator::Euler => c + k_eff * (max_conc - c) * dt,
                    Integrator::Rk4 => {
                        let k1 = k_eff * (max_conc - c) * dt;
                        let k2 = k_eff * (max_conc - (c + k1 / 2.0)) * dt;
                        let k3 = k_eff * (max_conc - (c + k2 / 2.0)) * dt;
                        let k4 = k_eff * (max_conc - (c + k3)) * dt;
                        c + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                    }
                }
                .clamp(0.0, max_conc);
            }
            traces.push((gate.output_name.clone(), conc));
        }
        Ok(SimulationResult { time, traces })
    }

    fn compute_k_eff(&self, gate: &DNAGate, inputs: &[(&str, f64)]) -> f64 {
        let k_hyb = self.arrhenius_scale(self.rate_hybridization);
        let k_disp = self.arrhenius_scale(self.rate_displacement);
        let input = |name: &str| -> f64 {
            inputs
                .iter()
                .find_map(|(key, value)| (*key == name).then_some(*value))
                .unwrap_or(0.0)
        };
        let k_eff = match gate.gate_type {
            GateType::And => {
                let a = input(&gate.input_names[0]);
                let b = input(&gate.input_names[1]);
                if a > 0.0 && b > 0.0 {
                    k_hyb * f64::min(a, b) * 1e-9
                } else {
                    0.0
                }
            }
            GateType::Or => {
                let max_input = gate
                    .input_names
                    .iter()
                    .map(|name| input(name))
                    .fold(0.0, f64::max);
                k_hyb * max_input * 1e-9
            }
            GateType::Not => {
                let inp = input(&gate.input_names[0]);
                k_disp * (1.0 - (inp / 200.0).clamp(0.0, 1.0))
            }
            GateType::Threshold => {
                let inp = input(&gate.input_names[0]);
                k_hyb * f64::max(0.0, inp - gate.threshold * 200.0) * 1e-9
            }
            GateType::Mux => {
                let sel = input(&gate.input_names[0]);
                let a = input(&gate.input_names[1]);
                let b = input(&gate.input_names[2]);
                let sel_frac = (sel / 200.0).clamp(0.0, 1.0);
                k_hyb * (sel_frac * a + (1.0 - sel_frac) * b) * 1e-9
            }
            GateType::Amplifier => k_hyb * input(&gate.input_names[0]) * 1e-9 * 5.0,
            GateType::Buffer => k_disp * (input(&gate.input_names[0]) / 200.0).clamp(0.0, 1.0),
            GateType::Nand | GateType::Xor | GateType::Catalytic => {
                k_hyb * input(&gate.input_names[0]) * 1e-9
            }
        };
        k_eff + gate.leak_rate
    }

    fn arrhenius_scale(&self, k_ref: f64) -> f64 {
        let t_ref = 310.15;
        let t_op = self.temperature_c + 273.15;
        k_ref * (-(15.0 / R_GAS) * (1.0 / t_op - 1.0 / t_ref)).exp()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimulationResult {
    pub time: Vec<f64>,
    pub traces: Vec<(String, Vec<f64>)>,
}

impl SimulationResult {
    pub fn final_concentration(&self, output_name: &str) -> Option<f64> {
        self.traces
            .iter()
            .find_map(|(name, trace)| (name == output_name).then(|| trace.last().copied()))
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateSpec {
    pub gate_type: GateType,
    pub inputs: Vec<String>,
    pub output: String,
    pub threshold: Option<f64>,
}

impl GateSpec {
    pub fn new(gate_type: GateType, inputs: &[&str], output: &str) -> Self {
        Self {
            gate_type,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: output.to_string(),
            threshold: None,
        }
    }

    pub fn with_threshold(inputs: &[&str], output: &str, threshold: f64) -> Self {
        Self {
            gate_type: GateType::Threshold,
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            output: output.to_string(),
            threshold: Some(threshold),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BitstreamToDNA {
    method: CompilationMethod,
    designer: SequenceDesigner,
    displacement: StrandDisplacementCompiler,
    enzymatic: EnzymaticGateCompiler,
    temperature_c: f64,
}

impl BitstreamToDNA {
    pub fn try_new(
        method: CompilationMethod,
        seed: u64,
        temperature_c: f64,
    ) -> Result<Self, String> {
        let designer =
            SequenceDesigner::new(seed, (GC_TARGET_LOW, GC_TARGET_HIGH), MAX_HOMOPOLYMER)?;
        let displacement = StrandDisplacementCompiler::new(designer.clone(), temperature_c)?;
        let enzymatic = EnzymaticGateCompiler::new(designer.clone());
        Ok(Self {
            method,
            designer,
            displacement,
            enzymatic,
            temperature_c,
        })
    }

    pub fn compile_network(
        &mut self,
        gates: &[GateSpec],
        input_names: &[&str],
        output_names: &[&str],
        name: &str,
    ) -> Result<DNACircuitDesign, String> {
        let mut design = DNACircuitDesign::new(name, self.method, self.temperature_c);
        for input in input_names {
            let seq = self
                .designer
                .generate_recognition(&format!("input_{input}"))?;
            let toehold = self
                .designer
                .generate_toehold(&format!("input_{input}_th"))?;
            design.input_strands.push(DNAStrand::new(
                format!("signal_{input}"),
                toehold + &seq,
                StrandRole::Signal,
                200.0,
            )?);
        }

        for spec in gates {
            let gate = match self.method {
                CompilationMethod::Displacement | CompilationMethod::Hybrid => {
                    self.compile_displacement_gate(spec)?
                }
                CompilationMethod::Enzymatic => self.compile_enzymatic_gate(spec)?,
            };
            design.gates.push(gate);
        }

        for output in output_names {
            let seq = self
                .designer
                .generate_recognition(&format!("output_{output}"))?;
            design.output_strands.push(DNAStrand::new(
                format!("output_{output}"),
                seq,
                StrandRole::Output,
                0.0,
            )?);
        }
        Ok(design)
    }

    pub fn validate(&self, design: &DNACircuitDesign) -> Result<Vec<String>, String> {
        Ok(NupackInterface::new(self.temperature_c, 1.0)?.validate_design(design))
    }

    fn compile_displacement_gate(&mut self, spec: &GateSpec) -> Result<DNAGate, String> {
        match spec.gate_type {
            GateType::And => self.require_arity(spec, 2).and_then(|_| {
                self.displacement
                    .compile_and(&spec.inputs[0], &spec.inputs[1], &spec.output)
            }),
            GateType::Or => self.require_arity(spec, 2).and_then(|_| {
                self.displacement
                    .compile_or(&spec.inputs[0], &spec.inputs[1], &spec.output)
            }),
            GateType::Not => self
                .require_arity(spec, 1)
                .and_then(|_| self.displacement.compile_not(&spec.inputs[0], &spec.output)),
            GateType::Mux => self.require_arity(spec, 3).and_then(|_| {
                self.displacement.compile_mux(
                    &spec.inputs[0],
                    &spec.inputs[1],
                    &spec.inputs[2],
                    &spec.output,
                )
            }),
            GateType::Amplifier => self.require_arity(spec, 1).and_then(|_| {
                self.displacement
                    .compile_amplifier(&spec.inputs[0], &spec.output)
            }),
            GateType::Buffer => self.require_arity(spec, 1).and_then(|_| {
                self.displacement
                    .compile_buffer(&spec.inputs[0], &spec.output)
            }),
            GateType::Threshold => self.require_arity(spec, 1).and_then(|_| {
                self.displacement.compile_threshold(
                    &spec.inputs[0],
                    &spec.output,
                    spec.threshold.unwrap_or(0.5),
                )
            }),
            other => Err(format!("Unsupported displacement gate: {other:?}")),
        }
    }

    fn compile_enzymatic_gate(&mut self, spec: &GateSpec) -> Result<DNAGate, String> {
        match spec.gate_type {
            GateType::Nand => self.require_arity(spec, 2).and_then(|_| {
                self.enzymatic
                    .compile_nand(&spec.inputs[0], &spec.inputs[1], &spec.output)
            }),
            GateType::Xor => self.require_arity(spec, 2).and_then(|_| {
                self.enzymatic
                    .compile_xor(&spec.inputs[0], &spec.inputs[1], &spec.output)
            }),
            other => Err(format!("Unsupported enzymatic gate: {other:?}")),
        }
    }

    fn require_arity(&self, spec: &GateSpec, expected: usize) -> Result<(), String> {
        if spec.inputs.len() == expected {
            Ok(())
        } else {
            Err(format!(
                "{:?} requires {expected} inputs, got {}",
                spec.gate_type,
                spec.inputs.len()
            ))
        }
    }
}

pub fn validate_dna_mapper(state: &PlateLayout) -> bool {
    state.total_gates() == state.gates.len()
        && state.all_strands().iter().all(|strand| {
            strand
                .sequence
                .bytes()
                .all(|base| matches!(base, b'A' | b'C' | b'G' | b'T'))
        })
}

fn reverse_complement(sequence: &str) -> String {
    sequence
        .bytes()
        .rev()
        .map(|base| match base {
            b'A' => 'T',
            b'C' => 'G',
            b'G' => 'C',
            b'T' => 'A',
            _ => 'N',
        })
        .collect()
}

fn max_homopolymer_run(sequence: &str) -> usize {
    if sequence.is_empty() {
        return 0;
    }
    let mut max_run = 1usize;
    let mut current_run = 1usize;
    let bytes = sequence.as_bytes();
    for idx in 1..bytes.len() {
        if bytes[idx] == bytes[idx - 1] {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 1;
        }
    }
    max_run
}

fn nn_dg(pair: &str) -> f64 {
    match pair {
        "AA" | "TT" => -1.00,
        "AT" => -0.88,
        "TA" => -0.58,
        "CA" | "TG" => -1.45,
        "GT" | "AC" => -1.44,
        "CT" | "AG" => -1.28,
        "GA" | "TC" => -1.30,
        "CG" => -2.17,
        "GC" => -2.24,
        "GG" | "CC" => -1.84,
        _ => -1.0,
    }
}

fn nn_dh(pair: &str) -> f64 {
    match pair {
        "AA" | "TT" => -7.9,
        "AT" | "TA" => -7.2,
        "CA" | "TG" => -8.5,
        "GT" | "AC" => -8.4,
        "CT" | "AG" => -7.8,
        "GA" | "TC" => -8.2,
        "CG" => -10.6,
        "GC" => -9.8,
        "GG" | "CC" => -8.0,
        _ => -8.0,
    }
}

fn nn_ds(pair: &str) -> f64 {
    match pair {
        "AA" | "TT" => -22.2,
        "AT" => -20.4,
        "TA" => -21.3,
        "CA" | "TG" => -22.7,
        "GT" | "AC" => -22.4,
        "CT" | "AG" => -21.0,
        "GA" | "TC" => -22.2,
        "CG" => -27.2,
        "GC" => -24.4,
        "GG" | "CC" => -19.9,
        _ => -22.0,
    }
}

fn is_watson_crick(left: u8, right: u8) -> bool {
    matches!(
        (left, right),
        (b'A', b'T') | (b'T', b'A') | (b'C', b'G') | (b'G', b'C')
    )
}

fn weighted_base(state: &mut u64, weights: [f64; 4]) -> u8 {
    let total = weights.iter().sum::<f64>();
    let mut draw = next_f64(state) * total.max(f64::EPSILON);
    let bases = *b"ACGT";
    for (idx, weight) in weights.iter().enumerate() {
        draw -= *weight;
        if draw <= 0.0 && *weight > 0.0 {
            return bases[idx];
        }
    }
    bases
        .iter()
        .zip(weights)
        .find_map(|(base, weight)| (weight > 0.0).then_some(*base))
        .unwrap_or(b'A')
}

fn nuc_index(base: u8) -> usize {
    match base {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 0,
    }
}

fn next_f64(state: &mut u64) -> f64 {
    *state = splitmix64(*state);
    ((*state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dna_mapper_new() {
        let state = PlateLayout::new("empty", CompilationMethod::Displacement, 37.0);
        assert!(validate_dna_mapper(&state));
    }

    #[test]
    fn test_dna_strand_contracts_and_thermodynamics() {
        let strand = DNAStrand::new("x", "AACCCGT", StrandRole::Signal, 100.0).unwrap();
        assert_eq!(strand.length(), 7);
        assert!((strand.gc_content() - 4.0 / 7.0).abs() < 1e-12);
        assert_eq!(strand.complement(), "ACGGGTT");
        assert_eq!(strand.max_homopolymer_run(), 3);
        assert!(strand.delta_g_37() < 0.0);

        let gc = DNAStrand::new("gc", "GCGCGCGCGCGCGCGC", StrandRole::Signal, 100.0).unwrap();
        let at = DNAStrand::new("at", "ATATATATATATATAT", StrandRole::Signal, 100.0).unwrap();
        assert!(
            gc.melting_temperature(1.0, 2.5e-7).unwrap()
                > at.melting_temperature(1.0, 2.5e-7).unwrap()
        );
        assert!(strand.melting_temperature(0.0, 2.5e-7).is_err());
        assert!(DNAStrand::new("bad", "ACGX", StrandRole::Signal, 1.0).is_err());
    }

    #[test]
    fn test_sequence_designer_is_deterministic_and_constrained() {
        let mut a = SequenceDesigner::new(42, (0.40, 0.60), 3).unwrap();
        let mut b = SequenceDesigner::new(42, (0.40, 0.60), 3).unwrap();
        let sa = a.generate(32, "input_A").unwrap();
        let sb = b.generate(32, "input_A").unwrap();
        assert_eq!(sa, sb);
        let strand = DNAStrand::new("generated", &sa, StrandRole::Signal, 1.0).unwrap();
        assert!((0.40..=0.60).contains(&strand.gc_content()));
        assert!(strand.max_homopolymer_run() <= 3);
    }

    #[test]
    fn test_displacement_compiler_and_kinetic_truth_table() {
        let mut compiler =
            BitstreamToDNA::try_new(CompilationMethod::Displacement, 42, 37.0).unwrap();
        let design = compiler
            .compile_network(
                &[GateSpec::new(GateType::And, &["A", "B"], "C")],
                &["A", "B"],
                &["C"],
                "and_gate",
            )
            .unwrap();

        assert_eq!(design.total_gates(), 1);
        assert!(design.total_strands() > 0);
        assert!(design.total_nucleotides() > 0);
        assert!(design.validate().is_empty());

        let mut sim = KineticSimulator::new(3e5, 1.0, 37.0, Integrator::Euler).unwrap();
        let high = sim
            .simulate(&design, &[("A", 200.0), ("B", 200.0)], 1800.0, 1.0)
            .unwrap();
        let low = sim
            .simulate(&design, &[("A", 200.0), ("B", 0.0)], 1800.0, 1.0)
            .unwrap();
        assert!(high.final_concentration("C").unwrap() > 50.0);
        assert!(low.final_concentration("C").unwrap() < 50.0);
    }

    #[test]
    fn test_gate_suite_and_enzymatic_sites() {
        let mut displacement =
            StrandDisplacementCompiler::new(SequenceDesigner::default(), 37.0).unwrap();
        assert_eq!(
            displacement.compile_or("A", "B", "Y").unwrap().gate_type,
            GateType::Or
        );
        assert_eq!(
            displacement
                .compile_not("A", "Y")
                .unwrap()
                .input_names
                .len(),
            1
        );
        assert_eq!(
            displacement
                .compile_threshold("A", "Y", 0.7)
                .unwrap()
                .threshold,
            0.7
        );
        assert_eq!(
            displacement
                .compile_mux("S", "A", "B", "Y")
                .unwrap()
                .input_names
                .len(),
            3
        );
        assert!(displacement
            .compile_amplifier("A", "Y")
            .unwrap()
            .strands
            .iter()
            .any(|s| s.role == StrandRole::Fuel));
        assert_eq!(
            displacement.compile_buffer("A", "Y").unwrap().gate_type,
            GateType::Buffer
        );

        let mut enzymatic = EnzymaticGateCompiler::new(SequenceDesigner::default());
        let nand = enzymatic.compile_nand("A", "B", "Y").unwrap();
        assert_eq!(nand.gate_type, GateType::Nand);
        assert!(nand.strands[0].sequence.contains("GAATTC"));
        assert!(nand.strands[0].sequence.contains("GGATCC"));
        assert_eq!(
            enzymatic.compile_xor("A", "B", "Y").unwrap().gate_type,
            GateType::Xor
        );
    }

    #[test]
    fn test_thermodynamic_validator_cross_hybridisation() {
        let nupack = NupackInterface::new(37.0, 1.0).unwrap();
        let (energy, structure) = nupack.compute_mfe("GCGCGCGC").unwrap();
        assert!(energy < 0.0);
        assert_eq!(structure, "........");
        let probs = nupack.compute_pair_probabilities("ACGT").unwrap();
        assert_eq!(probs.len(), 4);
        assert!(probs.iter().all(|row| row.len() == 4));
    }
}
