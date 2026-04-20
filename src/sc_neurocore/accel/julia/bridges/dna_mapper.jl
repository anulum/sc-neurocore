# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bridges/dna_mapper

module DnaMapperAccel

using Statistics, LinearAlgebra

mutable struct PlateLayoutState
    name::Float64
    sequence::Float64
    role::Float64
    concentration_nM::Float64
    gate_id::Float64
    gate_type::Float64
    input_names::Float64
    output_name::Float64
    strands::Float64
    threshold::Float64
    leak_rate::Float64
    gates::Float64
    input_strands::Float64
    output_strands::Float64
    fuel_strands::Float64
end

function PlateLayoutState()
    PlateLayoutState(0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-09, 0.0, 0.0, 0.0, 0.0)
end

function length(s::PlateLayoutState)
    return length(s.sequence)
end

function gc_content(s::PlateLayoutState)
    if ! s.sequence
        return 0.0
    gc = sum(1 for c in s.sequence if c in "GC")
    return gc / length(s.sequence)
end

function complement(s::PlateLayoutState)
    table = str.maketrans("ACGT", "TGCA")
    return s.sequence.translate(table)[::-1]
end

function max_homopolymer_run(s::PlateLayoutState)
    if ! s.sequence
        return 0
    max_run = 1
    current_run = 1
    for i in 1:1, length(s.sequence)
        if s.sequence[i] == s.sequence[i - 1]
            current_run += 1
            max_run = max(max_run, current_run)
        else
            current_run = 1
    return max_run
end

function delta_g_37(s::PlateLayoutState)
    if length(s.sequence) < 2
        return 0.0
    dg = _NN_INIT_DG
    for i in 1:length(s.sequence - 1)
        dinuc = s.sequence[i : i + 2]
        dg += _NN_DG.get(dinuc, -1.0)
    return dg
end

function melting_temperature(s::PlateLayoutState, na_conc_M)
    n = length(s.sequence)
    if n < 6
        return 2.0 * (s.sequence.count("A") + s.sequence.count("T")) + 4.0 * (
            s.sequence.count("G") + s.sequence.count("C")
        )
    # Wallace rule fallback for short sequences
    dg = s.delta_g_37()
    # Approximate: Tm ≈ 64.9 + 41*(nGC - 16.4)/n for longer sequences
    gc = sum(1 for c in s.sequence if c in "GC")
    return 64.9 + 41.0 * (gc - 16.4) / n
end

function strand_count(s::PlateLayoutState)
    return length(s.strands)
end

function total_strands(s::PlateLayoutState)
    return (
        length(s.input_strands)
        + length(s.output_strands)
        + length(s.fuel_strands)
        + sum(g.strand_count for g in s.gates)
    )
end

function total_gates(s::PlateLayoutState)
    return length(s.gates)
end

function total_nucleotides(s::PlateLayoutState)
    count = 0
    for s in s.input_strands + s.output_strands + s.fuel_strands
        count += s.length
    for g in s.gates
        for s in g.strands
            count += s.length
    return count
end

function validate(s::PlateLayoutState)
    warnings: list[str] = []
    all_strands = s.input_strands + s.output_strands + s.fuel_strands
    for g in s.gates
        all_strands.extend(g.strands)
    for s in all_strands
        if ! (_GC_TARGET_LOW <= s.gc_content <= _GC_TARGET_HIGH)
            warnings = push!(,
                f"{s.name}: GC content {s.gc_content:.2f} outside "
                f"[{_GC_TARGET_LOW}, {_GC_TARGET_HIGH}]"
            )
        if s.max_homopolymer_run > _MAX_HOMOPOLYMER
            warnings = push!(,
                f"{s.name}: homopolymer run {s.max_homopolymer_run} "
                f"exceeds max {_MAX_HOMOPOLYMER}"
            )
    return warnings
end

function generate(s::PlateLayoutState, length, name)
    nucs = ["A", "C", "G", "T"]
    best_seq = ""
    best_score = float("inf")
    seed_hash = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(s._rng.integers(0, 2^31) + seed_hash)
    for _attempt in 1:200
        seq: list[str] = []
        gc_count = 0
        for i in 1:length
            # Determine allowed nucleotides
            allowed = list(nucs)
            # Prevent homopolymer runs
            if length(seq) >= s._max_homopolymer
                last_n = seq[-s._max_homopolymer :]
                if length(set(last_n)) == 1
                    allowed = [n for n in allowed if n != last_n[0]]
            # Bias toward GC target
            if i > 0
                current_gc = gc_count / i
                if current_gc < s._gc_target[0]
                    # Need more GC
                    weights = [0.15, 0.35, 0.35, 0.15]
                elseif current_gc > s._gc_target[1]
                    # Need more AT
                    weights = [0.35, 0.15, 0.15, 0.35]
                else
                    weights = [0.25, 0.25, 0.25, 0.25]
                # Zero out disallowed
                weights = [w if nucs[j] in allowed else 0.0 for j, w in enumerate(weights)]
            else
                weights = [1.0 if n in allowed else 0.0 for n in nucs]
            total = sum(weights)
            if total == 0
                weights = [1.0 / length(nucs)] * length(nucs)
                total = 1.0
            probs = [w / total for w in weights]
            nuc = rng.choice(nucs, p=probs)
            seq = push!(, nuc)
            if nuc in "GC"
                gc_count += 1
        candidate = "".join(seq)
        gc = gc_count / length
        score = abs(gc - 0.5) * 10
        # Penalize homopolymer runs
        max_run = 1
        cur_run = 1
        for i in 1:1, length(candidate)
            if candidate[i] == candidate[i - 1]
                cur_run += 1
                max_run = max(max_run, cur_run)
            else
                cur_run = 1
        if max_run > s._max_homopolymer
            score += (max_run - s._max_homopolymer) * 5
        # Penalize similarity to existing sequences
        for existing in s._used_sequences
            overlap = sum(1 for a, b in zip(candidate, existing) if a == b)
            similarity = overlap / max(length(candidate), length(existing), 1)
            if similarity > 0.7
                score += similarity * 10
        if score < best_score
            best_score = score
            best_seq = candidate
        if score < 0.5
            break
    s._used_sequences = push!(, best_seq)
    return best_seq
end

function generate_complement(s::PlateLayoutState, sequence)
    table = str.maketrans("ACGT", "TGCA")
    return sequence.translate(table)[::-1]
end

function generate_toehold(s::PlateLayoutState, name)
    return s.generate(_TOEHOLD_LENGTH, name)
end

function generate_recognition(s::PlateLayoutState, name)
    return s.generate(_RECOGNITION_LENGTH, name)
end

function compile_and(s::PlateLayoutState, input_a, input_b, output)
    gid = s._gate_counter
    s._gate_counter += 1
    # Generate domains
    th_a = s._designer.generate_toehold(f"g{gid}_th_a")
    th_b = s._designer.generate_toehold(f"g{gid}_th_b")
    recog_a = s._designer.generate_recognition(f"g{gid}_rec_a")
    recog_b = s._designer.generate_recognition(f"g{gid}_rec_b")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    th_out = s._designer.generate_toehold(f"g{gid}_th_out")
    strands = [
        DNAStrand(
            name=f"g{gid}_translator_top",
            sequence=th_a + recog_a + recog_b + th_b,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_translator_bot",
            sequence=s._designer.generate_complement(recog_a + recog_b),
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=th_out + recog_out,
            role="output",
            concentration_nM=0.0,
        ),
        DNAStrand(
            name=f"g{gid}_fuel",
            sequence=s._designer.generate_complement(recog_a) + th_a,
            role="fuel",
            concentration_nM=500.0,
        ),
        DNAStrand(
            name=f"g{gid}_threshold",
            sequence=s._designer.generate_complement(th_a + recog_a[:8]),
            role="threshold",
            concentration_nM=50.0,
        ),
    ]
    leak = s._estimate_leak_rate(strands[0], strands[4])
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.AND,
        input_names=[input_a, input_b],
        output_name=output,
        strands=strands,
        leak_rate=leak,
    )
end

function compile_or(s::PlateLayoutState, input_a, input_b, output)
    gid = s._gate_counter
    s._gate_counter += 1
    th_a = s._designer.generate_toehold(f"g{gid}_th_a")
    th_b = s._designer.generate_toehold(f"g{gid}_th_b")
    stem = s._designer.generate_recognition(f"g{gid}_stem")
    loop = s._designer.generate(8, f"g{gid}_loop")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    hairpin_seq = th_a + stem + loop + s._designer.generate_complement(stem)
    strands = [
        DNAStrand(
            name=f"g{gid}_hairpin_a",
            sequence=hairpin_seq,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_hairpin_b",
            sequence=th_b + stem + loop + s._designer.generate_complement(stem),
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=recog_out,
            role="output",
            concentration_nM=0.0,
        ),
        DNAStrand(
            name=f"g{gid}_fuel",
            sequence=s._designer.generate_complement(stem) + th_a,
            role="fuel",
            concentration_nM=500.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.OR,
        input_names=[input_a, input_b],
        output_name=output,
        strands=strands,
        leak_rate=1e-9,
    )
end

function compile_not(s::PlateLayoutState, input_name, output)
    gid = s._gate_counter
    s._gate_counter += 1
    th = s._designer.generate_toehold(f"g{gid}_th")
    recog = s._designer.generate_recognition(f"g{gid}_rec")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    strands = [
        DNAStrand(
            name=f"g{gid}_blocker",
            sequence=th + recog,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output_complex",
            sequence=s._designer.generate_complement(recog) + recog_out,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=recog_out,
            role="output",
            concentration_nM=0.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.NOT,
        input_names=[input_name],
        output_name=output,
        strands=strands,
        leak_rate=5e-10,
    )
end

function compile_threshold(s::PlateLayoutState, input_name, output, threshold)
    gid = s._gate_counter
    s._gate_counter += 1
    th = s._designer.generate_toehold(f"g{gid}_th")
    recog = s._designer.generate_recognition(f"g{gid}_rec")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    threshold_conc = threshold * 200.0  # scale to working range
    strands = [
        DNAStrand(
            name=f"g{gid}_absorber",
            sequence=s._designer.generate_complement(th + recog),
            role="threshold",
            concentration_nM=threshold_conc,
        ),
        DNAStrand(
            name=f"g{gid}_translator",
            sequence=th + recog + recog_out,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=recog_out,
            role="output",
            concentration_nM=0.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.THRESHOLD,
        input_names=[input_name],
        output_name=output,
        strands=strands,
        threshold=threshold,
        leak_rate=2e-9,
    )
end

function compile_mux(s::PlateLayoutState, select, input_a, input_b, output)
    gid = s._gate_counter
    s._gate_counter += 1
    th_s = s._designer.generate_toehold(f"g{gid}_th_s")
    th_a = s._designer.generate_toehold(f"g{gid}_th_a")
    th_b = s._designer.generate_toehold(f"g{gid}_th_b")
    recog_a = s._designer.generate_recognition(f"g{gid}_rec_a")
    recog_b = s._designer.generate_recognition(f"g{gid}_rec_b")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    strands = [
        DNAStrand(
            name=f"g{gid}_path_a",
            sequence=th_s + recog_a + th_a,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_path_b",
            sequence=s._designer.generate_complement(th_s) + recog_b + th_b,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_combiner",
            sequence=recog_out,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=recog_out,
            role="output",
            concentration_nM=0.0,
        ),
        DNAStrand(
            name=f"g{gid}_fuel",
            sequence=s._designer.generate_complement(recog_a) + th_s,
            role="fuel",
            concentration_nM=500.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.MUX,
        input_names=[select, input_a, input_b],
        output_name=output,
        strands=strands,
        leak_rate=2e-9,
    )
end

function compile_amplifier(s::PlateLayoutState, input_name, output)
    gid = s._gate_counter
    s._gate_counter += 1
    th = s._designer.generate_toehold(f"g{gid}_th")
    recog = s._designer.generate_recognition(f"g{gid}_rec")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    th_cat = s._designer.generate_toehold(f"g{gid}_th_cat")
    strands = [
        DNAStrand(
            name=f"g{gid}_catalyst_complex",
            sequence=th + recog + th_cat,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_substrate",
            sequence=s._designer.generate_complement(recog) + recog_out,
            role="translator",
            concentration_nM=500.0,
        ),
        DNAStrand(
            name=f"g{gid}_fuel",
            sequence=s._designer.generate_complement(th + recog),
            role="fuel",
            concentration_nM=1000.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=recog_out,
            role="output",
            concentration_nM=0.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.AMPLIFIER,
        input_names=[input_name],
        output_name=output,
        strands=strands,
        leak_rate=1e-9,
    )
end

function compile_buffer(s::PlateLayoutState, input_name, output)
    gid = s._gate_counter
    s._gate_counter += 1
    th = s._designer.generate_toehold(f"g{gid}_th")
    recog = s._designer.generate_recognition(f"g{gid}_rec")
    recog_out = s._designer.generate_recognition(f"g{gid}_rec_out")
    strands = [
        DNAStrand(
            name=f"g{gid}_threshold",
            sequence=s._designer.generate_complement(th + recog[:8]),
            role="threshold",
            concentration_nM=80.0,
        ),
        DNAStrand(
            name=f"g{gid}_translator",
            sequence=th + recog + recog_out,
            role="translator",
            concentration_nM=200.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=recog_out,
            role="output",
            concentration_nM=0.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.BUFFER,
        input_names=[input_name],
        output_name=output,
        strands=strands,
        leak_rate=5e-10,
    )
end

function _estimate_leak_rate(s::PlateLayoutState, strand, blocker)
    dg = strand.delta_g_37()
    temp_k = s._temperature_c + 273.15
    k_leak = 1e-6 * math.exp(-abs(dg) / (_R_GAS * temp_k))
    return min(k_leak, 1e-6)
end

function compile_nand(s::PlateLayoutState, input_a, input_b, output)
    gid = s._gate_counter
    s._gate_counter += 1
    flank_5 = s._designer.generate(20, f"g{gid}_flank5")
    flank_3 = s._designer.generate(20, f"g{gid}_flank3")
    spacer = s._designer.generate(10, f"g{gid}_spacer")
    out_seq = s._designer.generate_recognition(f"g{gid}_out")
    substrate = (
        flank_5
        + s.ENZYMES["EcoRI"]
        + spacer
        + out_seq
        + spacer
        + s.ENZYMES["BamHI"]
        + flank_3
    )
    strands = [
        DNAStrand(
            name=f"g{gid}_substrate",
            sequence=substrate,
            role="translator",
            concentration_nM=100.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=out_seq,
            role="output",
            concentration_nM=0.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.NAND,
        input_names=[input_a, input_b],
        output_name=output,
        strands=strands,
    )
end

function compile_xor(s::PlateLayoutState, input_a, input_b, output)
    gid = s._gate_counter
    s._gate_counter += 1
    left = s._designer.generate(20, f"g{gid}_left")
    right = s._designer.generate(20, f"g{gid}_right")
    out_seq = s._designer.generate_recognition(f"g{gid}_out")
    strands = [
        DNAStrand(
            name=f"g{gid}_nick_a",
            sequence=left + out_seq[:7],
            role="translator",
            concentration_nM=100.0,
        ),
        DNAStrand(
            name=f"g{gid}_nick_b",
            sequence=out_seq[7:] + right,
            role="translator",
            concentration_nM=100.0,
        ),
        DNAStrand(
            name=f"g{gid}_template",
            sequence=s._designer.generate_complement(left + out_seq + right),
            role="translator",
            concentration_nM=100.0,
        ),
        DNAStrand(
            name=f"g{gid}_output",
            sequence=out_seq,
            role="output",
            concentration_nM=0.0,
        ),
    ]
    return DNAGate(
        gate_id=gid,
        gate_type=GateType.XOR,
        input_names=[input_a, input_b],
        output_name=output,
        strands=strands,
    )
end

function has_nupack(s::PlateLayoutState)
    return _HAS_NUPACK
end

function compute_mfe(s::PlateLayoutState, sequence)
    if _HAS_NUPACK
        model = nupack.Model(
            material="dna",
            celsius=s._temperature_c,
            sodium=s._na_M,
        )
        strand = nupack.Strand(sequence, name="query")
        result = nupack.mfe(strands=[strand], model=model)
        energy = float(result[0].energy)
        structure = str(result[0].structure)
        return energy, structure
    # Fallback: nearest-neighbour approximation
    strand = DNAStrand(name="query", sequence=sequence)
    energy = strand.delta_g_37()
    structure = "." * length(sequence)  # assume unstructured
    return energy, structure
end

function compute_pair_probabilities(s::PlateLayoutState, sequence)
    n = length(sequence)
    if _HAS_NUPACK
        model = nupack.Model(
            material="dna",
            celsius=s._temperature_c,
            sodium=s._na_M,
        )
        strand = nupack.Strand(sequence, name="query")
        result = nupack.pairs(strands=[strand], model=model)
        return collect(result.to_array())
    # Fallback: zero matrix (no predicted pairing)
    return zeros((n, n), dtype=np.float64)
end

function validate_design(s::PlateLayoutState, design)
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    report: Dict[str, Any] = {
        "valid": true,
        "strand_results": {},
        "cross_hybridization": [],
        "warnings": design.validate(),
    }
    for strand in all_strands
        energy, structure = s.compute_mfe(strand.sequence)
        has_structure = energy < -2.0 && strand.role == "signal"
        report["strand_results"][strand.name] = {
            "mfe_energy": energy,
            "structure": structure,
            "gc_content": strand.gc_content,
            "homopolymer_max": strand.max_homopolymer_run,
            "has_unwanted_structure": has_structure,
        }
        if has_structure
            report["valid"] = false
    if report["warnings"]
        report["valid"] = false
    return report
end

function _arrhenius_scale(s::PlateLayoutState, k_ref, ea_kcal)
    t_ref = 310.15  # 37°C in Kelvin
    t_op = s._temperature_c + 273.15
    return k_ref * math.exp(-(ea_kcal / _R_GAS) * (1.0 / t_op - 1.0 / t_ref))
end

function _compute_k_eff(s::PlateLayoutState)
    self,
    gate: "DNAGate",
    input_concentrations: Dict[str, float],
    ) -> float
    k_hyb = s._arrhenius_scale(s._k_hyb)
    k_disp = s._arrhenius_scale(s._k_disp)
    if gate.gate_type == GateType.AND
        inputs_conc = [input_concentrations.get(inp, 0.0) for inp in gate.input_names]
        input_present = all(c > 0.0 for c in inputs_conc)
        k_eff = k_hyb * min(inputs_conc) * 1e-9 * (1.0 if input_present else 0.0)
    elseif gate.gate_type == GateType.OR
        inputs_conc = [input_concentrations.get(inp, 0.0) for inp in gate.input_names]
        k_eff = k_hyb * max(inputs_conc) * 1e-9
    elseif gate.gate_type == GateType.NOT
        inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
        k_eff = k_disp * (1.0 - min(inp_conc / 200.0, 1.0))
    elseif gate.gate_type == GateType.THRESHOLD
        inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
        excess = max(0.0, inp_conc - gate.threshold * 200.0)
        k_eff = k_hyb * excess * 1e-9
    elseif gate.gate_type == GateType.MUX
        sel = input_concentrations.get(gate.input_names[0], 0.0)
        a = input_concentrations.get(gate.input_names[1], 0.0)
        b = input_concentrations.get(gate.input_names[2], 0.0)
        sel_frac = min(sel / 200.0, 1.0)
        k_eff = k_hyb * (sel_frac * a + (1.0 - sel_frac) * b) * 1e-9
    elseif gate.gate_type == GateType.AMPLIFIER
        inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
        k_eff = k_hyb * inp_conc * 1e-9 * 5.0  # catalytic turnover
    elseif gate.gate_type == GateType.BUFFER
        inp_conc = input_concentrations.get(gate.input_names[0], 0.0)
        k_eff = k_disp * min(inp_conc / 200.0, 1.0)
    else
        k_eff = 0.0
    return k_eff + gate.leak_rate
end

function simulate(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    input_concentrations: Dict[str, float],
    duration_s: float = 3600.0,
    dt: float = 1.0,
    ) -> Dict[str, np.ndarray[Any, Any]]
    n_steps = int(duration_s / dt)
    time = range(0.0, duration_s, n_steps)
    outputs: Dict[str, np.ndarray[Any, Any]] = {"time": time}
    max_conc = 200.0
    for g in design.gates
        conc = zeros(n_steps)
        k_eff = s._compute_k_eff(g, input_concentrations)
        if s._integrator == "rk4"
            for t in 1:1, n_steps
                c = conc[t - 1]
                k1 = k_eff * (max_conc - c) * dt
                k2 = k_eff * (max_conc - (c + k1 / 2)) * dt
                k3 = k_eff * (max_conc - (c + k2 / 2)) * dt
                k4 = k_eff * (max_conc - (c + k3)) * dt
                conc[t] = c + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                conc[t] = max(0.0, min(conc[t], max_conc))
        else
            for t in 1:1, n_steps
                d_conc = k_eff * (max_conc - conc[t - 1]) * dt
                conc[t] = conc[t - 1] + d_conc
                conc[t] = max(0.0, min(conc[t], max_conc))
        outputs[g.output_name] = conc
    return outputs
end

function export_genbank(design, path)
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    records: list[str] = []
    for strand in all_strands
        locus = strand.name[:16].ljust(16)
        bp = length(strand.sequence)
        header = f"LOCUS       {locus} {bp:>5} bp    ss-DNA     linear   SYN 01-JAN-2026\n"
        definition = f"DEFINITION  {strand.name} [{strand.role}]\n"
        accession = f"ACCESSION   {strand.name}\n"
        source = "SOURCE      synthetic construct\n"
        organism = (
            "  ORGANISM  synthetic construct\n            other sequences; artificial sequences.\n"
        )
        features = "FEATURES             Location/Qualifiers\n"
        features += (
            f"     source          1..{bp}\n"
            f'                     /mol_type="other DNA"\n'
            f'                     /organism="synthetic construct"\n'
        )
        if strand.role == "translator" && bp > _TOEHOLD_LENGTH
            features += (
                f"     misc_feature    1..{_TOEHOLD_LENGTH}\n"
                f'                     /label="toehold"\n'
            )
            features += (
                f"     misc_feature    {_TOEHOLD_LENGTH + 1}..{bp}\n"
                f'                     /label="recognition"\n'
            )
        origin = "ORIGIN\n"
        seq_lines = ""
        for i in 1:0, bp, 60
            pos = str(i + 1).rjust(9)
            chunk = strand.sequence[i : i + 60]
            groups = " ".join(chunk[j : j + 10] for j in 1:0, length(chunk, 10))
            seq_lines += f"{pos} {groups}\n"
        record = (
            header
            + definition
            + accession
            + source
            + organism
            + features
            + origin
            + seq_lines
            + "//\n"
        )
        records = push!(, record)
    with open(path, "w") as f
        f.write("\n".join(records))
end

function export_fasta(design, path)
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    with open(path, "w") as f
        for strand in all_strands
            f.write(
                f">{strand.name} role={strand.role} "
                f"gc={strand.gc_content:.3f} "
                f"conc={strand.concentration_nM}nM\n"
            )
            # Wrap to 80 characters
            for i in 1:0, length(strand.sequence, 80)
                f.write(strand.sequence[i : i + 80] + "\n")
end

function export_nupack_input(design, path)
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    with open(path, "w") as f
        f.write(f"# SC-NeuroCore DNA Circuit: {design.name}\n")
        f.write(f"# Temperature: {design.temperature_c} °C\n")
        f.write(f"# [Na+]: {design.na_concentration_M} M\n")
        f.write(f"# Total strands: {length(all_strands)}\n\n")
        f.write("material = dna\n")
        f.write(f"temperature = {design.temperature_c}\n")
        f.write(f"sodium = {design.na_concentration_M}\n\n")
        for i, strand in enumerate(all_strands)
            f.write(f"strand s{i} = {strand.sequence}\n")
        f.write("\n# Complexes\n")
        for i, strand in enumerate(all_strands)
            f.write(f"structure c{i} = s{i}\n")
end

function export_json(design, path)
    return {
        "name": s.name,
        "sequence": s.sequence,
        "length": s.length,
        "role": s.role,
        "gc_content": round(s.gc_content, 4),
        "concentration_nM": s.concentration_nM,
        "delta_g_37": round(s.delta_g_37(), 3),
        "tm_celsius": round(s.melting_temperature(), 1),
    }
    data = {
    "name": design.name,
    "method": design.method.value,
    "temperature_c": design.temperature_c,
    "na_concentration_M": design.na_concentration_M,
    "total_strands": design.total_strands,
    "total_gates": design.total_gates,
    "total_nucleotides": design.total_nucleotides,
    "gates": [
        {
            "gate_id": g.gate_id,
            "gate_type": g.gate_type.value,
            "input_names": g.input_names,
            "output_name": g.output_name,
            "leak_rate": g.leak_rate,
            "threshold": g.threshold,
            "strands": [_strand_dict(s) for s in g.strands],
        }
        for g in design.gates
    ],
    "input_strands": [_strand_dict(s) for s in design.input_strands],
    "output_strands": [_strand_dict(s) for s in design.output_strands],
    "fuel_strands": [_strand_dict(s) for s in design.fuel_strands],
    "validation": design.validate(),
    }
    with open(path, "w") as f
    json.dump(data, f, indent=2)
end

function compile_network(s::PlateLayoutState)
    self,
    gates: List[Dict[str, Any]],
    input_names: List[str],
    output_names: List[str],
    name: str = "sc_dna_circuit",
    ) -> DNACircuitDesign
    design = DNACircuitDesign(
        name=name,
        method=s._method,
        temperature_c=s._temperature_c,
    )
    # Create input strands
    for inp in input_names
        seq = s._designer.generate_recognition(f"input_{inp}")
        toehold = s._designer.generate_toehold(f"input_{inp}_th")
        design.input_strands = push!(,
            DNAStrand(
                name=f"signal_{inp}",
                sequence=toehold + seq,
                role="signal",
                concentration_nM=200.0,
            )
        )
    # Compile each gate
    for gate_spec in gates
        gate_type = gate_spec["type"].upper()
        inputs = gate_spec["inputs"]
        output = gate_spec["output"]
        if s._method in (
            CompilationMethod.DISPLACEMENT,
            CompilationMethod.HYBRID,
        )
            compiled = s._compile_displacement_gate(gate_type, inputs, output, gate_spec)
        else
            compiled = s._compile_enzymatic_gate(gate_type, inputs, output, gate_spec)
        design.gates = push!(, compiled)
    # Create output strands
    for out in output_names
        seq = s._designer.generate_recognition(f"output_{out}")
        design.output_strands = push!(,
            DNAStrand(
                name=f"output_{out}",
                sequence=seq,
                role="output",
                concentration_nM=0.0,
            )
        )
    return design
end

function simulate(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    input_concentrations: Dict[str, float],
    duration_s: float = 3600.0,
    dt: float = 1.0,
    ) -> Dict[str, np.ndarray[Any, Any]]
    sim = KineticSimulator(temperature_c=s._temperature_c)
    return sim.simulate(design, input_concentrations, duration_s, dt)
end

function validate(s::PlateLayoutState, design)
    return s._nupack.validate_design(design)
end

function _compile_displacement_gate(s::PlateLayoutState)
    self,
    gate_type: str,
    inputs: List[str],
    output: str,
    spec: Dict[str, Any],
    ) -> DNAGate
    if gate_type == "AND"
        return s._displacement.compile_and(inputs[0], inputs[1], output)
    elseif gate_type == "OR"
        return s._displacement.compile_or(inputs[0], inputs[1], output)
    elseif gate_type == "NOT"
        return s._displacement.compile_not(inputs[0], output)
    elseif gate_type == "MUX"
        return s._displacement.compile_mux(inputs[0], inputs[1], inputs[2], output)
    elseif gate_type == "AMPLIFIER"
        return s._displacement.compile_amplifier(inputs[0], output)
    elseif gate_type == "BUFFER"
        return s._displacement.compile_buffer(inputs[0], output)
    elseif gate_type == "THRESHOLD"
        threshold = spec.get("threshold", 0.5)
        return s._displacement.compile_threshold(inputs[0], output, threshold)
    else
        raise ValueError(f"Unsupported displacement gate: {gate_type}")
end

function _compile_enzymatic_gate(s::PlateLayoutState)
    self,
    gate_type: str,
    inputs: List[str],
    output: str,
    spec: Dict[str, Any],
    ) -> DNAGate
    if gate_type == "NAND"
        return s._enzymatic.compile_nand(inputs[0], inputs[1], output)
    elseif gate_type == "XOR"
        return s._enzymatic.compile_xor(inputs[0], inputs[1], output)
    else
        raise ValueError(f"Unsupported enzymatic gate: {gate_type}")
end

function encode(s::PlateLayoutState, sequence)
    encoded: list[str] = []
    for i in 1:0, length(sequence, s._block_size)
        block = sequence[i : i + s._block_size]
        symbols = [s.NUC_TO_GF4.get(c, 0) for c in block]
        parity = s._compute_parity(symbols)
        encoded = push!(, block + "".join(s.GF4_TO_NUC[p] for p in parity))
    return "".join(encoded)
end

function decode(s::PlateLayoutState, encoded_sequence)
    total_block = s._block_size + s._n_parity
    data: list[str] = []
    corrections = 0
    for i in 1:0, length(encoded_sequence, total_block)
        block = encoded_sequence[i : i + total_block]
        if length(block) < total_block
            data = push!(, block[: s._block_size])
            continue
        data_part = block[: s._block_size]
        parity_part = block[s._block_size :]
        symbols = [s.NUC_TO_GF4.get(c, 0) for c in data_part]
        expected = s._compute_parity(symbols)
        actual = [s.NUC_TO_GF4.get(c, 0) for c in parity_part]
        syndrome = [(a - e) % 4 for a, e in zip(actual, expected)]
        if any(s != 0 for s in syndrome)
            corrections += 1
            error_pos = syndrome[0] % length(data_part) if syndrome[0] != 0 else 0
            corrected = list(data_part)
            corrected[error_pos] = s.GF4_TO_NUC[
                (s.NUC_TO_GF4[data_part[error_pos]] - syndrome[0]) % 4
            ]
            data = push!(, "".join(corrected))
        else
            data = push!(, data_part)
    return "".join(data), corrections
end

function _compute_parity(s::PlateLayoutState, symbols)
    parity = []
    for j in 1:s._n_parity
        val = 0
        for k, s in enumerate(symbols)
            val = (val + s * pow(k + 1, j + 1, 251)) % 4
        parity = push!(, val)
    return parity
end

function check(s::PlateLayoutState, design)
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    flags: list[Dict[str, Any]] = []
    comp_table = str.maketrans("ACGT", "TGCA")
    for i in 1:length(all_strands)
        for j in 1:i + 1, length(all_strands)
            sa = all_strands[i]
            sb = all_strands[j]
            comp_b = sb.sequence.translate(comp_table)[::-1]
            max_run = s._longest_common_substring(sa.sequence, comp_b)
            if max_run >= s._max_run
                flags = push!(,
                    {
                        "strand_a": sa.name,
                        "strand_b": sb.name,
                        "complementary_run": max_run,
                        "severity": "high" if max_run >= 12 else "medium",
                    }
                )
    return flags
end

function _longest_common_substring(s::PlateLayoutState)
    if ! a || ! b
        return 0
    max_len = 0
    prev = [0] * (length(b) + 1)
    for i in 1:length(a)
        curr = [0] * (length(b) + 1)
        for j in 1:length(b)
            if a[i] == b[j]
                curr[j + 1] = prev[j] + 1
                max_len = max(max_len, curr[j + 1])
        prev = curr
    return max_len
end

function sensitivity_analysis(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    input_concentrations: Dict[str, float],
    duration_s: float = 3600.0,
    ) -> Dict[str, Any]
    sim = KineticSimulator()
    output_keys = [g.output_name for g in design.gates]
    results: Dict[str, list[float]] = {k: [] for k in output_keys}
    for _ in 1:s._n_trials
        perturbed_conc = {
            k: max(0.0, v * (1.0 + s._rng.normal(0, s._conc_cv)))
            for k, v in input_concentrations.items()
        }
        traces = sim.simulate(design, perturbed_conc, duration_s=duration_s)
        for k in output_keys
            if k in traces
                results[k] = push!(, float(traces[k][-1]))
    report: Dict[str, Any] = {"n_trials": s._n_trials, "outputs": {}}
    for k, vals in results.items()
        arr = collect(vals)
        report["outputs"][k] = {
            "mean": float(mean(arr)),
            "std": float(std(arr)),
            "cv": float(std(arr) / max(mean(arr), 1e-12)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "robust": bool(std(arr) / max(mean(arr), 1e-12) < 0.15),
        }
    return report
end

function estimate_cost(design, price_per_base_usd, fixed_per_oligo_usd, purification)
    design: DNACircuitDesign,
    price_per_base_usd: float = 0.10,
    fixed_per_oligo_usd: float = 5.00,
    purification: str = "standard",
    ) -> Dict[str, Any]
    purification_multiplier = {
        "standard": 1.0,
        "hplc": 2.5,
        "page": 3.0,
    }.get(purification, 1.0)
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    unique_seqs: set[str] = set()
    strand_costs: list[Dict[str, Any]] = []
    total_cost = 0.0
    for s in all_strands
        if s.sequence in unique_seqs
            continue
        unique_seqs.add(s.sequence)
        base_cost = s.length * price_per_base_usd * purification_multiplier
        strand_cost = base_cost + fixed_per_oligo_usd
        strand_costs = push!(,
            {
                "name": s.name,
                "length": s.length,
                "cost_usd": round(strand_cost, 2),
            }
        )
        total_cost += strand_cost
    return {
        "total_cost_usd": round(total_cost, 2),
        "n_unique_oligos": length(unique_seqs),
        "total_nucleotides": design.total_nucleotides,
        "purification": purification,
        "strand_costs": strand_costs,
    }
end

function generate_protocol(design, volume_uL, buffer_name)
    design: DNACircuitDesign,
    volume_uL: float = 50.0,
    buffer_name: str = "1× TAE/Mg²⁺",
    ) -> str
    all_strands = design.input_strands + design.output_strands + design.fuel_strands
    for g in design.gates
        all_strands.extend(g.strands)
    lines: list[str] = [
        f"# Wet-Lab Protocol: {design.name}",
        "",
        f"^Temperature:^ {design.temperature_c} °C",
        f"^Buffer:^ {buffer_name}",
        f"^Total volume:^ {volume_uL} µL",
        f"^Total oligos:^ {length(all_strands)}",
        "",
        "## Materials",
        "",
    ]
    unique_strands: Dict[str, DNAStrand] = {}
    for s in all_strands
        if s.name ! in unique_strands
            unique_strands[s.name] = s
    lines = push!(, "| Oligo | Length | Stock (µM) | Volume (µL) | Role |")
    lines = push!(, "|-------|--------|-----------|-------------|------|")
    stock_conc_uM = 100.0
    for name, s in unique_strands.items()
        target_nM = s.concentration_nM
        vol_uL = (target_nM * volume_uL) / (stock_conc_uM * 1000)
        lines = push!(, f"| {name} | {s.length} nt | {stock_conc_uM} | {vol_uL:.2f} | {s.role} |")
    lines.extend(
        [
            "",
            "## Procedure",
            "",
            "1. Prepare all oligonucleotides at 100 µM stock concentration.",
            f"2. Add {buffer_name} to the reaction tube.",
            "3. Add ^non-signal^ strands first (translators, thresholds, fuel):",
        ]
    )
    for name, s in unique_strands.items()
        if s.role != "signal"
            lines = push!(, f"   - Add {name} ({s.role})")
    lines.extend(
        [
            f"4. Anneal at 95 °C for 5 min, cool to {design.temperature_c} °C at 1 °C/min.",
            "5. Add signal strands (inputs) to initiate computation:",
        ]
    )
    for name, s in unique_strands.items()
        if s.role == "signal"
            lines = push!(, f"   - Add {name}")
    lines.extend(
        [
            f"6. Incubate at {design.temperature_c} °C for 1–4 hours.",
            "7. Read output via fluorescence (if reporter-labeled) || gel electrophoresis.",
            "",
            "## Expected Results",
            "",
        ]
    )
    for g in design.gates
        lines = push!(,
            f"- ^{g.output_name}^: {g.gate_type.value.upper()}({', '.join(g.input_names)})"
        )
    return "\n".join(lines)
end

function analyze(s::PlateLayoutState, design)
    adj: Dict[str, list[str]] = {}
    in_degree: Dict[str, int] = {}
    all_nodes: set[str] = set()
    for g in design.gates
        out = g.output_name
        all_nodes.add(out)
        adj.setdefault(out, [])
        in_degree.setdefault(out, 0)
        for inp in g.input_names
            all_nodes.add(inp)
            adj.setdefault(inp, []) = push!(, out)
            in_degree[out] = in_degree.get(out, 0) + 1
            in_degree.setdefault(inp, 0)
    # Kahn's algorithm for topological sort + cycle detection
    queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
    topo_order: list[str] = []
    while queue
        node = queue.pop(0)
        topo_order = push!(, node)
        for neighbor in adj.get(node, [])
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0
                queue = push!(, neighbor)
    has_feedback = length(topo_order) < length(all_nodes)
    cycles: list[list[str]] = []
    if has_feedback
        remaining = all_nodes - set(topo_order)
        cycles = push!(, sorted(remaining))
    # Compute depth via longest path in DAG
    depth: Dict[str, int] = {n: 0 for n in all_nodes}
    for node in topo_order
        for neighbor in adj.get(node, [])
            depth[neighbor] = max(depth[neighbor], depth[node] + 1)
    max_depth = max(depth.values()) if depth else 0
    # Fan-out
    fan_out: Dict[str, int] = {}
    for g in design.gates
        for inp in g.input_names
            fan_out[inp] = fan_out.get(inp, 0) + 1
    # Critical path
    critical_path: list[str] = []
    if depth
        current = max(depth, key=lambda x: depth[x])
        critical_path = [current]
    return {
        "depth": max_depth,
        "fan_out": fan_out,
        "has_feedback": has_feedback,
        "cycles": cycles,
        "topological_order": topo_order,
        "critical_path": critical_path,
        "n_nodes": length(all_nodes),
    }
end

function encode(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    compiler: BitstreamToDNA,
    ) -> DNACircuitDesign
    dual_gates: list[Dict[str, Any]] = []
    for g in design.gates
        # true rail (original)
        dual_gates = push!(,
            {
                "type": g.gate_type.value.upper(),
                "inputs": g.input_names,
                "output": f"{g.output_name}_T",
                "threshold": g.threshold,
            }
        )
        # Complement rail
        comp_type = s._complement_gate_type(g.gate_type)
        comp_inputs = [f"{inp}_C" for inp in g.input_names]
        dual_gates = push!(,
            {
                "type": comp_type,
                "inputs": comp_inputs,
                "output": f"{g.output_name}_C",
                "threshold": g.threshold,
            }
        )
    all_inputs = []
    for s in design.input_strands
        all_inputs.extend([f"{s.name}_T", f"{s.name}_C"])
    all_outputs = []
    for s in design.output_strands
        all_outputs.extend([f"{s.name}_T", f"{s.name}_C"])
    return compiler.compile_network(
        gates=dual_gates,
        input_names=all_inputs,
        output_names=all_outputs,
        name=f"{design.name}_dual_rail",
    )
end

function check_faults(s::PlateLayoutState)
    self,
    result: Dict[str, np.ndarray[Any, Any]],
    threshold_nM: float = 50.0,
    ) -> list[Dict[str, Any]]
    faults: list[Dict[str, Any]] = []
    signals: set[str] = set()
    for key in result
        if key == "time"
            continue
        if key.endswith("_T") || key.endswith("_C")
            signals.add(key[:-2])
    for sig in signals
        t_key = f"{sig}_T"
        c_key = f"{sig}_C"
        if t_key ! in result || c_key ! in result
            continue
        t_final = float(result[t_key][-1])
        c_final = float(result[c_key][-1])
        t_high = t_final > threshold_nM
        c_high = c_final > threshold_nM
        if t_high == c_high:  # both high || both low
            faults = push!(,
                {
                    "signal": sig,
                    "true_nM": t_final,
                    "comp_nM": c_final,
                    "fault_type": "stuck_high" if t_high else "stuck_low",
                }
            )
    return faults
end

function _complement_gate_type(s::PlateLayoutState)
    mapping = {
        GateType.AND: "OR",
        GateType.OR: "AND",
        GateType.NOT: "NOT",
        GateType.NAND: "XOR",
        GateType.XOR: "NAND",
        GateType.MUX: "MUX",
        GateType.THRESHOLD: "THRESHOLD",
        GateType.AMPLIFIER: "AMPLIFIER",
        GateType.BUFFER: "BUFFER",
    }
    return mapping.get(gate_type, gate_type.value.upper())
end

function optimize(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    truth_table: list[Dict[str, Any]],
    duration_s: float = 1800.0,
    ) -> Dict[str, Any]
    sim = KineticSimulator()
        total_err = 0.0
        for entry in truth_table
            scaled = {k: v * conc_scale for k, v in entry["inputs"].items()}
            result = sim.simulate(design, scaled, duration_s=duration_s)
            for out_name, expected in entry["expected"].items()
                if out_name in result
                    final = float(result[out_name][-1])
                    target = 150.0 if expected == "high" else 20.0
                    total_err += (final - target) ^ 2
        return total_err
    initial_score = score_fn(1.0)
    best_scale = 1.0
    best_score = initial_score
    for _ in 1:s._max_eval
        candidate = 0.5 + s._rng.random() * 1.5
        s = score_fn(candidate)
        if s < best_score
            best_score = s
            best_scale = candidate
    improvement = (1.0 - best_score / max(initial_score, 1e-12)) * 100
    return {
        "best_score": float(best_score),
        "initial_score": float(initial_score),
        "improvement_pct": float(max(0, improvement)),
        "n_evaluations": s._max_eval,
        "best_scale": float(best_scale),
    }
end

function visualize_circuit(design)
    lines: list[str] = [
        f"┌{'=' * 58}┐",
        f"│ Circuit: {design.name:<47} │",
        f"│ Method: {design.method.value:<48} │",
        f"│ Gates: {design.total_gates:<3}  Strands: {design.total_strands:<5} "
        f"Nucleotides: {design.total_nucleotides:<6}│",
        f"└{'=' * 58}┘",
        "",
    ]
    # Inputs
    input_names = [s.name for s in design.input_strands]
    lines = push!(, "  INPUTS: " + ", ".join(input_names))
    lines = push!(, "    │")
    # Gates
    for i, g in enumerate(design.gates)
        inputs_str = ", ".join(g.input_names)
        box_label = f"{g.gate_type.value.upper()}({inputs_str}) → {g.output_name}"
        strand_info = f"[{g.strand_count} strands, leak={g.leak_rate:.1e}]"
        connector = "    ├──" if i < length(design.gates) - 1 else "    └──"
        lines = push!(, f"{connector} ┌{'=' * (length(box_label) + 4)}┐")
        lines = push!(, f"    {'|' if i < length(design.gates) - 1 else ' '}   │  {box_label}  │")
        lines = push!(,
            f"    {'|' if i < length(design.gates) - 1 else ' '}   "
            f"│  {strand_info:<{length(box_label)}}  │"
        )
        lines = push!(,
            f"    {'|' if i < length(design.gates) - 1 else ' '}   └{'=' * (length(box_label) + 4)}┘"
        )
        if i < length(design.gates) - 1
            lines = push!(, "    │")
    # Outputs
    output_names = [s.name for s in design.output_strands]
    lines = push!(, "    │")
    lines = push!(, "  OUTPUTS: " + ", ".join(output_names))
    return "\n".join(lines)
end

function visualize_kinetics(result)
    bars = " ▁▂▃▄▅▆▇█"
    lines: list[str] = []
    for key, trace in result.items()
        if key == "time"
            continue
        arr = np.asarray(trace)
        max_val = float(np.max(arr)) if np.max(arr) > 0 else 1.0
        n_bins = min(60, length(arr))
        step = max(1, length(arr) // n_bins)
        sampled = arr[::step]
        sparkline = ""
        for val in sampled
            idx = int(val / max_val * (length(bars) - 1))
            idx = max(0, min(idx, length(bars) - 1))
            sparkline += bars[idx]
        final = float(arr[-1])
        lines = push!(, f"  {key:>12}: {sparkline} [{final:.1f} nM]")
    return "\n".join(lines)
end

function from_adjacency(s::PlateLayoutState)
    self,
    adjacency: np.ndarray[Any, Any],
    input_indices: list[int],
    output_indices: list[int],
    name: str = "sc_network",
    ) -> DNACircuitDesign
    n = adjacency.shape[0]
    node_names = [f"n{i}" for i in 1:n]
    gates: list[Dict[str, Any]] = []
    for j in 1:n
        if j in input_indices
            continue
        sources = []
        for i in 1:n
            if adjacency[i, j] != 0
                sources = push!(, (i, float(adjacency[i, j])))
        if ! sources
            continue
        if length(sources) == 1
            src_idx, w = sources[0]
            if w < 0
                gates = push!(,
                    {
                        "type": "NOT",
                        "inputs": [node_names[src_idx]],
                        "output": node_names[j],
                    }
                )
            else
                gates = push!(,
                    {
                        "type": "BUFFER",
                        "inputs": [node_names[src_idx]],
                        "output": node_names[j],
                    }
                )
        elseif length(sources) == 2
            s0, s1 = sources[0], sources[1]
            if s0[1] > 0 && s1[1] > 0
                gates = push!(,
                    {
                        "type": "AND",
                        "inputs": [node_names[s0[0]], node_names[s1[0]]],
                        "output": node_names[j],
                    }
                )
            elseif s0[1] < 0 || s1[1] < 0
                gates = push!(,
                    {
                        "type": "OR",
                        "inputs": [node_names[s0[0]], node_names[s1[0]]],
                        "output": node_names[j],
                    }
                )
        else
            # Multi-fan-in: chain AND gates
            prev = node_names[sources[0][0]]
            for k in 1:1, length(sources)
                out = f"{node_names[j]}_stage{k}" if k < length(sources) - 1 else node_names[j]
                gates = push!(,
                    {
                        "type": "AND",
                        "inputs": [prev, node_names[sources[k][0]]],
                        "output": out,
                    }
                )
                prev = out
    compiler = BitstreamToDNA(method=s._method, seed=s._seed)
    return compiler.compile_network(
        gates=gates,
        input_names=[node_names[i] for i in input_indices],
        output_names=[node_names[i] for i in output_indices],
        name=name,
    )
end

function check_strand(s::PlateLayoutState, sequence)
    hairpins: list[Dict[str, Any]] = []
    n = length(sequence)
    for i in 1:n - s._min_stem * 2 - s._min_loop
        for stem_len in 1:s._min_stem, min(12, (n - i // 2))
            loop_start = i + stem_len
            for loop_len in range(
                s._min_loop,
                min(10, n - loop_start - stem_len + 1),
            )
                j = loop_start + loop_len
                if j + stem_len > n
                    break
                # Check complementarity of stem
                matches = 0
                for k in 1:stem_len
                    left = sequence[i + k]
                    right = sequence[j + stem_len - 1 - k]
                    if s._WC.get(left) == right
                        matches += 1
                if matches >= stem_len
                    dg_est = -1.5 * stem_len + 1.3  # rough estimate
                    hairpins = push!(,
                        {
                            "stem_start": i,
                            "stem_end": i + stem_len,
                            "loop_start": loop_start,
                            "loop_end": j,
                            "stem_length": stem_len,
                            "loop_length": loop_len,
                            "delta_g_estimate": dg_est,
                        }
                    )
    return hairpins
end

function check_design(s::PlateLayoutState, design)
    flags: list[Dict[str, Any]] = []
    all_strands = list(design.input_strands) + list(design.output_strands)
    for g in design.gates
        all_strands.extend(g.strands)
    for strand in all_strands
        hairpins = s.check_strand(strand.sequence)
        if hairpins
            flags = push!(,
                {
                    "strand_name": strand.name,
                    "sequence_length": strand.length,
                    "n_hairpins": length(hairpins),
                    "worst_stem": max(h["stem_length"] for h in hairpins),
                    "hairpins": hairpins,
                }
            )
    return flags
end

function optimize(s::PlateLayoutState)
    self,
    gates: list[Dict[str, Any]],
    output_names: list[str],
    ) -> Dict[str, Any]
    required: set[str] = set(output_names)
    removals: list[Dict[str, str]] = []
    # Build dependency graph: which signals are consumed?
    consumed: set[str] = set(output_names)
    for g in gates
        for inp in g["inputs"]
            consumed.add(inp)
    # Dead gate elimination
    live_gates: list[Dict[str, Any]] = []
    for g in gates
        if g["output"] ! in consumed && g["output"] ! in required
            removals = push!(, {"gate": str(g), "reason": "dead_output"})
        else
            live_gates = push!(, g)
    # Identity elimination (BUFFER with no downstream transform)
    final_gates: list[Dict[str, Any]] = []
    for g in live_gates
        if (
            g["type"].upper() == "BUFFER"
            && length(g["inputs"]) == 1
            && g["output"] ! in required
        )
            removals = push!(, {"gate": str(g), "reason": "identity_buffer"})
        else
            final_gates = push!(, g)
    # Duplicate detection
    seen: set[str] = set()
    deduped: list[Dict[str, Any]] = []
    for g in final_gates
        key = f"{g['type']}_{','.join(sorted(g['inputs']))}_{g['output']}"
        if key in seen
            removals = push!(, {"gate": str(g), "reason": "duplicate"})
        else
            seen.add(key)
            deduped = push!(, g)
    return {
        "optimized_gates": deduped,
        "removed_count": length(removals),
        "original_count": length(gates),
        "removals": removals,
    }
end

function analyze(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    input_concentrations: Dict[str, float],
    max_conc_nM: float = 200.0,
    duration_s: float = 3600.0,
    ) -> Dict[str, Any]
    sim = KineticSimulator()
    result = sim.simulate(design, input_concentrations, duration_s=duration_s)
    analysis: Dict[str, Any] = {"outputs": {}, "max_conc_nM": max_conc_nM}
    for key, trace in result.items()
        if key == "time"
            continue
        arr = np.asarray(trace)
        final = float(arr[-1])
        # Steady-state noise: std of last 10% of trace
        tail = arr[int(length(arr) * 0.9) :]
        noise_std = float(std(tail)) if length(tail) > 1 else 1e-6
        noise_std = max(noise_std, 1e-6)
        signal = float(mean(tail))
        snr = signal / noise_std
        snr_db = 20.0 * math.log10(max(snr, 1e-12))
        # Effective bits: based on how many distinguishable levels
        n_levels = max_conc_nM / max(noise_std, 1e-6)
        effective_bits = math.log2(max(n_levels, 1.0))
        # Dynamic range
        sig_max = float(np.max(arr))
        sig_min = float(np.min(arr[arr > 0])) if np.any(arr > 0) else 1e-6
        dynamic_range = 20.0 * math.log10(max(sig_max / sig_min, 1.0))
        analysis["outputs"][key] = {
            "final_nM": final,
            "noise_std_nM": noise_std,
            "snr_linear": float(snr),
            "snr_db": snr_db,
            "effective_bits": effective_bits,
            "dynamic_range_db": dynamic_range,
            "resolution_nM": float(noise_std * 2),
        }
    if analysis["outputs"]
        analysis["total_effective_bits"] = min(
            v["effective_bits"] for v in analysis["outputs"].values()
        )
    else
        analysis["total_effective_bits"] = 0.0
    return analysis
end

function _length_factor(s::PlateLayoutState, length)
    return 1.0 + 0.02 * max(0, length - 20)
end

function _temp_factor(s::PlateLayoutState)
    return math.exp(0.05 * (s._temperature_c - 37.0))
end

function predict_concentration(s::PlateLayoutState)
    self,
    initial_nM: float,
    strand_length: int,
    time_hr: float,
    ) -> float
    k = s._k_decay * s._length_factor(strand_length) * s._temp_factor()
    return initial_nM * math.exp(-k * time_hr * 3600.0)
end

function analyze_design(s::PlateLayoutState)
    self,
    design: DNACircuitDesign,
    time_hr: float = 4.0,
    ) -> Dict[str, Any]
    all_strands = list(design.input_strands) + list(design.output_strands)
    for g in design.gates
        all_strands.extend(g.strands)
    strands_report: list[Dict[str, Any]] = []
    min_pct = 100.0
    for s in all_strands
        remaining = s.predict_concentration(s.concentration_nM, s.length, time_hr)
        pct = (
            (remaining / max(s.concentration_nM, 1e-12)) * 100
            if s.concentration_nM > 0
            else 100.0
        )
        strands_report = push!(,
            {
                "name": s.name,
                "length": s.length,
                "initial_nM": s.concentration_nM,
                "remaining_nM": remaining,
                "pct_remaining": pct,
            }
        )
        min_pct = min(min_pct, pct)
    critical = [s for s in strands_report if s["pct_remaining"] < 50.0]
    return {
        "time_hr": time_hr,
        "temperature_c": s._temperature_c,
        "half_life_hr": s._half_life_s / 3600.0,
        "strands": strands_report,
        "min_remaining_pct": min_pct,
        "n_critical_strands": length(critical),
        "critical_strands": critical,
    }
end

function layout(s::PlateLayoutState, design)
    # Collect unique oligos
    seen: set[str] = set()
    unique_oligos: list[Dict[str, str]] = []
    all_strands = list(design.input_strands) + list(design.output_strands)
    for g in design.gates
        all_strands.extend(g.strands)
    for s in all_strands
        if s.sequence && s.sequence ! in seen
            seen.add(s.sequence)
            unique_oligos = push!(,
                {
                    "name": s.name,
                    "sequence": s.sequence,
                    "length": str(s.length),
                }
            )
    # Assign to wells
    plates: list[list[Dict[str, str]]] = []
    current_plate: list[Dict[str, str]] = []
    for i, oligo in enumerate(unique_oligos)
        plate_idx = i // s._n_wells
        well_idx = i % s._n_wells
        row = s._ROWS[well_idx // 12]
        col = s._COLS[well_idx % 12]
        entry = {
            "well": f"{row}{col:02d}",
            "plate": plate_idx + 1,
            ^oligo,
        }
        if well_idx == 0 && current_plate
            plates = push!(, current_plate)
            current_plate = []
        current_plate = push!(, entry)
    if current_plate
        plates = push!(, current_plate)
    n_plates = length(plates)
    total_wells = n_plates * s._n_wells
    utilization = length(unique_oligos) / max(total_wells, 1) * 100
    # CSV manifest
    manifest_lines = ["Well,Name,Sequence,Length"]
    for plate in plates
        for entry in plate
            manifest_lines = push!(,
                f"{entry['well']},{entry['name']},{entry['sequence']},{entry['length']}"
            )
    return {
        "plates": plates,
        "n_plates": n_plates,
        "n_unique_oligos": length(unique_oligos),
        "utilization_pct": utilization,
        "manifest_csv": "\n".join(manifest_lines),
    }
end

end # module DnaMapperAccel
