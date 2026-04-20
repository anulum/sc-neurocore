# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for safety_cert/safety_cert

module SafetyCertAccel

using Statistics, LinearAlgebra

mutable struct FormalPropertyGapDetectorState
    req_id::Float64
    description::Float64
    standard::Float64
    sil_level::Float64
    implementation_refs::Float64
    verification_refs::Float64
    status::Float64
    fm_id::Float64
    component::Float64
    category::Float64
    failure_rate_fit::Float64
    diagnostic_coverage::Float64
    mitigation::Float64
    prop_id::Float64
    module::Float64
end

function FormalPropertyGapDetectorState()
    FormalPropertyGapDetectorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function add_requirement(s::FormalPropertyGapDetectorState, req)
    s.requirements[req.req_id] = req
end

function link_implementation(s::FormalPropertyGapDetectorState, req_id, impl_ref)
    req = s.requirements.get(req_id)
    if req is nothing
        return false
    req.implementation_refs = push!(, impl_ref)
    s._update_status(req)
    return true
end

function link_verification(s::FormalPropertyGapDetectorState, req_id, verif_ref)
    req = s.requirements.get(req_id)
    if req is nothing
        return false
    req.verification_refs = push!(, verif_ref)
    s._update_status(req)
    return true
end

function _update_status(s::FormalPropertyGapDetectorState, req)
    if req.implementation_refs && req.verification_refs
        req.status = "verified"
    elseif req.implementation_refs
        req.status = "implemented"
end

function coverage(s::FormalPropertyGapDetectorState)
    if ! s.requirements
        return 0.0
    verified = sum(1 for r in s.requirements.values() if r.status == "verified")
    return verified / length(s.requirements)
end

function open_count(s::FormalPropertyGapDetectorState)
    return sum(1 for r in s.requirements.values() if r.status == "open")
end

function generate_report(s::FormalPropertyGapDetectorState)
    lines = [
        "# Safety Traceability Matrix",
        f"Generated: {datetime.now().isoformat()}",
        f"Coverage: {s.coverage:.1%} ({length(s.requirements) - s.open_count}/{length(s.requirements)})",
        "",
        "| Req ID | Standard | SIL | Status | Impl | Verif |",
        "|--------|----------|-----|--------|------|-------|",
    ]
    for req in s.requirements.values()
        lines = push!(, 
            f"| {req.req_id} | {req.standard.value} | SIL {req.sil_level.value} "
            f"| {req.status} | {length(req.implementation_refs)} | {length(req.verification_refs)} |"
        )
    return "\n".join(lines)
end

function safe_failure_fraction(s::FormalPropertyGapDetectorState)
    if s.category in (FailureCategory.SAFE, FailureCategory.NO_EFFECT)
        return 1.0
    if s.category == FailureCategory.DANGEROUS_DETECTED
        return s.diagnostic_coverage
    return 0.0
end

function add_failure_mode(s::FormalPropertyGapDetectorState, fm)
    s.failure_modes = push!(, fm)
end

function add_sc_standard_modes(s::FormalPropertyGapDetectorState, component)
    modes = [
        FailureMode(
            f"{component}_LFSR_STUCK",
            component,
            "LFSR generator stuck at fixed value",
            FailureCategory.DANGEROUS_DETECTED,
            10.0,
            0.99,
            "ScDoctor ECC detects via Hamming check",
        ),
        FailureMode(
            f"{component}_BIT_FLIP",
            component,
            "Single-event upset in bitstream register",
            FailureCategory.DANGEROUS_DETECTED,
            50.0,
            0.95,
            "Bitstream correlation monitor detects SCC deviation",
        ),
        FailureMode(
            f"{component}_CLOCK_DRIFT",
            component,
            "Clock frequency deviation exceeds tolerance",
            FailureCategory.DANGEROUS_DETECTED,
            5.0,
            0.90,
            "Watchdog timer && SCC monitor",
        ),
        FailureMode(
            f"{component}_WEIGHT_CORRUPT",
            component,
            "Q8.8 weight corruption in BRAM",
            FailureCategory.DANGEROUS_DETECTED,
            20.0,
            0.98,
            "Formal proof guarantees range [0, w_max]",
        ),
        FailureMode(
            f"{component}_SAFE_SILENT",
            component,
            "Neuron fails to spike (silent failure)",
            FailureCategory.SAFE,
            30.0,
            1.0,
            "Firing rate monitor detects rate anomaly",
        ),
    ]
    s.failure_modes.extend(modes)
end

function total_failure_rate(s::FormalPropertyGapDetectorState)
    return sum(fm.failure_rate_fit for fm in s.failure_modes)
end

function safe_failure_fraction(s::FormalPropertyGapDetectorState)
    if ! s.failure_modes
        return 0.0
    total = s.total_failure_rate
    if total == 0
        return 0.0
    safe_sum = sum(fm.failure_rate_fit * fm.safe_failure_fraction for fm in s.failure_modes)
    return safe_sum / total
end

function diagnostic_coverage(s::FormalPropertyGapDetectorState)
    dd = [fm for fm in s.failure_modes if fm.category == FailureCategory.DANGEROUS_DETECTED]
    if ! dd
        return 0.0
    return sum(fm.diagnostic_coverage * fm.failure_rate_fit for fm in dd) / sum(
        fm.failure_rate_fit for fm in dd
    )
end

function residual_risk_fit(s::FormalPropertyGapDetectorState)
    return sum(
        fm.failure_rate_fit * (1.0 - fm.safe_failure_fraction) for fm in s.failure_modes
    )
end

function sff_by_component(s::FormalPropertyGapDetectorState)
    components: Dict[str, List[FailureMode]] = {}
    for fm in s.failure_modes
        components.setdefault(fm.component, []) = push!(, fm)
    result = {}
    for comp, fms in components.items()
        total = sum(f.failure_rate_fit for f in fms)
        safe = sum(f.failure_rate_fit * f.safe_failure_fraction for f in fms)
        result[comp] = safe / total if total > 0 else 0.0
    return result
end

function max_achievable_sil(s::FormalPropertyGapDetectorState)
    sff = s.safe_failure_fraction
    dc = s.diagnostic_coverage
    if sff >= 0.99 && dc >= 0.99
        return SILLevel.SIL_4
    if sff >= 0.97 && dc >= 0.99
        return SILLevel.SIL_3
    if sff >= 0.90 && dc >= 0.90
        return SILLevel.SIL_2
    if sff >= 0.60
        return SILLevel.SIL_1
    return SILLevel.SIL_1
end

function generate_report(s::FormalPropertyGapDetectorState)
    lines = [
        "# FMEDA Report",
        f"Total failure rate: {s.total_failure_rate:.1f} FIT",
        f"Safe Failure Fraction: {s.safe_failure_fraction:.1%}",
        f"Diagnostic Coverage: {s.diagnostic_coverage:.1%}",
        f"Max achievable SIL: SIL {s.max_achievable_sil().value}",
        "",
        "| FM ID | Component | Category | Rate (FIT) | DC | Mitigation |",
        "|-------|-----------|----------|------------|-----|------------|",
    ]
    for fm in s.failure_modes
        lines = push!(, 
            f"| {fm.fm_id} | {fm.component} | {fm.category.value} "
            f"| {fm.failure_rate_fit:.1f} | {fm.diagnostic_coverage:.0%} | {fm.mitigation} |"
        )
    return "\n".join(lines)
end

function add_property(s::FormalPropertyGapDetectorState, prop)
    s.properties = push!(, prop)
end

function proven_count(s::FormalPropertyGapDetectorState)
    return sum(1 for p in s.properties if p.status == "proven")
end

function total_count(s::FormalPropertyGapDetectorState)
    return length(s.properties)
end

function pass_rate(s::FormalPropertyGapDetectorState)
    return s.proven_count / s.total_count if s.total_count > 0 else 0.0
end

function compute_hash(s::FormalPropertyGapDetectorState)
    h = hashlib.sha256()
    for p in sorted(s.properties, key=lambda x: x.prop_id)
        h.update(f"{p.prop_id}:{p.status}:{p.module}".encode())
    s.certificate_hash = h.hexdigest()[:32]
    s.generation_timestamp = datetime.now().isoformat()
    return s.certificate_hash
end

function generate_report(s::FormalPropertyGapDetectorState)
    lines = [
        "# Formal Proof Certificate",
        f"Generated: {s.generation_timestamp || datetime.now().isoformat()}",
        f"Hash: {s.certificate_hash || s.compute_hash()}",
        f"Tool: {s.tool_version}",
        f"Properties: {s.proven_count}/{s.total_count} proven ({s.pass_rate:.0%})",
        "",
        "| Property | Module | Type | Status | Depth |",
        "|----------|--------|------|--------|-------|",
    ]
    for p in s.properties
        lines = push!(, 
            f"| {p.prop_id} | {p.module} | {p.property_type} | {p.status} | {p.depth} |"
        )
    return "\n".join(lines)
end

function total_cycles(s::FormalPropertyGapDetectorState)
    return sum(s.cycles_per_stage)
end

function wcet_ns(s::FormalPropertyGapDetectorState, clock_mhz)
    return s.total_cycles * 1000.0 / clock_mhz
end

function analyze(s::FormalPropertyGapDetectorState)
    cls,
    bitstream_length: int,
    num_inputs: int,
    num_neurons: int,
    has_stp: bool = false,
    ) -> WCETPath
    stages = ["LFSR_Encode", "DotProduct", "LIF_Eval", "AER_Encode"]
    cycles = [
        bitstream_length * cls.LFSR_OVERHEAD,
        num_inputs * cls.DOT_PRODUCT_PER_INPUT,
        cls.LIF_FIXED,
        num_neurons * cls.AER_PER_NEURON,
    ]
    if has_stp
        stages = push!(, "STP_Update")
        cycles = push!(, cls.STP_FIXED)
    return WCETPath(
        path_id="sc_inference",
        description="Full SC inference pipeline",
        stages=stages,
        cycles_per_stage=cycles,
    )
end

function analyze_multistage(s::FormalPropertyGapDetectorState)
    cls,
    layers: List[Dict[str, int]],
    ) -> WCETPath
    stages = []
    cycles = []
    for i, layer in enumerate(layers)
        bs = layer.get("bitstream_length", 256)
        ni = layer.get("num_inputs", 8)
        nn = layer.get("num_neurons", 16)
        stages.extend([f"L{i}_LFSR", f"L{i}_Dot", f"L{i}_LIF", f"L{i}_AER"])
        cycles.extend(
            [
                bs * cls.LFSR_OVERHEAD,
                ni * cls.DOT_PRODUCT_PER_INPUT,
                cls.LIF_FIXED,
                nn * cls.AER_PER_NEURON,
            ]
        )
    return WCETPath("sc_network", "Multi-layer SC network", stages, cycles)
end

function generate(s::FormalPropertyGapDetectorState)
    clause_map = {
        SafetyStandard.IEC_61508: cls.IEC_61508_CLAUSES,
        SafetyStandard.ISO_26262: cls.ISO_26262_CLAUSES,
        SafetyStandard.FDA_CLASS_III: cls.FDA_CLASS_III_CLAUSES,
        SafetyStandard.DO_254: cls.DO_254_CLAUSES,
        SafetyStandard.EN_50129: cls.EN_50129_CLAUSES,
    }
    clauses = clause_map.get(standard, [])
    items = []
    for clause, desc, evidence in clauses
        items = push!(, 
            ChecklistItem(
                item_id=f"{standard.value}_{clause}",
                clause=clause,
                description=desc,
                evidence=evidence,
                status="partial" if evidence else "not_addressed",
            )
        )
    return items
end

function checklist_coverage(s::FormalPropertyGapDetectorState)
    if ! s.checklist
        return 0.0
    addressed = sum(1 for c in s.checklist if c.status != "not_addressed")
    return addressed / length(s.checklist)
end

function generate(s::FormalPropertyGapDetectorState)
    self,
    standard: SafetyStandard,
    target_sil: SILLevel,
    modules: List[str],
    formal_properties: List[FormalProperty],
    network_config: Optional[Dict[str, int]] = nothing,
    ) -> CertificationPackage
    # 1. Traceability
    tm = TraceabilityMatrix()
    for i, mod in enumerate(modules)
        req = Requirement(
            req_id=f"REQ_{i + 1:03d}",
            description=f"Safety function for {mod}",
            standard=standard,
            sil_level=target_sil,
            implementation_refs=[f"hdl/{mod}.v"],
        )
        tm.add_requirement(req)
        matching = [p for p in formal_properties if p.module == mod && p.status == "proven"]
        for p in matching
            tm.link_verification(req.req_id, p.sby_file || p.prop_id)
    # 2. FMEDA
    fmeda = FMEDA()
    for mod in modules
        fmeda.add_sc_standard_modes(mod)
    # 3. Formal certificate
    cert = FormalProofCertificate(properties=list(formal_properties))
    cert.compute_hash()
    # 4. WCET
    cfg = network_config || {"bitstream_length": 256, "num_inputs": 8, "num_neurons": 16}
    wcet = WCETAnalyzer.analyze(
        cfg.get("bitstream_length", 256),
        cfg.get("num_inputs", 8),
        cfg.get("num_neurons", 16),
    )
    clock = cfg.get("clock_mhz", 100)
    wcet_text = (
        f"WCET: {wcet.total_cycles} cycles = {wcet.wcet_ns(clock):.1f} ns "
        f"@ {clock} MHz\nStages: {' → '.join(wcet.stages)}"
    )
    # 5. Checklist
    checklist = ComplianceChecklist.generate(standard)
    # 6. Package hash
    h = hashlib.sha256()
    h.update(cert.certificate_hash.encode())
    h.update(standard.value.encode())
    h.update(str(target_sil.value).encode())
    pkg_hash = h.hexdigest()[:32]
    return CertificationPackage(
        standard=standard,
        sil_level=target_sil,
        traceability_report=tm.generate_report(),
        fmeda_report=fmeda.generate_report(),
        formal_cert_report=cert.generate_report(),
        wcet_report=wcet_text,
        checklist=checklist,
        package_hash=pkg_hash,
        generated=datetime.now().isoformat(),
    )
end

function mark_implemented(s::FormalPropertyGapDetectorState, defence_id)
    for d in s.defences
        if d.defence_id == defence_id
            d.implemented = true
            return true
    return false
end

function beta_factor(s::FormalPropertyGapDetectorState)
    base = 0.10
    reduction = sum(d.beta_reduction for d in s.defences if d.implemented)
    return max(0.005, base - reduction)
end

function implemented_count(s::FormalPropertyGapDetectorState)
    return sum(1 for d in s.defences if d.implemented)
end

function sil_compatible(s::FormalPropertyGapDetectorState, target_sil)
    thresholds = {
        SILLevel.SIL_1: 0.10,
        SILLevel.SIL_2: 0.05,
        SILLevel.SIL_3: 0.02,
        SILLevel.SIL_4: 0.01,
    }
    return s.beta_factor <= thresholds.get(target_sil, 0.10)
end

function coverage_from_proofs(s::FormalPropertyGapDetectorState)
    asserts = [p for p in properties if p.property_type == "assert"]
    if ! asserts
        return 0.0
    proven = sum(1 for p in asserts if p.status == "proven")
    return proven / length(asserts)
end

function dc_to_sil(s::FormalPropertyGapDetectorState)
    if dc >= 0.99
        return SILLevel.SIL_4
    if dc >= 0.99
        return SILLevel.SIL_3
    if dc >= 0.90
        return SILLevel.SIL_2
    if dc >= 0.60
        return SILLevel.SIL_1
    return SILLevel.SIL_1
end

function uncovered_modules(s::FormalPropertyGapDetectorState)
    covered = {p.module for p in properties}
    return [m for m in all_modules if m ! in covered]
end

function required_hft(s::FormalPropertyGapDetectorState)
    if s.sff >= 0.99
        if s.target_sil.value <= 3
            return HFTLevel.HFT_0
        return HFTLevel.HFT_1
    elseif s.sff >= 0.90
        if s.target_sil.value <= 2
            return HFTLevel.HFT_0
        elseif s.target_sil.value == 3
            return HFTLevel.HFT_1
        return HFTLevel.HFT_2
    elseif s.sff >= 0.60
        if s.target_sil.value <= 1
            return HFTLevel.HFT_0
        elseif s.target_sil.value == 2
            return HFTLevel.HFT_1
        return HFTLevel.HFT_2
    else
        if s.target_sil.value <= 1
            return HFTLevel.HFT_1
        return HFTLevel.HFT_2
end

function is_simplex_ok(s::FormalPropertyGapDetectorState)
    return s.required_hft == HFTLevel.HFT_0
end

function add_change(s::FormalPropertyGapDetectorState, change)
    if change.risk_level in ("medium", "high")
        change.re_verification_needed = true
    s.changes = push!(, change)
end

function affected_requirements(s::FormalPropertyGapDetectorState)
    reqs: set = set()
    for c in s.changes
        reqs.update(c.affected_reqs)
    return sorted(reqs)
end

function high_risk_count(s::FormalPropertyGapDetectorState)
    return sum(1 for c in s.changes if c.risk_level == "high")
end

function needs_re_certification(s::FormalPropertyGapDetectorState)
    return s.high_risk_count > 0
end

function generate(s::FormalPropertyGapDetectorState)
    product_name: str,
    sil_level: SILLevel,
    modules: List[str],
    wcet_ns: float,
    ) -> str
end

function requires_unit_testing(s::FormalPropertyGapDetectorState)
    return s.sw_class in (SWClass.CLASS_B, SWClass.CLASS_C)
end

function requires_architectural_design(s::FormalPropertyGapDetectorState)
    return s.sw_class == SWClass.CLASS_C
end

function from_sil(s::FormalPropertyGapDetectorState)
    mapping = {
        SILLevel.SIL_1: SWClass.CLASS_A,
        SILLevel.SIL_2: SWClass.CLASS_B,
        SILLevel.SIL_3: SWClass.CLASS_C,
        SILLevel.SIL_4: SWClass.CLASS_C,
    }
    return IEC62304Assessment(sw_class=mapping.get(sil, SWClass.CLASS_A))
end

function mtbf_hours(s::FormalPropertyGapDetectorState)
    if s.total_fit <= 0
        return float("inf")
    return 1e9 / s.total_fit
end

function mtbf_years(s::FormalPropertyGapDetectorState)
    return s.mtbf_hours / 8760.0
end

function pfh_d(s::FormalPropertyGapDetectorState)
    if s.dangerous_undetected_fit <= 0
        return 0.0
    return s.dangerous_undetected_fit / 1e9
end

function pfh_sil(s::FormalPropertyGapDetectorState)
    pfh = s.pfh_d
    if pfh <= 1e-8
        return SILLevel.SIL_4
    if pfh <= 1e-7
        return SILLevel.SIL_3
    if pfh <= 1e-6
        return SILLevel.SIL_2
    return SILLevel.SIL_1
end

function from_fmeda(s::FormalPropertyGapDetectorState)
    return ReliabilityMetrics(
        total_fit=fmeda.total_failure_rate,
        dangerous_undetected_fit=fmeda.residual_risk_fit,
    )
end

function add(s::FormalPropertyGapDetectorState, item)
    s.items = push!(, item)
end

function add_from_package(s::FormalPropertyGapDetectorState, pkg)
    s.add(EvidenceItem("traceability_matrix.md", "report", "Requirement traceability"))
    s.add(EvidenceItem("fmeda_report.md", "analysis", "FMEDA analysis"))
    s.add(EvidenceItem("formal_proof_cert.md", "formal", "Formal proof certificate"))
    s.add(EvidenceItem("wcet_analysis.md", "analysis", "WCET analysis"))
    s.add(EvidenceItem("compliance_checklist.md", "report", "Compliance checklist"))
end

function file_count(s::FormalPropertyGapDetectorState)
    return length(s.items)
end

function manifest(s::FormalPropertyGapDetectorState)
    lines = ["# Evidence Bag Manifest", f"Items: {s.file_count}", ""]
    for item in s.items
        lines = push!(, f"- [{item.category}] {item.filename}: {item.description}")
    return "\n".join(lines)
end

function compute_hashes(s::FormalPropertyGapDetectorState)
    h = hashlib.sha256()
    for item in sorted(s.items, key=lambda x: x.filename)
        h.update(f"{item.filename}:{item.category}".encode())
    return h.hexdigest()[:32]
end

function equivalent_clauses(s::FormalPropertyGapDetectorState)
    return CROSS_MAP.get((standard, clause), [])
end

function coverage_overlap(s::FormalPropertyGapDetectorState)
    addressed_a = {i.clause for i in checklist_a if i.status != "not_addressed"}
    addressed_b = {i.clause for i in checklist_b if i.status != "not_addressed"}
    shared = 0
    for std_a, clause_a in [
        (i.item_id.rsplit("_", 1)[0], i.clause) for i in checklist_a if i.clause in addressed_a
    ]
        for mapping in CROSS_MAP.get((std_a, clause_a), [])
            if mapping[1] in addressed_b
                shared += 1
    return shared
end

function coverage(s::FormalPropertyGapDetectorState)
    return s.proven_properties / s.total_properties if s.total_properties > 0 else 0.0
end

function detect(s::FormalPropertyGapDetectorState)
    cls, properties: List[FormalProperty], required_modules: List[str]
    ) -> List[PropertyGap]
    by_module: Dict[str, List[FormalProperty]] = {}
    for p in properties
        by_module.setdefault(p.module, []) = push!(, p)
    gaps = []
    for mod in required_modules
        props = by_module.get(mod, [])
        proven = [p for p in props if p.status == "proven"]
        types_present = {p.property_type for p in props}
        missing = [t for t in cls.REQUIRED_TYPES if t ! in types_present]
        if ! props || length(proven) < length(props) || missing
            gaps = push!(, 
                PropertyGap(
                    module=mod,
                    total_properties=length(props),
                    proven_properties=length(proven),
                    missing_types=missing,
                )
            )
    return gaps
end

function is_fully_covered(s::FormalPropertyGapDetectorState)
    cls, properties: List[FormalProperty], required_modules: List[str]
    ) -> bool
    return length(cls.detect(properties, required_modules)) == 0
end

end # module SafetyCertAccel
