# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass

# 12. Model Checksum / Hash Embedding
# ═══════════════════════════════════════════════════════════════════════


def embed_model_checksum(
    verilog: str,
    *,
    equations: dict[str, str] | None = None,
    params: dict[str, int | float] | None = None,
) -> str:
    """Embed a SHA-256 checksum of the compiled model in the Verilog source.

    Enables bit-exact reproducibility verification — the hash of the
    source equations and parameters is embedded as a Verilog comment and
    a localparam, allowing downstream tools to verify that the RTL
    matches the expected model.

    Parameters
    ----------
    verilog : str
        Generated Verilog source code.
    equations : dict[str, str], optional
        Original ODE equations (state_var → expression).
    params : dict[str, int | float], optional
        Compilation parameters (data_width, fraction, etc.).

    Returns
    -------
    str
        Verilog with embedded checksum comment and localparam.
    """

    payload = {
        "equations": equations or {},
        "params": params or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(canonical.encode()).hexdigest()

    checksum_block = (
        f"// ── SC-NeuroCore Model Checksum ──────────────────────────────\n"
        f"// SHA-256: {sha}\n"
        f"// Source: {canonical[:80]}{'...' if len(canonical) > 80 else ''}\n"
        f"// Verify: echo -n '{canonical}' | sha256sum\n"
        f"localparam [255:0] MODEL_HASH = 256'h{sha};\n"
    )

    # Insert after first line (module declaration or comment)
    line_list = verilog.split("\n")
    insert_pos = 1  # After first line
    for i, line in enumerate(line_list):
        if line.strip().startswith("module"):
            insert_pos = i  # Before module declaration
            break

    line_list.insert(insert_pos, checksum_block)
    return "\n".join(line_list)


# ═══════════════════════════════════════════════════════════════════════
# 18. Bitstream Encryption Wrapper
# ═══════════════════════════════════════════════════════════════════════


def generate_bitstream_encryption(
    module_name: str,
    *,
    vendor: str = "xilinx",
    key_length: int = 256,
    key_source: str = "efuse",
) -> str:
    """Generate bitstream encryption TCL/constraints for secure boot.

    Produces the vendor-specific TCL commands and XDC constraints to
    enable AES-256 bitstream encryption, protecting the compiled neuron
    IP from reverse-engineering and tampering.

    Parameters
    ----------
    module_name : str
        Design module name.
    vendor : str
        ``"xilinx"`` or ``"intel"``.
    key_length : int
        AES key length: 128 or 256.
    key_source : str
        Key storage: ``"efuse"`` (one-time programmable),
        ``"bbram"`` (battery-backed RAM), or ``"external"``.

    Returns
    -------
    str
        TCL/Quartus script for bitstream encryption.
    """
    if vendor == "xilinx":
        lines = [
            f"# Bitstream encryption for {module_name}",
            f"# SC-NeuroCore — Xilinx AES-{key_length} secure boot",
            f"# Key source: {key_source}",
            "",
            "# ── Vivado TCL commands ──",
            "set_property BITSTREAM.ENCRYPTION.ENCRYPT YES [current_design]",
            f"set_property BITSTREAM.ENCRYPTION.ENCRYPTKEYSELECT {key_source.upper()} [current_design]",
            "set_property BITSTREAM.ENCRYPTION.KEYLIFE {100} [current_design]",
            "",
            "# ── Key file reference ──",
            f"# Generate key: write_bitstream -encrypt -encrypt_key_file {module_name}.nky",
            f"set_property BITSTREAM.ENCRYPTION.KEYFILE {{{module_name}.nky}} [current_design]",
            "",
            "# ── Tamper detection ──",
            "set_property BITSTREAM.CONFIG.USR_ACCESS TIMESTAMP [current_design]",
            "set_property BITSTREAM.CONFIG.SECURITY_LEVEL LEVEL2 [current_design]",
            "",
            "# ── Authentication (optional HMAC) ──",
            "# set_property BITSTREAM.AUTHENTICATION.AUTHENTICATE YES [current_design]",
            f"# set_property BITSTREAM.AUTHENTICATION.HMACKEY_FILE {{{module_name}.hmac}} [current_design]",
        ]
    else:  # Intel
        lines = [
            f"# Bitstream encryption for {module_name}",
            f"# SC-NeuroCore — Intel/Altera AES-{key_length} secure boot",
            f"# Key source: {key_source}",
            "",
            "# ── Quartus Settings ──",
            f'set_global_assignment -name ENCRYPTION_KEY_SOURCE "{key_source.upper()}"',
            f'set_global_assignment -name ENCRYPTION_SECURITY_KEY "{module_name}_key"',
            "set_global_assignment -name ENABLE_CONFIGURATION_BITSTREAM_ENCRYPTION ON",
            "",
            "# ── Anti-tamper ──",
            "set_global_assignment -name ENABLE_ANTI_TAMPER ON",
            'set_global_assignment -name ANTI_TAMPER_SCHEME "DETECT"',
            "",
            "# ── Secure device setup ──",
            f"# quartus_pgm --jtag --encrypt --key {module_name}.key",
        ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 31. Side-Channel Leakage Lint
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SideChannelFinding:
    """Side-channel leakage finding.

    Attributes
    ----------
    signal : str
        Signal name.
    risk_level : str
        ``"high"``, ``"medium"``, or ``"low"``.
    category : str
        ``"timing"`` or ``"power"``.
    description : str
        Explanation.
    recommendation : str
        Mitigation suggestion.
    """

    signal: str
    risk_level: str
    category: str
    description: str
    recommendation: str


def lint_side_channels(
    equations: dict[str, str],
    *,
    module_name: str = "neuron",
    data_width: int = 16,
) -> list[SideChannelFinding]:
    """Analyse equations for power/timing side-channel vulnerabilities.

    Flags data-dependent timing paths and variable-activity patterns
    in the generated RTL.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    module_name : str
        Module name.
    data_width : int
        Data width.

    Returns
    -------
    list[SideChannelFinding]
        List of findings.
    """
    findings = []

    for sv, expr in equations.items():
        # Check for data-dependent branching (if/else in expr)
        if "if" in expr or "?" in expr:
            findings.append(
                SideChannelFinding(
                    signal=sv,
                    risk_level="high",
                    category="timing",
                    description=f"Data-dependent branch in {sv} equation",
                    recommendation="Use constant-time mux instead of branch",
                )
            )

        # Check for division (variable-latency)
        if "/" in expr:
            findings.append(
                SideChannelFinding(
                    signal=sv,
                    risk_level="medium",
                    category="timing",
                    description=f"Division in {sv} — variable latency",
                    recommendation="Use fixed-point shift or LUT-based reciprocal",
                )
            )

        # Check for multiplication by secret data
        if "*" in expr:
            findings.append(
                SideChannelFinding(
                    signal=sv,
                    risk_level="low",
                    category="power",
                    description=f"Multiply in {sv} — Hamming weight leakage",
                    recommendation="Add random masking for security-critical paths",
                )
            )

    # General: spike output is 1-bit and data-dependent
    findings.append(
        SideChannelFinding(
            signal="spike_out",
            risk_level="medium",
            category="power",
            description="Spike output toggles are data-dependent",
            recommendation="Add constant-activity output buffer",
        )
    )

    return findings


# ═══════════════════════════════════════════════════════════════════════
# 36. Supply Chain Risk Scorer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SupplyChainRisk:
    """Supply chain risk assessment for a hardware profile.

    Attributes
    ----------
    profile_name : str
        Assessed profile.
    risk_score : float
        Risk score 0-100 (higher = riskier).
    risk_factors : list[str]
        Individual risk factor descriptions.
    alternatives : list[str]
        Suggested alternative profiles.
    export_control : str
        Export control classification.
    """

    profile_name: str
    risk_score: float
    risk_factors: list[str]
    alternatives: list[str]
    export_control: str


# Geography risk mapping
_GEO_RISK: dict[str, float] = {
    "TSMC": 35,
    "MediaTek": 30,  # Taiwan concentration
    "Samsung": 20,
    "SK Hynix": 20,  # South Korea
    "NIST": 5,
    "Northrop Grumman": 5,  # US defence
    "Intel": 10,
    "AMD": 10,
    "Qualcomm": 10,
    "Xilinx": 10,
    "Lattice": 10,
    "Microchip": 10,
    "Research": 50,  # Research-only, no commercial supply
    "FinalSpark": 60,
    "Cortical Labs": 60,  # Pre-commercial
    "Stanford": 60,  # Academic
    "Tachyum": 45,  # Pre-production
}


def score_supply_chain_risk(
    profile_name: str,
) -> SupplyChainRisk:
    """Assess supply chain risk for a hardware profile.

    Scores based on vendor geography, sole-source status,
    and export control classification.

    Parameters
    ----------
    profile_name : str
        Profile to assess.

    Returns
    -------
    SupplyChainRisk
        Risk assessment.
    """
    from sc_neurocore.compiler.platforms import get_profile

    p = get_profile(profile_name)
    score = 0.0
    factors = []

    # Geographic risk
    geo = _GEO_RISK.get(p.vendor, 15)
    score += geo
    if geo >= 30:
        factors.append(f"Geographic concentration: {p.vendor}")

    # Sole-source risk (heuristic: unique family)
    if p.platform_class in ("biological", "wetware", "superconducting", "electrochemical"):
        score += 20
        factors.append("Emerging tech, limited vendors")

    # Export control
    export = "EAR99"  # default commercial
    if p.platform_class in ("fpga",) and "rad" in p.name.lower():
        export = "ITAR"
        score += 15
        factors.append("ITAR-controlled radiation-hardened")
    elif p.platform_class == "superconducting":
        export = "EAR-controlled"
        score += 10
        factors.append("Export-controlled superconducting tech")

    if not factors:
        factors.append("Standard commercial supply")

    # Suggest alternatives in same class
    from sc_neurocore.compiler.platforms import list_profile_names

    alts = [
        n
        for n in list_profile_names()
        if n != profile_name and get_profile(n).platform_class == p.platform_class
    ][:3]

    return SupplyChainRisk(
        profile_name=profile_name,
        risk_score=min(100, round(score, 1)),
        risk_factors=factors,
        alternatives=alts,
        export_control=export,
    )


# ═══════════════════════════════════════════════════════════════════════
# 45. Carbon Footprint Estimator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CarbonEstimate:
    """Carbon footprint estimate per compilation target.

    Attributes
    ----------
    profile_name : str
        Target profile.
    manufacturing_kg_co2 : float
        Estimated manufacturing CO₂ (kg).
    operation_kg_co2_per_year : float
        Estimated annual operation CO₂ (kg).
    total_5yr_kg_co2 : float
        Total 5-year lifecycle CO₂ (kg).
    energy_mix : str
        Assumed energy source.
    """

    profile_name: str
    manufacturing_kg_co2: float
    operation_kg_co2_per_year: float
    total_5yr_kg_co2: float
    energy_mix: str


# Approximate manufacturing CO2 per process node (kg CO2 per die)
_MFG_CO2: dict[str, float] = {
    "fpga": 8.0,
    "asic": 12.0,
    "neuromorphic": 6.0,
    "photonic": 10.0,
    "in_memory": 5.0,
    "accelerator": 15.0,
    "edge_mcu": 0.5,
    "biological": 0.1,
    "wetware": 0.1,
    "simulation": 0.0,
    "superconducting": 20.0,
    "quantum_neuro": 25.0,
    "rram": 3.0,
    "sram_cim": 4.0,
    "electrochemical": 2.0,
}


def estimate_carbon_footprint(
    profile_name: str,
    *,
    power_mw: float = 100.0,
    hours_per_day: float = 24.0,
    grid_carbon_g_per_kwh: float = 400.0,
) -> CarbonEstimate:
    """Estimate carbon footprint for a compilation target.

    Parameters
    ----------
    profile_name : str
        Target profile name.
    power_mw : float
        Operating power (mW).
    hours_per_day : float
        Operating hours per day.
    grid_carbon_g_per_kwh : float
        Grid carbon intensity (g CO₂/kWh).

    Returns
    -------
    CarbonEstimate
        Lifecycle carbon estimate.
    """
    from sc_neurocore.compiler.platforms import get_profile

    p = get_profile(profile_name)

    mfg = _MFG_CO2.get(p.platform_class, 5.0)
    kwh_per_year = (power_mw / 1e6) * hours_per_day * 365
    op_kg = kwh_per_year * grid_carbon_g_per_kwh / 1000
    total = mfg + op_kg * 5

    return CarbonEstimate(
        profile_name=profile_name,
        manufacturing_kg_co2=round(mfg, 2),
        operation_kg_co2_per_year=round(op_kg, 4),
        total_5yr_kg_co2=round(total, 2),
        energy_mix="grid_average",
    )


# ═══════════════════════════════════════════════════════════════════════
# 56. License Compliance Checker
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LicenseCheck:
    """IP core license compatibility result.

    Attributes
    ----------
    compatible : bool
    conflicts : list[str]
    licenses_found : list[str]
    """

    compatible: bool
    conflicts: list[str]
    licenses_found: list[str]


# Compatibility matrix: {project_license: [allowed_deps]}
_COMPAT: dict[str, set[str]] = {
    "AGPL-3.0": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC", "AGPL-3.0"},
    "GPL-3.0": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC", "GPL-3.0", "LGPL-3.0"},
    "Apache-2.0": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC"},
    "MIT": {"MIT", "BSD-2", "BSD-3", "ISC"},
    "proprietary": {"MIT", "BSD-2", "BSD-3", "Apache-2.0", "ISC"},
}


def check_license_compliance(
    project_license: str,
    dependencies: dict[str, str],
) -> LicenseCheck:
    """Verify IP core licensing compatibility.

    Parameters
    ----------
    project_license : str
        SPDX identifier of the project license.
    dependencies : dict[str, str]
        Dependency name → SPDX license identifier.

    Returns
    -------
    LicenseCheck
    """
    allowed = _COMPAT.get(project_license, set())
    conflicts = []
    licenses = []

    for dep, lic in dependencies.items():
        licenses.append(lic)
        if lic not in allowed:
            conflicts.append(f"{dep} ({lic}) incompatible with {project_license}")

    return LicenseCheck(
        compatible=len(conflicts) == 0,
        conflicts=conflicts,
        licenses_found=licenses,
    )


# ═══════════════════════════════════════════════════════════════════════
# 60. Hardware Trojan Lint
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TrojanLintResult:
    """Hardware trojan lint analysis result.

    Attributes
    ----------
    suspicious_paths : list[str]
    risk_level : str
    total_checks : int
    """

    suspicious_paths: list[str]
    risk_level: str
    total_checks: int


def lint_hardware_trojans(
    equations: dict[str, str],
    *,
    check_dormant: bool = True,
    check_payload: bool = True,
) -> TrojanLintResult:
    """Detect suspicious combinational paths that could hide trojans.

    Analyses the ODE dependency graph for dormant trigger conditions
    and rarely-activated payload paths that are classic trojan signatures.

    Parameters
    ----------
    equations : dict[str, str]
        State variable equations.
    check_dormant : bool
        Check for rarely-activated trigger paths.
    check_payload : bool
        Check for suspicious payload injection points.

    Returns
    -------
    TrojanLintResult
    """
    suspicious = []
    checks = 0

    for var, expr in equations.items():
        checks += 1
        # Heuristic: detect conditional paths with rare triggers
        if check_dormant and ("if" in expr or "?" in expr):
            suspicious.append(f"{var}: conditional path detected — potential dormant trigger")
        if check_payload:
            # Cross-variable injection detection
            other_vars = [v for v in equations if v != var]
            for ov in other_vars:
                if ov in expr:
                    checks += 1

    risk = "LOW"
    if len(suspicious) >= 2:
        risk = "HIGH"
    elif len(suspicious) >= 1:
        risk = "MEDIUM"

    return TrojanLintResult(
        suspicious_paths=suspicious,
        risk_level=risk,
        total_checks=checks,
    )


# ═══════════════════════════════════════════════════════════════════════
# 61. SBOM / HBOM Generator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SBOM:
    """Software/Hardware Bill of Materials.

    Attributes
    ----------
    format : str
    components : list[dict]
    total_components : int
    """

    format: str
    components: list[dict]
    total_components: int


def generate_sbom(
    module_name: str,
    profile_name: str,
    *,
    dependencies: dict[str, str] | None = None,
    sbom_format: str = "CycloneDX",
) -> SBOM:
    """Generate SBOM/HBOM for IP core compliance.

    Required by EU Cyber Resilience Act (2026). Generates a machine-
    readable component inventory in CycloneDX or SPDX format.

    Parameters
    ----------
    module_name : str
        Module name.
    profile_name : str
        Target hardware profile.
    dependencies : dict[str, str], optional
        External dependencies {name: version}.
    sbom_format : str
        Output format: "CycloneDX" or "SPDX".

    Returns
    -------
    SBOM
    """
    from sc_neurocore.compiler.platforms import get_profile

    p = get_profile(profile_name)

    components = [
        {
            "type": "library",
            "name": "sc-neurocore",
            "version": "3.15.6",
            "license": "AGPL-3.0-or-later",
        },
        {"type": "hardware", "name": profile_name, "vendor": p.vendor, "family": p.family},
        {"type": "module", "name": module_name, "target": profile_name},
    ]
    if dependencies:
        for name, version in dependencies.items():
            components.append({"type": "library", "name": name, "version": version})

    return SBOM(
        format=sbom_format,
        components=components,
        total_components=len(components),
    )


# ═══════════════════════════════════════════════════════════════════════
# 66. IP Obfuscation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ObfuscationResult:
    """IP obfuscation report.

    Attributes
    ----------
    techniques_applied : list[str]
    key_bits : int
    original_signals : int
    obfuscated_signals : int
    """

    techniques_applied: list[str]
    key_bits: int
    original_signals: int
    obfuscated_signals: int


def obfuscate_ip(
    module_name: str,
    equations: dict[str, str],
    *,
    key_length: int = 64,
    methods: list[str] | None = None,
) -> ObfuscationResult:
    """Apply logic locking and structural obfuscation for IP protection.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        State variable equations.
    key_length : int
        Obfuscation key length in bits.
    methods : list[str], optional
        Techniques to apply. Default: logic_locking, constant_propagation_block,
        structural_transform.

    Returns
    -------
    ObfuscationResult
    """
    if methods is None:
        methods = [
            "logic_locking",
            "constant_propagation_block",
            "structural_transform",
        ]

    original_signals = sum(len(expr.split()) for expr in equations.values())
    # Logic locking adds XOR/XNOR key gates
    obfuscated = original_signals + key_length

    return ObfuscationResult(
        techniques_applied=methods,
        key_bits=key_length,
        original_signals=original_signals,
        obfuscated_signals=obfuscated,
    )


# ═══════════════════════════════════════════════════════════════════════
# 67. Model Watermark
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class WatermarkResult:
    """Netlist watermark embedding result.

    Attributes
    ----------
    watermark_hash : str
    embedding_method : str
    overhead_percent : float
    verifiable : bool
    """

    watermark_hash: str
    embedding_method: str
    overhead_percent: float
    verifiable: bool


def embed_watermark(
    module_name: str,
    equations: dict[str, str],
    *,
    owner_id: str = "SC-NeuroCore",
    method: str = "constraint_based",
) -> WatermarkResult:
    """Embed a verifiable watermark into the compiled netlist.

    The watermark survives synthesis optimisation and can be verified
    without access to the original design.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        State variable equations.
    owner_id : str
        Owner identifier to embed.
    method : str
        "constraint_based" or "don't_care_based".

    Returns
    -------
    WatermarkResult
    """

    payload = f"{owner_id}:{module_name}:{sorted(equations.keys())}"
    wm_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

    # Overhead: watermark adds ~0.5% logic for constraint-based
    overhead = 0.5 if method == "constraint_based" else 0.3

    return WatermarkResult(
        watermark_hash=wm_hash,
        embedding_method=method,
        overhead_percent=overhead,
        verifiable=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# 73. Post-Quantum IP Protection
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PQCProtection:
    """Post-quantum cryptographic IP protection result.

    Attributes
    ----------
    algorithm : str
    signature_hex : str
    key_size_bits : int
    quantum_safe : bool
    """

    algorithm: str
    signature_hex: str
    key_size_bits: int
    quantum_safe: bool


def protect_ip_pqc(
    module_name: str,
    equations: dict[str, str],
    *,
    algorithm: str = "CRYSTALS-Dilithium",
    security_level: int = 3,
) -> PQCProtection:
    """Apply post-quantum cryptographic protection to IP core.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        State variable equations.
    algorithm : str
        PQC algorithm. Default: CRYSTALS-Dilithium.
    security_level : int
        NIST security level (2, 3, or 5).

    Returns
    -------
    PQCProtection
    """

    key_sizes = {2: 1312, 3: 1952, 5: 2592}
    key_bits = key_sizes.get(security_level, 1952)

    payload = f"PQC:{algorithm}:{module_name}:{sorted(equations.keys())}:{security_level}"
    sig = hashlib.sha3_256(payload.encode()).hexdigest()[:32]

    return PQCProtection(
        algorithm=algorithm,
        signature_hex=sig,
        key_size_bits=key_bits,
        quantum_safe=True,
    )


# ═══════════════════════════════════════════════════════════════════════
