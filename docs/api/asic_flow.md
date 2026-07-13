# ASIC Flow

The ASIC-flow package materialises deterministic Yosys, OpenROAD, OpenSTA,
KLayout, Magic, and Netgen inputs for Sky130, GF180MCU, commercial-template,
and custom process configurations. It does not execute those external tools or
claim physical area, power, timing, DRC, LVS, or GDSII closure.

## Architecture

The historical `sc_neurocore.asic_flow.asic_flow` import remains available, but
implementation ownership is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `pdk` | PDK presets, path resolution, installation checks, and validation |
| `design` | physical-design and stochastic-synthesis parameters |
| `decks` | Yosys, OpenROAD, SDC, and GDSII scripts |
| `signoff` | STA/DRC/LVS scripts, PVT corners, OCV, and summaries |
| `constraints` | CDC, IR-drop, IO-placement, and equivalence scripts |
| `estimation` | deterministic pre-synthesis screening estimates |
| `flow` | complete deck generation, bundle writes, and evidence manifests |
| `hierarchy` | per-block synthesis and hard-macro top integration |
| `readiness` | evidence-derived tape-out checklist state |

The modules form an acyclic import graph. All 38 historical definitions retain
their original qualified names and pickle paths. The package root intentionally
keeps the narrower one-command API:

```python
from sc_neurocore.asic_flow import ASICFlowBundle, generate_asic_flow_bundle
```

## Generate a bundle

```python
from sc_neurocore.asic_flow.asic_flow import DesignParams, generate_asic_flow_bundle

bundle = generate_asic_flow_bundle(
    "build/asic/sky130_demo",
    pdk_type="sky130",
    design=DesignParams(
        top_module="sc_neurocore_top",
        rtl_files=["rtl/top.sv"],
    ),
    pdk_root="/opt/pdks",
    require_pdk_files=True,
    n_neurons=32,
    n_synapses=512,
    bitstream_width=256,
    n_aer_ports=8,
    formal_evidence_artifacts=[
        "formal/sc_neurocore_top.sby",
        "formal/report.json",
    ],
)

print(bundle.manifest_path)
print(bundle.pdk_resolution.usable_for_synthesis)
```

The bundle contains nine generated flow files and
`asic_flow_manifest.json`. When `require_pdk_files=True`, missing Liberty,
cell-LEF, technology-LEF, setup, DRC, and LVS inputs are recorded rather than
hidden. Formal evidence is complete for a claim only when at least one proof
source (`.sby`, `.sv`, or `.sva`) and one report (`.json`, `.txt`, or `.log`)
are attached.

The manifest always reports `external_eda_executed: false` and
`physical_ppa_claim_allowed: false`. A downstream run must attach exact tool,
container, PDK revision, command, and signoff artefacts before physical claims
are made.

## Polyglot boundary

Deck construction is typed text and filesystem orchestration, not a numerical
kernel. Earlier Rust, Go, Julia, and Mojo files named for `asic_flow` were
nonfunctional generated mirrors and have been removed. The maintained execution
boundary is the external EDA toolchain; no language-speed comparison is claimed
for removed code.

## Verification and benchmark evidence

The focused suite contains 100 tests and covers all 580 statements and 66
branches across the 11 package files. It locks the historical facade, exclusive
symbol ownership, the real import DAG, pickle paths, package API, responsibility
size limits, removed false mirrors, and live benchmark hashes.

`benchmarks/bench_asic_flow.py` compares the pre-refactor source archive with
the modular candidate over 30 interleaved cold processes. Both variants emit
the same 10,465-byte canonical payload at SHA-256
`ae901f9b10bdc61f0997964d6143568994625bdf89080f02cd58efbc83099653`.
The committed capture used CPU affinity without exclusive isolation while the
workstation was under load, so its timing medians are local regression context,
not publishable throughput evidence.

## API reference

::: sc_neurocore.asic_flow.asic_flow
