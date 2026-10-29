// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware registration browser

import { useEffect, useState } from "react";

import { fetchHardwareCatalogue } from "../api/hardwareApi";
import type { HardwareCatalogueEntry, HardwareCatalogueResponse } from "../hardwareCatalogueTypes";

/**
 * Show one registration, keeping planning and physical evidence distinct.
 *
 * @param props - Registration, current FPGA target and model-mode restrictions.
 * @returns Source, constraints and an explicit target-selection action for FPGA profiles.
 */
function HardwareEntry({ entry, selectedTarget, modelMode, onSelectTarget }: {
  entry: HardwareCatalogueEntry;
  selectedTarget: string;
  modelMode: boolean;
  onSelectTarget: (target: string) => void;
}) {
  const profile = entry.synthesis_profile;
  const nir = entry.nir_profile;
  return (
    <article aria-label={entry.name} style={{ borderTop: "1px solid var(--border)", padding: "10px 0" }}>
      <h4 style={{ margin: "0 0 4px" }}>{entry.name}</h4>
      <p>Registered {entry.kind} path · Physical evidence: no receipts attached</p>
      {entry.planning_profile && <p>Resource-planning profile; its assumptions are not measured specifications.</p>}
      {nir !== null && (
        <div>
          <p>NIR manifest: {nir.backend_status}. Declared nodes: {nir.supported_nir_nodes.join(", ")}.</p>
          <p>Unsupported nodes: {nir.unsupported_nir_nodes.join(", ")}.</p>
          <p>Declared transport: {nir.sc_constraints.stream_transport}.</p>
          <p>Precision modes: {nir.sc_constraints.precision_modes.join(", ")}.</p>
          <p>Bitstream lengths: {nir.sc_constraints.bitstream_lengths.join(", ")}.</p>
        </div>
      )}
      {entry.adapter_package && <p>Adapter handoff package exists; vendor runtime execution is not attested.</p>}
      {entry.driver !== null && <p>Python driver: <code>{entry.driver}</code></p>}
      {profile !== null && (
        <div>
          <p>Synthesis command: <code>{profile.synthesis_command}</code>.</p>
          <p>Device selector: {profile.device_selector ?? "not device-bound"}.
            {" "}Package selector: {profile.package_selector ?? "not specified by this registration"}.
            {" "}Routing tool: {profile.routing_tool ?? "not offered by this profile"}.</p>
          <p>Reference capacity ({profile.capacity_device}): {Object.entries(profile.capacity)
            .map(([name, value]) => `${name}: ${String(value)}`).join(", ")}.</p>
          <button
            aria-pressed={selectedTarget === profile.target}
            disabled={modelMode && profile.target !== "ice40" && profile.target !== "ecp5"}
            onClick={() => { onSelectTarget(profile.target); }}
          >Use {profile.target.toUpperCase()} target</button>
          <p>Selection changes the synthesis target only. It does not start a tool or program a board.
            {" "}Catalogue-model routing remains limited to its admitted targets.</p>
        </div>
      )}
      <p>Required: {entry.requirements.join("; ")}.</p>
      <ul>{entry.limitations.map((limit) => <li key={limit}>{limit}</li>)}</ul>
      <p>Registration source: {entry.registration_sources.join("; ")}.</p>
    </article>
  );
}

/**
 * Browse current device, NIR, synthesis and board-driver registrations.
 *
 * @param props - Existing synthesis selection and supported model-mode restrictions.
 * @returns Accessible catalogue loading, refusal, filtering and selection controls.
 */
export default function HardwareCatalogue({ selectedTarget, modelMode, onSelectTarget, onTargetsLoaded, defaultExpanded = false }: {
  selectedTarget: string;
  modelMode: boolean;
  onSelectTarget: (target: string) => void;
  onTargetsLoaded?: (targets: string[]) => void;
  defaultExpanded?: boolean;
}) {
  const [catalogue, setCatalogue] = useState<HardwareCatalogueResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState("all");

  useEffect(() => {
    let current = true;
    setCatalogue(null);
    setError(null);
    void fetchHardwareCatalogue().then((response) => {
      if (current) {
        setCatalogue(response);
        onTargetsLoaded?.(response.entries.flatMap((entry) =>
          entry.synthesis_profile === null ? [] : [entry.synthesis_profile.target],
        ));
      }
    }).catch((cause: unknown) => {
      if (current) setError(cause instanceof Error ? cause.message : "Hardware catalogue request failed.");
    });
    return () => { current = false; };
  }, [reload, onTargetsLoaded]);

  const query = search.trim().toLowerCase();
  const entries = catalogue === null ? [] : catalogue.entries.filter((entry) =>
    (kind === "all" || entry.kind === kind)
    && `${entry.name} ${entry.target_id}`.toLowerCase().includes(query),
  );

  return (
    <section aria-label="Hardware catalogue" style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)" }}>
      <details open={defaultExpanded}>
        <summary>Hardware catalogue</summary>
        <p>Browse registered software paths before choosing hardware.
          Tool reachability is shown separately by the synthesis tool status.
          Registration is not a vendor conformance or physical measurement verdict.</p>
        {error !== null && (
          <div role="alert">
            <p>{error}</p>
            <button onClick={() => { setReload((value) => value + 1); }}>Retry hardware catalogue</button>
          </div>
        )}
        {catalogue === null && error === null && <p role="status">Loading hardware catalogue…</p>}
        {catalogue !== null && (
          <div>
            <p>{catalogue.evidence_scope}</p>
            <label>Search hardware
              <input value={search} onChange={(event) => { setSearch(event.target.value); }} />
            </label>
            <label>Hardware category
              <select value={kind} onChange={(event) => { setKind(event.target.value); }}>
                <option value="all">All registrations</option>
                <option value="neuromorphic">Neuromorphic profiles</option>
                <option value="fpga">FPGA profiles</option>
                <option value="board">Board drivers</option>
              </select>
            </label>
            <p role="status">{entries.length} matching registrations</p>
            {entries.map((entry) => (
              <HardwareEntry key={entry.target_id} entry={entry} selectedTarget={selectedTarget}
                modelMode={modelMode} onSelectTarget={onSelectTarget} />
            ))}
          </div>
        )}
      </details>
    </section>
  );
}
