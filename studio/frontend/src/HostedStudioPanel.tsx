// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hosted metadata view without a backend

import { useEffect, useState } from "react";
import { PANEL_TITLES } from "./capabilityShell";
import { HostCatalogueRefusal, requestHostFederation, type HostedStudio } from "./hostBridge";

/** Lifecycle of the single host metadata request; none grants backend access. */
type HostState =
  | { status: "loading" }
  | { status: "ready"; studio: HostedStudio }
  | { status: "unavailable"; message: string };

/**
 * Show the host's SC-NeuroCore metadata and the hosted execution boundary.
 *
 * Starts one bounded bridge request per mount; unmount cancels it. A refused,
 * missing or malformed catalogue stays unavailable. Neither success nor retry
 * mounts the standalone workspace or grants any backend feature.
 *
 * @returns The read-only hosted Studio view, with explicit feature availability.
 */
export default function HostedStudioPanel() {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<HostState>({ status: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });
    void requestHostFederation(controller.signal).then((studios) => {
      if (controller.signal.aborted) return;
      const own = studios.filter((studio) => studio.studio === "sc-neurocore");
      const studio = own[0];
      setState(own.length === 1 && studio !== undefined
        ? { status: "ready", studio }
        : { status: "unavailable", message: "Host catalogue does not uniquely identify SC-NeuroCore." });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setState({ status: "unavailable", message: error instanceof HostCatalogueRefusal
          ? error.message : "Host catalogue is unavailable." });
      }
    });
    return () => { controller.abort(); };
  }, [attempt]);

  return (
    <section className="panel-section" aria-label="Hosted SC-NeuroCore Studio">
      <h1>SC-NeuroCore Studio</h1>
      <p>Hosted view · no backend connected</p>
      <p>This view displays the Hub's declared catalogue metadata. It does not
        attest installed runtime, scientific validity or hardware support.</p>
      <section aria-label="Host catalogue">
        <h2>Host catalogue</h2>
        {state.status === "loading" ? <p role="status">Reading host catalogue…</p>
          : state.status === "unavailable" ? <>
            <p role="alert">{state.message}</p>
            <button type="button" onClick={() => { setAttempt((value) => value + 1); }}>
              Retry host catalogue
            </button>
          </> : <dl>
            <dt>Studio</dt><dd>{state.studio.studio}</dd>
            <dt>Version</dt><dd>{state.studio.version}</dd>
            <dt>Contract era</dt><dd>{state.studio.era}</dd>
            <dt>Declared evidence types</dt>
            <dd>{state.studio.evidenceTypes.join(", ") || "None declared"}</dd>
          </dl>}
      </section>
      <section aria-label="Hosted feature availability">
        <h2>Workspace availability</h2>
        <p>Simulation, analysis, training, compilation, co-simulation, FPGA synthesis,
          jobs and artefact downloads are not available in hosted view.</p>
        <p>The model catalogue, projects, sessions, evidence export, authentication
          and operator controls are not available in hosted view.</p>
        <details>
          <summary>Workbench views — not available in hosted view</summary>
          <ul>{Object.entries(PANEL_TITLES).map(([key, title]) => (
            <li key={key}>{title}: not available in hosted view</li>
          ))}</ul>
        </details>
        <p>Use the standalone Studio with its configured local backend for these workflows.</p>
      </section>
    </section>
  );
}
