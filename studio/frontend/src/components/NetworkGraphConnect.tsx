// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Keyboard path for connecting two populations

import { useState } from "react";

import type { PopulationNode } from "../api/client";

const control: React.CSSProperties = {
  fontSize: 10,
  fontFamily: "var(--font-mono)",
  padding: "3px 4px",
  background: "var(--bg-tertiary)",
  color: "var(--text-primary)",
  border: "1px solid var(--control-border)",
  borderRadius: "var(--radius)",
};

/** What the keyboard connect controls need to create a projection. */
export interface NetworkGraphConnectProps {
  /** Populations available as either end of the new projection. */
  populations: PopulationNode[];
  /** Creates the projection. The same store action the canvas drag calls. */
  onConnect: (sourceId: string, targetId: string) => void;
}

/**
 * Create a projection without a pointer.
 *
 * On the canvas a projection is made by dragging between node handles, which
 * no keyboard reaches. Populations could be added, edited, deleted and undone
 * from the keyboard, but not connected, so a network could not be built
 * without a mouse at all.
 *
 * Both ends are chosen by name rather than by position, and the button calls
 * the same store action the drag calls — including its Dale's-principle sign
 * derivation and its failure reporting — so the two paths cannot diverge.
 *
 * @param props - The populations to offer and the action that connects them.
 * @returns The connect controls, or a note when there is nothing to connect.
 */
export default function NetworkGraphConnect({
  populations,
  onConnect,
}: NetworkGraphConnectProps) {
  const [sourceId, setSourceId] = useState("");
  const [targetId, setTargetId] = useState("");

  if (populations.length < 2) {
    return (
      <p data-testid="graph-connect-empty" style={{ fontSize: 10, color: "var(--text-muted)" }}>
        Add a second population to connect one.
      </p>
    );
  }

  const ready = sourceId !== "" && targetId !== "";
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center", marginTop: 6 }}>
      <label style={{ fontSize: 10 }}>
        Source{" "}
        <select
          aria-label="Projection source population"
          value={sourceId}
          onChange={(e) => { setSourceId(e.target.value); }}
          style={control}
        >
          <option value="">choose</option>
          {populations.map((population) => (
            <option key={population.id} value={population.id}>{population.label}</option>
          ))}
        </select>
      </label>
      <label style={{ fontSize: 10 }}>
        Target{" "}
        <select
          aria-label="Projection target population"
          value={targetId}
          onChange={(e) => { setTargetId(e.target.value); }}
          style={control}
        >
          <option value="">choose</option>
          {populations.map((population) => (
            <option key={population.id} value={population.id}>{population.label}</option>
          ))}
        </select>
      </label>
      <button
        type="button"
        data-testid="graph-connect-submit"
        disabled={!ready}
        aria-label="Connect the chosen source population to the chosen target population"
        onClick={() => { onConnect(sourceId, targetId); }}
        style={{ ...control, cursor: ready ? "pointer" : "default" }}
      >
        Connect
      </button>
    </div>
  );
}
