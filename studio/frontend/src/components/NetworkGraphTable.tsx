// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Keyboard and screen-reader equivalent of the Network Canvas

/**
 * The canvas as a table.
 *
 * Everything the diagram shows is here in text and reachable by keyboard: what
 * each population is, what drives it, what reaches it and what it reaches, and
 * a control to delete it that says what deleting it takes with it.
 */

import type { CSSProperties } from "react";

import type { PopulationNode, ProjectionEdge } from "../api/client";
import type { StudioGraphIssueLocation } from "../studioGraphValidation";
import {
  STUDIO_GRAPH_TABLE_COLUMNS,
  studioGraphTable,
  studioGraphTableRemoveLabel,
  type StudioGraphTableConnection,
} from "../studioGraphTable";

const cell: CSSProperties = {
  border: "1px solid var(--border)",
  fontSize: 10,
  padding: "3px 6px",
  textAlign: "left",
  verticalAlign: "top",
};

const headerCell: CSSProperties = { ...cell, fontWeight: 600 };

/** Text placed for a screen reader without taking visual space. */
const offscreen: CSSProperties = {
  clip: "rect(0 0 0 0)",
  clipPath: "inset(50%)",
  height: 1,
  overflow: "hidden",
  position: "absolute",
  whiteSpace: "nowrap",
  width: 1,
};

/**
 * List the projections at one end of a population, or say there are none.
 *
 * An empty cell reads as missing information; `none` reads as an answer.
 *
 * @returns The connections as a list, or the word `none`.
 */
function Connections({ connections }: { connections: StudioGraphTableConnection[] }) {
  if (connections.length === 0) {
    return <span style={{ color: "var(--text-muted)" }}>none</span>;
  }
  return (
    <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
      {connections.map((connection) => (
        <li key={connection.id}>
          {connection.populationLabel}{" "}
          <span style={{ color: "var(--text-muted)" }}>{connection.detail}</span>
          <Problems messages={connection.issues} />
        </li>
      ))}
    </ul>
  );
}

/**
 * What validation refused about one object, or nothing at all.
 *
 * @returns The refusals as a list, or nothing at all when there are none.
 */
function Problems({ messages }: { messages: string[] }) {
  if (messages.length === 0) {
    return null;
  }
  return (
    <ul style={{ color: "var(--danger, #c0392b)", listStyle: "none", margin: 0, padding: 0 }}>
      {messages.map((message) => (
        <li key={message}>{message}</li>
      ))}
    </ul>
  );
}

/**
 * What the table needs to stand in for the canvas.
 *
 * It takes the graph and the validation issues rather than reading them from
 * the store, so the same table can be rendered for a graph that is not the one
 * currently being edited.
 */
export interface NetworkGraphTableProps {
  populations: PopulationNode[];
  projections: ProjectionEdge[];
  /** Located validation failures; empty when the graph has not been refused. */
  issues?: StudioGraphIssueLocation[];
  onRemovePopulation: (id: string) => void;
}

/**
 * Render the graph as a table with a caption stating its topology.
 *
 * The population is the row header, so a screen reader announces which
 * population a cell belongs to, and each row also carries one sentence
 * describing it in full for a reader who does not want to walk the cells.
 *
 * @returns The topology as a table, captioned with its size.
 */
export default function NetworkGraphTable({
  populations,
  projections,
  issues = [],
  onRemovePopulation,
}: NetworkGraphTableProps) {
  const table = studioGraphTable(populations, projections, issues);
  return (
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <caption style={{ captionSide: "top", fontSize: 11, padding: "4px 0", textAlign: "left" }}>
        {table.caption}
      </caption>
      <thead>
        <tr>
          {STUDIO_GRAPH_TABLE_COLUMNS.map((column) => (
            <th key={column} scope="col" style={headerCell}>
              {column}
            </th>
          ))}
          <th scope="col" style={headerCell}>
            Actions
          </th>
        </tr>
      </thead>
      <tbody>
        {table.rows.map((row) => (
          <tr key={row.id}>
            <th scope="row" style={headerCell}>
              <span>{row.label}</span>
              <span style={offscreen}> — {row.description}</span>
            </th>
            <td style={cell}>{row.model}</td>
            <td style={cell}>{row.count}</td>
            <td style={cell}>{row.neuronType}</td>
            <td style={cell}>{row.drive}</td>
            <td style={cell}>
              <Connections connections={row.incoming} />
            </td>
            <td style={cell}>
              <Connections connections={row.outgoing} />
            </td>
            <td style={cell}>
              {row.issues.length === 0 ? (
                <span style={{ color: "var(--text-muted)" }}>none</span>
              ) : (
                <Problems messages={row.issues} />
              )}
            </td>
            <td style={cell}>
              <button
                type="button"
                onClick={() => { onRemovePopulation(row.id); }}
                aria-label={studioGraphTableRemoveLabel(row)}
                style={{
                  background: "transparent",
                  border: "1px solid var(--control-border)",
                  borderRadius: 3,
                  color: "var(--text-muted)",
                  cursor: "pointer",
                  fontSize: 10,
                  padding: "2px 8px",
                }}
              >
                Delete
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
