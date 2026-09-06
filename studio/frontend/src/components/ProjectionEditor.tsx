// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The inputs for the fields a projection is run with

/**
 * The connection contract, as a form.
 *
 * Every input names its field, states the contract it must satisfy, and
 * carries whatever the server said about that field the last time the graph
 * was validated. The statement and the error are both tied to the input with
 * `aria-describedby`, so a screen reader reads them with the field rather than
 * leaving them somewhere on the page to be found.
 *
 * Typing is kept local until it parses. A user typing `-0.5` passes through
 * `-`, which is not a number; rewriting the box on every keystroke would fight
 * them. The graph is updated as soon as the text is a value, and the server is
 * asked whether that value is admissible.
 */

import { useEffect, useState } from "react";
import type { CSSProperties } from "react";

import type { PopulationNode, ProjectionEdge } from "../api/client";
import type { StudioGraphIssueLocation } from "../studioGraphValidation";
import {
  studioProjectionEdit,
  studioProjectionFieldErrors,
  studioProjectionFields,
  studioProjectionTitle,
  type StudioProjectionField,
  type StudioProjectionFieldModel,
} from "../studioProjectionEditor";

const row: CSSProperties = { display: "flex", flexDirection: "column", gap: 2, padding: "4px 0" };
const labelStyle: CSSProperties = { color: "var(--text-primary)", fontSize: 11, fontWeight: 600 };
const helpStyle: CSSProperties = { color: "var(--text-muted)", fontSize: 10 };
const errorStyle: CSSProperties = { color: "var(--danger, #c0392b)", fontSize: 10 };
const inputStyle: CSSProperties = {
  background: "var(--bg-primary)",
  border: "1px solid var(--control-border)",
  borderRadius: 3,
  color: "var(--text-primary)",
  fontSize: 11,
  padding: "2px 6px",
};

/**
 * What the projection editor needs to edit one projection.
 *
 * The populations are passed in whole because a projection is edited against
 * its endpoints — the source's neuron type decides the sign its weight may
 * take.
 */
export interface ProjectionEditorProps {
  projection: ProjectionEdge;
  populations: PopulationNode[];
  /** Located validation failures for the whole graph. */
  issues?: StudioGraphIssueLocation[];
  onChange: (id: string, update: Partial<ProjectionEdge>) => void;
  /** Ask the server whether the graph is admissible as it now stands. */
  onValidate: () => void;
}

/**
 * Render one editable field of the projection, with everything bound to it.
 *
 * The contract statement and any failure are tied to the input with
 * `aria-describedby`, so a screen reader reads them with the field rather than
 * leaving them somewhere on the page to be found.
 *
 * @returns One labelled input with its contract and any refusal bound to it.
 */
function Field({
  model,
  draft,
  reason,
  onEdit,
}: {
  model: StudioProjectionFieldModel;
  draft: string | undefined;
  reason: string | undefined;
  onEdit: (field: StudioProjectionField, raw: string | boolean) => void;
}) {
  const helpId = `projection-${model.field}-help`;
  const errorId = `projection-${model.field}-error`;
  const messages = reason === undefined ? model.errors : [reason, ...model.errors];
  const described = messages.length > 0 ? `${helpId} ${errorId}` : helpId;
  return (
    <div style={row}>
      <label htmlFor={`projection-${model.field}`} style={labelStyle}>
        {model.label}
        {model.applies ? "" : " (not used by this rule)"}
      </label>
      {model.kind === "choice" ? (
        <select
          id={`projection-${model.field}`}
          value={model.value}
          aria-describedby={described}
          aria-invalid={messages.length > 0}
          onChange={(event) => { onEdit(model.field, event.target.value); }}
          style={inputStyle}
        >
          {model.choices.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      ) : model.kind === "checkbox" ? (
        <input
          id={`projection-${model.field}`}
          type="checkbox"
          checked={model.checked}
          aria-describedby={described}
          aria-invalid={messages.length > 0}
          onChange={(event) => { onEdit(model.field, event.target.checked); }}
          style={{ alignSelf: "flex-start" }}
        />
      ) : (
        <input
          id={`projection-${model.field}`}
          type="text"
          inputMode="decimal"
          value={draft ?? model.value}
          disabled={!model.applies}
          aria-describedby={described}
          aria-invalid={messages.length > 0}
          onChange={(event) => { onEdit(model.field, event.target.value); }}
          style={inputStyle}
        />
      )}
      <span id={helpId} style={helpStyle}>
        {model.help}
      </span>
      {messages.length > 0 && (
        <span id={errorId} style={errorStyle} role="alert">
          {messages.join(" ")}
        </span>
      )}
    </div>
  );
}

/**
 * Edit one projection's executed fields.
 *
 * @returns The editor for one projection's executed fields.
 */
export default function ProjectionEditor({
  projection,
  populations,
  issues = [],
  onChange,
  onValidate,
}: ProjectionEditorProps) {
  const [drafts, setDrafts] = useState<Partial<Record<StudioProjectionField, string>>>({});
  const [reasons, setReasons] = useState<Partial<Record<StudioProjectionField, string>>>({});

  // A different projection is a different set of values; carrying the previous
  // one's half-typed text into it would show one edge the other's numbers.
  useEffect(() => {
    setDrafts({});
    setReasons({});
  }, [projection.id]);

  const source = populations.find((population) => population.id === projection.source);
  const models = studioProjectionFields(projection, source, issues);
  const wholeProjection = studioProjectionFieldErrors(issues, projection.id).get("") ?? [];

  const onEdit = (field: StudioProjectionField, raw: string | boolean): void => {
    if (typeof raw === "string") {
      setDrafts((current) => ({ ...current, [field]: raw }));
    }
    const edit = studioProjectionEdit(field, raw);
    if (!edit.ok) {
      setReasons((current) => ({ ...current, [field]: edit.reason }));
      return;
    }
    setReasons((current) =>
      Object.fromEntries(Object.entries(current).filter(([name]) => name !== field)),
    );
    onChange(projection.id, edit.update);
    onValidate();
  };

  return (
    <section
      aria-label={`Projection ${studioProjectionTitle(projection, populations)}`}
      style={{
        borderLeft: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        gap: 2,
        minWidth: 220,
        overflow: "auto",
        padding: "8px 12px",
      }}
    >
      <h3 style={{ fontSize: 12, margin: 0 }}>
        {studioProjectionTitle(projection, populations)}
      </h3>
      {wholeProjection.length > 0 && (
        <span style={errorStyle} role="alert">
          {wholeProjection.join(" ")}
        </span>
      )}
      {models.map((model) => (
        <Field
          key={model.field}
          model={model}
          draft={drafts[model.field]}
          reason={reasons[model.field]}
          onEdit={onEdit}
        />
      ))}
    </section>
  );
}
