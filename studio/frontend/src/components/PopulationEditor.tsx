// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The inputs for the fields a population is run with

/**
 * The model contract, as a form.
 *
 * Identity, external input and the model's own parameters, each input naming
 * its field, stating what it must satisfy, and carrying whatever the server
 * said about that field the last time the graph was validated — all tied to
 * the input with `aria-describedby`.
 *
 * The parameters section waits for the contract rather than guessing: which
 * constructor fields a population may override is the server's answer. The
 * fields that are **not** inputs are listed with the reason each is not,
 * because a user who cannot find one should read why rather than conclude the
 * editor is incomplete.
 */

import { useEffect, useState } from "react";
import type { CSSProperties } from "react";

import type { PopulationModelContract, PopulationNode } from "../api/client";
import type { StudioGraphIssueLocation } from "../studioGraphValidation";
import {
  studioPopulationDriveFields,
  studioPopulationEdit,
  studioPopulationFieldErrors,
  studioPopulationFields,
  studioPopulationParameterFields,
  studioPopulationUnsupported,
  type StudioPopulationFieldModel,
} from "../studioPopulationEditor";

const row: CSSProperties = { display: "flex", flexDirection: "column", gap: 2, padding: "3px 0" };
const labelStyle: CSSProperties = { color: "var(--text-primary)", fontSize: 11, fontWeight: 600 };
const helpStyle: CSSProperties = { color: "var(--text-muted)", fontSize: 10 };
const errorStyle: CSSProperties = { color: "var(--danger, #c0392b)", fontSize: 10 };
const headingStyle: CSSProperties = { fontSize: 11, margin: "8px 0 0", textTransform: "uppercase" };
const inputStyle: CSSProperties = {
  background: "var(--bg-primary)",
  border: "1px solid var(--control-border)",
  borderRadius: 3,
  color: "var(--text-primary)",
  fontSize: 11,
  padding: "2px 6px",
};

/**
 * Turn a request attribute into an element id a label can point at.
 *
 * @param field - The request attribute, as the server names it.
 * @returns An element id a label can point at.
 */
function elementId(field: string): string {
  return `population-${field.replace(/\./g, "-")}`;
}

/**
 *
 */
export interface PopulationEditorProps {
  population: PopulationNode;
  /** Model names the server admits for a population. */
  models: string[];
  /** The contract of this population's model, or `null` until it arrives. */
  contract: PopulationModelContract | null;
  /** Located validation failures for the whole graph. */
  issues?: StudioGraphIssueLocation[];
  onChange: (id: string, update: Partial<PopulationNode>) => void;
  /** Ask the server whether the graph is admissible as it now stands. */
  onValidate: () => void;
}

/**
 * Render one editable field of the population, with everything bound to it.
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
  model: StudioPopulationFieldModel;
  draft: string | undefined;
  reason: string | undefined;
  onEdit: (field: string, raw: string) => void;
}) {
  const id = elementId(model.field);
  const helpId = `${id}-help`;
  const errorId = `${id}-error`;
  const messages = reason === undefined ? model.errors : [reason, ...model.errors];
  const described = messages.length > 0 ? `${helpId} ${errorId}` : helpId;
  return (
    <div style={row}>
      <label htmlFor={id} style={labelStyle}>
        {model.label}
      </label>
      {model.kind === "choice" ? (
        <select
          id={id}
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
      ) : (
        <input
          id={id}
          type="text"
          inputMode={model.kind === "number" ? "decimal" : "text"}
          value={draft ?? model.value}
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
 * Edit one population's executed fields.
 *
 * @returns The editor for one population's executed fields.
 */
export default function PopulationEditor({
  population,
  models,
  contract,
  issues = [],
  onChange,
  onValidate,
}: PopulationEditorProps) {
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [reasons, setReasons] = useState<Record<string, string>>({});

  // A different population is a different set of values; so is a different
  // model, whose parameters are not the previous one's.
  useEffect(() => {
    setDrafts({});
    setReasons({});
  }, [population.id, population.model]);

  const identity = studioPopulationFields(population, models, issues);
  const drive = studioPopulationDriveFields(population, issues);
  const parameters = studioPopulationParameterFields(population, contract, issues);
  const unsupported = studioPopulationUnsupported(contract);
  const wholePopulation = studioPopulationFieldErrors(issues, population.id).get("") ?? [];

  const onEdit = (field: string, raw: string): void => {
    setDrafts((current) => ({ ...current, [field]: raw }));
    const edit = studioPopulationEdit(population, field, raw);
    if (!edit.ok) {
      setReasons((current) => ({ ...current, [field]: edit.reason }));
      return;
    }
    setReasons((current) =>
      Object.fromEntries(Object.entries(current).filter(([name]) => name !== field)),
    );
    onChange(population.id, edit.update);
    onValidate();
  };

  const render = (model: StudioPopulationFieldModel) => (
    <Field
      key={model.field}
      model={model}
      draft={drafts[model.field]}
      reason={reasons[model.field]}
      onEdit={onEdit}
    />
  );

  return (
    <section
      aria-label={`Population ${population.label}`}
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
      <h3 style={{ fontSize: 12, margin: 0 }}>{population.label}</h3>
      {wholePopulation.length > 0 && (
        <span style={errorStyle} role="alert">
          {wholePopulation.join(" ")}
        </span>
      )}
      {identity.map(render)}
      <h4 style={headingStyle}>Input</h4>
      {drive.map(render)}
      <h4 style={headingStyle}>Parameters</h4>
      {contract?.model !== population.model ? (
        <span style={helpStyle}>
          Waiting for the model contract; parameters are not offered until the server states
          which of them a population may override.
        </span>
      ) : (
        parameters.map(render)
      )}
      {unsupported.length > 0 && contract?.model === population.model && (
        <>
          <h4 style={headingStyle}>Not editable</h4>
          <ul style={{ ...helpStyle, listStyle: "none", margin: 0, padding: 0 }}>
            {unsupported.map((entry) => (
              <li key={entry.name}>
                {entry.name}: {entry.reason}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
