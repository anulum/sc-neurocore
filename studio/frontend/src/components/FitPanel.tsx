// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parameter fitting panel

/**
 * Fit a model's parameters to recordings split into training and hold-out.
 *
 * The model is the selected catalogue model's canonical schema or the
 * workspace's candidate draft. Recordings are CSV files of current and observed
 * value per step, each assigned to training or hold-out; the fit never sees the
 * hold-out recordings. The result states what the data constrain: fitted
 * values with standard errors only when every parameter is identifiable, the
 * unconstrained combinations otherwise, and the hold-out error per recording.
 */

import { useState, type CSSProperties } from "react";

import { submitFitReplay, submitFit, type FitReplay, type FitResult, type FitConstraint } from "../api/fitsApi";
import { StudioRequestError } from "../api/http";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import { parseCandidateText } from "../candidateWorkbench";
import {
  fitIdentifiabilityNotes,
  fitParameterRows,
  parseFixed,
  parseRecordingCsv,
} from "../fitWorkbench";
import type { FitRecording } from "../api/fitsApi";
import { useLaboratoryJob } from "../useLaboratoryJob";
import CohortPanel from "./CohortPanel";
import { useStudioStore } from "../stores/studio";

const button: CSSProperties = {
  background: "transparent", border: "1px solid var(--control-border)", borderRadius: 3,
  color: "var(--text-secondary)", cursor: "pointer", fontSize: "var(--fs-body)", padding: "2px 8px",
};
const cell: CSSProperties = { border: "1px solid var(--border)", fontSize: "var(--fs-body)", padding: "2px 6px", textAlign: "left" };

/** One parameter row as typed. */
interface DomainRow {
  name: string;
  low: string;
  high: string;
  log: boolean;
}

/** One imported recording and the set it belongs to. */
interface CohortRow {
  recording: FitRecording;
  split: "train" | "holdout";
}

/**
 * Read a failure into a sentence: the server's own reason when it gave one.
 *
 * @param error - What the request rejected with.
 * @returns The sentence.
 */
function failureMessage(error: unknown): string {
  if (error instanceof StudioRequestError) {
    const detail = error.detail as { message?: unknown } | null;
    if (detail !== null && typeof detail === "object" && typeof detail.message === "string") {
      return detail.message;
    }
  }
  return error instanceof Error ? error.message : String(error);
}

/**
 * The parameter fitting panel.
 *
 * @returns The panel.
 */
export default function FitPanel() {
  const { selectedModelName, candidates } = useStudioStore();
  const [source, setSource] = useState<"catalogue" | "candidate">("catalogue");
  const [observable, setObservable] = useState("v");
  const [domains, setDomains] = useState<DomainRow[]>([{ name: "", low: "", high: "", log: false }]);
  const [fixedText, setFixedText] = useState("");
  const [constraintText, setConstraintText] = useState("[]");
  const task = useLaboratoryJob<FitResult | (FitReplay & { result: FitResult })>("sc-studio-fit-job");
  const [cohort, setCohort] = useState<CohortRow[]>([]);
  const [seed, setSeed] = useState("0");
  const [generations, setGenerations] = useState("40");
  const [population, setPopulation] = useState("12");
  const result = task.result === null ? null : "fitted" in task.result ? task.result : task.result.result;
  const replay = task.result !== null && "reproduced" in task.result ? task.result : null;
  const [message, setMessage] = useState<string | null>(null);
  const busy = task.busy;

  const updateDomain = (index: number, patch: Partial<DomainRow>) => {
    setDomains((rows) => rows.map((row, at) => (at === index ? { ...row, ...patch } : row)));
  };

  /** Build the request from the form, or say what stops it. */
  async function fit(): Promise<void> {

    const fixed = parseFixed(fixedText);
    if (!fixed.ok) {
      setMessage(fixed.message);
      return;
    }
    let model: { catalogue_model: string } | { schema: Record<string, unknown> };
    if (source === "catalogue") {
      if (selectedModelName === "") {
        setMessage("Select a catalogue model first.");
        return;
      }
      model = { catalogue_model: selectedModelName };
    } else {
      const parsed = parseCandidateText(candidates[0]?.text ?? "");
      const schema = parsed.ok ? (parsed.document as { model?: unknown } | null)?.model : undefined;
      if (typeof schema !== "object" || schema === null) {
        setMessage("The candidate draft has no model to fit.");
        return;
      }
      model = { schema: schema as Record<string, unknown> };
    }
    setMessage(null);
    try {
      const constraints = JSON.parse(constraintText) as FitConstraint[];
      await task.start(() => submitFit({
        ...model,
        observable,
        domains: domains.map((row) => ({
          name: row.name.trim(),
          low: Number(row.low),
          high: Number(row.high),
          scale: row.log ? "log" : "linear",
        })),
        fixed: fixed.value,
        constraints,
        train: cohort.filter((row) => row.split === "train").map((row) => row.recording),
        holdout: cohort.filter((row) => row.split === "holdout").map((row) => row.recording),
        seed: Number(seed),
        generations: Number(generations),
        population: Number(population),
      }));
      setMessage(null);
    } catch (error) {
      setMessage(failureMessage(error));
    }
  }

  const notes = result === null ? [] : fitIdentifiabilityNotes(result);

  return (
    <section aria-label="Parameter fitting" style={{ flex: 1, overflow: "auto", padding: "8px 12px", display: "flex", flexDirection: "column", gap: 8, fontSize: "var(--fs-body)" }}>
      <h2 style={{ fontSize: 13, margin: 0 }}>Parameter fitting</h2>
      <p style={{ margin: 0, color: "var(--text-secondary)" }}>
        The fit sees the training recordings only; hold-out recordings measure how the fitted
        parameters generalise.
      </p>

      {/* The observed variable is its own row: set inline after the radio
          buttons it read as a third model choice. */}
      <fieldset style={{ border: "1px solid var(--border)", display: "flex", gap: 16, flexWrap: "wrap" }}>
        <legend>Model</legend>
        <label><input type="radio" name="fit-source" checked={source === "catalogue"}
          onChange={() => { setSource("catalogue"); }} /> Selected catalogue model ({selectedModelName || "none"})</label>
        <label><input type="radio" name="fit-source" checked={source === "candidate"}
          onChange={() => { setSource("candidate"); }} /> Candidate draft</label>
      </fieldset>
      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        <label htmlFor="fit-observable">Observed variable</label>
        <input id="fit-observable" value={observable} onChange={(event) => { setObservable(event.target.value); }} style={{ width: 80 }} />
        <span className="panel-note">the state the recordings' observed column is compared with</span>
      </div>

      <table style={{ borderCollapse: "collapse" }}>
        <caption style={{ captionSide: "top", textAlign: "left" }}>Parameters to fit</caption>
        <thead>
          <tr>{["Name", "Low", "High", "Log scale", ""].map((column) => <th key={column} scope="col" style={cell}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {domains.map((row, index) => (
            <tr key={index}>
              <td style={cell}><input aria-label={`Parameter ${index + 1} name`} value={row.name}
                onChange={(event) => { updateDomain(index, { name: event.target.value }); }} /></td>
              <td style={cell}><input aria-label={`Parameter ${index + 1} low bound`} value={row.low}
                onChange={(event) => { updateDomain(index, { low: event.target.value }); }} style={{ width: 70 }} /></td>
              <td style={cell}><input aria-label={`Parameter ${index + 1} high bound`} value={row.high}
                onChange={(event) => { updateDomain(index, { high: event.target.value }); }} style={{ width: 70 }} /></td>
              <td style={cell}><input type="checkbox" aria-label={`Parameter ${index + 1} on a log scale`} checked={row.log}
                onChange={(event) => { updateDomain(index, { log: event.target.checked }); }} /></td>
              <td style={cell}><button type="button" style={button} aria-label={`Remove parameter ${index + 1}`}
                onClick={() => { setDomains((rows) => rows.filter((_, at) => at !== index)); }}>Remove</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div>
        <button type="button" style={button}
          onClick={() => { setDomains((rows) => [...rows, { name: "", low: "", high: "", log: false }]); }}>
          Add parameter
        </button>
      </div>

      <label htmlFor="fit-fixed">Fixed parameters (name=value, one per line)</label>
      <textarea id="fit-fixed" rows={3} value={fixedText} onChange={(event) => { setFixedText(event.target.value); }}
        style={{ fontFamily: "var(--font-mono)", fontSize: "var(--fs-body)" }} />

      <label htmlFor="fit-constraints">Parameter constraints (JSON: name, coefficients, low, high)</label>
      <textarea id="fit-constraints" rows={3} value={constraintText} onChange={(event) => { setConstraintText(event.target.value); }} />

      <div>
        <label htmlFor="fit-recordings">Recordings (CSV: current,observed per step)</label>{" "}
        <input id="fit-recordings" type="file" multiple accept=".csv,text/csv"
          onChange={(event) => {
            const files = [...(event.target.files ?? [])];
            event.target.value = "";
            void Promise.all(files.map(async (file) => parseRecordingCsv(file.name, await file.text())))
              .then((parsed) => {
                const imported: CohortRow[] = [];
                for (const entry of parsed) {
                  if (!entry.ok) {
                    setMessage(entry.message);
                    return;
                  }
                  imported.push({ recording: { ...entry.value, group: entry.value.name }, split: "train" });
                }
                setCohort((rows) => [...rows, ...imported]);
                setMessage(null);
              });
          }} />
      </div>
      {cohort.length > 0 && (
        <table style={{ borderCollapse: "collapse" }}>
          <caption style={{ captionSide: "top", textAlign: "left" }}>Cohort</caption>
          <thead>
            <tr>{["Recording", "Samples", "Acquisition group", "Set", ""].map((column) => <th key={column} scope="col" style={cell}>{column}</th>)}</tr>
          </thead>
          <tbody>
            {cohort.map((row, index) => (
              <tr key={row.recording.name}>
                <th scope="row" style={cell}>{row.recording.name}</th>
                <td style={cell}>{row.recording.current.length}</td>
                <td style={cell}><input aria-label={`Acquisition group of ${row.recording.name}`} value={row.recording.group ?? ""}
                  onChange={(event) => { const group = event.target.value; setCohort((rows) => rows.map((each, at) => at === index ? { ...each, recording: { ...each.recording, group } } : each)); }} /></td>
                <td style={cell}>
                  <select aria-label={`Set of ${row.recording.name}`} value={row.split}
                    onChange={(event) => {
                      const split = event.target.value === "holdout" ? "holdout" : "train";
                      setCohort((rows) => rows.map((each, at) => (at === index ? { ...each, split } : each)));
                    }}>
                    <option value="train">training</option>
                    <option value="holdout">hold-out</option>
                  </select>
                </td>
                <td style={cell}><button type="button" style={button} aria-label={`Remove ${row.recording.name}`}
                  onClick={() => { setCohort((rows) => rows.filter((_, at) => at !== index)); }}>Remove</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
        <label htmlFor="fit-seed">Seed</label>
        <input id="fit-seed" type="number" value={seed} onChange={(event) => { setSeed(event.target.value); }} style={{ width: 60 }} />
        <label htmlFor="fit-generations">Generations</label>
        <input id="fit-generations" type="number" value={generations} onChange={(event) => { setGenerations(event.target.value); }} style={{ width: 60 }} />
        <label htmlFor="fit-population">Population</label>
        <input id="fit-population" type="number" value={population} onChange={(event) => { setPopulation(event.target.value); }} style={{ width: 60 }} />
        <button type="button" style={button} disabled={busy} onClick={() => { void fit(); }}>Run fit</button>
      </div>

      <div id="fit-outcome" role="status">
        {message !== null && <p style={{ margin: 0 }}>{message}</p>}
        {task.id !== null && <p>Background fit: {task.status} ({task.id})</p>}
        {task.error !== null && <p>{task.error}</p>}
        {busy && <button type="button" style={button} onClick={() => { void task.cancel(); }}>Cancel fit</button>}
        {result !== null && (
          <div>
            <table style={{ borderCollapse: "collapse" }}>
              <caption style={{ captionSide: "top", textAlign: "left" }}>Fitted parameters</caption>
              <thead>
                <tr>{["Parameter", "Value", "Standard error", "Effective search domain"].map((column) => <th key={column} scope="col" style={cell}>{column}</th>)}</tr>
              </thead>
              <tbody>
                {fitParameterRows(result).map((row) => (
                  <tr key={row.name}>
                    <th scope="row" style={cell}>{row.name}</th>
                    <td style={cell}>{row.value.toPrecision(6)}</td>
                    <td style={cell}>{row.standardError === null ? "not stated" : row.standardError.toPrecision(3)}</td>
                    <td style={cell}>{result.problem.domains.filter((domain) => domain.name === row.name).map((domain) => `${domain.low}…${domain.high} (${domain.scale})`).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.uncertainty.reason !== undefined && <p>{result.uncertainty.reason}</p>}
            {notes.map((note) => <p key={note} style={{ margin: 0 }}>{note}</p>)}
            <p style={{ margin: 0 }}>
              Training loss {result.training_loss?.toPrecision(4) ?? "no feasible finite trial"}; {result.optimiser.generations_run} generations,
              {" "}{result.optimiser.evaluations} trials, {result.optimiser.failed_trials} failed;
              {" "}{result.optimiser.converged ? "converged" : `not converged (${result.optimiser.message})`}.
            </p>
            <details><summary>Optimiser history</summary><ol aria-label="Optimiser generation losses">
              {result.optimiser.history.map((row) => <li key={row.generation}>Generation {row.generation}: {row.loss ?? "no feasible trial"}</li>)}
            </ol></details>
            <ul aria-label="Hold-out error" style={{ margin: 0, paddingLeft: 16 }}>
              {result.holdout.map((row) => (
                <li key={row.recording}>
                  {row.recording}: {row.rmse === null ? "diverged" : `RMSE ${row.rmse.toPrecision(4)}`}
                </li>
              ))}
            </ul>
            <div style={{ display: "flex", gap: 6 }}>
              <button type="button" style={button} onClick={() => {
                downloadBrowserArtefact(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }), "fit-result.json");
              }}>Export result</button>
              <button type="button" style={button} disabled={busy} onClick={() => {
                void task.start(() => submitFitReplay(result));
              }}>Replay</button>
            </div>
            {replay !== null && (
              <p style={{ margin: 0 }}>
                Replay {replay.reproduced ? "reproduced" : "did not reproduce"} the result
                ({replay.replayed_sha256.slice(0, 16)}…).
              </p>
            )}
          </div>
        )}
      </div>
      <CohortPanel />
    </section>
  );
}
