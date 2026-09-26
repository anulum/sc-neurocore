// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Shared-sample experiment comparison

import { useState } from "react";
import { compareMeasurements, replayCohort, submitCohort, type CohortResult, type MeasuredPareto } from "../api/fitsApi";
import { downloadBrowserArtefact } from "../browserArtefactDownload";
import { useLaboratoryJob } from "../useLaboratoryJob";

/**
 * Execute complete imported protocols, preserving all trial outcomes and receipts.
 *
 * @returns The cohort workbench.
 */
export default function CohortPanel() {
  const [document, setDocument] = useState<unknown>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [comparison, setComparison] = useState<MeasuredPareto | null>(null);
  const [replaying, setReplaying] = useState(false);
  const task = useLaboratoryJob<CohortResult>("sc-studio-cohort-job");
  const result = task.result;

  /**
   * Parse a protocol before sending it through scientific admission.
   *
   * @param file - Complete versioned cohort document.
   */
  async function importDocument(file: File): Promise<void> {
    try {
      const parsed: unknown = JSON.parse(await file.text());
      setDocument(parsed); setComparison(null); setMessage(`Imported ${file.name}; execution validates units, splits and budgets.`);
    } catch (caught) { setMessage(caught instanceof Error ? caught.message : String(caught)); }
  }

  return <section aria-label="Experiment cohorts">
    <h3>Experiment cohorts</h3>
    <p>Compare models over the same explicit input and noise samples. Sweep values, acquisition groups,
      training selection and held-out metrics are preserved in the full experiment document.</p>
    <label htmlFor="cohort-import">Import experiment cohort JSON</label>{" "}
    <input id="cohort-import" type="file" accept=".json,application/json" onChange={(event) => {
      const file = event.target.files?.[0]; event.target.value = "";
      if (file !== undefined) void importDocument(file);
    }} />
    <button type="button" disabled={document === null || task.busy} onClick={() => {
      setComparison(null); void task.start(() => submitCohort(document));
    }}>Run cohort</button>
    {task.busy && <button type="button" onClick={() => { void task.cancel(); }}>Cancel cohort</button>}
    <div role="status" aria-label="Cohort status">
      {task.id !== null && <p>Background cohort: {task.status} ({task.id})</p>}
      {message !== null && <p>{message}</p>}{task.error !== null && <p>{task.error}</p>}
    </div>
    {result !== null && <>
      <table>
        <caption>Complete sweep trials</caption>
        <thead><tr>{["Model", "Parameters", "Metric / unit", "Status", "Samples / split / value"].map((heading) => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
        <tbody>{result.trials.map((trial) => <tr key={trial.trial_sha256}>
          <th scope="row">{trial.model}</th>
          <td>{Object.entries(trial.parameters).map(([name, value]) => `${name}=${value}`).join(", ")}</td>
          <td>{trial.metric.kind} / {trial.metric.unit}</td>
          <td>{trial.status}{trial.rejected_constraints.length > 0 && ` (${trial.rejected_constraints.join(", ")})`}</td>
          <td>{trial.samples.map((sample) => `${sample.sample}: ${sample.split} ${sample.failed ? "failed" : sample.value?.toPrecision(5)}`).join("; ")}</td>
        </tr>)}</tbody>
      </table>
      <ul aria-label="Training-only model selection">{result.selection.map((row) => <li key={row.model}>
        {row.model}: {row.training_metric === undefined ? row.reason : `training metric ${row.training_metric.toPrecision(5)}`}
      </li>)}</ul>
      <p>{result.measurement_status}</p>
      <button type="button" onClick={() => {
        downloadBrowserArtefact(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }), "cohort-result.json");
      }}>Export full cohort result</button>{" "}
      <button type="button" disabled={replaying} onClick={() => {
        setReplaying(true); void replayCohort(result).then((replay) => { setMessage(`Cohort replay ${replay.reproduced ? "reproduced" : "did not reproduce"} the full result.`); })
          .catch((caught: unknown) => { setMessage(caught instanceof Error ? caught.message : String(caught)); }).finally(() => { setReplaying(false); });
      }}>Replay full cohort</button>
      <p>Hardware tradeoffs require comparable externally acquired receipts. Simulation steps are never converted to energy.</p>
      <label htmlFor="measurement-import">Import measurement receipts JSON</label>{" "}
      <input id="measurement-import" type="file" accept=".json,application/json" onChange={(event) => {
        const file = event.target.files?.[0]; event.target.value = "";
        if (file !== undefined) void file.text().then(async (text) => {
          const receipts: unknown = JSON.parse(text);
          setComparison(await compareMeasurements(result, receipts));
        }).catch((caught: unknown) => { setMessage(caught instanceof Error ? caught.message : String(caught)); });
      }} />
      {comparison !== null && <div aria-label="Measured tradeoffs">
        <p>{comparison.custody}</p>
        {!comparison.comparable ? <p>No frontier: {comparison.reason}</p> : <table>
          <caption>Comparable supplied measurements (all axes minimised)</caption>
          <thead><tr>{["Model", `Holdout error (${comparison.metric?.unit ?? ""})`, "Latency (ms)", `Resources (${comparison.contract?.resource_unit ?? ""})`, "Energy (J)", "Nondominated"].map((heading) => <th key={heading} scope="col">{heading}</th>)}</tr></thead>
          <tbody>{comparison.rows.map((row) => <tr key={row.trial_sha256}><th scope="row">{row.model}</th>
            <td>{row.holdout_error}</td><td>{row.latency_ms}</td><td>{row.resources}</td><td>{row.energy_j}</td><td>{row.nondominated ? "yes" : "no"}</td></tr>)}</tbody>
        </table>}
      </div>}
    </>}
  </section>;
}
