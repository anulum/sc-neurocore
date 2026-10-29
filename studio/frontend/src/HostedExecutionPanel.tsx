// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — hosted local scientific workspace

import { useEffect, useRef, useState } from "react";
import { describeHost, downloadHostArtifact, hostedTextDigest, HostApiRefusal, readHostJob, readHostJson, submitHostJob } from "./hostApi";
import { type HostJob, type HostJson, type HostOperation } from "./hostApiTypes";
import HostedJobResult, { hostedCompilerView } from "./HostedJobResult";
import { hostedExample, HOST_OPERATION_LABELS } from "./hostedExamples";

/** Active worker states; only actual terminal records permit another submission. */
const ACTIVE = new Set(["pending", "running", "cancelling"]);

/**
 * Operate one real local job at a time through the host's closed bridge.
 *
 * @returns Scientific editor, installed operation availability, real jobs and verified results.
 */
export default function HostedExecutionPanel() {
  const lifetime = useRef(new AbortController());
  const [operations, setOperations] = useState<HostOperation[] | null>(null);
  const [operation, setOperation] = useState<HostOperation>("simulate");
  const [source, setSource] = useState(hostedExample("simulate", ""));
  const [job, setJob] = useState<HostJob | null>(null);
  const [result, setResult] = useState<HostJson | null>(null);
  const [catalogue, setCatalogue] = useState<HostJson | null>(null);
  const [graph, setGraph] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController(); lifetime.current = controller;
    void describeHost(controller.signal).then((available) => {
      if (!controller.signal.aborted) setOperations(available);
    }).catch(() => {
      if (!controller.signal.aborted) setError("Local runtime is unavailable. Ask the operator to check its configuration.");
    });
    return () => { controller.abort(); };
  }, []);

  /**
   * Bind user work to this mount and hide unexpected browser diagnostics.
   *
   * @param action - One user-selected typed client operation.
   * @returns Completion after updating visible progress or a safe refusal.
   */
  async function perform(action: (signal: AbortSignal) => Promise<void>): Promise<void> {
    const signal = lifetime.current.signal;
    setBusy(true); setError("");
    await action(signal).catch((failure: unknown) => {
      if (!signal.aborted) setError(failure instanceof HostApiRefusal ? failure.message : "Runtime operation could not be completed.");
    }).finally(() => { if (!signal.aborted) setBusy(false); });
  }

  /**
   * Read an actual completed artifact and retain only its accepted graph for reuse.
   *
   * @param current - Actual matching terminal job.
   * @param signal - Mounted lifetime.
   * @returns Completion after verifying and displaying the result.
   */
  async function collect(current: HostJob, signal: AbortSignal): Promise<void> {
    if (current.status !== "completed") return;
    const document = await readHostJson({ action: "result", jobId: current.jobId }, signal);
    if (signal.aborted) return;
    const compiled = hostedCompilerView(document);
    setResult(document);
    if (compiled?.irText) setGraph(compiled.irText);
  }

  /**
   * Refresh explicitly, so no polling timer outlives an unavailable backend.
   *
   * @param selected - Actual job selected by the visible mounted job controls.
   */
  function refresh(selected: HostJob): void {
    void perform(async (signal) => {
      const current = await readHostJob(selected.jobId, "status", signal);
      if (signal.aborted) return;
      setJob(current); await collect(current, signal);
    });
  }

  return <section aria-label="Local scientific workspace">
    <h2>Local scientific workspace</h2>
    {operations === null ? <p role="status">{error ? "Local runtime connection unavailable." : "Checking installed local runtime…"}</p>
      : <p>Connected through SCPN Studio · {operations.length} available actions</p>}
    {error && <p role="alert">{error}</p>}
    {operations === null && error && <button type="button" disabled={busy} onClick={() => {
      void perform(async (signal) => {
        const available = await describeHost(signal);
        if (!signal.aborted) setOperations(available);
      });
    }}>Retry runtime connection</button>}
    <p>Edit scientific request JSON directly; original number text and declaration order are retained.
      For graph inspection, replay or emission, load the last accepted graph or paste another graph in the request.</p>
    <label>Scientific action <select aria-label="Scientific action" value={operation} disabled={operations === null || busy}
      onChange={(event) => {
        const selected = operations?.find((item) => item === event.target.value);
        if (selected !== undefined) setOperation(selected);
      }}>{(operations ?? []).map((item) => <option key={item} value={item}>{HOST_OPERATION_LABELS[item]}</option>)}</select></label>
    <button type="button" disabled={busy || operations === null} onClick={() => {
      setSource(hostedExample(operation, graph));
    }}>Load example / last graph</button>
    <label style={{ display: "block" }}>Scientific request JSON<textarea aria-label="Scientific request JSON" rows={16}
      style={{ width: "100%", boxSizing: "border-box", fontFamily: "var(--font-mono)" }} value={source}
      onChange={(event) => { setSource(event.target.value); }} /></label>
    <button type="button" disabled={busy || !operations?.includes(operation) || (job !== null && ACTIVE.has(job.status))}
      onClick={() => { void perform(async (signal) => {
        const admitted = await submitHostJob(operation, source, signal);
        if (!signal.aborted) { setJob(admitted); setResult(null); await collect(admitted, signal); }
      }); }}>Run scientific action</button>
    <button type="button" disabled={busy || operations === null} onClick={() => { void perform(async (signal) => {
      const document = await readHostJson({ action: "models" }, signal);
      if (!signal.aborted) setCatalogue(document);
    }); }}>Browse model catalogue</button>
    <button type="button" disabled={busy || operations === null} onClick={() => { void perform(async (signal) => {
      const document = await readHostJson({ action: "templates" }, signal);
      if (!signal.aborted) setCatalogue(document);
    }); }}>Browse equation templates</button>
    {busy && <p role="status">Waiting for local runtime…</p>}
    {job !== null && <section aria-label="Current scientific job">
      <h3>{job.kind} job</h3><p>{job.jobId} · <strong>{job.status}</strong></p>
      <button type="button" disabled={busy} onClick={() => { refresh(job); }}>Refresh job and result</button>
      <button type="button" disabled={busy || !ACTIVE.has(job.status)} onClick={() => {
        void perform(async (signal) => {
          const current = await readHostJob(job.jobId, "cancel", signal);
          if (!signal.aborted) { setJob(current); await collect(current, signal); }
        });
      }}>Cancel job</button>
      {!["completed", "pending", "running", "cancelling"].includes(job.status)
        && <p role="status">This job produced no completed result. Review the request or contact the operator.</p>}
      <ul aria-label="Declared job artifacts">{job.artifacts.map((artifact, index) => <li key={`${artifact.digest}-${index}`}>
        {artifact.result ? "Scientific result" : "Job artifact"} · {artifact.bytes} bytes · <code>{artifact.digest}</code>
        <button type="button" disabled={busy || artifact.bytes > 8 * 1024 * 1024} onClick={() => {
          void perform(async (signal) => {
            await downloadHostArtifact(job.jobId, artifact, signal);
          });
        }}>Download artifact {index + 1}</button>
      </li>)}</ul>
    </section>}
    {result !== null && job !== null && <HostedJobResult document={result} jobId={job.jobId} busy={busy}
      onDownload={(selection) => { void perform(async (signal) => {
        const digest = "digest" in selection ? selection.digest : await hostedTextDigest(selection.text);
        const artifact = job.artifacts.find((item) => item.digest === digest);
        if (artifact === undefined) throw new HostApiRefusal("The selected result has no declared downloadable artifact.");
        await downloadHostArtifact(job.jobId, artifact, signal);
      }); }} />}
    {catalogue !== null && <details open><summary>Installed catalogue JSON</summary>
      <pre style={{ maxHeight: 360, overflow: "auto", whiteSpace: "pre-wrap" }}>{catalogue.json}</pre></details>}
  </section>;
}
