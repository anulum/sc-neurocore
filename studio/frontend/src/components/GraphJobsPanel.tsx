// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Recoverable background graph calculation controls

import { graphJobActive } from "../graphJobs";
import { useGraphJobs } from "../useGraphJobs";
import { useStudioStore } from "../stores/studio";
import "./GraphJobsPanel.css";

/**
 * Submit and recover actual supervised graph work from the maintained network canvas.
 *
 * @returns Background controls, truthful observed status, recovery and verified export.
 */
export default function GraphJobsPanel() {
  const jobs = useGraphJobs();
  const empty = useStudioStore((state) => state.graphPopulations.length === 0);
  return (
    <section aria-label="Background graph jobs" className="graph-jobs-panel" style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
        <h3 style={{ margin: 0, fontSize: "var(--fs-body)" }}>Background graph jobs</h3>
        <button disabled={!jobs.ready || jobs.busy || empty || jobs.unconfirmed || jobs.blockedRecovery}
          onClick={() => { void jobs.submit(false); }}>Run background job</button>
        <button disabled={!jobs.ready || jobs.busy} onClick={jobs.refresh}>Refresh jobs</button>
        {jobs.unconfirmed && <button disabled={jobs.busy} onClick={() => { void jobs.submit(true); }}>Recover original submission</button>}
        {jobs.selected !== null && graphJobActive(jobs.selected) && (
          <button disabled={jobs.busy} onClick={() => { void jobs.cancel(); }}>Cancel selected job</button>
        )}
        <button disabled={jobs.artifact === null} onClick={jobs.exportResult}>Export verified result</button>
        <button disabled={!jobs.ready || jobs.busy} onClick={jobs.forget}>Forget local selection</button>
      </div>
      {!jobs.ready && <p role="status">Checking the Studio account before loading graph jobs.</p>}
      {jobs.busy && <p role="status">Waiting for the graph job authority.</p>}
      {jobs.unconfirmed && <p role="status">The original submission is retained. Recover it before starting another calculation.</p>}
      {jobs.restoredSnapshot && <p role="status">Opened saved graph recovery. Its selection takes precedence over this tab's recovery cache.
        The browser cache is kept until you explicitly run, recover, select or forget a job.</p>}
      {jobs.error !== null && <p role="alert">{jobs.error}</p>}
      {jobs.resultError !== null && <p role="status">{jobs.resultError}</p>}
      {jobs.selected !== null && (
        <p role="status" data-testid="selected-graph-job">
          {jobs.selected.job_id}: {jobs.selected.status} · {jobs.selected.admission.projected_neuron_steps} neuron steps
          {jobs.artifact !== null && ` · ${jobs.artifact.result.n_spikes} spikes · backend ${jobs.artifact.result.execution.backend.selected}`}
          {jobs.selected.error !== null && ` · ${jobs.selected.error}`}
        </p>
      )}
      <details>
        <summary>Graph job history ({jobs.history.length})</summary>
        <p>Opening a job observes its stored calculation. Forgetting this selection keeps server history.</p>
        <ul aria-label="Stored graph calculations">
          {jobs.history.map((job) => (
            <li key={job.job_id}>
              <button disabled={jobs.busy || jobs.unconfirmed} aria-pressed={jobs.selected?.job_id === job.job_id}
                onClick={() => { void jobs.select(job.job_id); }}>{job.job_id}: {job.status}</button>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
