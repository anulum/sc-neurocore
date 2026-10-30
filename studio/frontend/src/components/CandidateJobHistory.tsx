// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate job history and supervisor controls

import { candidateJobActive, type CandidateJob } from "../candidateJobs";

/** Observation actions owned by the candidate panel's lifecycle hook. */
export interface CandidateJobHistoryProps {
  history: CandidateJob[];
  selected: CandidateJob | null;
  error: string | null;
  refresh: () => void;
  select: (jobId: string) => Promise<void>;
  cancel: () => Promise<void>;
  download: (job: CandidateJob, path: string) => Promise<void>;
}

/**
 * Expose durable history, actual supervisor state and declared artifacts.
 *
 * @param props - Checked records and explicit user actions.
 * @returns Accessible job observation and control elements.
 */
export default function CandidateJobHistory(props: CandidateJobHistoryProps) {
  const { selected } = props;
  return <section aria-label="Candidate job history">
    <h3>Candidate jobs</h3>
    <button type="button" onClick={props.refresh}>Refresh candidate jobs</button>
    {props.error !== null && <p role="alert">{props.error}</p>}
    {selected !== null && <div aria-label="Selected candidate job" role="status">
      <p>{selected.admission.candidate_name} · {selected.admission.candidate_operation} · {selected.status}</p>
      <p>Job <code>{selected.job_id}</code> · candidate <code>{selected.admission.candidate_sha256}</code></p>
      <p>Request <code>{selected.admission.request_sha256}</code> · projected steps {selected.admission.projected_steps}</p>
      {selected.error !== null && <p>{selected.error}</p>}
      {(selected.status === "interrupted" || selected.status === "unknown")
        && <p>Execution could not be confirmed. Inspect retained inputs before starting a new job.</p>}
      <button type="button" disabled={!candidateJobActive(selected) || selected.status === "cancelling"}
        onClick={() => { void props.cancel(); }}>Cancel candidate job</button>
      {selected.artifacts.map((artifact) => <button type="button" key={artifact.relative_path}
        onClick={() => { void props.download(selected, artifact.relative_path); }}>Download {artifact.relative_path}</button>)}
    </div>}
    {props.history.length === 0 ? <p>No candidate jobs loaded.</p> : <ul aria-label="Stored candidate jobs">
      {props.history.map((job) => <li key={job.job_id}>
        <button type="button" aria-pressed={selected?.job_id === job.job_id}
          onClick={() => { void props.select(job.job_id); }}>
          {job.admission.candidate_name} · {job.admission.candidate_operation} · {job.status} · {job.job_id}
        </button>
      </li>)}
    </ul>}
  </section>;
}
