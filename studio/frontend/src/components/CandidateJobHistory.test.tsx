// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate history presentation and explicit controls
// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";
import { parseCandidateJob } from "../candidateJobs";
import CandidateJobHistory, { type CandidateJobHistoryProps } from "./CandidateJobHistory";

it("renders exact custody, recovery states and named user actions", async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  const job = parseCandidateJob({
    artifacts: [{ relative_path: "candidate.json", sha256: "a".repeat(64), size_bytes: 5 }],
    created_at_utc: "2026-10-30T00:00:00Z", error: null, execution_model: "process",
    finished_at_utc: null, job_id: "sj_0123456789abcdef", kind: "analysis", owner: "alice",
    request_id: null, result: null, started_at_utc: null, status: "pending",
    experiment_sha256: "b".repeat(64), admission: {
      candidate_operation: "simulate", candidate_name: "OriginalCandidate",
      candidate_sha256: "a".repeat(64), request_sha256: "b".repeat(64), projected_steps: 20,
    },
  });
  const container = document.createElement("div"); document.body.append(container);
  const root = createRoot(container);
  const actions: string[] = [];
  const props: CandidateJobHistoryProps = {
    history: [], selected: null, error: null,
    refresh: () => { actions.push("refresh"); },
    select: (id) => { actions.push(id); return Promise.resolve(); },
    cancel: () => { actions.push("cancel"); return Promise.resolve(); },
    download: (_job, path) => { actions.push(path); return Promise.resolve(); },
  };
  try {
    await act(async () => { root.render(<CandidateJobHistory {...props} />); });
    expect(container.textContent).toContain("No candidate jobs loaded.");
    await act(async () => { root.render(<CandidateJobHistory {...props} history={[job]} selected={job} />); });
    expect(container.textContent).toContain(job.admission.candidate_sha256);
    expect(container.textContent).toContain(job.admission.request_sha256);
    expect(container.textContent).toContain("projected steps 20");
    for (const button of container.querySelectorAll("button")) {
      await act(async () => { button.click(); });
    }
    expect(actions).toEqual(["refresh", "cancel", "candidate.json", job.job_id]);
    for (const status of ["cancelling", "cancelled", "interrupted", "unknown"] as const) {
      await act(async () => {
        root.render(<CandidateJobHistory {...props} history={[job]} selected={{ ...job, status, error: "Retained failure" }} error="Observation failed" />);
      });
      expect(container.querySelector('[role="alert"]')?.textContent).toBe("Observation failed");
      expect(container.textContent).toContain("Retained failure");
      const cancel = [...container.querySelectorAll("button")].find((button) => button.textContent === "Cancel candidate job");
      expect(cancel?.disabled).toBe(true);
      expect(container.textContent.includes("Execution could not be confirmed")).toBe(status === "interrupted" || status === "unknown");
    }
    await act(async () => { root.render(<CandidateJobHistory {...props} history={[job]} />); });
    expect(container.querySelector('[aria-pressed="false"]')).not.toBeNull();
  } finally {
    await act(async () => { root.unmount(); }); container.remove();
  }
});
