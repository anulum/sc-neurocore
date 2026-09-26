// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Recoverable owned laboratory job lifecycle

import { useEffect, useState } from "react";
import { StudioRequestError } from "./api/http";
import { cancelLaboratoryJob, laboratoryJob, type LaboratorySubmission } from "./api/fitsApi";


/** Actual server-owned task state and workbench actions. */
interface LaboratoryState<T> {
  id: string | null; status: string; result: T | null; error: string | null; busy: boolean;
  start: (submit: () => Promise<LaboratorySubmission>) => Promise<void>;
  cancel: () => Promise<void>;
}

/**
 * Prefer the server's explanatory message when admission has a structured reason.
 *
 * @param caught - Transport or execution error.
 * @returns A sentence suitable for the workbench status.
 */
function laboratoryError(caught: unknown): string {
  if (caught instanceof StudioRequestError && typeof caught.detail === "object" && caught.detail !== null && "message" in caught.detail && typeof caught.detail.message === "string") return caught.detail.message;
  return caught instanceof Error ? caught.message : String(caught);
}

/**
 * Poll one submitted task and recover its identity when the panel is reopened.
 *
 * @param storageKey - Session-specific key for this independent workbench.
 * @returns Actual job status, result and submission/cancellation operations.
 */
export function useLaboratoryJob<T>(storageKey: string): LaboratoryState<T> {
  const [id, setId] = useState<string | null>(() => sessionStorage.getItem(storageKey));
  const [status, setStatus] = useState<string>(id === null ? "idle" : "recovering");
  const [result, setResult] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  useEffect(() => {
    if (id === null) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const record = await laboratoryJob<T>(id);
        if (disposed) return;
        if (record.job_id !== id) throw new Error("The response belongs to a different job.");
        setStatus(record.status);
        setError(record.error);
        if (["completed", "failed", "cancelled", "timed_out"].includes(record.status)) {
          if (record.status === "completed") setResult(record.result);
          return;
        }
      } catch (caught) {
        if (disposed) return;
        setError(laboratoryError(caught));
        if (caught instanceof StudioRequestError && [403, 404].includes(caught.status)) { setStatus("unavailable"); return; }
        setStatus("status unavailable; retrying");
      }
      timer = setTimeout(() => { void poll(); }, 400);
    };
    void poll();
    return () => { disposed = true; if (timer !== undefined) clearTimeout(timer); };
  }, [id]);

  /**
   * Preserve receipt identity before starting observation of the actual job.
   *
   * @param submit - API call returning the actual job receipt.
   */
  async function start(submit: () => Promise<LaboratorySubmission>): Promise<void> {
    setSubmitting(true); setError(null); setResult(null);
    try {
      const receipt = await submit();
      sessionStorage.setItem(storageKey, receipt.job_id);
      setStatus("pending"); setId(receipt.job_id);
    } catch (caught) {
      setError(laboratoryError(caught));
    } finally { setSubmitting(false); }
  }

  /** Request cancellation without describing it as complete before the supervisor does. */
  async function cancel(): Promise<void> {
    if (id === null) return;
    try {
      const record = await cancelLaboratoryJob(id);
      setStatus(record.status);
    } catch (caught) { setError(laboratoryError(caught)); }
  }
  const busy = submitting || (id !== null && !["completed", "failed", "cancelled", "timed_out", "unavailable"].includes(status));
  return { id, status, result, error, busy, start, cancel };
}
