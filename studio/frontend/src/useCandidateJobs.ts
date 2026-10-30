// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Recoverable candidate job observation

import { useCallback, useEffect, useRef, useState } from "react";
import { cancelCandidateJob, getCandidateJob, listCandidateJobs } from "./api/candidatesApi";
import { bindCandidateJobObservation, candidateJobActive, type CandidateJob, type CandidateJobReceipt } from "./candidateJobs";

/**
 * Recover server history and observe one selected job without resubmitting it.
 *
 * History survives browser restarts in the server ledger. Transport failures
 * stop automatic observation and expose an explicit retry; they never start
 * another calculation or turn an unobserved job into a failed calculation.
 *
 * @returns History, selection, observation errors and explicit user actions.
 */
export function useCandidateJobs() {
  const [history, setHistory] = useState<CandidateJob[]>([]);
  const [selected, setSelected] = useState<CandidateJob | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [observationError, setObservationError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState(0);
  const mounted = useRef(false);
  const selection = useRef(0);

  const refresh = useCallback((): void => { setRefreshToken((value) => value + 1); }, []);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; selection.current += 1; };
  }, []);

  useEffect(() => {
    let disposed = false;
    void listCandidateJobs().then((jobs) => {
      if (!disposed) { setHistory(jobs); setHistoryError(null); }
    }).catch((failure: unknown) => {
      if (!disposed) setHistoryError(`Could not load candidate history: ${String(failure)}`);
    });
    return () => { disposed = true; };
  }, [refreshToken]);

  const select = useCallback(async (jobId: string): Promise<void> => {
    const expected = selected?.job_id === jobId ? selected : history.find((job) => job.job_id === jobId);
    const generation = ++selection.current;
    setSelected(null);
    setObservationError(null);
    try {
      const observed = await getCandidateJob(jobId);
      const job = expected === undefined ? observed : bindCandidateJobObservation(expected, observed);
      if (mounted.current && generation === selection.current) setSelected(job);
    } catch (failure) {
      if (mounted.current && generation === selection.current) setObservationError(`Could not observe candidate job: ${String(failure)}`);
    }
  }, [history, selected]);

  useEffect(() => {
    if (selected === null || !candidateJobActive(selected)) return;
    let disposed = false;
    const generation = selection.current;
    const timer = window.setTimeout(() => {
      void getCandidateJob(selected.job_id).then((observed) => {
        const job = bindCandidateJobObservation(selected, observed);
        if (!disposed && generation === selection.current) {
          setSelected(job);
          setHistory((jobs) => jobs.map((item) => item.job_id === job.job_id ? { ...job, result: null } : item));
          setObservationError(null);
        }
      }).catch((failure: unknown) => {
        if (!disposed && generation === selection.current) setObservationError(`Observation stopped; refresh to retry: ${String(failure)}`);
      });
    }, 400);
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [selected, refreshToken]);

  const accept = useCallback((receipt: CandidateJobReceipt): void => {
    selection.current += 1;
    setSelected(receipt.job);
    setObservationError(null);
    refresh();
  }, [refresh]);

  const cancel = useCallback(async (): Promise<void> => {
    if (selected === null || !candidateJobActive(selected)) return;
    const generation = ++selection.current;
    try {
      const job = bindCandidateJobObservation(selected, await cancelCandidateJob(selected.job_id));
      if (mounted.current && generation === selection.current) {
        setSelected(job); setObservationError(null); refresh();
      }
    } catch (failure) {
      if (mounted.current && generation === selection.current) setObservationError(`Could not request cancellation: ${String(failure)}`);
    }
  }, [selected, refresh]);

  return { history, selected, error: observationError ?? historyError, refresh, select, accept, cancel };
}
