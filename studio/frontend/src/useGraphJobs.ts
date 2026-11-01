// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Recoverable graph jobs bound to live editor and account context

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { authHeaders } from "./api/http";
import {
  cancelGraphJob, downloadGraphJobArtifact, getGraphJob, listGraphJobs, prepareGraphJob, submitGraphJob,
  type PreparedGraphJob,
} from "./api/graphJobsApi";
import { downloadBrowserArtefact } from "./browserArtefactDownload";
import { graphJobInputKey } from "./graphJobRequest";
import type { GraphJobArtifact } from "./graphJobResultTransport";
import {
  bindGraphJobObservation, bindGraphJobPreparation, graphJobActive, type GraphJob,
} from "./graphJobs";
import {
  loadGraphJobRecovery, prepareGraphJobRecovery, readGraphJobRecovery, storeGraphJobRecovery,
  type GraphJobRecovery,
} from "./graphJobRecovery";
import { studioGraphRequest } from "./studioGraphRequests";
import { useStudioStore } from "./stores/studio";

/**
 * Capture current editable inputs through the same builder used by quick simulation.
 *
 * @returns Current graph and shared duration, timestep and seed.
 */
function currentGraph() {
  const state = useStudioStore.getState();
  return studioGraphRequest(state.graphPopulations, state.graphProjections, state.duration, state.dt, state.seed);
}

/** Observations owned by one account and restored workspace lifetime. */
interface GraphJobUiState {
  history: GraphJob[];
  selected: GraphJob | null;
  recovery: GraphJobRecovery | null;
  artifact: GraphJobArtifact | null;
  error: string | null;
  historyError: string | null;
  resultError: string | null;
  blockedRecovery: boolean;
  busy: boolean;
}

/**
 * Run, recover and observe graph jobs without adopting another editor context's result.
 *
 * Browser recovery is written before an allocating POST. History, restoration
 * and ordinary polling never submit work. Every asynchronous write is fenced
 * by account, workspace, restore token and selected calculation; scientific
 * adoption additionally requires fresh server interpretation of the live graph.
 *
 * @returns Checked history, selection, exact verified artifact and explicit user actions.
 */
export function useGraphJobs() {
  const authSession = useStudioStore((state) => state.authSession);
  const authLoading = useStudioStore((state) => state.authLoading);
  const workspace = useStudioStore((state) => state.graphWorkspaceId);
  const token = useStudioStore((state) => state.graphContextToken);
  const authorization = authHeaders().Authorization;
  const context = useMemo(() => ({ authSession, authLoading, workspace, token, authorization,
    owner: authSession === null || authLoading ? null : authSession.principal_id ?? "local" }),
  [authSession, authLoading, workspace, token, authorization]);
  const [refreshToken, setRefreshToken] = useState(0);
  const lifetime = useMemo(() => ({ context, refreshToken, controller: new AbortController() }), [context, refreshToken]);
  const empty = useMemo<GraphJobUiState>(() => ({ history: [], selected: null, recovery: null,
    artifact: null, error: null, historyError: null, resultError: null, blockedRecovery: false, busy: false }), [lifetime]);
  const [observation, setObservation] = useState(() => ({ lifetime, value: empty }));
  const { history, selected, recovery, artifact, error, historyError, resultError, blockedRecovery, busy }
    = observation.lifetime === lifetime ? observation.value : empty;
  const update = useCallback((patch: Partial<GraphJobUiState> | ((before: GraphJobUiState) => Partial<GraphJobUiState>)): void => {
    setObservation((previous) => {
      const before = previous.lifetime === lifetime ? previous.value : empty;
      return { lifetime, value: { ...before, ...typeof patch === "function" ? patch(before) : patch } };
    });
  }, [lifetime, empty]);
  const sequence = useRef(0);
  const busyRef = useRef(false);
  const applied = useRef<{ result: GraphJobArtifact["result"]; key: string } | null>(null);

  /** Withdraw only this hook's adopted result, preserving a later quick simulation. */
  const withdraw = useCallback((): void => {
    if (applied.current !== null && useStudioStore.getState().graphSimResult === applied.current.result) {
      useStudioStore.setState({ graphSimResult: null });
    }
    applied.current = null;
  }, []);

  /** Whether this exact live account and restored workspace still owns a reply. */
  const current = useCallback((): boolean => {
    const state = useStudioStore.getState();
    return context.owner !== null && !lifetime.controller.signal.aborted
      && state.authSession === context.authSession && state.authLoading === context.authLoading
      && state.graphWorkspaceId === context.workspace && state.graphContextToken === context.token
      && authHeaders().Authorization === context.authorization;
  }, [context, lifetime]);

  /** Persist exact recovery metadata before exposing it to store snapshots. */
  const retain = useCallback((value: GraphJobRecovery): void => {
    storeGraphJobRecovery(context.workspace, value);
    useStudioStore.setState({ graphJobRecovery: value });
    update({ recovery: value });
  }, [context.workspace, update]);

  useEffect(() => {
    if (lifetime.controller.signal.aborted) lifetime.controller = new AbortController();
    const controller = lifetime.controller;
    const generation = ++sequence.current;
    const accepts = (): boolean => !controller.signal.aborted && current() && generation === sequence.current;
    busyRef.current = false;
    update(empty);
    const owner = context.owner;
    if (owner !== null) {
      void listGraphJobs(owner, controller.signal).then((jobs) => {
        if (!controller.signal.aborted && current()) update({ history: jobs });
      }).catch((failure: unknown) => {
        if (!controller.signal.aborted && current()) update({ historyError: `History could not be loaded: ${String(failure)}` });
      });
      void (async () => {
        try {
          const saved = readGraphJobRecovery(useStudioStore.getState().graphJobRecovery);
          const cached = context.token.restored ? null : loadGraphJobRecovery(context.workspace, owner);
          const fromSnapshot = saved?.owner === owner ? saved : null;
          const restored = context.token.restored ? fromSnapshot : cached ?? fromSnapshot;
          const prepared = restored?.submission === null || restored?.submission === undefined
            ? null : await prepareGraphJobRecovery(restored.submission);
          if (!accepts()) return;
          update({ recovery: restored });
          useStudioStore.setState({ graphJobRecovery: restored });
          if (restored?.selectedJobId !== null && restored?.selectedJobId !== undefined) {
            const observed = await getGraphJob(restored.selectedJobId, owner, controller.signal);
            const job = prepared === null ? observed : bindGraphJobPreparation(observed, prepared.identity);
            if (accepts()) update({ selected: job });
          }
        } catch (failure) {
          if (accepts()) {
            update({ blockedRecovery: true, error: `Recovery could not be read: ${String(failure)}` });
          }
        }
      })();
    }
    const unsubscribe = useStudioStore.subscribe(() => {
      if (applied.current === null) return;
      let matches: boolean;
      try { matches = current() && graphJobInputKey(currentGraph()) === applied.current.key; } catch { matches = false; }
      if (!matches) { withdraw(); update({ resultError: "The graph or workspace changed; the previous job result was withdrawn." }); }
    });
    return () => { controller.abort(); unsubscribe(); sequence.current += 1; withdraw(); };
  }, [context, current, lifetime, empty, update, withdraw]);

  useEffect(() => {
    if (selected === null || !graphJobActive(selected)) return;
    const owner = selected.owner;
    const signal = lifetime.controller.signal;
    const generation = sequence.current;
    let disposed = false;
    const timer = window.setTimeout(() => {
      void getGraphJob(selected.job_id, owner, signal).then((observed) => {
        const job = bindGraphJobObservation(selected, observed);
        if (!disposed && current() && generation === sequence.current) {
          update({ selected: job, error: null });
          update(({ history: jobs }) => ({ history: jobs.map((entry) => entry.job_id === job.job_id ? { ...job, result: null } : entry) }));
        }
      }).catch((failure: unknown) => {
        if (!disposed && current() && generation === sequence.current) update({ error: `Observation stopped; refresh to retry: ${String(failure)}` });
      });
    }, 400);
    return () => { disposed = true; window.clearTimeout(timer); };
  }, [selected, context.owner, current, lifetime, update]);

  useEffect(() => {
    if (selected?.status !== "completed") return;
    const signal = lifetime.controller.signal;
    const generation = sequence.current;
    let disposed = false;
    const accepts = (): boolean => !disposed && current() && generation === sequence.current;
    void (async () => {
      try {
        const graph = currentGraph();
        const key = graphJobInputKey(graph);
        const complete = await downloadGraphJobArtifact(selected, signal);
        if (!accepts()) return;
        update({ artifact: complete });
        const checked = await prepareGraphJob(graph, signal);
        if (!accepts()) return;
        if (key !== graphJobInputKey(currentGraph()) || checked.identity.graph_sha256 !== selected.admission.graph_sha256) {
          withdraw(); update({ resultError: "This job belongs to a different graph. Its verified original result can still be exported." }); return;
        }
        applied.current = { result: complete.result, key };
        useStudioStore.setState({ graphSimResult: complete.result });
        update({ resultError: null });
      } catch (failure) {
        if (accepts()) update({ resultError: `Result could not be adopted: ${String(failure)}` });
      }
    })();
    return () => { disposed = true; };
  }, [selected, current, lifetime, update, withdraw]);

  /** Start or explicitly retry retained original input under the same durable key. */
  const submit = useCallback(async (retry: boolean): Promise<void> => {
    const owner = context.owner;
    const signal = lifetime.controller.signal;
    if (owner === null || !current() || busyRef.current) return;
    busyRef.current = true; update({ busy: true, selected: null, error: null, resultError: null, artifact: null }); withdraw();
    const generation = ++sequence.current;
    let retained = recovery;
    let sent = false;
    const admission = new AbortController();
    const stopForContext = (): void => { admission.abort(); };
    signal.addEventListener("abort", stopForContext, { once: true });
    let stopForEdit: (() => void) | null = null;
    try {
      let prepared: PreparedGraphJob;
      let retryKey: string;
      if (retry) {
        if (retained?.owner !== owner || retained.submission === null) throw new Error("No retained graph submission is available.");
        retryKey = retained.submission.retryKey;
        prepared = await prepareGraphJobRecovery(retained.submission);
      } else {
        if (blockedRecovery || recovery?.submission !== null && recovery?.submission !== undefined && recovery.selectedJobId === null) {
          throw new Error("Recover or forget the unconfirmed submission before starting another calculation.");
        }
        const graph = currentGraph();
        const key = graphJobInputKey(graph);
        stopForEdit = useStudioStore.subscribe(() => {
          try { if (graphJobInputKey(currentGraph()) !== key) admission.abort(); }
          catch { admission.abort(); }
        });
        prepared = await prepareGraphJob(graph, admission.signal);
        if (!current() || generation !== sequence.current) return;
        if (key !== graphJobInputKey(currentGraph())) throw new Error("The graph changed before submission; prepare it again.");
        retryKey = crypto.randomUUID();
        retained = { schema_version: "studio.graph.job-recovery.v1", owner, selectedJobId: null,
          submission: { requestJson: prepared.request.json, retryKey, validation: {
            schema_version: "studio.graph.job.validation.v1", execution_mode: "async_job", ...prepared.identity,
          } } };
      }
      if (!current() || generation !== sequence.current) return;
      retain(retained);
      sent = true;
      const receipt = await submitGraphJob(prepared, owner, retryKey, admission.signal);
      if (!current() || generation !== sequence.current) return;
      update({ selected: receipt.job });
      retain({ ...retained, selectedJobId: receipt.job.job_id });
      update(({ history: jobs }) => ({ history: [receipt.job, ...jobs.filter((job) => job.job_id !== receipt.job.job_id)] }));
    } catch (failure) {
      if (current() && generation === sequence.current) update({ error: sent
        ? `Submission acknowledgement is unconfirmed; recover the retained input with the same key: ${String(failure)}`
        : `Graph submission stopped before admission: ${String(failure)}` });
    } finally {
      stopForEdit?.();
      signal.removeEventListener("abort", stopForContext);
      if (current() && generation === sequence.current) { busyRef.current = false; update({ busy: false }); }
    }
  }, [context.owner, current, lifetime, recovery, retain, withdraw, blockedRecovery, update]);

  /** Observe one explicit history selection without resubmitting its calculation. */
  const select = useCallback(async (jobId: string): Promise<void> => {
    const owner = context.owner;
    const signal = lifetime.controller.signal;
    if (owner === null || busyRef.current || !current()) return;
    if (recovery?.submission !== null && recovery?.submission !== undefined && recovery.selectedJobId === null) {
      update({ error: "Recover or forget the unconfirmed submission before selecting another job." }); return;
    }
    const expected = history.find((job) => job.job_id === jobId);
    const generation = ++sequence.current;
    withdraw(); update({ selected: null, artifact: null, error: null, resultError: null });
    try {
      const observed = await getGraphJob(jobId, owner, signal);
      const job = expected === undefined ? observed : bindGraphJobObservation(expected, observed);
      if (current() && generation === sequence.current) {
        retain({ schema_version: "studio.graph.job-recovery.v1", owner, selectedJobId: job.job_id, submission: null });
        update({ selected: job });
      }
    } catch (failure) {
      if (current() && generation === sequence.current) update({ error: `Could not select graph job: ${String(failure)}` });
    }
  }, [context.owner, current, lifetime, history, retain, withdraw, recovery, update]);

  /** Ask the authority to cancel, retaining its actual response rather than inventing a terminal state. */
  const cancel = useCallback(async (): Promise<void> => {
    const owner = context.owner;
    const signal = lifetime.controller.signal;
    if (selected === null || !graphJobActive(selected) || owner === null || !current() || busyRef.current) return;
    const generation = ++sequence.current;
    busyRef.current = true; update({ busy: true });
    try {
      const job = bindGraphJobObservation(selected, await cancelGraphJob(selected.job_id, owner, signal));
      if (current() && generation === sequence.current) {
        update({ selected: job, error: null });
        update(({ history: jobs }) => ({ history: jobs.map((entry) => entry.job_id === job.job_id ? { ...job, result: null } : entry) }));
      }
    } catch (failure) {
      if (current() && generation === sequence.current) update({ error: `Cancellation was not confirmed; refresh actual status: ${String(failure)}` });
    } finally {
      if (current() && generation === sequence.current) { busyRef.current = false; update({ busy: false }); }
    }
  }, [selected, context.owner, current, lifetime, update]);

  /** Forget only this browser recovery pointer; actual jobs remain in server history. */
  const forget = useCallback((): void => {
    if (context.owner === null || !current() || busyRef.current) return;
    try {
      retain({ schema_version: "studio.graph.job-recovery.v1", owner: context.owner, selectedJobId: null, submission: null });
      sequence.current += 1; withdraw(); update({ selected: null, artifact: null, blockedRecovery: false, error: null, resultError: null });
    } catch (failure) { update({ error: `Recovery could not be reset: ${String(failure)}` }); }
  }, [context.owner, current, retain, withdraw, update]);

  /** Export the verified original bytes, never a reserialised approximation of their digest. */
  const exportResult = useCallback((): void => {
    if (artifact === null || !current()) return;
    update({ error: null });
    try { downloadBrowserArtefact(new Blob([artifact.bytes], { type: "application/json" }), "network-result.json"); }
    catch (failure) { update({ error: `Export failed: ${String(failure)}` }); }
  }, [artifact, current, update]);

  const unconfirmed = recovery?.submission !== null && recovery?.submission !== undefined && recovery.selectedJobId === null;
  return { history, selected, artifact, busy, ready: context.owner !== null, unconfirmed, blockedRecovery,
    restoredSnapshot: context.token.restored, error: error ?? historyError, resultError, submit, select, cancel, forget, exportResult,
    refresh: (): void => { setRefreshToken((value) => value + 1); } };
}
