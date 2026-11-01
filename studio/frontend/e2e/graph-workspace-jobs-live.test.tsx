// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Graph editor recovery through actual HTTP, storage and supervised workers
// @vitest-environment happy-dom

import { randomUUID } from "node:crypto";
import { startCandidateTestServer } from "../src/candidateTestServer";
import { act, StrictMode } from "react";
import { expect, it } from "vitest";
import { listProjects, loadProject } from "../src/api/projectApi";
import { STUDIO_SAVED_SESSIONS_KEY } from "../src/studioSavedSessions";
import { loadGraphJobRecovery, storeGraphJobRecovery, graphWorkspaceId } from "../src/graphJobRecovery";
import { studioProjectRestoreState, studioProjectSaveState } from "../src/studioProjectState";
import { useStudioStore } from "../src/stores/studio";
import GraphJobsPanel from "../src/components/GraphJobsPanel";
import { server, root, container, dom, committedViews, current, responseGate, until, authorityJobCount } from "./graphJobTestContext";

it("adopts only the actual current graph, preserves layout moves and withdraws scientific edits", async () => {
  const before = await authorityJobCount();
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null && useStudioStore.getState().graphSimResult !== null);
  expect(current().selected?.status).toBe("completed");
  expect(current().artifact?.result.n_total).toBe(2);
  expect(await authorityJobCount()).toBe(before + 1);
  const verified = useStudioStore.getState().graphSimResult;
  await act(async () => { useStudioStore.getState().updatePopulation("e", { position: { x: 900, y: 700 } }); });
  expect(useStudioStore.getState().graphSimResult).toBe(verified);
  await act(async () => { useStudioStore.getState().updatePopulation("e", { params: { tau: 11 } }); });
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  expect(current().resultError).toContain("withdrawn");
  await act(async () => { current().refresh(); });
  await until(() => current().resultError?.includes("different graph") === true);
  expect(current().artifact?.result.n_total).toBe(2);
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  expect(await authorityJobCount()).toBe(before + 1);
}, 90_000);

it("keeps graph and selected job through actual named-session and server-project save/open without another graph admission", async () => {
  await act(async () => { await current().submit(false); });
  await until(() => useStudioStore.getState().graphSimResult !== null);
  const jobId = current().selected?.job_id;
  const count = await authorityJobCount();
  let oldToken = useStudioStore.getState().graphContextToken;
  await act(async () => { expect(useStudioStore.getState().saveSession("graph-recovery").status).toBe("ok"); });
  await act(async () => { useStudioStore.setState({ graphPopulations: [] }); useStudioStore.getState().loadSession("graph-recovery"); });
  expect(useStudioStore.getState().graphContextToken).not.toBe(oldToken);
  expect(useStudioStore.getState().graphPopulations[0]?.count).toBe(2);
  await until(() => current().selected?.job_id === jobId && useStudioStore.getState().graphSimResult !== null);
  await until(() => useStudioStore.getState().result !== null);
  expect(useStudioStore.getState().result).toMatchObject({ model_name: "PoissonNeuron" });
  expect(useStudioStore.getState().error).toBeNull();
  const sessionResult = useStudioStore.getState().result;
  await act(async () => { await useStudioStore.getState().saveProjectToServer("graph-recovery-project"); });
  expect(useStudioStore.getState().projectSaveResult?.name).toBe("graph-recovery-project");
  oldToken = useStudioStore.getState().graphContextToken;
  await act(async () => { useStudioStore.setState({ graphPopulations: [] }); await useStudioStore.getState().loadProjectFromServer("graph-recovery-project"); });
  expect(useStudioStore.getState().graphContextToken).not.toBe(oldToken);
  await until(() => current().selected?.job_id === jobId && useStudioStore.getState().graphSimResult !== null);
  await until(() => useStudioStore.getState().result !== null && useStudioStore.getState().result !== sessionResult);
  expect(useStudioStore.getState().result).toMatchObject({ model_name: "PoissonNeuron" });
  expect(useStudioStore.getState().error).toBeNull();
  expect(await authorityJobCount()).toBe(count);
}, 90_000);

it("reports malformed snapshot input without overwriting a named session or writing a server project", async () => {
  await act(async () => {
    expect(useStudioStore.getState().saveSession("kept-graph").status).toBe("ok");
    await useStudioStore.getState().saveProjectToServer("kept-graph-project");
  });
  const storedSessions = localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY);
  const savedPage = useStudioStore.getState().savedSessions;
  const project = await loadProject("kept-graph-project");
  const projects = await listProjects();
  await act(async () => {
    useStudioStore.setState({ graphWorkspaceId: "unreadable" });
    const failed = useStudioStore.getState().saveSession("kept-graph");
    expect(failed.status).toBe("failed");
    expect(failed.message).toContain("workspace identity is invalid");
    await useStudioStore.getState().saveProjectToServer("unwritten-graph-project");
  });
  expect(useStudioStore.getState().error).toContain("workspace identity is invalid");
  expect(useStudioStore.getState().savedSessions).toEqual(savedPage);
  expect(localStorage.getItem(STUDIO_SAVED_SESSIONS_KEY)).toBe(storedSessions);
  expect(await listProjects()).toEqual(projects);
  expect(await loadProject("kept-graph-project")).toEqual(project);
  await act(async () => { current().forget(); });
  expect(current().error).toContain("Recovery could not be reset");
  expect(current().blockedRecovery).toBe(true);
});

it.each([false, true])("recovers a real acted-on lost acknowledgement with the original persisted body and key (previous calculation=%s)", async (previous) => {
  if (previous) {
    await act(async () => { await current().submit(false); });
    await until(() => current().artifact !== null);
  }
  const before = await authorityJobCount();
  const gate = await responseGate("POST", (path) => path === "/api/graph/jobs");
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  let submission: Promise<void> | null = null;
  await act(async () => { submission = current().submit(false); });
  await until(gate.held);
  expect(await authorityJobCount()).toBe(before + 1);
  const retained = useStudioStore.getState().graphJobRecovery;
  expect(retained?.submission?.requestJson).toContain('"tau":10');
  expect(retained?.selectedJobId).toBeNull();
  expect(current().selected).toBeNull();
  await act(async () => { gate.release(true); await submission; });
  await until(() => current().unconfirmed && !current().busy);
  expect(current().error).toContain("unconfirmed");
  await act(async () => { await current().select("sj_0000000000000000"); });
  expect(current().error).toContain("Recover or forget");
  await act(async () => { await current().submit(false); });
  expect(await authorityJobCount()).toBe(before + 1);
  await act(async () => { current().refresh(); });
  await until(() => current().unconfirmed && !current().blockedRecovery);
  await act(async () => { root.render(<StrictMode><GraphJobsPanel /></StrictMode>); });
  await until(() => Array.from(container.querySelectorAll("button")).some((button) => button.textContent === "Recover original submission"));
  await act(async () => {
    const recover = Array.from(container.querySelectorAll("button")).find((button) => button.textContent === "Recover original submission");
    if (recover === undefined) throw new Error("No visible graph recovery control");
    recover.click();
  });
  await until(() => container.querySelector('[data-testid="selected-graph-job"]')?.textContent.includes(": completed") === true);
  expect(await authorityJobCount()).toBe(before + 1);
  expect(useStudioStore.getState().graphJobRecovery?.submission?.retryKey).toBe(retained?.submission?.retryKey);
  expect(useStudioStore.getState().graphJobRecovery?.submission?.requestJson).toBe(retained?.submission?.requestJson);
}, 90_000);

it.each([3, Number.NaN])("stops a scientific edit while real preparation is delayed, before any allocating POST (count=%s)", async (count) => {
  const before = await authorityJobCount();
  const gate = await responseGate("POST", (path) => path === "/api/graph/jobs/validate");
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  let submission: Promise<void> | null = null;
  await act(async () => { submission = current().submit(false); });
  await until(gate.held);
  await act(async () => { useStudioStore.getState().updatePopulation("e", { count }); gate.release(); await submission; });
  expect(await authorityJobCount()).toBe(before);
  expect(current().error).toContain("before admission");
}, 90_000);

it.each(["same-workspace-restore", "account-switch"] as const)("refuses a delayed old complete artifact after %s", async (change) => {
  const gate = await responseGate("GET", (path) => path.endsWith("/result"));
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await act(async () => { await current().submit(false); });
  await until(gate.held);
  const snapshot = studioProjectSaveState(useStudioStore.getState());
  await act(async () => {
    if (change === "same-workspace-restore") useStudioStore.setState(studioProjectRestoreState(snapshot));
    else useStudioStore.setState({ authSession: { authenticated: true, principal_id: "bob", roles: ["viewer"] } });
  });
  if (change === "same-workspace-restore") await until(() => useStudioStore.getState().graphSimResult !== null);
  const adopted = useStudioStore.getState().graphSimResult;
  await act(async () => { gate.release(); await dom.happyDOM.waitUntilComplete(); });
  expect(useStudioStore.getState().graphSimResult).toBe(adopted);
  if (change === "account-switch") { expect(adopted).toBeNull(); expect(current().selected).toBeNull(); expect(current().artifact).toBeNull(); }
}, 90_000);

it("cancels actual running above-sync work and preserves nonexportable cancelled state", async () => {
  await act(async () => {
    useStudioStore.setState({ dt: 1, duration: 1000 }); useStudioStore.getState().updatePopulation("e", { count: 2000 });
    await current().submit(false);
  });
  await until(() => current().selected?.status === "running");
  await act(async () => { await current().cancel(); });
  await until(() => current().selected?.status === "cancelled");
  expect(current().artifact).toBeNull();
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  await act(async () => { await current().cancel(); current().forget(); });
  expect(current().selected).toBeNull();
}, 90_000);

it.each(["account", "workspace"] as const)("clears the very first committed %s view and refuses callbacks retained by the previous context", async (change) => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null && current().history.length > 0);
  const previous = current();
  const count = await authorityJobCount();
  const nextWorkspace = change === "workspace" ? graphWorkspaceId() : useStudioStore.getState().graphWorkspaceId;
  const first = committedViews.length;
  await act(async () => {
    if (change === "account") useStudioStore.setState({ authSession: { authenticated: true, principal_id: "bob", roles: ["viewer"] } });
    else useStudioStore.setState({ graphWorkspaceId: nextWorkspace, graphJobRecovery: null });
  });
  const view = committedViews.slice(first).find((entry) => change === "account" ? entry.owner === "bob" : entry.workspace === nextWorkspace);
  expect(view).toMatchObject({ selected: null, history: 0, artifact: false });
  await act(async () => {
    await previous.submit(false); await previous.select(previous.selected?.job_id ?? "missing");
    await previous.cancel(); previous.forget(); previous.exportResult();
  });
  expect(await authorityJobCount()).toBe(count);
  expect(current().artifact).toBeNull();
  expect(current().selected).toBeNull();
  expect(useStudioStore.getState().graphSimResult).toBeNull();
}, 90_000);

it("reports actual history transport loss and a missing selection, then refreshes without submitting", async () => {
  const count = await authorityJobCount();
  const gate = await responseGate("GET", (path) => path === "/api/graph/jobs");
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await until(gate.held);
  await act(async () => { gate.release(true); });
  await until(() => current().error?.includes("History could not be loaded") === true);
  await act(async () => { await current().select("sj_0000000000000000"); });
  expect(current().error).toContain("Could not select graph job");
  await act(async () => { await current().submit(true); });
  expect(current().error).toContain("No retained graph submission");
  dom.happyDOM.setURL(server.origin);
  await act(async () => { current().refresh(); });
  await until(() => current().error === null);
  expect(await authorityJobCount()).toBe(count);
});

it("keeps a completed graph nonexportable after actual artifact transport loss and recovers by observation", async () => {
  const gate = await responseGate("GET", (path) => path.endsWith("/result"));
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await act(async () => { await current().submit(false); });
  await until(gate.held);
  const count = await authorityJobCount();
  const id = current().selected?.job_id;
  if (id === undefined) throw new Error("Missing completed selection");
  await act(async () => { gate.release(true); });
  await until(() => current().resultError?.includes("Result could not be adopted") === true);
  expect(current().artifact).toBeNull();
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  await act(async () => { await current().select(id); });
  await until(() => current().artifact !== null && useStudioStore.getState().graphSimResult !== null);
  expect(await authorityJobCount()).toBe(count);
}, 90_000);

it("stops observation on a real poll disconnect and refreshes the actual terminal authority state", async () => {
  const gate = await responseGate("GET", (path) => /^\/api\/graph\/jobs\/sj_[0-9a-f]{16}$/.test(path));
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await act(async () => {
    useStudioStore.setState({ dt: 1, duration: 1000 }); useStudioStore.getState().updatePopulation("e", { count: 2000 });
    await current().submit(false);
  });
  await until(gate.held);
  const count = await authorityJobCount();
  await act(async () => { gate.release(true); });
  await until(() => current().error?.includes("Observation stopped") === true);
  const stopped = current().selected;
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 600)); });
  expect(current().selected).toBe(stopped);
  await act(async () => { current().refresh(); });
  await until(() => current().selected?.status === "completed");
  expect(await authorityJobCount()).toBe(count);
}, 90_000);

it("reports a lost cancellation acknowledgement and refreshes the real cancellation without another admission", async () => {
  const gate = await responseGate("POST", (path) => path.endsWith("/cancel"));
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await act(async () => {
    useStudioStore.setState({ dt: 1, duration: 1000 }); useStudioStore.getState().updatePopulation("e", { count: 2000 });
    await current().submit(false);
  });
  await until(() => current().selected?.status === "running");
  const count = await authorityJobCount();
  let cancellation: Promise<void> | null = null;
  await act(async () => { cancellation = current().cancel(); });
  await until(gate.held);
  await act(async () => { gate.release(true); await cancellation; });
  expect(current().error).toContain("Cancellation was not confirmed");
  await act(async () => { current().refresh(); });
  await until(() => current().selected?.status === "cancelled");
  expect(current().artifact).toBeNull();
  expect(await authorityJobCount()).toBe(count);
}, 90_000);

it("withdraws an adopted result when a public editor action produces a nonfinite draft", async () => {
  await act(async () => { await current().submit(false); });
  await until(() => useStudioStore.getState().graphSimResult !== null);
  await act(async () => { useStudioStore.getState().updatePopulation("e", { params: { tau: Number.NaN } }); });
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  expect(current().resultError).toContain("withdrawn");
});

it("preserves an unsaved canvas, undo history and selection when opening a graph-less legacy named session", async () => {
  await act(async () => {
    useStudioStore.getState().updatePopulation("e", { count: 3 });
    useStudioStore.setState({ selectedPopulationId: "e", selectedPopulationIds: ["e"],
      savedSessions: [{ name: "legacy-model", state: { selectedModelName: "PoissonNeuron", dt: 0.1, duration: 20, seed: 42 } }] });
  });
  const before = useStudioStore.getState();
  await act(async () => { useStudioStore.getState().loadSession("legacy-model"); });
  const after = useStudioStore.getState();
  expect(after.graphPopulations).toBe(before.graphPopulations);
  expect(after.graphProjections).toBe(before.graphProjections);
  expect(after.graphHistory).toBe(before.graphHistory);
  expect(after.graphWorkspaceId).toBe(before.graphWorkspaceId);
  expect(after.graphContextToken).toBe(before.graphContextToken);
  expect(after.selectedPopulationId).toBe("e");
  await act(async () => { useStudioStore.getState().undoGraphEdit(); });
  expect(useStudioStore.getState().graphPopulations[0]?.count).toBe(2);
});

it("clears account-owned recovery atomically before another account can save, including an unmounted graph panel", async () => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null);
  const alice = useStudioStore.getState().graphJobRecovery;
  expect(alice?.submission?.requestJson).toContain('"tau":10');
  await act(async () => { root.unmount(); });
  useStudioStore.setState({ authSession: { authenticated: true, principal_id: "bob", roles: ["viewer"] } });
  expect(useStudioStore.getState().graphJobRecovery).toBeNull();
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  useStudioStore.setState((state) => ({ ...state, graphJobRecovery: alice }), true);
  expect(useStudioStore.getState().graphJobRecovery).toBeNull();
  expect(useStudioStore.getState().saveSession("bob-workspace").status).toBe("ok");
  await useStudioStore.getState().saveProjectToServer("bob-workspace-project");
  expect(useStudioStore.getState().savedSessions.find((session) => session.name === "bob-workspace")?.state.graphJobRecovery).toBeNull();
  const project = await loadProject("bob-workspace-project");
  expect(JSON.stringify(project)).not.toContain(alice?.submission?.retryKey);
  expect(JSON.stringify(project)).toContain('"graphJobRecovery":null');
});

it("keeps a different unconfirmed browser retry while visibly opening a saved selection", async () => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null);
  const snapshot = studioProjectSaveState(useStudioStore.getState());
  const saved = useStudioStore.getState().graphJobRecovery;
  if (saved?.submission === null || saved?.submission === undefined) throw new Error("No original submission");
  const cached = { ...saved, selectedJobId: null, submission: { ...saved.submission, retryKey: randomUUID() } };
  const workspace = useStudioStore.getState().graphWorkspaceId;
  storeGraphJobRecovery(workspace, cached);
  await act(async () => { useStudioStore.setState(studioProjectRestoreState(snapshot)); });
  await until(() => current().selected?.job_id === saved.selectedJobId);
  expect(loadGraphJobRecovery(workspace, "local")).toEqual(cached);
  await act(async () => { root.render(<StrictMode><GraphJobsPanel /></StrictMode>); });
  await until(() => container.textContent.includes("takes precedence over this tab's recovery cache"));
  expect(loadGraphJobRecovery(workspace, "local")?.submission?.retryKey).toBe(cached.submission.retryKey);
});

it("prevents actual Alice login/logout recovery from entering Bob's actual server and browser saves", async () => {
  const password = randomUUID();
  const authenticated = await startCandidateTestServer(password);
  try {
    dom.happyDOM.setURL(authenticated.origin);
    await act(async () => { await useStudioStore.getState().loginBrowserUser("alice", password); });
    expect(useStudioStore.getState().authSession?.principal_id).toBe("alice");
    await act(async () => { await current().submit(false); });
    await until(() => current().artifact !== null);
    const original = useStudioStore.getState().graphJobRecovery;
    expect(original?.owner).toBe("alice");
    await act(async () => { await useStudioStore.getState().logoutBrowserUser(); });
    expect(useStudioStore.getState().graphJobRecovery).toBeNull();
    expect(current().selected).toBeNull();
    await act(async () => { await useStudioStore.getState().loginBrowserUser("bob", password); });
    expect(useStudioStore.getState().authSession?.principal_id).toBe("bob");
    await until(() => current().ready && current().history.length === 0);
    expect(current().selected).toBeNull();
    expect(useStudioStore.getState().graphJobRecovery).toBeNull();
    await act(async () => {
      expect(useStudioStore.getState().saveSession("bob-private").status).toBe("ok");
      await useStudioStore.getState().saveProjectToServer("bob-private-project");
    });
    const project = await loadProject("bob-private-project");
    const saved = useStudioStore.getState().savedSessions.find((entry) => entry.name === "bob-private");
    expect(saved?.state.graphJobRecovery).toBeNull();
    expect(JSON.stringify(project)).toContain('"graphJobRecovery":null');
    expect(JSON.stringify(project)).not.toContain(original?.submission?.retryKey);
    await act(async () => { await useStudioStore.getState().logoutBrowserUser(); });
  } finally {
    await authenticated.stop();
    dom.happyDOM.setURL(server.origin);
  }
}, 90_000);

it.each([false, true])("reports a changed graph recovery during an actual project export (disconnect=%s)", async (disconnect) => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null);
  const gate = await responseGate("POST", (path) => path === "/api/studio/evidence/bundle");
  dom.happyDOM.setURL(gate.origin);
  let exporting: Promise<void> | null = null;
  await act(async () => {
    exporting = useStudioStore.getState().createEvidenceBundleForSurface("project", {
      audit_limit: 10, analysis_results: [], command_replay: null,
      default_flow_attestations: [], default_flow_runs: [], include_audit: true,
      job_ids: [], model_scan_results: [], project_name: null, simulation_results: [],
      weight_restore_results: [], weight_restore_attach_results: [],
    });
  });
  await until(gate.held);
  await act(async () => { current().forget(); gate.release(disconnect); await exporting; });
  expect(useStudioStore.getState().projectEvidenceBundle).toBeNull();
  expect(useStudioStore.getState().projectEvidenceBundleLoading).toBe(false);
  expect(useStudioStore.getState().projectEvidenceBundleError).toContain("workspace changed during export");
});

it("reports an unsealable graph clock during an actual in-flight project export", async () => {
  const gate = await responseGate("POST", (path) => path === "/api/studio/evidence/bundle");
  dom.happyDOM.setURL(gate.origin);
  let exporting: Promise<void> | null = null;
  await act(async () => {
    exporting = useStudioStore.getState().createEvidenceBundleForSurface("project", {
      audit_limit: 10, analysis_results: [], command_replay: null,
      default_flow_attestations: [], default_flow_runs: [], include_audit: true,
      job_ids: [], model_scan_results: [], project_name: null, simulation_results: [],
      weight_restore_results: [], weight_restore_attach_results: [],
    });
  });
  await until(gate.held);
  await act(async () => { useStudioStore.getState().setDuration(Number.NaN); gate.release(); await exporting; });
  expect(useStudioStore.getState().projectEvidenceBundle).toBeNull();
  expect(useStudioStore.getState().projectEvidenceBundleError).toContain("NaN");
  expect(useStudioStore.getState().projectEvidenceBundleLoading).toBe(false);
});
