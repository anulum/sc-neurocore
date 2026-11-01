// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Actual late graph responses cannot overwrite newer context or authority state
// @vitest-environment happy-dom

import { act } from "react";
import { expect, it } from "vitest";
import { graphWorkspaceId } from "../src/graphJobRecovery";
import { useStudioStore } from "../src/stores/studio";
import { current, responseGate, dom, until, authorityJobCount, whileDigestPending } from "./graphJobTestContext";

it.each([false, true])("cannot revive a cancelled graph from a late poll (disconnect=%s)", async (disconnect) => {
  const gate = await responseGate("GET", (path) => /^\/api\/graph\/jobs\/sj_[0-9a-f]{16}$/.test(path));
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await act(async () => {
    useStudioStore.setState({ dt: 1, duration: 1000 }); useStudioStore.getState().updatePopulation("e", { count: 2000 });
    await current().submit(false);
  });
  await until(gate.held);
  await act(async () => { await current().cancel(); });
  await until(() => current().selected?.status === "cancelled");
  const cancelled = current().selected;
  await act(async () => { gate.release(disconnect); await dom.happyDOM.waitUntilComplete(); });
  expect(current().selected).toBe(cancelled);
  expect(current().error).toBeNull();
  expect(current().artifact).toBeNull();
  expect(useStudioStore.getState().graphSimResult).toBeNull();
}, 90_000);

it.each([false, true])("keeps the newer selected calculation after an old selection response (disconnect=%s)", async (disconnect) => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null);
  const first = current().selected?.job_id;
  await act(async () => {
    useStudioStore.getState().updatePopulation("e", { params: { tau: 11 } });
    await current().submit(false);
  });
  await until(() => current().artifact !== null);
  const second = current().selected?.job_id;
  if (first === undefined || second === undefined) throw new Error("Missing actual graph selections");
  const count = await authorityJobCount();
  const gate = await responseGate("GET", (path) => path === `/api/graph/jobs/${first}`);
  dom.happyDOM.setURL(gate.origin);
  let selecting: Promise<void> | null = null;
  await act(async () => { selecting = current().select(first); });
  await until(gate.held);
  await act(async () => { await current().select(second); });
  await until(() => current().artifact !== null && current().selected?.job_id === second);
  const selected = current().selected;
  const verified = current().artifact;
  await act(async () => { gate.release(disconnect); await selecting; await dom.happyDOM.waitUntilComplete(); });
  expect(current().selected).toBe(selected);
  expect(current().artifact).toBe(verified);
  expect(current().error).toBeNull();
  expect(await authorityJobCount()).toBe(count);
}, 90_000);

it.each(["validation", "admission"] as const)("keeps a restored workspace independent of an old %s response", async (stage) => {
  const before = await authorityJobCount();
  const workspace = useStudioStore.getState().graphWorkspaceId;
  const gate = await responseGate("POST", (path) => path === (stage === "validation" ? "/api/graph/jobs/validate" : "/api/graph/jobs"));
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  let submitting: Promise<void> | null = null;
  await act(async () => { submitting = current().submit(false); });
  await until(gate.held);
  expect(await authorityJobCount()).toBe(before + (stage === "admission" ? 1 : 0));
  await act(async () => {
    useStudioStore.setState({ graphWorkspaceId: graphWorkspaceId(), graphJobRecovery: null, graphContextToken: { restored: true } });
  });
  await act(async () => { gate.release(); await submitting; await dom.happyDOM.waitUntilComplete(); });
  expect(current().selected).toBeNull();
  expect(current().artifact).toBeNull();
  expect(current().error).toBeNull();
  expect(current().busy).toBe(false);
  expect(useStudioStore.getState().graphJobRecovery).toBeNull();
  if (stage === "admission") {
    await act(async () => {
      useStudioStore.setState({ graphWorkspaceId: workspace, graphJobRecovery: null, graphContextToken: { restored: false } });
    });
    await until(() => current().unconfirmed);
    await act(async () => { await current().submit(true); });
    await until(() => current().artifact !== null);
    expect(await authorityJobCount()).toBe(before + 1);
  } else expect(await authorityJobCount()).toBe(before);
}, 90_000);

it("does not carry a delayed cancellation error or busy state into a restored workspace", async () => {
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
  await act(async () => {
    useStudioStore.setState({ graphWorkspaceId: graphWorkspaceId(), graphJobRecovery: null, graphContextToken: { restored: true } });
  });
  await act(async () => { gate.release(true); await cancellation; await dom.happyDOM.waitUntilComplete(); });
  expect(current().selected).toBeNull();
  expect(current().artifact).toBeNull();
  expect(current().error).toBeNull();
  expect(current().busy).toBe(false);
  expect(await authorityJobCount()).toBe(count);
}, 90_000);

it("withdraws adoption when the graph changes while its real completed-result revalidation is held", async () => {
  let preparations = 0;
  const gate = await responseGate("POST", (path) => path === "/api/graph/jobs/validate" && ++preparations === 2);
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await act(async () => { await current().submit(false); });
  await until(gate.held);
  const verified = current().artifact;
  expect(verified).not.toBeNull();
  expect(useStudioStore.getState().graphSimResult).toBeNull();
  await act(async () => { useStudioStore.getState().updatePopulation("e", { params: { tau: 11 } }); gate.release(); });
  await until(() => current().resultError?.includes("different graph") === true);
  expect(current().artifact).toBe(verified);
  expect(useStudioStore.getState().graphSimResult).toBeNull();
}, 90_000);

it("restores a selected history pointer without retained preparation and ignores its delayed old read after restore", async () => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null);
  const id = current().selected?.job_id;
  if (id === undefined) throw new Error("No actual graph to restore");
  await act(async () => { current().forget(); await current().select(id); });
  await until(() => current().artifact !== null);
  expect(useStudioStore.getState().graphJobRecovery?.submission).toBeNull();
  await act(async () => { current().refresh(); });
  await until(() => current().artifact !== null && current().selected?.job_id === id);
  const gate = await responseGate("GET", (path) => path === `/api/graph/jobs/${id}`);
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await until(gate.held);
  await act(async () => {
    useStudioStore.setState({ graphWorkspaceId: graphWorkspaceId(), graphJobRecovery: null, graphContextToken: { restored: true } });
  });
  await act(async () => { gate.release(); await dom.happyDOM.waitUntilComplete(); });
  expect(current().selected).toBeNull();
  expect(current().artifact).toBeNull();
  expect(current().blockedRecovery).toBe(false);
  expect(current().error).toBeNull();
}, 90_000);

it.each(["preparation", "admission", "artifact", "revalidation", "cancellation", "restoration", "scientific-edit"] as const)(
  "refuses an old %s after its complete HTTP body arrived but actual WebCrypto is still pending", async (stage) => {
    let restoredId = "";
    if (stage === "restoration") {
      await act(async () => { await current().submit(false); });
      await until(() => current().artifact !== null);
      restoredId = current().selected?.job_id ?? "";
      await act(async () => { current().forget(); await current().select(restoredId); });
      await until(() => current().artifact !== null);
    }
    const before = await authorityJobCount();
    let preparations = 0;
    const method = stage === "artifact" || stage === "restoration" ? "GET" : "POST";
    const gate = await responseGate(method, (path) => {
      if (stage === "preparation" || stage === "scientific-edit") return path === "/api/graph/jobs/validate";
      if (stage === "revalidation") return path === "/api/graph/jobs/validate" && ++preparations === 2;
      if (stage === "admission") return path === "/api/graph/jobs";
      if (stage === "artifact") return path.endsWith("/result");
      if (stage === "cancellation") return path.endsWith("/cancel");
      return path === `/api/graph/jobs/${restoredId}`;
    });
    dom.happyDOM.setURL(gate.origin);
    await act(async () => { current().refresh(); });
    await until(() => current().history.length === before);
    let pending: Promise<void> | null = null;
    if (stage !== "restoration") {
      await act(async () => {
        if (stage === "cancellation") {
          useStudioStore.setState({ dt: 1, duration: 1000 }); useStudioStore.getState().updatePopulation("e", { count: 2000 });
        }
        pending = current().submit(false);
      });
      if (stage === "cancellation") {
        await until(() => current().selected?.status === "running");
        await act(async () => { pending = current().cancel(); });
      }
    }
    await until(gate.held);
    await whileDigestPending(() => { gate.release(); }, () => {
      if (stage === "scientific-edit") useStudioStore.getState().updatePopulation("e", { count: 3 });
      else useStudioStore.setState({ graphWorkspaceId: graphWorkspaceId(), graphJobRecovery: null, graphContextToken: { restored: true } });
    });
    await act(async () => { await pending; await dom.happyDOM.waitUntilComplete(); });
    expect(current().selected).toBeNull();
    expect(current().artifact).toBeNull();
    if (stage === "scientific-edit") expect(current().error).toContain("The graph changed before submission");
    else expect(current().error).toBeNull();
    expect(current().busy).toBe(false);
    expect(current().blockedRecovery).toBe(false);
    expect(useStudioStore.getState().graphSimResult).toBeNull();
    expect(useStudioStore.getState().graphJobRecovery).toBeNull();
    expect(await authorityJobCount()).toBe(before + (stage === "preparation" || stage === "restoration" || stage === "scientific-edit" ? 0 : 1));
  }, 90_000,
);

it("observes a known durable selection even when the real history response has not arrived", async () => {
  await act(async () => { await current().submit(false); });
  await until(() => current().artifact !== null);
  const id = current().selected?.job_id;
  if (id === undefined) throw new Error("No durable graph selection");
  const count = await authorityJobCount();
  const gate = await responseGate("GET", (path) => path === "/api/graph/jobs");
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  await until(gate.held);
  expect(current().history).toEqual([]);
  await act(async () => { await current().select(id); });
  await until(() => current().artifact !== null);
  expect(current().selected?.job_id).toBe(id);
  await act(async () => { gate.release(); await dom.happyDOM.waitUntilComplete(); });
  expect(await authorityJobCount()).toBe(count);
});

it("refuses a retained retry when its actual input-seal digest completes in another workspace", async () => {
  const before = await authorityJobCount();
  const gate = await responseGate("POST", (path) => path === "/api/graph/jobs");
  dom.happyDOM.setURL(gate.origin);
  await act(async () => { current().refresh(); });
  let submitting: Promise<void> | null = null;
  await act(async () => { submitting = current().submit(false); });
  await until(gate.held);
  await act(async () => { gate.release(true); await submitting; });
  await until(() => current().unconfirmed && !current().busy);
  await act(async () => { current().refresh(); });
  await until(() => current().unconfirmed);
  const original = useStudioStore.getState().graphJobRecovery;
  await whileDigestPending(() => { submitting = current().submit(true); }, () => {
    useStudioStore.setState({ graphWorkspaceId: graphWorkspaceId(), graphJobRecovery: null, graphContextToken: { restored: true } });
  });
  await act(async () => { await submitting; await dom.happyDOM.waitUntilComplete(); });
  expect(original?.submission).not.toBeNull();
  expect(current().selected).toBeNull();
  expect(current().artifact).toBeNull();
  expect(current().error).toBeNull();
  expect(current().busy).toBe(false);
  expect(current().unconfirmed).toBe(false);
  expect(await authorityJobCount()).toBe(before + 1);
}, 90_000);
