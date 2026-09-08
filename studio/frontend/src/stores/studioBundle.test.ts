// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evidence bundle request lifecycle through public store

import { afterEach, expect, it, vi } from "vitest";
import { useStudioStore } from "./studio";
const initial = useStudioStore.getState();
const cases = [["admin", "evidenceBundle", "evidenceBundleError"],
  ["project", "projectEvidenceBundle", "projectEvidenceBundleError"],
  ["compile", "compileEvidenceBundle", "compileEvidenceBundleError"],
  ["synthesis", "synthesisEvidenceBundle", "synthesisEvidenceBundleError"]] as const;
const request = { audit_limit: 10, analysis_results: [], command_replay: null,
  default_flow_attestations: [], default_flow_runs: [], include_audit: true,
  job_ids: [], model_scan_results: [], project_name: null, simulation_results: [],
  weight_restore_results: [], weight_restore_attach_results: [] };
afterEach(() => { useStudioStore.setState(initial, true); vi.unstubAllGlobals(); });

/**
 * Control export completion while keeping the real HTTP client and store.
 *
 * @returns A transport completion handle, not an evidence verification fixture.
 */
function transport() {
  let finish = (_fail?: boolean): void => { throw new Error("request absent"); };
  const fetch = vi.fn<typeof globalThis.fetch>((_url, init) => {
    if (init?.method !== "POST") return Promise.resolve(new Response(JSON.stringify({ audit: {}, jobs: [] })));
    return new Promise<Response>((resolve, reject) => {
      finish = (fail = false) => {
        if (fail) reject(new Error("export offline"));
        else resolve(new Response(JSON.stringify({ job_id: "sj_bundle", artifacts: [] })));
      };
    });
  });
  vi.stubGlobal("fetch", fetch);
  return { finish: (fail = false) => { finish(fail); }, fetch };
}

it.each(cases)("%s withdraws the old bundle during a failed rerun", async (surface, field) => {
  let io = transport();
  let run = useStudioStore.getState().createEvidenceBundleForSurface(surface, request);
  io.finish(); await run;
  expect(useStudioStore.getState()[field]).not.toBeNull();
  io = transport();
  run = useStudioStore.getState().createEvidenceBundleForSurface(surface, request);
  const during = useStudioStore.getState()[field];
  io.finish(true); await run;
  expect(during).toBeNull();
  expect(useStudioStore.getState()[field]).toBeNull();
});

it.each(cases.slice(1))("%s drops stale bundle success and failure", async (surface, field, errorField) => {
  for (const fail of [false, true]) {
    const io = transport();
    const run = useStudioStore.getState().createEvidenceBundleForSurface(surface, request);
    useStudioStore.getState().setSourceMode(useStudioStore.getState().sourceMode === "model" ? "ode" : "model");
    io.finish(fail); await run;
    expect(useStudioStore.getState()[field]).toBeNull();
    expect(useStudioStore.getState()[errorField]).toBeNull();
  }
});

it.each(cases)("%s refuses a duplicate export without state mutation", async (surface) => {
  const io = transport();
  const run = useStudioStore.getState().createEvidenceBundleForSurface(surface, request);
  const state = useStudioStore.getState();
  const duplicate = useStudioStore.getState().createEvidenceBundleForSurface(surface, request);
  const after = useStudioStore.getState();
  io.finish();
  expect(after).toBe(state);
  await run; await duplicate;
});

it("admin export remains valid while the selected experiment changes", async () => {
  const io = transport();
  const run = useStudioStore.getState().createEvidenceBundle(request);
  useStudioStore.getState().setSourceMode("ode");
  io.finish(); await run;
  expect(useStudioStore.getState().evidenceBundle?.job_id).toBe("sj_bundle");
});

it("independent surface exports can overlap", async () => {
  const first = transport();
  const a = useStudioStore.getState().createEvidenceBundleForSurface("compile", request);
  const second = transport();
  const b = useStudioStore.getState().createEvidenceBundleForSurface("synthesis", request);
  first.finish(); second.finish(); await Promise.all([a, b]);
  expect(useStudioStore.getState().compileEvidenceBundle?.job_id).toBe("sj_bundle");
  expect(useStudioStore.getState().synthesisEvidenceBundle?.job_id).toBe("sj_bundle");
});

it("drops synthesis bundle completion after job identity changes during refresh", async () => {
  const refresh: ((value: Response) => void)[] = [];
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>((_url, init) => {
    if (init?.method === "POST") return Promise.resolve(new Response(JSON.stringify({ job_id: "sj_bundle", artifacts: [] })));
    return new Promise<Response>((resolve) => { refresh.push(resolve); });
  }));
  const run = useStudioStore.getState().createEvidenceBundleForSurface("synthesis", request);
  await vi.waitFor(() => { expect(refresh).toHaveLength(2); });
  useStudioStore.setState({ latestSynthesisJobId: "sj_changed" });
  for (const resolve of refresh) resolve(new Response(JSON.stringify({ audit: {}, jobs: [] })));
  await run;
  expect(useStudioStore.getState().synthesisEvidenceBundle).toBeNull();
  expect(useStudioStore.getState().synthesisEvidenceBundleLoading).toBe(false);
});

it("reports unsealable project input without making a request", async () => {
  const io = transport();
  useStudioStore.setState({ dt: Number.NaN });
  await useStudioStore.getState().createEvidenceBundleForSurface("project", request);
  expect(io.fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().projectEvidenceBundleError).toContain("NaN");
  expect(useStudioStore.getState().projectEvidenceBundleLoading).toBe(false);
});
