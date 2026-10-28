// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Inspector controls with real supervised native compiler jobs
// @vitest-environment happy-dom

import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, mkdtempSync } from "node:fs";
import { get as httpGet } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import CompilerInspector from "../src/components/CompilerInspector";
import { emitSVDirect, fetchCosimDetail, fetchModelDetail, fetchStudioJobArtifact } from "../src/api/client";
import { defaultEquationIrConfiguration } from "../src/equationIrConfig";
import { useStudioStore } from "../src/stores/studio";
import { studioBundleIsCurrent } from "../src/studioBundleContext";

const initial = useStudioStore.getState();
const origin = new URL(process.env.SC_NEUROCORE_STUDIO_COMPILER_API_ORIGIN ?? "http://127.0.0.1:24022");
const namespace = mkdtempSync(join(tmpdir(), "studio-inspector-live-"));
let server: ChildProcess | null = null;
let closed = Promise.resolve();
let root: Root;
let container: HTMLDivElement;
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

beforeAll(async () => {
  if (origin.hostname !== "127.0.0.1" || origin.protocol !== "http:") {
    throw new Error("The Inspector integration server requires a test-owned IPv4 loopback origin.");
  }
  const log = createWriteStream(join(namespace, "server.log"));
  const running = spawn(process.env.SC_NEUROCORE_STUDIO_COMPILER_PYTHON ?? "python", [
    "-m", "uvicorn", "sc_neurocore.studio.app:create_app", "--factory",
    "--host", origin.hostname, "--port", origin.port, "--log-level", "warning",
  ], { env: { ...process.env,
    SC_NEUROCORE_STUDIO_CORS_ORIGINS: origin.origin,
    SC_NEUROCORE_STUDIO_JOB_ROOT: join(namespace, "jobs"),
    SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH: join(namespace, "audit.jsonl"),
    SC_NEUROCORE_STUDIO_PROJECT_ROOT: join(namespace, "projects"),
  }, stdio: ["ignore", "pipe", "pipe"] });
  server = running;
  running.stdout.pipe(log, { end: false });
  running.stderr.pipe(log, { end: false });
  closed = new Promise((resolve) => { running.once("close", () => { log.end(); resolve(); }); });
  await vi.waitFor(async () => {
    if (running.exitCode !== null) throw new Error(`Inspector API exited ${String(running.exitCode)}; see ${namespace}`);
    const healthy = await new Promise<boolean>((resolve) => {
      const request = httpGet(new URL("/api/health", origin), (response) => {
        response.resume(); resolve(response.statusCode === 200);
      });
      request.once("error", () => { resolve(false); });
      request.setTimeout(1000, () => { request.destroy(); });
    });
    expect(healthy).toBe(true);
  }, { timeout: 30_000, interval: 500 });
}, 35_000);

afterAll(async () => {
  const running = server;
  if (running === null) return;
  running.kill("SIGINT");
  const deadline = setTimeout(() => { running.kill("SIGKILL"); }, 10_000);
  try { await closed; } finally { clearTimeout(deadline); }
});

beforeEach(() => {
  container = document.createElement("div"); document.body.append(container);
  act(() => {
    useStudioStore.setState({ ...initial, sourceMode: "ode", equations: ["dv/dt = I"],
      threshold: "", reset: "", odeInit: { v: 7 }, odeParams: {}, dt: 0.5,
      equationIrConfig: defaultEquationIrConfiguration() }, true);
    root = createRoot(container); root.render(<CompilerInspector />);
  });
});

afterEach(() => {
  act(() => { root.unmount(); useStudioStore.setState(initial, true); });
  container.remove(); vi.restoreAllMocks();
});

/**
 * Click the rendered control and wait for its actual store operation to settle.
 *
 * @param label - Visible button text.
 * @param busy - Public loading flag owned by the operation.
 */
async function click(label: string, busy: "isSimulating" | "compileEvidenceBundleLoading" = "isSimulating"): Promise<void> {
  const button = [...container.querySelectorAll("button")].find((item) => item.textContent === label);
  if (button === undefined || button.disabled) throw new Error(`Unavailable Inspector action: ${label}`);
  await act(async () => {
    button.click();
    await expect.poll(() => useStudioStore.getState()[busy], { timeout: 60_000 }).toBe(false);
  });
  expect(useStudioStore.getState().error).toBeNull();
}

it("compiles, checks, replays and exports the authored source through Inspector controls", async () => {
  await click("Build IR");
  const built = useStudioStore.getState().irResult;
  expect(built?.valid).toBe(true);
  expect(built?.studio_job_receipt.job_id).toMatch(/^sj_/);
  expect(built?.source_replay).toBe("not_requested");
  await click("Check graph structure");
  expect(useStudioStore.getState().irResult?.source_sha256).toBe(built?.source_sha256);
  await click("Replay source");
  expect(useStudioStore.getState().irResult?.source_replay).toBe("passed");
  await click("Emit SV");
  const state = useStudioStore.getState();
  expect(state.svSource).toContain("module sc_equation_neuron");
  expect(state.irResult?.source_graph_sha256).toBe(built?.source_graph_sha256);

  // Observe actual Blob arguments while retaining the DOM's object-URL implementation.
  const urls = vi.spyOn(URL, "createObjectURL");
  const downloads: [string, string | null][] = [
    ["Download IR", state.irText], ["Download SystemVerilog", state.svSource],
    ["Download compiler request", state.irRequestKey],
  ];
  for (const [label, expected] of downloads) {
    await click(label);
    const payload = urls.mock.calls.at(-1)?.[0];
    if (!(payload instanceof Blob)) throw new Error("Download did not receive a Blob");
    expect(await payload.text()).toBe(expected);
  }
  await click("Export compile evidence bundle", "compileEvidenceBundleLoading");
  const exported = useStudioStore.getState();
  expect(exported.compileEvidenceBundleError).toBeNull();
  expect(studioBundleIsCurrent("compile", exported)).toBe(true);
  expect(studioBundleIsCurrent("compile", { ...exported, irText: exported.irText + "\ud800" })).toBe(false);
  const bundle = exported.compileEvidenceBundle;
  expect(bundle?.summary.source_job_count).toBe(1);
  expect(container.textContent).toContain(`bundle ${String(bundle?.bundle_id)}`);
  const artifact = container.querySelector<HTMLButtonElement>('[aria-label="Download compile evidence artifact evidence/manifest.json"]');
  if (artifact === null) throw new Error("Exported manifest control absent");
  await act(async () => {
    artifact.click();
    await expect.poll(() => urls.mock.calls.length, { timeout: 10_000 }).toBe(4);
  });
  const manifestBlob = urls.mock.calls.at(-1)?.[0];
  if (!(manifestBlob instanceof Blob) || bundle === null) throw new Error("Manifest download absent");
  const bytes = Buffer.from(await manifestBlob.arrayBuffer());
  const manifest: unknown = JSON.parse(bytes.toString("utf-8"));
  const { manifest_artifact: artifactReceipt, ...sealed } = bundle.manifest;
  expect(manifest).toEqual(sealed);
  expect(artifactReceipt).toMatchObject({ sha256: createHash("sha256").update(bytes).digest("hex"), size_bytes: bytes.length });
  await act(async () => {
    await useStudioStore.getState().createEvidenceBundleForSurface("compile", {
      audit_limit: 100, analysis_results: [], command_replay: null,
      default_flow_attestations: [], default_flow_runs: [], include_audit: true,
      job_ids: ["sj_missing_inspector_export"], model_scan_results: [], project_name: null,
      simulation_results: [], weight_restore_results: [], weight_restore_attach_results: [],
    });
  });
  const failed = useStudioStore.getState();
  expect(failed.compileEvidenceBundle).toBeNull();
  expect(failed.compileEvidenceBundleError).toBeTruthy();
  expect(container.textContent).toContain(failed.compileEvidenceBundleError);
}, 180_000);

it.each(["ode", "model"] as const)("retains a reproducible %s compile export alongside native equation authoring", async (sourceMode) => {
  if (sourceMode === "model") {
    const detail = await fetchModelDetail("AdaptiveThresholdIFNeuron");
    act(() => { useStudioStore.setState({ sourceMode, selectedModelName: detail.name,
      modelDetail: detail, modelParams: {}, modelIntegrator: "map", dt: detail.dt,
      modelQFormat: "Q8.8" }); });
  }
  await act(async () => { await useStudioStore.getState().runCompile(); });
  const compiled = useStudioStore.getState();
  expect(compiled.error).toBeNull();
  expect(compiled.verilogSrc).toContain("module ");
  expect(compiled.compileTraceability?.source).toBe(sourceMode);
  expect(container.textContent).toContain("Generated source; native graph validation unavailable.");
  await click("Export compile evidence bundle", "compileEvidenceBundleLoading");
  const exported = useStudioStore.getState();
  expect(exported.compileEvidenceBundleError).toBeNull();
  const bundle = exported.compileEvidenceBundle;
  if (bundle === null) throw new Error("Compile evidence export absent");
  const command = await fetchStudioJobArtifact(bundle.job_id, "evidence/command-replay.json");
  const replay: unknown = JSON.parse(await command.text());
  const trace = compiled.compileTraceability;
  if (trace === null) throw new Error("Compile source trace absent");
  expect(replay).toMatchObject({ method: "POST", route: sourceMode === "ode" ? "/api/compile" : "/api/models/compile",
    body: { module_name: trace.output.module_name },
    source_payload_sha256: trace.input_sha256 });
  if (typeof replay !== "object" || replay === null || !("route" in replay)
    || typeof replay.route !== "string" || !("body" in replay)) throw new Error("Replay request absent");
  const reproduced = await fetch(replay.route, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(replay.body) });
  const reproducedText = await reproduced.text();
  expect(reproduced.ok, reproducedText).toBe(true);
  const reproducedResult: unknown = JSON.parse(reproducedText);
  expect(reproducedResult).toMatchObject({ verilog: compiled.verilogSrc,
    compile_traceability: { input_sha256: trace.input_sha256 } });
}, 120_000);

it("keeps direct equation RTL and actual fixed-point comparison available", async () => {
  const request = { equations: ["dv/dt = I"], init: { v: 7 }, params: {},
    threshold: "", reset: "", module_name: "inspector_direct" };
  const direct = await emitSVDirect(request);
  expect(direct.verilog).toContain("module inspector_direct");
  expect(direct.compile_traceability.source_payload).toMatchObject({ equations: request.equations, init: request.init });
  const comparison = await fetchCosimDetail({ equations: request.equations, init: request.init,
    dt: 0.5, duration: 1, current: 1, protocol: "constant", frequency_hz: 10,
    q_format: "Q8.8", overflow: "saturate", rounding: "truncate" });
  expect(comparison.float_result.states.v).toEqual([7.5, 8]);
  expect(comparison.fixed_result.states.v).toEqual(comparison.float_result.states.v);
  expect(comparison.error.max_error).toBe(0);
}, 120_000);
