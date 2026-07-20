// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Configure React 19 act environment for createRoot mounts.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — real React DOM mount of useStudioAnalysisJobIntegration
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, createElement, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";

import type {
  AnalysisJobReceipt,
  AnalysisJobRequestBody,
  FICurveResponse,
  StudioJobRecord,
} from "./api/client";
import {
  createAnalysisJobSession,
  type AnalysisJobApi,
  type AnalysisJobSessionOptions,
} from "./analysisJob";
import type { StudioSimulationConfigInput } from "./studioSimulationConfig";
import {
  resolveStudioAnalysisJobIntegration,
  studioAnalysisJobIntegrationCanSubmit,
  useStudioAnalysisJobIntegration,
  type StudioAnalysisJobIntegrationInput,
  type UseStudioAnalysisJobIntegrationOptions,
  type UseStudioAnalysisJobIntegrationResult,
} from "./useStudioAnalysisJobIntegration";

const modelInput: StudioSimulationConfigInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams: { tau: 10, capacitance: 1 },
  equations: ["dv/dt = 0"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { tau: 20, e_l: -65 },
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
};

const baseInput: StudioAnalysisJobIntegrationInput = {
  simulation: modelInput,
  analysis: "fi_curve",
  sweepParam: "tau",
  sweepParamY: "capacitance",
};

const fiResult: FICurveResponse = {
  analysis_metadata: {
    analysis_type: "fi_curve",
    evidence_classification: "analysis",
    input_sha256: "a".repeat(64),
    output_keys: ["currents", "rates"],
    result_sha256: "b".repeat(64),
    schema_version: "studio.analysis-result.v1",
    source: "model",
    status: "completed",
  },
  currents: [0, 1],
  rates: [0, 5],
};

function jobRecord(
  o: Partial<StudioJobRecord> & Pick<StudioJobRecord, "status">,
): StudioJobRecord {
  return {
    artifacts: [], created_at_utc: "2026-07-20T00:00:00Z", error: null,
    execution_model: "thread", finished_at_utc: null, job_id: "sj_dom",
    kind: "analysis", owner: "studio", request_id: null, result: null,
    started_at_utc: null, ...o,
  };
}

function receipt(): AnalysisJobReceipt {
  return {
    analysis: "fi_curve", execution_mode: "async_job",
    job: jobRecord({ status: "pending" }), job_id: "sj_dom",
    schema_version: "studio.analysis.job.v1",
    status_route: "/api/studio/jobs/sj_dom",
  };
}

describe("resolveStudioAnalysisJobIntegration (pure)", () => {
  it("builds request and applies capability/disabled gates", () => {
    const ok = resolveStudioAnalysisJobIntegration(baseInput);
    expect(ok.request.ok).toBe(true);
    expect(ok.disabled).toBe(false);
    expect(
      resolveStudioAnalysisJobIntegration(baseInput, {
        capabilityEnabled: false,
      }).disabled,
    ).toBe(true);
    expect(
      studioAnalysisJobIntegrationCanSubmit({
        sessionCanSubmit: true, disabled: true, requestOk: true,
      }),
    ).toBe(false);
  });
});

describe("useStudioAnalysisJobIntegration real React DOM mount", () => {
  let root: Root | null = null;
  let host: HTMLDivElement | null = null;
  let latest: UseStudioAnalysisJobIntegrationResult | null = null;

  afterEach(() => {
    vi.useRealTimers();
    if (root !== null) {
      act(() => {
        root?.unmount();
      });
    }
    root = null;
    host?.remove();
    host = null;
    latest = null;
  });

  function mountHook(
    input: StudioAnalysisJobIntegrationInput,
    options: UseStudioAnalysisJobIntegrationOptions = {},
  ): void {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    function Host() {
      const value = useStudioAnalysisJobIntegration(input, options);
      useEffect(() => {
        latest = value;
      });
      return createElement(
        "div",
        { "data-testid": "hook-host" },
        value.selectedAnalysisLabel ?? "none",
      );
    }
    act(() => {
      root?.render(createElement(Host));
    });
  }

  it("mounts the production hook and disables submit when capability is off", () => {
    mountHook(baseInput, { capabilityEnabled: false });
    expect(latest).not.toBeNull();
    expect(latest?.disabled).toBe(true);
    expect(latest?.canSubmit).toBe(false);
    expect(latest?.request.ok).toBe(true);
    expect(host?.textContent).toBe("f-I curve");
  });

  it("submits async job, applies completion patch, disposes, suppresses stale updates", async () => {
    vi.useFakeTimers();
    let disposed = false;
    let onChange: ((s: ReturnType<typeof createAnalysisJobSession> extends never ? never : import("./analysisJob").AnalysisJobViewState) => void) | undefined;
    const polls: StudioJobRecord[] = [
      jobRecord({ status: "pending" }),
      jobRecord({
        status: "completed",
        result: { ...fiResult },
        finished_at_utc: "2026-07-20T00:00:02Z",
      }),
    ];
    let idx = 0;
    const api: AnalysisJobApi = {
      submit: async () => receipt(),
      fetchJob: async () => {
        const next = polls[Math.min(idx, polls.length - 1)]!;
        idx += 1;
        return next;
      },
    };
    const patches: unknown[] = [];
    const createSession = (opts: AnalysisJobSessionOptions) => {
      onChange = opts.onChange;
      const session = createAnalysisJobSession({
        ...opts,
        setTimeoutFn: setTimeout as typeof setTimeout,
        clearTimeoutFn: clearTimeout as typeof clearTimeout,
      });
      return {
        dispose: () => {
          disposed = true;
          session.dispose();
        },
        getState: () => session.getState(),
        startJob: async (request: AnalysisJobRequestBody) => {
          await session.startJob(request);
        },
      };
    };

    mountHook(baseInput, {
      applyPatch: (p) => {
        patches.push(p);
      },
      hookOptions: { api, createSession, pollIntervalMs: 10 },
    });
    expect(latest?.canSubmit).toBe(true);
    expect(latest?.request.ok).toBe(true);

    act(() => {
      if (latest?.request.ok) {
        latest.startJob(latest.request.value);
      }
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(10);
      await vi.advanceTimersByTimeAsync(10);
    });

    expect(latest?.state.phase).toBe("completed");
    expect(patches.some((p) =>
      typeof p === "object" && p !== null && "activeTab" in p
      && (p as { activeTab: string }).activeTab === "fi-curve"
    )).toBe(true);

    const staleBefore = patches.length;
    act(() => {
      root?.unmount();
    });
    root = null;
    expect(disposed).toBe(true);

    // Stale session onChange after dispose must not patch further.
    onChange?.({
      analysis: "fi_curve",
      error: null,
      jobId: "sj_stale",
      phase: "completed",
      result: fiResult,
      statusRoute: "/api/studio/jobs/sj_stale",
    });
    expect(patches.length).toBe(staleBefore);
  });
});
