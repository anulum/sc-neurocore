// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AnalysisJobWorkbench composition tests
import type { ComponentProps } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  analysisJobPhaseLabel,
  canSubmitAnalysisJob,
  initialAnalysisJobState,
  isAnalysisJobBusy,
  type AnalysisJobApi,
  type AnalysisJobSession,
  type AnalysisJobSessionOptions,
  type AnalysisJobViewState,
} from "../analysisJob";
import {
  buildAnalysisJobRequest,
  type AnalysisJobSelection,
} from "../analysisJobRequest";
import type { StudioSimulationConfigInput } from "../studioSimulationConfig";
import {
  attachAnalysisJobReactBinding,
  type UseAnalysisJobOptions,
  type UseAnalysisJobResult,
} from "../useAnalysisJob";
import { isAnalysisJobControlSubmitEnabled } from "./AnalysisJobControl";
import AnalysisJobWorkbench from "./AnalysisJobWorkbench";

const modelInput: StudioSimulationConfigInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams: { tau: 10, capacitance: 1 },
  equations: ["dv/dt = -(v - e_l) / tau + i"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { tau: 20, e_l: -65 },
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
};

const odeInput: StudioSimulationConfigInput = {
  ...modelInput,
  sourceMode: "ode",
  selectedModelName: "",
};

const selections: Array<{ selection: AnalysisJobSelection; label: string }> = [
  { selection: { analysis: "fi_curve" }, label: "f-I curve" },
  { selection: { analysis: "sensitivity" }, label: "sensitivity" },
  {
    selection: {
      analysis: "bifurcation",
      sweep: { sweepParam: "tau", parameterValue: 10 },
    },
    label: "bifurcation",
  },
  {
    selection: {
      analysis: "heatmap",
      sweep: {
        sweepParamX: "tau",
        parameterValueX: 10,
        sweepParamY: "capacitance",
        parameterValueY: 2,
      },
    },
    label: "heatmap",
  },
];

const useAnalysisJobMock = vi.hoisted(() =>
  vi.fn(
    (_options?: UseAnalysisJobOptions): UseAnalysisJobResult => ({
      busy: false,
      canSubmit: true,
      startJob: vi.fn(),
      state: initialAnalysisJobState(),
    }),
  ),
);

vi.mock("../useAnalysisJob", async () => {
  const actual = await vi.importActual<typeof import("../useAnalysisJob")>(
    "../useAnalysisJob",
  );
  return {
    ...actual,
    useAnalysisJob: (options?: UseAnalysisJobOptions) =>
      useAnalysisJobMock(options),
  };
});

function hookResult(
  overrides: Partial<UseAnalysisJobResult> = {},
): UseAnalysisJobResult {
  return {
    busy: overrides.busy ?? false,
    canSubmit: overrides.canSubmit ?? true,
    startJob: overrides.startJob ?? vi.fn(),
    state: overrides.state ?? initialAnalysisJobState(),
  };
}

function renderWorkbench(
  props: Partial<ComponentProps<typeof AnalysisJobWorkbench>> = {},
) {
  return renderToStaticMarkup(
    <AnalysisJobWorkbench
      simulationInput={props.simulationInput ?? modelInput}
      selection={props.selection ?? { analysis: "fi_curve" }}
      selectedAnalysisLabel={props.selectedAnalysisLabel ?? "f-I curve"}
      hookOptions={props.hookOptions}
    />,
  );
}

describe("AnalysisJobWorkbench composition", () => {
  afterEach(() => {
    useAnalysisJobMock.mockReset();
    useAnalysisJobMock.mockImplementation(() => hookResult());
  });

  it("composes model and ODE via W09 oracle for all four kinds", () => {
    for (const input of [modelInput, odeInput]) {
      for (const { selection, label } of selections) {
        useAnalysisJobMock.mockReturnValue(hookResult());
        const html = renderWorkbench({
          simulationInput: input,
          selection,
          selectedAnalysisLabel: label,
        });
        const expected = buildAnalysisJobRequest(input, selection);
        expect(expected.ok).toBe(true);
        expect(html).toContain('data-testid="analysis-job-workbench"');
        expect(html).toContain('data-testid="analysis-job-control"');
        expect(html).toContain(`Selected: ${label}`);
        expect(html).toContain(`Phase: ${analysisJobPhaseLabel("idle")}`);
        expect(html).toContain('data-phase="idle"');
        expect(
          isAnalysisJobControlSubmitEnabled({
            busy: false,
            canSubmit: true,
            request: expected,
          }),
        ).toBe(true);
        expect(html).not.toContain('disabled=""');
      }
    }
  });

  it("propagates invalid W09 request and blocks submit", () => {
    const startJob = vi.fn();
    useAnalysisJobMock.mockReturnValue(hookResult({ startJob }));
    const invalidInput = { ...modelInput, dt: Number.NaN };
    const expected = buildAnalysisJobRequest(invalidInput, {
      analysis: "fi_curve",
    });
    expect(expected).toEqual({
      ok: false,
      error: "analysis_request_dt_invalid",
    });
    const html = renderWorkbench({
      simulationInput: invalidInput,
      selection: { analysis: "fi_curve" },
    });
    expect(html).toContain("analysis_request_dt_invalid");
    expect(html).toContain("disabled");
    expect(
      isAnalysisJobControlSubmitEnabled({
        busy: false,
        canSubmit: true,
        request: expected,
      }),
    ).toBe(false);
    expect(startJob).not.toHaveBeenCalled();
  });

  it("forwards injected W08 options by reference and disposes via W08 binding", () => {
    let disposed = false;
    const unused = async () => {
      throw new Error("unused");
    };
    const api: AnalysisJobApi = { submit: vi.fn(unused), fetchJob: vi.fn(unused) };
    const createSession = vi.fn(
      (_opts: AnalysisJobSessionOptions): AnalysisJobSession => ({
        dispose: () => {
          disposed = true;
        },
        getState: () => initialAnalysisJobState(),
        startJob: async () => undefined,
      }),
    );
    const hookOptions: UseAnalysisJobOptions = {
      api,
      createSession,
      pollIntervalMs: 77,
    };
    useAnalysisJobMock.mockReturnValue(hookResult());
    renderWorkbench({ hookOptions });
    expect(useAnalysisJobMock).toHaveBeenCalledWith(hookOptions);
    expect(useAnalysisJobMock.mock.calls[0]?.[0]).toBe(hookOptions);
    const binding = attachAnalysisJobReactBinding(hookOptions);
    expect(createSession).toHaveBeenCalled();
    binding.dispose();
    expect(disposed).toBe(true);
  });

  it("blocks submission while busy with real phase label", () => {
    const startJob = vi.fn();
    const busyState: AnalysisJobViewState = {
      ...initialAnalysisJobState(),
      phase: "running",
      jobId: "sj_1",
    };
    useAnalysisJobMock.mockReturnValue(hookResult({
      busy: isAnalysisJobBusy(busyState.phase),
      canSubmit: canSubmitAnalysisJob(busyState),
      startJob,
      state: busyState,
    }));
    const html = renderWorkbench({ selection: { analysis: "sensitivity" } });
    expect(html).toContain('data-phase="running"');
    expect(html).toContain(`Phase: ${analysisJobPhaseLabel("running")}`);
    expect(html).toContain("disabled");
    expect(startJob).not.toHaveBeenCalled();
  });
});
