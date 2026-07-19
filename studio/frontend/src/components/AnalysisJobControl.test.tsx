// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AnalysisJobControl presentational tests
import type { ComponentProps } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { FICurveResponse } from "../api/client";
import {
  analysisJobPhaseLabel,
  initialAnalysisJobState,
  type AnalysisJobPhase,
  type AnalysisJobViewState,
} from "../analysisJob";
import type { AnalysisJobRequestBuildResult } from "../analysisJobRequest";
import AnalysisJobControl, {
  decideAnalysisJobControlStart,
  isAnalysisJobControlSubmitEnabled,
} from "./AnalysisJobControl";

const validRequest: AnalysisJobRequestBuildResult = {
  ok: true,
  value: {
    analysis: "fi_curve",
    payload: { model_name: "lif", i_min: 0, i_max: 24, i_steps: 25 },
  },
};

const invalidRequest: AnalysisJobRequestBuildResult = {
  ok: false,
  error: "analysis_request_dt_invalid",
};

const completedResult: FICurveResponse = {
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

function state(phase: AnalysisJobPhase, extra: Partial<AnalysisJobViewState> = {}): AnalysisJobViewState {
  return {
    ...initialAnalysisJobState(),
    phase,
    ...extra,
  };
}

function renderControl(props: Partial<ComponentProps<typeof AnalysisJobControl>> = {}) {
  return renderToStaticMarkup(
    <AnalysisJobControl
      busy={props.busy ?? false}
      canSubmit={props.canSubmit ?? true}
      request={props.request ?? validRequest}
      selectedAnalysisLabel={props.selectedAnalysisLabel ?? "f-I curve"}
      startJob={props.startJob ?? vi.fn()}
      state={props.state ?? initialAnalysisJobState()}
    />,
  );
}

describe("AnalysisJobControl pure start helper", () => {
  it("starts only for valid request when canSubmit and not busy", () => {
    const startJob = vi.fn();
    expect(
      decideAnalysisJobControlStart({
        busy: false,
        canSubmit: true,
        request: validRequest,
        startJob,
      }),
    ).toBe("started");
    expect(startJob).toHaveBeenCalledWith(validRequest.value);
  });

  it("blocks invalid and busy inputs without calling startJob", () => {
    const startJob = vi.fn();
    expect(
      decideAnalysisJobControlStart({
        busy: false,
        canSubmit: true,
        request: invalidRequest,
        startJob,
      }),
    ).toBe("blocked_invalid");
    expect(
      decideAnalysisJobControlStart({
        busy: true,
        canSubmit: true,
        request: validRequest,
        startJob,
      }),
    ).toBe("blocked_busy");
    expect(
      decideAnalysisJobControlStart({
        busy: false,
        canSubmit: false,
        request: validRequest,
        startJob,
      }),
    ).toBe("blocked_busy");
    expect(startJob).not.toHaveBeenCalled();
  });
});

describe("AnalysisJobControl render", () => {
  it("renders idle SSR surface with enabled submit for valid request", () => {
    const html = renderControl();
    expect(html).toContain("data-testid=\"analysis-job-control\"");
    expect(html).toContain("Selected: f-I curve");
    expect(html).toContain(`Phase: ${analysisJobPhaseLabel("idle")}`);
    expect(html).toContain("data-phase=\"idle\"");
    expect(html).toContain("Run async analysis");
    expect(html).not.toContain("disabled=\"\"");
    expect(isAnalysisJobControlSubmitEnabled({
      busy: false,
      canSubmit: true,
      request: validRequest,
    })).toBe(true);
  });

  it("disables submit and shows exact policy error for invalid request", () => {
    const html = renderControl({ request: invalidRequest });
    expect(html).toContain("data-testid=\"analysis-job-control-request-error\"");
    expect(html).toContain("analysis_request_dt_invalid");
    expect(html).toContain("disabled");
    expect(isAnalysisJobControlSubmitEnabled({
      busy: false,
      canSubmit: true,
      request: invalidRequest,
    })).toBe(false);
  });

  it("renders every busy and terminal phase label", () => {
    const phases: AnalysisJobPhase[] = [
      "submitting",
      "pending",
      "running",
      "completed",
      "failed",
      "cancelled",
      "timed_out",
      "malformed",
    ];
    for (const phase of phases) {
      const html = renderControl({
        busy: phase === "submitting" || phase === "pending" || phase === "running",
        canSubmit: !(phase === "submitting" || phase === "pending" || phase === "running"),
        state: state(
          phase,
          phase === "completed"
            ? { result: completedResult }
            : phase === "failed" || phase === "malformed"
              ? { error: "budget_exceeded" }
              : {},
        ),
      });
      expect(html).toContain(`data-phase=\"${phase}\"`);
      expect(html).toContain(`Phase: ${analysisJobPhaseLabel(phase)}`);
    }
  });

  it("shows path-safe public error and completed metadata without raw payload dump", () => {
    const failed = renderControl({
      state: state("failed", { error: "budget_exceeded" }),
    });
    expect(failed).toContain("data-testid=\"analysis-job-control-error\"");
    expect(failed).toContain("budget_exceeded");
    expect(failed).not.toContain("/home/");
    expect(failed).not.toContain("currents");

    const completed = renderControl({
      state: state("completed", { result: completedResult }),
    });
    expect(completed).toContain("data-testid=\"analysis-job-control-completed-summary\"");
    expect(completed).toContain("fi_curve");
    expect(completed).toContain("analysis");
    expect(completed).toContain("studio.analysis-result.v1");
    expect(completed).toContain("completed");
    expect(completed).not.toContain("currents");
    expect(completed).not.toContain(JSON.stringify(completedResult.currents));
  });

  it("disables submit while busy", () => {
    expect(isAnalysisJobControlSubmitEnabled({
      busy: true,
      canSubmit: true,
      request: validRequest,
    })).toBe(false);
    const html = renderControl({
      busy: true,
      canSubmit: false,
      state: state("running"),
    });
    expect(html).toContain("disabled");
    expect(html).toContain("aria-busy=\"true\"");
  });
});
