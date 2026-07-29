// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio operator workbench panel tests
import { act } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { OperatorWorkbenchState } from "../operatorWorkbenchState";
import OperatorWorkbenchPanel from "./OperatorWorkbenchPanel";

function workbenchState(overrides: Partial<OperatorWorkbenchState> = {}): OperatorWorkbenchState {
  return {
    cards: overrides.cards ?? [
      {
        action: "Save project",
        detail: "0 server projects, 0 local sessions",
        key: "workspace",
        status: "warning",
        title: "Workspace",
        value: "Unsaved",
      },
      {
        action: "Run simulation",
        detail: "No path-free simulation metadata is loaded",
        key: "simulation",
        status: "blocked",
        title: "Simulation",
        value: "Not run",
      },
      {
        action: "Open admin",
        detail: "8/8 capabilities healthy",
        key: "evidence",
        status: "ready",
        title: "Evidence health",
        value: "development",
      },
      {
        action: "Export bundle",
        detail: "Requires Synthesise",
        key: "export",
        status: "blocked",
        title: "Export",
        value: "blocked",
      },
    ],
    evidenceActionEnabled: overrides.evidenceActionEnabled ?? false,
    evidenceExportTarget: overrides.evidenceExportTarget ?? null,
    headline: overrides.headline ?? "Next: Design",
    subhead: overrides.subhead ?? "0/7 lifecycle steps complete",
  };
}

describe("OperatorWorkbenchPanel", () => {
  it("renders first-screen operator cards with progress and disabled export", () => {
    const html = renderToStaticMarkup(
      <OperatorWorkbenchPanel
        onExportEvidence={() => undefined}
        onOpenAdmin={() => undefined}
        onOpenCompiler={() => undefined}
        onOpenSynthesis={() => undefined}
        onOpenProjects={() => undefined}
        onRunSimulation={() => undefined}
        state={workbenchState()}
      />,
    );

    expect(html).toContain("Operator workbench");
    expect(html).toContain("Next: Design");
    expect(html).toContain("0/7 lifecycle steps complete");
    expect(html).toContain('data-card="workspace"');
    expect(html).toContain("Evidence health");
    expect(html).toContain("Export bundle");
    expect(html).toContain("disabled");
  });

  it("enables the export action when bundle evidence can be produced", () => {
    const html = renderToStaticMarkup(
      <OperatorWorkbenchPanel
        onExportEvidence={() => undefined}
        onOpenAdmin={() => undefined}
        onOpenCompiler={() => undefined}
        onOpenSynthesis={() => undefined}
        onOpenProjects={() => undefined}
        onRunSimulation={() => undefined}
        state={workbenchState({ evidenceActionEnabled: true, evidenceExportTarget: "project" })}
      />,
    );

    expect(html).toContain("Export bundle");
    expect(html).not.toContain("disabled");
  });

  it("opens synthesis when the hardware card advertises that action", async () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    const onOpenCompiler = vi.fn();
    const onOpenSynthesis = vi.fn();
    const state = workbenchState({
      cards: [
        {
          action: "Open synthesis",
          detail: "Synthesise is next",
          key: "compile",
          status: "ready",
          title: "Hardware path",
          value: "Compiled",
        },
      ],
    });

    await act(async () => {
      root.render(
        <OperatorWorkbenchPanel
          onExportEvidence={() => undefined}
          onOpenAdmin={() => undefined}
          onOpenCompiler={onOpenCompiler}
          onOpenProjects={() => undefined}
          onOpenSynthesis={onOpenSynthesis}
          onRunSimulation={() => undefined}
          state={state}
        />,
      );
    });
    container.querySelector("button")?.click();

    expect(onOpenSynthesis).toHaveBeenCalledOnce();
    expect(onOpenCompiler).not.toHaveBeenCalled();
    await act(async () => root.unmount());
  });
});
