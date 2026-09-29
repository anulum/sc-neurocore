// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — a refused link stays said when a run starts

import { afterEach, describe, expect, it } from "vitest";

import type { ModelSummary } from "./api/client";
import { useStudioStore } from "./stores/studio";
import { studioAnalysisStartState } from "./studioAnalysisState";

describe("applyShareLink refusal", () => {
  const initial = useStudioStore.getState();
  afterEach(() => {
    useStudioStore.setState(initial, true);
    window.location.hash = "";
  });

  it("outlives the start of the startup simulation", async () => {
    // The live suite caught this: the link was refused, then the automatic
    // simulation of the default model started a second later, cleared
    // `error`, and the message was gone.
    window.location.hash = "#model=NoSuchNeuron";
    useStudioStore.setState({ models: [{ name: "LIFNeuron" } as ModelSummary] });

    await useStudioStore.getState().applyShareLink();
    useStudioStore.setState(studioAnalysisStartState());

    const state = useStudioStore.getState();
    expect(state.linkNotice).toBe(
      'This link opens "NoSuchNeuron", which this catalogue does not hold. '
      + "It may have been renamed since the link was made.",
    );
    expect(state.error).toBeNull();
  });
});
