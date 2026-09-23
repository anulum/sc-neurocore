// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model selection response ordering

import { createStore } from "zustand/vanilla";
import { describe, expect, it, vi } from "vitest";

import type { ModelDetail } from "../api/client";
import { fetchModelDetail, fetchReplayPack } from "../api/client";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, fetchModelDetail: vi.fn(), fetchReplayPack: vi.fn() };
});

/** Build the contract fields consumed by model selection in this test. */
function detail(name: string, parameter: string): ModelDetail {
  return {
    name,
    dt: 0.05,
    integration_method: "euler",
    compile_configuration: null,
    params: [{ name: parameter, default: 1 }],
    state_vars: [],
  } as unknown as ModelDetail;
}

describe("Studio model selection", () => {
  it("keeps the last selected model when an older detail response arrives later", async () => {
    const replies = new Map<string, (value: ModelDetail) => void>();
    vi.mocked(fetchModelDetail).mockImplementation(
      (name) => new Promise((resolve) => { replies.set(name, resolve); }),
    );
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData,
      ...createStudioStoreActions(set, get),
    }));
    store.setState({ runSimulation: vi.fn(() => Promise.resolve()) });

    const first = store.getState().selectModel("ATypeKNeuron");
    await store.getState().exportReplayPack();
    expect(fetchReplayPack).not.toHaveBeenCalled();
    expect(store.getState().error).toBe("Selected model is still loading");
    const second = store.getState().selectModel("HodgkinHuxleyNeuron");
    replies.get("HodgkinHuxleyNeuron")?.(detail("HodgkinHuxleyNeuron", "g_na"));
    await second;
    replies.get("ATypeKNeuron")?.(detail("ATypeKNeuron", "g_a"));
    await first;

    expect(store.getState().selectedModelName).toBe("HodgkinHuxleyNeuron");
    expect(store.getState().modelDetail?.name).toBe("HodgkinHuxleyNeuron");
    expect(store.getState().modelParams).toEqual({ g_na: 1 });
    expect(store.getState().runSimulation).toHaveBeenCalledTimes(1);
  });
});
