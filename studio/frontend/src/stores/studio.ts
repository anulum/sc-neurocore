// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio Zustand store composition root (public entry for App/hooks).

import { create } from "zustand";
import { setStudioAuthToken } from "../api/client";
import { syncStoredStudioAuthToken } from "../studioAuthSession";
import { readStudioStartupHashState } from "../studioStartupRuntime";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

export type { SourceMode, ViewTab, StudioState } from "./studioTypes";
export type { EvidenceBundleSurface } from "../evidenceBundles";

syncStoredStudioAuthToken(setStudioAuthToken);

export const useStudioStore = create<StudioState>((set, get) => {
  const actions = createStudioStoreActions(set, get);
  const state: StudioState = {
    ...studioInitialData,
    ...actions,
  };
  return state;
});


const startupHashState = readStudioStartupHashState();
if (startupHashState !== null) {
  // Fire-and-forget on purpose: this runs while the module is being evaluated,
  // before anything can await it, and a share link that names a model the
  // catalogue no longer has should leave the Studio open on its defaults
  // rather than fail to start.
  void useStudioStore.getState().selectModel(startupHashState.selectedModelName);
  useStudioStore.setState({
    current: startupHashState.current,
    duration: startupHashState.duration,
    protocol: startupHashState.protocol,
  });
}
