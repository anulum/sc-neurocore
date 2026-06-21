// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio characterisation store state helper tests
import { describe, expect, it } from "vitest";

import type { CharacterizeResponse } from "./api/client";
import {
  characterizeCompleteState,
  characterizeFailureState,
  characterizeProgressMessageState,
  characterizeProgressState,
  characterizeRequestConfig,
  characterizeRunStartState,
} from "./characterizeStoreState";

function characterizeResponse(
  overrides: Partial<CharacterizeResponse> = {},
): CharacterizeResponse {
  return {
    fi_curve: overrides.fi_curve ?? { currents: [0, 10], rates: [0, 25] },
    max_rate: overrides.max_rate ?? 25,
    pattern: overrides.pattern ?? {
      description: "Regular spiking",
      pattern: "regular",
    },
    spike_count: overrides.spike_count ?? 5,
    state_ranges: overrides.state_ranges ?? {
      v: { max: -50, mean: -60, min: -70 },
    },
    stats: overrides.stats ?? {
      isi_cv: null,
      isi_histogram: null,
      isi_mean_ms: null,
      rate_hz: 25,
    },
    threshold_current: overrides.threshold_current ?? null,
    top_sensitivities: overrides.top_sensitivities ?? [
      { param: "tau_m", rate_change: 3.5 },
    ],
  };
}

describe("characterisation store state helpers", () => {
  it("builds request config and start state", () => {
    expect(characterizeRequestConfig({
      current: 9,
      dt: 0.05,
      duration: 120,
      modelParams: { tau_m: 10 },
      selectedModelName: "lif",
    })).toEqual({
      current: 9,
      dt: 0.05,
      duration: 120,
      name: "lif",
      params: { tau_m: 10 },
    });
    expect(characterizeRunStartState()).toEqual({
      activeTab: "characterize",
      error: null,
      isSimulating: true,
      progressMsg: "Starting characterisation...",
      progressPct: 0,
    });
  });

  it("builds bounded progress and completion patches", () => {
    const result = characterizeResponse();

    expect(characterizeProgressState({
      msg: "Scanning current",
      pct: 48.6,
      type: "progress",
    })).toEqual({
      progressMsg: "Scanning current",
      progressPct: 49,
    });
    expect(characterizeProgressState({ pct: 150, type: "progress" })).toEqual({
      progressMsg: "",
      progressPct: 100,
    });
    expect(characterizeCompleteState(result)).toEqual({
      charResult: result,
      isSimulating: false,
      progressMsg: "",
      progressPct: 100,
    });
  });

  it("normalises failures from errors, strings, and empty payloads", () => {
    expect(characterizeFailureState(new Error("socket closed"))).toEqual({
      error: "socket closed",
      isSimulating: false,
      progressMsg: "",
      progressPct: 0,
    });
    expect(characterizeFailureState("backend unavailable")).toEqual({
      error: "backend unavailable",
      isSimulating: false,
      progressMsg: "",
      progressPct: 0,
    });
    expect(characterizeFailureState("", "Fallback")).toEqual({
      error: "Fallback",
      isSimulating: false,
      progressMsg: "",
      progressPct: 0,
    });
  });

  it("maps websocket progress, complete, error, and heartbeat messages", () => {
    const result = characterizeResponse({
      stats: {
        isi_cv: 0.2,
        isi_histogram: { counts: [1, 2], edges: [0, 5, 10] },
        isi_mean_ms: 5,
        rate_hz: 50,
      },
      threshold_current: 1.5,
    });

    expect(characterizeProgressMessageState({
      msg: "Complete",
      pct: 100,
      type: "progress",
    })).toEqual({
      progressMsg: "Complete",
      progressPct: 100,
    });
    expect(characterizeProgressMessageState({
      result,
      type: "complete",
    })).toEqual(characterizeCompleteState(result));
    expect(characterizeProgressMessageState({
      msg: "worker failed",
      type: "error",
    })).toEqual(characterizeFailureState("worker failed"));
    expect(characterizeProgressMessageState({ type: "heartbeat" })).toBeNull();
  });

  it("fails closed on malformed complete payloads", () => {
    expect(characterizeProgressMessageState({
      result: { max_rate: Number.NaN },
      type: "complete",
    })).toEqual(characterizeFailureState("Malformed characterisation result"));
    expect(characterizeProgressMessageState({
      result: {
        ...characterizeResponse(),
        fi_curve: { currents: [0], rates: ["bad"] },
      },
      type: "complete",
    })).toEqual(characterizeFailureState("Malformed characterisation result"));
  });
});
