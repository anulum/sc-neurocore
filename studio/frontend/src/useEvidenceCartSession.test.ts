// @vitest-environment happy-dom
// SPDX-License-Identifier: AGPL-3.0-or-later
// Configure React 19 act environment for createRoot mounts.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evidence cart session await parity (W12-G)

/**
 * Real React DOM mount of useEvidenceCartSession.
 *
 * W12-G: runAnalysisIntoCart must await store.runFICurve (now async via
 * runStudioAnalysisJob) and only enqueue when fiResult identity changes after
 * the awaited completion — never race on pre-await state and never false-enqueue
 * on failure or stale leave-behind.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, createElement, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";

import type { FICurveResponse, SimulateResponse } from "./api/client";
import { analysisResultIdentity } from "./evidenceCartIdentity";
import {
  useEvidenceCartSession,
  type EvidenceCartSession,
} from "./useEvidenceCartSession";

const DIGEST_A = "a".repeat(64);
const DIGEST_B = "b".repeat(64);
const DIGEST_C = "c".repeat(64);

type StoreSlice = {
  fiResult: FICurveResponse | null;
  result: SimulateResponse | null;
  selectedModelName: string;
  sourceMode: "model" | "ode";
  runFICurve: () => Promise<void>;
  runSimulation: () => Promise<void>;
};

const mockStore = vi.hoisted(() => ({
  state: null as unknown as StoreSlice,
}));

vi.mock("./stores/studio", () => ({
  useStudioStore: {
    getState: () => mockStore.state,
  },
}));

function fiCurve(resultSha256: string, rates: number[] = [0, 5]): FICurveResponse {
  return {
    analysis_metadata: {
      analysis_type: "fi_curve",
      evidence_classification: "analysis",
      input_sha256: DIGEST_A,
      output_keys: ["currents", "rates"],
      result_sha256: resultSha256,
      schema_version: "studio.analysis-result.v1",
      source: "model",
      status: "completed",
    },
    currents: [0, 1],
    rates,
  };
}

function resetStore(partial: Partial<StoreSlice> = {}): void {
  mockStore.state = {
    fiResult: null,
    result: null,
    selectedModelName: "LIFNeuron",
    sourceMode: "model",
    runFICurve: async () => {
      /* default no-op */
    },
    runSimulation: async () => {
      /* default no-op */
    },
    ...partial,
  };
}

describe("useEvidenceCartSession runAnalysisIntoCart await parity (W12-G)", () => {
  let root: Root | null = null;
  let host: HTMLDivElement | null = null;
  let latest: EvidenceCartSession | null = null;

  beforeEach(() => {
    resetStore();
    latest = null;
  });

  afterEach(() => {
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

  function mountHook(): void {
    host = document.createElement("div");
    document.body.appendChild(host);
    root = createRoot(host);
    function Host() {
      const value = useEvidenceCartSession();
      useEffect(() => {
        latest = value;
      });
      return createElement(
        "div",
        { "data-testid": "evidence-cart-session" },
        String(value.cart.items.length),
      );
    }
    act(() => {
      root?.render(createElement(Host));
    });
  }

  it("enqueues after mocked async runFICurve sets a new fiResult identity", async () => {
    let resolveJob: (() => void) | null = null;
    const jobGate = new Promise<void>((resolve) => {
      resolveJob = resolve;
    });
    let runStarted = false;

    resetStore({
      fiResult: null,
      runFICurve: async () => {
        runStarted = true;
        await jobGate;
        // Mirror W12-D store path: identity lands only after async job complete.
        mockStore.state = {
          ...mockStore.state,
          fiResult: fiCurve(DIGEST_B, [0, 12]),
        };
      },
    });

    mountHook();
    expect(latest).not.toBeNull();
    expect(latest?.cart.items).toHaveLength(0);

    let runPromise: Promise<void> | undefined;
    act(() => {
      runPromise = latest?.runAnalysisIntoCart();
    });
    expect(runStarted).toBe(true);
    // Still empty while the async job is in flight — must not race pre-await.
    expect(latest?.cart.items).toHaveLength(0);

    await act(async () => {
      resolveJob?.();
      await runPromise;
    });

    expect(latest?.cart.items).toHaveLength(1);
    expect(latest?.error).toBeNull();
    const item = latest?.cart.items[0];
    expect(item?.kind).toBe("analysis");
    expect(item?.sourceName).toBe("LIFNeuron");
    const payload = item?.payload as {
      analysis_kind: string;
      result: FICurveResponse;
    };
    expect(payload.analysis_kind).toBe("fi_curve");
    expect(analysisResultIdentity(payload.result)).toBe(DIGEST_B);
    expect(payload.result.rates).toEqual([0, 12]);
    expect(host?.textContent).toBe("1");
  });

  it("does not false-enqueue when runFICurve fails without an identity change", async () => {
    const prior = fiCurve(DIGEST_A, [0, 1]);
    resetStore({
      fiResult: prior,
      runFICurve: async () => {
        // Failed async job: leave-behind prior result, no new digest.
        mockStore.state = {
          ...mockStore.state,
          fiResult: prior,
        };
      },
    });

    mountHook();
    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });

    expect(latest?.cart.items).toHaveLength(0);
    // Failed/unchanged paths stay quiet (no operator error spam).
    expect(latest?.error).toBeNull();
  });

  it("skips when identity is unchanged after await (stale leave-behind race)", async () => {
    const prior = fiCurve(DIGEST_C, [1, 2]);
    let awaitCount = 0;
    resetStore({
      fiResult: prior,
      runFICurve: async () => {
        awaitCount += 1;
        // Explicitly re-assign same object so the session must re-read identity.
        mockStore.state = {
          ...mockStore.state,
          fiResult: prior,
        };
      },
    });

    mountHook();
    const beforeId = analysisResultIdentity(mockStore.state.fiResult);
    expect(beforeId).toBe(DIGEST_C);

    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });

    expect(awaitCount).toBe(1);
    expect(analysisResultIdentity(mockStore.state.fiResult)).toBe(beforeId);
    expect(latest?.cart.items).toHaveLength(0);
    expect(latest?.error).toBeNull();
  });

  it("does not enqueue when runFICurve clears fiResult (no valid after identity)", async () => {
    resetStore({
      fiResult: fiCurve(DIGEST_A),
      runFICurve: async () => {
        mockStore.state = {
          ...mockStore.state,
          fiResult: null,
        };
      },
    });

    mountHook();
    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });

    expect(latest?.cart.items).toHaveLength(0);
    expect(latest?.error).toBeNull();
  });

  it("enqueues successive successful async jobs with distinct digests only", async () => {
    resetStore({
      fiResult: null,
      runFICurve: async () => {
        mockStore.state = {
          ...mockStore.state,
          fiResult: fiCurve(DIGEST_B, [0, 3]),
        };
      },
    });
    mountHook();

    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });
    expect(latest?.cart.items).toHaveLength(1);

    // Second run with same digest must not double-enqueue.
    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });
    expect(latest?.cart.items).toHaveLength(1);

    // Third run with a new digest enqueues once more.
    mockStore.state = {
      ...mockStore.state,
      runFICurve: async () => {
        mockStore.state = {
          ...mockStore.state,
          fiResult: fiCurve(DIGEST_C, [0, 9]),
          selectedModelName: "AdExNeuron",
        };
      },
    };
    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });
    expect(latest?.cart.items).toHaveLength(2);
    expect(latest?.cart.items[1]?.sourceName).toBe("AdExNeuron");
    const secondPayload = latest?.cart.items[1]?.payload as {
      result: FICurveResponse;
    };
    expect(analysisResultIdentity(secondPayload.result)).toBe(DIGEST_C);
  });

  it("propagates runFICurve rejection without mutating the cart", async () => {
    resetStore({
      fiResult: null,
      runFICurve: async () => {
        throw new Error("async_job_transport_failed");
      },
    });
    mountHook();

    let caught: unknown = null;
    await act(async () => {
      try {
        await latest?.runAnalysisIntoCart();
      } catch (error) {
        caught = error;
      }
    });

    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toBe("async_job_transport_failed");
    expect(latest?.cart.items).toHaveLength(0);
  });

  it("clears a residual session error before awaiting the analysis job", async () => {
    resetStore({
      fiResult: null,
      runFICurve: async () => {
        mockStore.state = {
          ...mockStore.state,
          fiResult: fiCurve(DIGEST_B),
        };
      },
    });
    mountHook();

    // Seed a residual error via empty-cart export path (throws).
    await act(async () => {
      try {
        await latest?.exportSessionCart();
      } catch {
        /* expected */
      }
    });
    expect(latest?.error).toBeTruthy();

    await act(async () => {
      await latest?.runAnalysisIntoCart();
    });

    expect(latest?.error).toBeNull();
    expect(latest?.cart.items).toHaveLength(1);
  });
});
