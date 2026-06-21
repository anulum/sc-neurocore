// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio startup hash browser runtime tests

import { describe, expect, it } from "vitest";

import { readStudioStartupHashState } from "./studioStartupRuntime";

describe("Studio startup hash browser runtime", () => {
  it("returns null when no browser runtime is available", () => {
    expect(readStudioStartupHashState(null)).toBeNull();
  });

  it("returns null for empty or malformed startup hashes", () => {
    expect(readStudioStartupHashState({ hash: "" })).toBeNull();
    expect(readStudioStartupHashState({ hash: "#bad" }, () => "{")).toBeNull();
  });

  it("decodes a valid startup hash from the supplied runtime", () => {
    const payload = { c: 15, d: 150, m: "model", mn: "lif", pr: "burst" };

    expect(readStudioStartupHashState({ hash: "#payload" }, () => JSON.stringify(payload)))
      .toEqual({
        current: 15,
        duration: 150,
        protocol: "burst",
        selectedModelName: "lif",
      });
  });
});
