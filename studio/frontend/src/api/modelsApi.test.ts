// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model catalogue API helper tests

import { describe, expect, it } from "vitest";

import { catalogueQueryString } from "./modelsApi";

describe("catalogue query string", () => {
  it("sends nothing when nothing filters", () => {
    expect(catalogueQueryString({})).toBe("");
    expect(catalogueQueryString({
      text: "",
      family: "",
      min_verified_science: 0,
      min_verified_silicon: 0,
      verified_perfect_only: false,
    })).toBe("");
  });

  it("encodes each filter the server reads, escaping text", () => {
    expect(catalogueQueryString({
      text: "leaky & fire",
      family: "Integrate-and-Fire",
      min_verified_science: 3,
      verified_perfect_only: true,
    })).toBe(
      "?text=leaky+%26+fire&family=Integrate-and-Fire&min_verified_science=3&verified_perfect_only=true",
    );
  });
});
