// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio cross-runtime evidence seal tests

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  EVIDENCE_SEAL_ALGORITHM,
  EVIDENCE_SEAL_SCHEMA_VERSION,
  EvidenceSealError,
  canonicalSealText,
  sealSha256,
} from "./evidenceSeal";

interface SealVector {
  canonical: string;
  value: unknown;
}

interface SealVectorDocument {
  schema_version: string;
  vectors: SealVector[];
}

const VECTORS: SealVectorDocument = JSON.parse(
  readFileSync(fileURLToPath(new URL("./evidenceSealVectors.json", import.meta.url)), "utf8"),
) as SealVectorDocument;

describe("the vectors the Python half wrote", () => {
  it("carries the contract this module implements", () => {
    expect(VECTORS.schema_version).toBe(EVIDENCE_SEAL_SCHEMA_VERSION);
    expect(VECTORS.vectors.length).toBeGreaterThan(0);
    expect(EVIDENCE_SEAL_ALGORITHM).toBe("sha256");
  });

  it("canonicalises every vector to the same text as the server", () => {
    for (const vector of VECTORS.vectors) {
      expect(canonicalSealText(vector.value), JSON.stringify(vector.value)).toBe(
        vector.canonical,
      );
    }
  });
});

describe("the canonical form", () => {
  it("writes an integral number as an integer", () => {
    // JSON.stringify already renders 1.0 as "1"; the seal must agree with the
    // server, which renders it as "1" too rather than as Python's "1.0".
    expect(canonicalSealText(1)).toBe("1");
    expect(canonicalSealText(-70)).toBe("-70");
    expect(canonicalSealText(-0)).toBe("0");
  });

  it("writes a large integral number in full rather than in exponent form", () => {
    expect(canonicalSealText(1e21)).toBe(`1${"0".repeat(21)}`);
  });

  it("writes a fraction in one scientific form", () => {
    expect(canonicalSealText(0.1)).toBe("1e-1");
    expect(canonicalSealText(1e-7)).toBe("1e-7");
    expect(canonicalSealText(0.30000000000000004)).toBe("3.0000000000000004e-1");
  });

  it("orders object keys by code point, not by UTF-16 unit", () => {
    const text = canonicalSealText({ z: 1, a: 2, "\u{1f600}": 3, "￿": 4 });

    expect(text.indexOf('"a"')).toBeLessThan(text.indexOf('"z"'));
    expect(text.indexOf('"￿"')).toBeLessThan(text.indexOf('"\u{1f600}"'));
  });

  it("distinguishes keys that share a prefix", () => {
    expect(canonicalSealText({ ab: 1, a: 2 })).toBe('{"a":2,"ab":1}');
  });

  it("writes no insignificant whitespace", () => {
    expect(canonicalSealText({ a: [1, 2], b: { c: null } })).toBe(
      '{"a":[1,2],"b":{"c":null}}',
    );
  });

  it("escapes only what has to be escaped", () => {
    expect(canonicalSealText('a"b\\c\n\t')).toBe('"a\\"b\\\\c\\n\\t"');
    expect(canonicalSealText("")).toBe('"\\u0001"');
    expect(canonicalSealText("ä☃")).toBe('"ä☃"');
  });

  it("writes booleans and null", () => {
    expect(canonicalSealText([true, false, null])).toBe("[true,false,null]");
  });
});

describe("what the seal refuses", () => {
  it("refuses a non-finite number rather than altering it", () => {
    for (const value of [Number.NaN, Infinity, -Infinity]) {
      expect(() => canonicalSealText(value)).toThrow(EvidenceSealError);
    }
  });

  it("refuses an unpaired surrogate", () => {
    expect(() => canonicalSealText("\ud800")).toThrow(/unpaired surrogate/);
  });

  it("refuses a value JSON has no place for", () => {
    for (const value of [undefined, () => 1, Symbol("x"), 1n]) {
      expect(() => canonicalSealText(value)).toThrow(EvidenceSealError);
    }
  });
});

describe("the digest", () => {
  it("is the SHA-256 of the canonical text", async () => {
    const digest = await sealSha256({ dt: 1, spikes: [0.5] });

    expect(digest).toMatch(/^[0-9a-f]{64}$/);
    expect(await sealSha256({ spikes: [0.5], dt: 1.0 })).toBe(digest);
  });

  it("changes when a value changes", async () => {
    expect(await sealSha256({ spike_count: 3 })).not.toBe(
      await sealSha256({ spike_count: 4 }),
    );
  });

  it("survives the round trip a payload makes through this runtime", async () => {
    const fromTheServer = { dt: 1.0, spike_count: 3.0, v: [0.1, -70.0, 1e-7] };
    const afterTheRoundTrip = JSON.parse(JSON.stringify(fromTheServer)) as unknown;

    expect(await sealSha256(afterTheRoundTrip)).toBe(await sealSha256(fromTheServer));
  });
});
