// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio RTL preview model-mode tests

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@monaco-editor/react", () => ({ default: () => <div>editor</div> }));
vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({ sourceMode: "model", verilogSrc: "" }),
}));

describe("VerilogPreview", () => {
  it("invites compilation of the selected catalogue model", async () => {
    const { default: VerilogPreview } = await import("./VerilogPreview");
    const html = renderToStaticMarkup(<VerilogPreview />);

    expect(html).toContain("selected model");
    expect(html).not.toContain("Switch to ODE mode");
  });
});
