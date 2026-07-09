// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ModelDocMarkdown } from "./ModelDocViewer";

describe("ModelDocViewer", () => {
  it("escapes hostile reference Markdown at the render sink", () => {
    const html = renderToStaticMarkup(
      <ModelDocMarkdown
        markdown={'# Unsafe\n<script>alert("x")</script>\n<img src=x onerror="alert(1)" />'}
      />,
    );

    expect(html).toContain("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;");
    expect(html).toContain("&lt;img src=x onerror=&quot;alert(1)&quot; /&gt;");
    expect(html).not.toContain("<script>");
    expect(html).not.toContain("<img ");
  });
});
