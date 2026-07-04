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
