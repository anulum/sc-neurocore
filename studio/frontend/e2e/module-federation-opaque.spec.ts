// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — production remote in a real opaque iframe and parent-message transport

import { expect, test, type Response as BrowserResponse } from "@playwright/test";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { extname, join, resolve, sep } from "node:path";
import { promisify } from "node:util";
import envelope from "../../../docs/_generated/studio_manifest.json" with { type: "json" };

const schema = envelope.schema_a;
const catalogue = [{ studio: schema.studio, version: schema.studio_version,
  era: schema.contract_era, evidenceTypes: schema.evidence_types }];
let output = "";
let origin = "";
let server: Server | undefined;

/**
 * Serve only the built fixture bytes, including modules requested by an opaque origin.
 *
 * @param request - A local browser request.
 * @param response - Its static response; no backend or identity routes are implemented.
 */
async function serve(request: IncomingMessage, response: ServerResponse): Promise<void> {
  const path = decodeURIComponent(new URL(request.url ?? "/", "http://fixture.local").pathname);
  const file = resolve(output, "." + path);
  if (!file.startsWith(output + sep)) {
    response.writeHead(404).end();
    return;
  }
  try {
    const bytes = await readFile(file);
    const contentType = extname(file) === ".js" ? "text/javascript"
      : extname(file) === ".css" ? "text/css" : "text/html";
    response.writeHead(200, { "Content-Type": contentType, "Access-Control-Allow-Origin": "*" });
    response.end(bytes);
  } catch {
    response.writeHead(404).end();
  }
}

test.beforeAll(async () => {
  const testInfo = test.info();
  await mkdir(testInfo.outputDir, { recursive: true });
  output = await mkdtemp(join(testInfo.outputDir, "opaque-fixture-"));
  const built = await promisify(execFile)(process.execPath, [
    "node_modules/vite/bin/vite.js", "build", "--config", "e2e/federation-host/vite.config.ts",
    "--outDir", output,
  ], { env: process.env });
  const buildLog = testInfo.outputPath("opaque-fixture-build.log");
  await writeFile(buildLog, built.stdout + built.stderr);
  await testInfo.attach("opaque-fixture-build", { path: buildLog, contentType: "text/plain" });
  server = createServer((request, response) => {
    void serve(request, response).catch(() => { response.writeHead(500).end(); });
  });
  const running = server;
  await new Promise<void>((ready, reject) => {
    running.once("error", reject);
    running.listen(0, "127.0.0.1", () => { running.off("error", reject); ready(); });
  });
  const address = running.address();
  if (address === null || typeof address === "string") throw new Error("Missing fixture address");
  origin = `http://127.0.0.1:${String(address.port)}`;
});

test.afterAll(async () => {
  const running = server;
  if (running !== undefined) {
    running.closeAllConnections();
    await new Promise<void>((closed, reject) => {
      running.close((error) => { if (error) reject(error); else closed(); });
    });
  }
});

for (const initialReply of ["metadata", "refuse"] as const) {
  test(`opaque production entry handles ${initialReply} without a backend`, async ({ page }, testInfo) => {
    const requests: string[] = [];
    const received: BrowserResponse[] = [];
    const sockets: string[] = [];
    const errors: string[] = [];
    page.on("request", (request) => { requests.push(request.url()); });
    page.on("response", (response) => {
      if (new URL(response.url()).origin === "http://127.0.0.1:5184") received.push(response);
    });
    page.on("websocket", (socket) => { sockets.push(socket.url()); });
    page.on("pageerror", (error) => { errors.push(error.message); });
    const query = new URLSearchParams({ catalogue: JSON.stringify(catalogue), reply: initialReply });
    await page.goto(`${origin}/opaque.html?${query.toString()}`);
    const frame = page.frameLocator('iframe[title="Opaque producer panel"]');
    await expect(frame.locator("#root")).toHaveAttribute("data-federation-status", "loaded");
    await expect(frame.getByText("Hosted view · no backend connected", { exact: true })).toBeVisible();
    if (initialReply === "refuse") {
      await expect(frame.getByRole("alert")).toHaveText("Host catalogue request was refused.");
      await expect(frame.locator("body")).not.toContainText("Private fixture diagnostic");
      await page.getByRole("button", { name: "Allow catalogue replies", exact: true }).click();
      await frame.getByRole("button", { name: "Retry host catalogue", exact: true }).click();
    }
    await expect(frame.locator("dl")).toContainText(schema.studio_version);
    await expect(frame.locator("dl")).toContainText(schema.contract_era);
    for (const evidence of schema.evidence_types) await expect(frame.locator("dl")).toContainText(evidence);
    await expect(frame.locator("button, input, select, textarea, canvas")).toHaveCount(0);
    await expect(frame.getByLabel("Save project", { exact: true })).toHaveCount(0);
    await frame.getByText("Workbench views — not available in hosted view", { exact: true }).click();
    await expect(frame.locator("li")).toHaveCount(25);
    for (const item of await frame.locator("li").all()) await expect(item).toContainText("not available in hosted view");
    const child = page.frames().find((entry) => entry.parentFrame() !== null);
    if (child === undefined) throw new Error("Opaque producer frame missing");
    const context = await child.evaluate(() => {
      let parentDomBlocked = false;
      try { window.parent.document.querySelector("body"); }
      catch (error: unknown) { parentDomBlocked = error instanceof DOMException && error.name === "SecurityError"; }
      return { origin: globalThis.origin, secure: isSecureContext, parentDomBlocked };
    });
    expect(context.origin).toBe("null");
    expect(context.parentDomBlocked).toBe(true);
    await expect(page.locator("iframe")).toHaveAttribute("sandbox", "allow-scripts");
    const messages = await page.locator("#bridge-requests li").evaluateAll((items) => items.map((item) => ({
      id: item.getAttribute("data-id"), capability: item.getAttribute("data-capability"),
      origin: item.getAttribute("data-origin"),
    })));
    expect(messages.length).toBeGreaterThanOrEqual(initialReply === "refuse" ? 2 : 1);
    expect(messages.every((message) => message.capability === "federation:config" && message.origin === "null")).toBe(true);
    expect(new Set(messages.map((message) => message.id)).size).toBe(messages.length);
    expect(requests.some((url) => new URL(url).pathname.endsWith("/remoteEntry.js"))).toBe(true);
    expect(requests.some((url) => /StandaloneStudioPanel|monaco|\/vs\//u.test(new URL(url).pathname))).toBe(false);
    expect(requests.filter((url) => /^\/(?:api|ws)(?:\/|$)/u.test(new URL(url).pathname))).toEqual([]);
    expect(requests.every((url) => [origin, "http://127.0.0.1:5184"].includes(new URL(url).origin))).toBe(true);
    expect(sockets).toEqual([]);
    expect(errors).toEqual([]);
    const assets = await Promise.all(received.map(async (response) => {
      const path = new URL(response.url()).pathname;
      expect(path).toMatch(/^\/studios\/sc-neurocore\//u);
      expect(response.ok()).toBe(true);
      const bytes = await response.body();
      const file = resolve("dist", path.slice("/studios/sc-neurocore/".length));
      expect(file.startsWith(resolve("dist") + sep)).toBe(true);
      const actual = createHash("sha256").update(bytes).digest("hex");
      expect(actual).toBe(createHash("sha256").update(await readFile(file)).digest("hex"));
      return { path, status: response.status(), bytes: bytes.length, sha256: actual };
    }));
    expect(assets.some((asset) => asset.path.endsWith("/remoteEntry.js"))).toBe(true);
    const metadataBytes = await readFile(new URL("../../../docs/_generated/studio_manifest.json", import.meta.url));
    const metadata: unknown = JSON.parse(metadataBytes.toString("utf8"));
    expect(metadata).toEqual(envelope);
    const metadataSha256 = createHash("sha256").update(metadataBytes).digest("hex");
    const receiptPath = testInfo.outputPath("opaque-producer-receipt.json");
    await writeFile(receiptPath, JSON.stringify({
      boundary: "Producer transport fixture, not SCPN-STUDIO admission/bridge/rollback/identity qualification",
      metadataSource: "docs/_generated/studio_manifest.json", metadataSha256, context, messages, requests, assets, sockets, errors,
    }, null, 2) + "\n");
    await testInfo.attach("opaque-producer-receipt", { contentType: "application/json", path: receiptPath });
  });
}
