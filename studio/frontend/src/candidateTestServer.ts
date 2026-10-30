// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Real candidate API for frontend lifecycle tests

import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { get } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

/** One actual FastAPI server, with an exclusively owned durable job root. */
export interface CandidateTestServer {
  origin: string;
  root: string;
  stop: () => Promise<void>;
}

/**
 * Reap the exact child started by this helper, including failed startup.
 *
 * @param child - Exact owned Python server process.
 */
async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const closed = once(child, "exit");
  child.kill("SIGTERM");
  const timer = setTimeout(() => { child.kill("SIGKILL"); }, 10_000);
  try { await closed; } finally { clearTimeout(timer); }
}

/**
 * Wait for the actual HTTP health route before any DOM test can start.
 *
 * @param origin - Kernel-selected loopback origin announced by the server.
 */
async function waitReady(origin: string): Promise<void> {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    const ready = await new Promise<boolean>((accept) => {
      const request = get(`${origin}/api/health`, (response) => {
        response.resume(); accept(response.statusCode === 200);
      });
      request.setTimeout(1000, () => { request.destroy(); });
      request.once("error", () => { accept(false); });
    });
    if (ready) return;
    await delay(50);
  }
  throw new Error("Candidate HTTP health did not become ready.");
}

/**
 * Start production HTTP routes instead of replacing fetch or job workers.
 *
 * The Python interpreter comes from the repository's existing environment.
 * The kernel chooses a free loopback port. Every caller must settle its jobs
 * and call stop; an outer test deadline additionally owns the process group.
 *
 * @returns The ready origin and its deterministic cleanup action.
 */
export async function startCandidateTestServer(): Promise<CandidateTestServer> {
  const repo = resolve(import.meta.dirname, "../../..");
  const root = await mkdtemp(join(tmpdir(), "candidate-ui-"));
  const code = [
    "import json, socket, sys, uvicorn",
    "from sc_neurocore.studio.app import create_app",
    "from sc_neurocore.studio.platform.settings import StudioRuntimeSettings",
    "sock=socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(128)",
    "origin='http://127.0.0.1:'+str(sock.getsockname()[1])",
    "print(json.dumps({'port':sock.getsockname()[1]}),flush=True)",
    "uvicorn.run(create_app(StudioRuntimeSettings(job_root_path=sys.argv[1]+'/jobs', audit_log_path=sys.argv[1]+'/audit.jsonl', cors_allowed_origins=(origin,))), fd=sock.fileno(), log_level='warning')",
  ].join("\n");
  const child = spawn(process.env.SC_NEUROCORE_STUDIO_CANDIDATE_PYTHON ?? "python", ["-u", "-c", code, root], {
    cwd: repo, env: { ...process.env, PYTHONPATH: `${repo}/src:${repo}/bridge:${repo}` },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
  try {
    const origin = await new Promise<string>((accept, reject) => {
      const timer = setTimeout(() => { reject(new Error(`Candidate server startup expired: ${stderr}`)); }, 30_000);
      let output = "";
      child.stdout.on("data", (chunk: Buffer) => {
        output += chunk.toString();
        if (!output.includes("\n")) return;
        try {
          const announcement: unknown = JSON.parse(output.split("\n")[0] ?? "");
          if (typeof announcement !== "object" || announcement === null || !("port" in announcement)
            || typeof announcement.port !== "number" || !Number.isInteger(announcement.port)) {
            throw new Error("Invalid candidate server port announcement.");
          }
          clearTimeout(timer);
          accept(`http://127.0.0.1:${String(announcement.port)}`);
        } catch (error) { clearTimeout(timer); reject(error instanceof Error ? error : new Error(String(error))); }
      });
      child.once("error", (error) => { clearTimeout(timer); reject(error); });
      child.once("exit", () => { clearTimeout(timer); reject(new Error(`Candidate server exited: ${stderr}`)); });
    });
    await waitReady(origin);
    return { origin, root, stop: async () => { await stopChild(child); await rm(root, { recursive: true }); } };
  } catch (error) {
    await stopChild(child);
    await rm(root, { recursive: true });
    throw error;
  }
}
