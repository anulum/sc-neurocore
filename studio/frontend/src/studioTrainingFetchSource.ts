// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — authenticated Training Monitor SSE transport

/** Read Studio SSE with the same bearer authority as other HTTP requests. */

import { authHeaders } from "./api/http";
import type { StudioTrainingStreamEventSource } from "./studioTrainingStream";

const MAX_FRAME_CHARACTERS = 1_048_576;

/**
 * Open one authenticated SSE stream and deliver complete data frames.
 *
 * Native EventSource cannot send the Studio bearer header. This adapter uses
 * the existing HTTP token store, rejects non-SSE responses, and never puts the
 * token in a URL. Closing the source aborts its pending read.
 *
 * @param url - Same-origin path for one server-selected training job.
 * @param request - Fetch implementation; the production default is browser fetch.
 * @returns EventSource-shaped handle consumed by the existing stream decoder.
 */
export function createFetchStudioTrainingSource(
  url: string,
  request: typeof fetch = fetch,
): StudioTrainingStreamEventSource {
  if (!/^\/api\/training\/stream\/[^/?#]+$/.test(url)) {
    throw new Error("Training event stream path is invalid");
  }
  const abort = new AbortController();
  let closed = false;
  const isClosed = (): boolean => closed;
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  const source: StudioTrainingStreamEventSource = {
    onerror: null,
    onmessage: null,
    close: () => {
      if (isClosed()) return;
      closed = true;
      abort.abort();
      if (reader !== null) void reader.cancel().catch(() => undefined);
    },
  };

  const read = async (): Promise<void> => {
    try {
      const response = await request(url, {
        cache: "no-store",
        headers: { Accept: "text/event-stream", ...authHeaders() },
        redirect: "error",
        signal: abort.signal,
      });
      if (isClosed()) return;
      if (!response.ok
        || response.body === null
        || !response.headers.get("content-type")?.startsWith("text/event-stream")) {
        throw new Error("Training event stream unavailable");
      }
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (!isClosed()) {
        const chunk = await reader.read();
        if (isClosed()) return;
        if (chunk.done) {
          source.onerror?.(new Event("error"));
          return;
        }
        pending = (pending + decoder.decode(chunk.value, { stream: true }))
          .replace(/\r\n/g, "\n");
        let boundary = pending.indexOf("\n\n");
        while (boundary >= 0) {
          const frame = pending.slice(0, boundary);
          pending = pending.slice(boundary + 2);
          if (frame.length > MAX_FRAME_CHARACTERS) {
            throw new Error("Training event frame exceeds limit");
          }
          const data = frame.split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart());
          if (data.length > 0 && !isClosed()) {
            source.onmessage?.(new MessageEvent("message", { data: data.join("\n") }));
          }
          if (isClosed()) return;
          boundary = pending.indexOf("\n\n");
        }
        if (pending.length > MAX_FRAME_CHARACTERS) {
          throw new Error("Training event frame exceeds limit");
        }
      }
    } catch {
      if (!isClosed()) source.onerror?.(new Event("error"));
    } finally {
      reader?.releaseLock();
    }
  };
  void read();
  return source;
}
