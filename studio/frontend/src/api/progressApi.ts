// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: progress endpoints.
import { progressWebSocketProtocols } from "./http";
import type {
  ProgressMessage,
} from "./types";

/**
 * Open a progress socket for one long-running operation.
 *
 * The token cannot travel as a header on a WebSocket, so it goes as a
 * subprotocol; see `progressWebSocketProtocols`. A frame that does not parse is
 * dropped rather than thrown, because a malformed progress update must not
 * take down the operation it is reporting on, and a failed connection is
 * reported to the same callback as an ordinary error message so callers have
 * one place to handle it.
 *
 * @param op - The operation to run.
 * @param config - The operation's own configuration, sent on open.
 * @param onMessage - Called for each progress update and for a failure.
 * @returns The open socket, for the caller to close.
 */
export function connectProgress(
  op: string,
  config: Record<string, unknown>,
  onMessage: (msg: ProgressMessage) => void,
): WebSocket {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(
    `${proto}//${window.location.host}/ws/progress`,
    progressWebSocketProtocols(),
  );
  ws.onopen = () => {
    ws.send(JSON.stringify({ op, config }));
  };
  ws.onmessage = (e: MessageEvent<string>) => {
    try {
      onMessage(JSON.parse(e.data) as ProgressMessage);
    } catch { /* a frame that does not parse is dropped, not thrown */ }
  };
  ws.onerror = () => {
    onMessage({ type: "error", msg: "WebSocket connection failed" });
  };
  return ws;
}
