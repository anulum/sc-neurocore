// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — bounded hosted federation metadata request

/** A deliberately authored refusal safe to display in the hosted view. */
export class HostCatalogueRefusal extends Error {}

/** The public metadata returned by the Hub's federation:config handler. */
export interface HostedStudio {
  studio: string;
  version: string;
  era: string;
  evidenceTypes: string[];
}

/**
 * Narrow structured-clone data before using it as host metadata.
 *
 * @param value - Untrusted message data.
 * @returns Whether this is a non-array object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validate Hub metadata without interpreting it as backend readiness.
 *
 * @param value - One untrusted catalogue entry.
 * @returns Whether its metadata fields match the host contract.
 */
function isHostedStudio(value: unknown): value is HostedStudio {
  return isRecord(value)
    && typeof value.studio === "string" && value.studio.length > 0
    && typeof value.version === "string" && value.version.length > 0
    && typeof value.era === "string" && value.era.length > 0
    && Array.isArray(value.evidenceTypes)
    && value.evidenceTypes.every((entry: unknown) => typeof entry === "string");
}

/**
 * Request catalogue metadata from the immediate parent, with a five-second limit.
 *
 * Only a matching response from that parent is accepted. Refusal and malformed
 * data use fixed messages, never host error text. Abort, completion and timeout
 * remove both listeners and the timer. This function performs no HTTP, socket,
 * credential or browser-storage operation.
 *
 * @param signal - Cancel the request when the hosted view unmounts.
 * @returns Validated host metadata; rejects on refusal, invalid data or timeout.
 */
export function requestHostFederation(signal: AbortSignal): Promise<HostedStudio[]> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new HostCatalogueRefusal("Host catalogue request cancelled."));
      return;
    }
    const id = crypto.randomUUID();
    const parent = window.parent;
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new HostCatalogueRefusal("Host catalogue request timed out."));
    }, 5_000);
    /** Release every request-owned callback before settling the promise. */
    function cleanup(): void {
      window.clearTimeout(timer);
      window.removeEventListener("message", receive);
      signal.removeEventListener("abort", cancel);
    }
    /** Prevent an unmounted view from accepting a late host answer. */
    function cancel(): void {
      cleanup();
      reject(new HostCatalogueRefusal("Host catalogue request cancelled."));
    }
    /**
     * Accept only a correlated answer from the actual immediate parent.
     *
     * @param event - A window message, including its browser-assigned source.
     */
    function receive(event: MessageEvent<unknown>): void {
      const data: unknown = event.data;
      if (event.source !== parent || !isRecord(data)
        || data.type !== "scpn:bridge:response" || data.id !== id) return;
      cleanup();
      if (data.ok !== true) {
        reject(new HostCatalogueRefusal("Host catalogue request was refused."));
      } else if (!Array.isArray(data.result) || !data.result.every(isHostedStudio)) {
        reject(new HostCatalogueRefusal("Host catalogue response is invalid."));
      } else {
        resolve(data.result);
      }
    }
    window.addEventListener("message", receive);
    signal.addEventListener("abort", cancel, { once: true });
    try {
      parent.postMessage({ type: "scpn:bridge:request", id,
        capability: "federation:config", payload: null }, "*");
    } catch {
      cleanup();
      reject(new HostCatalogueRefusal("Host catalogue request could not be sent."));
    }
  });
}
