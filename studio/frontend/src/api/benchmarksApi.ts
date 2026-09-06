// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: benchmarks endpoints.
import { post, get } from "./http";
import type {
  BenchmarkSubmission,
  DatabankLeaderboard,
} from "./types";

/**
 * Run a benchmark on the server and get back a submission it will accept.
 *
 * The result is not yet in the databank: contributing it is a separate,
 * deliberate step, so a measurement can be inspected before it is published.
 *
 * @param body - Channel count, tap count and repeat count to measure with.
 * @returns The measurement, shaped as a submission.
 */
export const runBenchmark = (body: { n_channels: number; n_taps: number; repeats: number }) =>
  post<BenchmarkSubmission>("/benchmarks/run", body);

/**
 * Publish a measurement to the shared databank under a handle.
 *
 * @param submission - A measurement returned by {@link runBenchmark}.
 * @param handle - The name to attribute it to.
 * @returns Whether the server stored it.
 */
export const contributeBenchmark = (submission: BenchmarkSubmission, handle: string) =>
  post<{ stored: boolean }>("/benchmarks/contribute", { submission, handle });

/**
 * Read the contributed benchmark databank.
 *
 * @returns The leaderboard as the server holds it.
 */
export const fetchDatabank = () => get<DatabankLeaderboard>("/benchmarks/databank");
