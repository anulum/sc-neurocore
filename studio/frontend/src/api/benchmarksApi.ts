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

export const runBenchmark = (body: { n_channels: number; n_taps: number; repeats: number }) =>
  post<BenchmarkSubmission>("/benchmarks/run", body);

export const contributeBenchmark = (submission: BenchmarkSubmission, handle: string) =>
  post<{ stored: boolean }>("/benchmarks/contribute", { submission, handle });

export const fetchDatabank = () => get<DatabankLeaderboard>("/benchmarks/databank");
