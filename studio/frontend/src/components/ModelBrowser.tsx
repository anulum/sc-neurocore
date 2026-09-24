// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useStudioStore } from "../stores/studio";
import {
    fetchModelFacets,
    queryModels,
    type CatalogueQueryParams,
    type CatalogueQueryResult,
    type ModelBehavior,
    type ModelFacets,
    type ModelScanMetadata,
} from "../api/client";
import { useModelScanJob } from "../useModelScanJob";
import EvidenceSummaryStrip, {
    type EvidenceSummaryItem,
} from "./EvidenceSummaryStrip";
import EvidenceTierBadge, { DualAxisBadge } from "./EvidenceTierBadge";

/**
 * Maturity dot colours.
 *
 * Rendered as a filled dot and, for the badge row, as text on --bg-tertiary or
 * as a ground under --bg-primary text. Every value clears WCAG 2.2 AA in both
 * roles; paletteContrast.test.ts fails if a new one does not.
 */
export const MATURITY_COLORS: Record<string, string> = {
    validated: "#81c784",
    experimental: "#ffb74d",
    reference: "#4fc3f7",
};

/**
 * Live-scan firing-pattern colours.
 *
 * Same two roles as MATURITY_COLORS. `silent` is the muted text token itself,
 * referenced rather than copied so the two cannot drift apart again.
 */
export const PATTERN_COLORS: Record<string, string> = {
    tonic: "#81c784",
    bursting: "#ffb74d",
    adapting: "#4fc3f7",
    irregular: "#ce93d8",
    chaotic: "#ff5252",
    silent: "var(--text-muted)",
    single_spike: "#90a4ae",
    error: "#8c8c8c",
};

/**
 * Measured behaviour-tag colours (descriptor facet, distinct from the live scan).
 *
 * Same two roles as MATURITY_COLORS.
 */
export const BEHAVIOR_COLORS: Record<string, string> = {
    excitable: "#81c784",
    quiescent: "#8c8c8c",
    tonic: "#81c784",
    adapting: "#4fc3f7",
    bursting: "#ffb74d",
    irregular: "#ce93d8",
    chaotic: "#ff5252",
    phasic: "#90a4ae",
    "rate-coded": "#4dd0e1",
    stochastic: "#bd6ecb",
};

/**
 * The leading characters of a digest, enough to recognise it by.
 *
 * @param value - The full digest.
 * @returns Its first ten characters.
 */
function shortDigest(value: string): string {
    return value.slice(0, 10);
}

/**
 * Summarise a catalogue scan for the evidence strip.
 *
 * @param metadata - The scan's metadata, or `null` when none has run.
 * @returns One labelled item per figure, empty when there is no scan.
 */
export function buildModelScanEvidenceItems(
    metadata: ModelScanMetadata | null,
): EvidenceSummaryItem[] {
    if (!metadata) return [];
    return [
        { label: "class", value: metadata.evidence_classification },
        { label: "status", value: metadata.status },
        { label: "models", value: String(metadata.model_count) },
        { label: "errors", value: String(metadata.error_count) },
        { label: "in", value: shortDigest(metadata.input_sha256) },
        { label: "out", value: shortDigest(metadata.result_sha256) },
    ];
}

/** What the list is narrowed by once the server has answered. */
interface ModelGroupFilters {
    /**
     * The names the server's catalogue query admitted, or `null` before it has
     * answered; the server holds the one filtering rule.
     */
    matched: ReadonlySet<string> | null;
    /** A live firing pattern from this browser's own scan, which only it holds. */
    patternFilter: string;
    behaviors: Record<string, ModelBehavior>;
}

/**
 * Keep the models the catalogue query admitted and group them by family.
 *
 * Search, family, behaviour and readiness are the server's decision; only the
 * live scan pattern, which exists in this browser alone, is applied here.
 *
 * @param models - The catalogue.
 * @param filters - The admitted names and the scan pattern.
 * @returns Family name to the models in it, families with none omitted.
 */
export function filterAndGroupModels<T extends { name: string; category: string }>(
    models: T[],
    filters: ModelGroupFilters,
): Record<string, T[]> {
    const { matched, patternFilter, behaviors } = filters;
    const groups: Record<string, T[]> = {};
    for (const m of models) {
        if (matched !== null && !matched.has(m.name)) continue;
        if (patternFilter && behaviors[m.name]?.pattern !== patternFilter) continue;
        (groups[m.category] ??= []).push(m);
    }
    return groups;
}

/** The proven-readiness filters the panel offers, as catalogue query fields. */
export interface ReadinessFilters {
    minVerifiedScience: number;
    minVerifiedSilicon: number;
    verifiedPerfectOnly: boolean;
}

/**
 * Build the catalogue query the panel's filters describe.
 *
 * @param text - The search text.
 * @param family - The chosen family, or `""`.
 * @param behavior - The chosen behaviour tag, or `""`.
 * @param readiness - The proven-readiness floors.
 * @returns The query; defaults are left for the encoder to drop.
 */
export function catalogueQueryFor(
    text: string,
    family: string,
    behavior: string,
    readiness: ReadinessFilters,
): CatalogueQueryParams {
    return {
        text,
        family,
        behavior,
        min_verified_science: readiness.minVerifiedScience,
        min_verified_silicon: readiness.minVerifiedSilicon,
        verified_perfect_only: readiness.verifiedPerfectOnly,
    };
}

/**
 * Order facet counts for display: largest first, then by name.
 *
 * @param counts - Value to count.
 * @returns The entries, in display order.
 */
export function facetEntries(counts: Record<string, number>): [string, number][] {
    return Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

/**
 * Names the models whose metadata could not be read.
 *
 * Rendered only when the corpus is degraded. The catalogue lists an unreadable
 * model rather than dropping it, so without this notice a fault would be
 * visible only as an entry that refuses to open.
 *
 * @param facets - Catalogue facets, or `null` before they have loaded.
 * @returns The notice, or `null` when the corpus reads cleanly.
 */
export function CatalogueHealth({ facets }: { facets: ModelFacets | null }) {
    if (!facets || facets.invalid_models.length === 0) return null;
    return (
        <div
            role="alert"
            data-testid="catalogue-health"
            style={{
                marginBottom: 4,
                padding: "3px 5px",
                fontSize: 9,
                fontFamily: "var(--font-mono)",
                color: "var(--error, #c0392b)",
                border: "1px solid var(--error, #c0392b)",
                borderRadius: "var(--radius)",
            }}
            title={facets.invalid_models.join(", ")}
        >
            {facets.invalid_models.length} of {facets.total} models have unreadable
            metadata and cannot be browsed: {facets.invalid_models.join(", ")}
        </div>
    );
}

/**
 * One toggle in a filter group: a real button, so a keyboard reaches it and a
 * screen reader hears whether it is on.
 *
 * @param props - Whether it is on, what pressing does, its tooltip, an optional
 *   accent colour and its label.
 * @returns The button.
 */
function FilterChip({
    pressed,
    onPress,
    title,
    colour,
    children,
}: {
    pressed: boolean;
    onPress: () => void;
    title: string;
    colour?: string;
    children: ReactNode;
}) {
    return (
        <button
            type="button"
            aria-pressed={pressed}
            onClick={onPress}
            title={title}
            style={{
                fontSize: 9,
                padding: "1px 6px",
                borderRadius: 3,
                cursor: "pointer",
                border: "1px solid var(--control-border)",
                background: pressed ? colour ?? "var(--accent)" : "var(--bg-tertiary)",
                color: pressed ? "var(--bg-primary)" : colour ?? "var(--text-muted)",
            }}
        >
            {children}
        </button>
    );
}

/**
 * The catalogue panel: search, facets, scan, and the model list.
 *
 * @returns The panel.
 */
export default function ModelBrowser() {
    const {
        models,
        selectedModelName,
        modelFilter,
        loadModels,
        selectModel,
        setModelFilter,
    } = useStudioStore();

    const modelScan = useModelScanJob();
    const behaviors = modelScan.state.behaviors;
    const scanMetadata = modelScan.state.scanMetadata;
    const [patternFilter, setPatternFilter] = useState<string>("");
    const [facets, setFacets] = useState<ModelFacets | null>(null);
    const [familyFilter, setFamilyFilter] = useState<string>("");
    const [behaviorFilter, setBehaviorFilter] = useState<string>("");
    const [readiness, setReadiness] = useState<ReadinessFilters>({
        minVerifiedScience: 0,
        minVerifiedSilicon: 0,
        verifiedPerfectOnly: false,
    });
    const [query, setQuery] = useState<CatalogueQueryResult | null>(null);
    const [queryError, setQueryError] = useState<string | null>(null);

    useEffect(() => {
        void loadModels();
    }, [loadModels]);
    useEffect(() => {
        fetchModelFacets()
            .then(setFacets)
            .catch(() => { setFacets(null); });
    }, []);
    useEffect(() => {
        let current = true;
        queryModels(catalogueQueryFor(modelFilter, familyFilter, behaviorFilter, readiness))
            .then((result) => {
                if (!current) return;
                setQuery(result);
                setQueryError(null);
            })
            .catch((error: unknown) => {
                if (!current) return;
                setQuery(null);
                setQueryError(error instanceof Error ? error.message : String(error));
            });
        // A slower answer to an older query must not replace a newer one.
        return () => { current = false; };
    }, [modelFilter, familyFilter, behaviorFilter, readiness]);

    const matched = useMemo(() => (query === null ? null : new Set(query.models)), [query]);
    const grouped = useMemo(
        () => filterAndGroupModels(models, { matched, patternFilter, behaviors }),
        [models, matched, patternFilter, behaviors],
    );

    const totalFiltered = Object.values(grouped).reduce(
        (s, g) => s + g.length,
        0,
    );
    const patterns = [
        ...new Set(Object.values(behaviors).map((b) => b.pattern)),
    ].sort();

    return (
        <div>
            <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
                <input
                    type="text"
                    placeholder="Search models..."
                    value={modelFilter}
                    onChange={(e) => { setModelFilter(e.target.value); }}
                    style={{
                        flex: 1,
                        padding: "4px 6px",
                        fontSize: 11,
                        background: "var(--bg-tertiary)",
                        color: "var(--text-primary)",
                        border: "1px solid var(--control-border)",
                        borderRadius: "var(--radius)",
                        // No inline `outline: none` here: an inline style beats
                        // the stylesheet, so it removed the focus ring that
                        // `input:focus-visible` puts back.
                        fontFamily: "var(--font-mono)",
                    }}
                />
                <button
                    type="button"
                    data-testid="model-scan-job-button"
                    onClick={() => {
                        if (modelScan.canSubmit) {
                            modelScan.startScan();
                        }
                    }}
                    disabled={modelScan.busy}
                    aria-busy={modelScan.busy}
                    style={{
                        fontSize: 9,
                        padding: "2px 6px",
                        background: "var(--bg-tertiary)",
                        color:
                            modelScan.state.phase === "completed"
                                ? "var(--accent)"
                                : "var(--text-muted)",
                        border: "1px solid var(--control-border)",
                        borderRadius: 3,
                        cursor: modelScan.busy ? "wait" : "pointer",
                        opacity: modelScan.busy ? 0.7 : 1,
                    }}
                    title="Submit one asynchronous full-catalogue model scan job and poll its production status route. Coloured dots already show each model's cached measured behaviour."
                >
                    {modelScan.phaseLabel}
                </button>
            </div>

            {modelScan.state.error !== null && (
                <div
                    data-testid="model-scan-job-error"
                    role="alert"
                    style={{
                        fontSize: 9,
                        color: "var(--error, #ff5252)",
                        marginBottom: 4,
                    }}
                >
                    {modelScan.state.error}
                </div>
            )}

            {(modelScan.state.phase === "pending"
                || modelScan.state.phase === "running"
                || modelScan.state.phase === "unknown"
                || modelScan.state.phase === "submitting") && (
                <div
                    data-testid="model-scan-job-status"
                    style={{
                        fontSize: 9,
                        color: "var(--text-muted)",
                        marginBottom: 4,
                    }}
                >
                    scan job: {modelScan.state.phase}
                    {modelScan.state.jobId !== null
                        ? ` · ${modelScan.state.jobId}`
                        : ""}
                </div>
            )}

            <CatalogueHealth facets={facets} />

            <select
                aria-label="Filter by family"
                value={familyFilter}
                onChange={(e) => { setFamilyFilter(e.target.value); }}
                style={{
                    width: "100%",
                    marginBottom: 4,
                    padding: "3px 4px",
                    fontSize: 10,
                    background: "var(--bg-tertiary)",
                    color: "var(--text-primary)",
                    border: "1px solid var(--control-border)",
                    borderRadius: "var(--radius)",
                    fontFamily: "var(--font-mono)",
                }}
            >
                <option value="">All families</option>
                {facetEntries(query?.facets.family ?? {}).map(([family, count]) => (
                    <option key={family} value={family}>
                        {family} ({count})
                    </option>
                ))}
                {familyFilter !== "" && !(familyFilter in (query?.facets.family ?? {})) && (
                    <option value={familyFilter}>{familyFilter} (0)</option>
                )}
            </select>

            {query !== null && Object.keys(query.facets.behavior).length > 0 && (
                <div
                    role="group"
                    aria-label="Filter by declared behaviour"
                    style={{ display: "flex", gap: 3, flexWrap: "wrap", marginBottom: 4 }}
                >
                    <FilterChip
                        pressed={!behaviorFilter}
                        onPress={() => { setBehaviorFilter(""); }}
                        title="Every declared behaviour"
                    >
                        any behaviour
                    </FilterChip>
                    {facetEntries(query.facets.behavior).map(([tag, count]) => (
                        <FilterChip
                            key={tag}
                            pressed={tag === behaviorFilter}
                            onPress={() => { setBehaviorFilter(tag === behaviorFilter ? "" : tag); }}
                            title={`${count} models under the other filters declare ${tag}`}
                            colour={BEHAVIOR_COLORS[tag]}
                        >
                            {tag} {count}
                        </FilterChip>
                    ))}
                </div>
            )}

            <div
                role="group"
                aria-label="Filter by proven science readiness"
                style={{ display: "flex", gap: 3, marginBottom: 4, flexWrap: "wrap" }}
            >
                {[
                    { tier: 0, label: "any science" },
                    { tier: 3, label: "S3+ proven" },
                    { tier: 5, label: "S5 proven" },
                ].map((o) => (
                    <FilterChip
                        key={`s-${o.tier}`}
                        pressed={readiness.minVerifiedScience === o.tier}
                        onPress={() => { setReadiness({ ...readiness, minVerifiedScience: o.tier }); }}
                        title={
                            o.tier === 0
                                ? "No science floor"
                                : `Science tier S${o.tier} or higher, proven by fresh facet receipts; a declared tier is not enough`
                        }
                    >
                        {o.label}
                    </FilterChip>
                ))}
            </div>

            <div
                role="group"
                aria-label="Filter by proven silicon readiness"
                style={{ display: "flex", gap: 3, marginBottom: 4, flexWrap: "wrap" }}
            >
                {[
                    { tier: 0, label: "any silicon" },
                    { tier: 1, label: "H1+ proven" },
                ].map((o) => (
                    <FilterChip
                        key={`h-${o.tier}`}
                        pressed={readiness.minVerifiedSilicon === o.tier}
                        onPress={() => { setReadiness({ ...readiness, minVerifiedSilicon: o.tier }); }}
                        title={
                            o.tier === 0
                                ? "No silicon floor (includes models not enrolled on silicon)"
                                : "Silicon tier H1 or higher, proven by fresh facet receipts"
                        }
                    >
                        {o.label}
                    </FilterChip>
                ))}
                <FilterChip
                    pressed={readiness.verifiedPerfectOnly}
                    onPress={() => {
                        setReadiness({ ...readiness, verifiedPerfectOnly: !readiness.verifiedPerfectOnly });
                    }}
                    title="Proven at S5 and at the model's declared terminal silicon tier"
                >
                    perfect, proven{query !== null ? ` ${query.facets.verified_perfect}` : ""}
                </FilterChip>
            </div>

            {queryError !== null && (
                <div role="alert" style={{ fontSize: 9, color: "var(--error, #ff5252)", marginBottom: 4 }}>
                    Catalogue query failed: {queryError}. The list below is not filtered.
                </div>
            )}

            {patterns.length > 0 && (
                <div
                    style={{
                        display: "flex",
                        gap: 3,
                        flexWrap: "wrap",
                        marginBottom: 4,
                    }}
                >
                    <span
                        onClick={() => { setPatternFilter(""); }}
                        style={{
                            fontSize: 9,
                            padding: "1px 5px",
                            borderRadius: 3,
                            cursor: "pointer",
                            background: !patternFilter
                                ? "var(--accent)"
                                : "var(--bg-tertiary)",
                            color: !patternFilter
                                ? "var(--bg-primary)"
                                : "var(--text-muted)",
                        }}
                    >
                        all
                    </span>
                    {patterns.map((p) => (
                        <span
                            key={p}
                            onClick={() => { setPatternFilter(p === patternFilter ? "" : p); }
                            }
                            style={{
                                fontSize: 9,
                                padding: "1px 5px",
                                borderRadius: 3,
                                cursor: "pointer",
                                background:
                                    p === patternFilter
                                        ? PATTERN_COLORS[p] ?? "var(--accent)"
                                        : "var(--bg-tertiary)",
                                color:
                                    p === patternFilter
                                        ? "var(--bg-primary)"
                                        : PATTERN_COLORS[p] ??
                                          "var(--text-muted)",
                            }}
                        >
                            {p}
                        </span>
                    ))}
                </div>
            )}

            {scanMetadata && (
                <EvidenceSummaryStrip
                    variant="grid"
                    items={buildModelScanEvidenceItems(scanMetadata)}
                />
            )}

            <div style={{ maxHeight: 200, overflowY: "auto" }}>
                {Object.entries(grouped)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([cat, ms]) => (
                        <div key={cat}>
                            <div
                                style={{
                                    fontSize: 9,
                                    fontWeight: 700,
                                    color: "var(--accent)",
                                    padding: "3px 4px 1px",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.05em",
                                }}
                            >
                                {cat} ({ms.length})
                            </div>
                            {ms.map((m) => {
                                const beh = behaviors[m.name];
                                return (
                                    <div
                                        key={m.name}
                                        onClick={() => { void selectModel(m.name); }}
                                        title={m.description || m.name}
                                        style={{
                                            padding: "2px 8px",
                                            fontSize: 10,
                                            fontFamily: "var(--font-mono)",
                                            cursor: "pointer",
                                            borderRadius: 3,
                                            background:
                                                m.name === selectedModelName
                                                    ? "var(--accent-dim)"
                                                    : "transparent",
                                            color:
                                                m.name === selectedModelName
                                                    ? "var(--accent)"
                                                    : "var(--text-secondary)",
                                            display: "flex",
                                            justifyContent: "space-between",
                                            alignItems: "center",
                                        }}
                                    >
                                        <span style={{ flex: 1, minWidth: 0 }}>
                                            <span>
                                                {m.name
                                                    .replace("Neuron", "")
                                                    .replace("Model", "")}
                                            </span>
                                            {m.metadata_state !== "available" && (
                                                <span
                                                    data-testid={`model-metadata-state-${m.name}`}
                                                    title={
                                                        m.metadata_error ??
                                                        "no committed descriptor; described by code introspection"
                                                    }
                                                    style={{
                                                        marginLeft: 4,
                                                        fontSize: 8,
                                                        color:
                                                            m.metadata_state === "invalid"
                                                                ? "var(--error, #c0392b)"
                                                                : "var(--text-muted)",
                                                    }}
                                                >
                                                    [{m.metadata_state}]
                                                </span>
                                            )}
                                            <span
                                                data-testid={`model-contract-${m.name}`}
                                                style={{
                                                    display: "block",
                                                    maxWidth: 330,
                                                    overflow: "hidden",
                                                    textOverflow: "ellipsis",
                                                    whiteSpace: "nowrap",
                                                    color: "var(--text-muted)",
                                                    fontSize: 8,
                                                }}
                                                title={`validation ${m.validation_metric}; integrator ${m.integration_method}; terminal ${m.terminal_silicon_tier || "none"}: ${m.terminal_reason || "no terminal silicon target declared"}`}
                                            >
                                                validation {m.validation_metric} · integrator{" "}
                                                {m.integration_method} · terminal{" "}
                                                {m.terminal_silicon_tier || "none"}:{" "}
                                                {m.terminal_reason ||
                                                    "no terminal silicon target declared"}
                                            </span>
                                        </span>
                                        <span
                                            style={{
                                                display: "flex",
                                                gap: 4,
                                                alignItems: "center",
                                            }}
                                        >
                                            <DualAxisBadge
                                                scienceLabel={
                                                    m.science_label
                                                }
                                                siliconLabel={
                                                    m.silicon_label
                                                }
                                                scienceTier={m.science_tier}
                                                siliconTier={m.silicon_tier}
                                            />
                                            <EvidenceTierBadge
                                                tier={m.tier}
                                                evidenceKind={m.evidence_kind}
                                            />
                                            {m.provenance?.doi && (
                                                <a
                                                    href={`https://doi.org/${m.provenance.doi}`}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    onClick={(e) => { e.stopPropagation(); }
                                                    }
                                                    title={`DOI ${m.provenance.doi}`}
                                                    style={{
                                                        fontSize: 8,
                                                        color: "var(--accent)",
                                                        textDecoration: "none",
                                                    }}
                                                >
                                                    DOI
                                                </a>
                                            )}
                                            {m.behavior_tags.length > 0 && (
                                                <span
                                                    title={`measured behaviour: ${m.behavior_tags.join(", ")}`}
                                                    style={{
                                                        display: "flex",
                                                        gap: 2,
                                                        alignItems: "center",
                                                    }}
                                                >
                                                    {m.behavior_tags
                                                        .slice(0, 4)
                                                        .map((t) => (
                                                            <span
                                                                key={t}
                                                                style={{
                                                                    width: 5,
                                                                    height: 5,
                                                                    borderRadius:
                                                                        "50%",
                                                                    background:
                                                                        BEHAVIOR_COLORS[
                                                                            t
                                                                        ] ??
                                                                        "var(--bg-tertiary)",
                                                                }}
                                                            />
                                                        ))}
                                                </span>
                                            )}
                                            <span
                                                title={`maturity: ${m.maturity}`}
                                                style={{
                                                    width: 6,
                                                    height: 6,
                                                    borderRadius: "50%",
                                                    background:
                                                        MATURITY_COLORS[
                                                            m.maturity
                                                        ] ??
                                                        "var(--bg-tertiary)",
                                                }}
                                            />
                                            {beh && (
                                                <span
                                                    style={{
                                                        fontSize: 8,
                                                        padding: "0 3px",
                                                        borderRadius: 2,
                                                        background:
                                                            PATTERN_COLORS[
                                                                beh.pattern
                                                            ] ??
                                                            "var(--bg-tertiary)",
                                                        color: "var(--bg-primary)",
                                                        fontWeight: 600,
                                                    }}
                                                >
                                                    {beh.pattern}
                                                </span>
                                            )}
                                            <span
                                                style={{
                                                    color: "var(--text-muted)",
                                                    fontSize: 9,
                                                }}
                                            >
                                                {m.state_var_names.join(",")}
                                                &middot;{m.n_params}p
                                            </span>
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    ))}
            </div>
            <div
                style={{
                    fontSize: 9,
                    color: "var(--text-muted)",
                    marginTop: 3,
                }}
            >
                {totalFiltered}/{models.length} models
                {Object.keys(behaviors).length > 0 &&
                    ` · ${Object.values(behaviors).filter((b) => b.pattern === "tonic").length} tonic · ${Object.values(behaviors).filter((b) => b.pattern === "bursting").length} bursting · ${Object.values(behaviors).filter((b) => b.pattern === "silent").length} silent`}
            </div>
        </div>
    );
}
