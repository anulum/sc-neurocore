import { useEffect, useMemo, useState } from "react";
import { useStudioStore } from "../stores/studio";
import {
    fetchModelFacets,
    fetchModelScan,
    type ModelBehavior,
    type ModelFacets,
    type ModelScanMetadata,
} from "../api/client";
import EvidenceSummaryStrip, {
    type EvidenceSummaryItem,
} from "./EvidenceSummaryStrip";
import EvidenceTierBadge from "./EvidenceTierBadge";

const MATURITY_COLORS: Record<string, string> = {
    validated: "#81c784",
    experimental: "#ffb74d",
    reference: "#4fc3f7",
};

const PATTERN_COLORS: Record<string, string> = {
    tonic: "#81c784",
    bursting: "#ffb74d",
    adapting: "#4fc3f7",
    irregular: "#ce93d8",
    chaotic: "#ff5252",
    silent: "#484f58",
    single_spike: "#90a4ae",
    error: "#616161",
};

// Measured behaviour-tag colours (descriptor facet, distinct from the live scan).
const BEHAVIOR_COLORS: Record<string, string> = {
    excitable: "#81c784",
    quiescent: "#616161",
    tonic: "#81c784",
    adapting: "#4fc3f7",
    bursting: "#ffb74d",
    irregular: "#ce93d8",
    chaotic: "#ff5252",
    phasic: "#90a4ae",
    "rate-coded": "#4dd0e1",
    stochastic: "#ba68c8",
};

function shortDigest(value: string): string {
    return value.slice(0, 10);
}

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

interface ModelGroupFilters {
    modelFilter: string;
    familyFilter: string;
    patternFilter: string;
    minTier: number;
    behaviors: Record<string, ModelBehavior>;
    behaviorFilter?: string;
}

/** Filter the catalogue by search text, family, measured behaviour tag, live
 *  firing pattern, and minimum evidence tier, then group the survivors by their
 *  displayed family. */
export function filterAndGroupModels<
    T extends {
        name: string;
        category: string;
        family: string;
        tier?: number;
        behavior_tags?: string[];
    },
>(models: T[], filters: ModelGroupFilters): Record<string, T[]> {
    let filtered = models;
    if (filters.modelFilter) {
        const q = filters.modelFilter.toLowerCase();
        filtered = filtered.filter(
            (m) =>
                m.name.toLowerCase().includes(q) ||
                m.category.toLowerCase().includes(q),
        );
    }
    if (filters.familyFilter) {
        filtered = filtered.filter((m) => m.family === filters.familyFilter);
    }
    if (filters.behaviorFilter) {
        filtered = filtered.filter((m) =>
            m.behavior_tags?.includes(filters.behaviorFilter!),
        );
    }
    if (filters.minTier > 0) {
        filtered = filtered.filter((m) => (m.tier ?? 0) >= filters.minTier);
    }
    if (filters.patternFilter) {
        filtered = filtered.filter(
            (m) => filters.behaviors[m.name]?.pattern === filters.patternFilter,
        );
    }
    const groups: Record<string, T[]> = {};
    for (const m of filtered) {
        (groups[m.category] ??= []).push(m);
    }
    return groups;
}

export default function ModelBrowser() {
    const {
        models,
        selectedModelName,
        modelFilter,
        loadModels,
        selectModel,
        setModelFilter,
    } = useStudioStore();

    const [behaviors, setBehaviors] = useState<Record<string, ModelBehavior>>(
        {},
    );
    const [scanMetadata, setScanMetadata] = useState<ModelScanMetadata | null>(
        null,
    );
    const [patternFilter, setPatternFilter] = useState<string>("");
    const [scanLoaded, setScanLoaded] = useState(false);
    const [facets, setFacets] = useState<ModelFacets | null>(null);
    const [familyFilter, setFamilyFilter] = useState<string>("");
    const [behaviorFilter, setBehaviorFilter] = useState<string>("");
    const [minTier, setMinTier] = useState<number>(0);

    useEffect(() => {
        loadModels();
    }, [loadModels]);
    useEffect(() => {
        fetchModelFacets()
            .then(setFacets)
            .catch(() => setFacets(null));
    }, []);

    function loadBehaviors() {
        if (scanLoaded) return;
        setScanLoaded(true);
        fetchModelScan()
            .then((data) => {
                const map: Record<string, ModelBehavior> = {};
                for (const b of data.models) map[b.name] = b;
                setBehaviors(map);
                setScanMetadata(data.scan_metadata);
            })
            .catch(() => setScanLoaded(false));
    }

    const grouped = useMemo(
        () =>
            filterAndGroupModels(models, {
                modelFilter,
                familyFilter,
                patternFilter,
                minTier,
                behaviors,
                behaviorFilter,
            }),
        [
            models,
            modelFilter,
            familyFilter,
            patternFilter,
            minTier,
            behaviors,
            behaviorFilter,
        ],
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
                    onChange={(e) => setModelFilter(e.target.value)}
                    style={{
                        flex: 1,
                        padding: "4px 6px",
                        fontSize: 11,
                        background: "var(--bg-tertiary)",
                        color: "var(--text-primary)",
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius)",
                        outline: "none",
                        fontFamily: "var(--font-mono)",
                    }}
                />
                <button
                    onClick={loadBehaviors}
                    style={{
                        fontSize: 9,
                        padding: "2px 6px",
                        background: "var(--bg-tertiary)",
                        color: scanLoaded
                            ? "var(--accent)"
                            : "var(--text-muted)",
                        border: "1px solid var(--border)",
                        borderRadius: 3,
                        cursor: "pointer",
                    }}
                    title="Optional live single-current firing-pattern scan (slow). The coloured dots already show each model's cached measured behaviour."
                >
                    {scanLoaded ? "Scanned" : "Scan"}
                </button>
            </div>

            {facets && (
                <select
                    aria-label="Filter by family"
                    value={familyFilter}
                    onChange={(e) => setFamilyFilter(e.target.value)}
                    style={{
                        width: "100%",
                        marginBottom: 4,
                        padding: "3px 4px",
                        fontSize: 10,
                        background: "var(--bg-tertiary)",
                        color: "var(--text-primary)",
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius)",
                        fontFamily: "var(--font-mono)",
                    }}
                >
                    <option value="">All families ({facets.total})</option>
                    {facets.families.map((f) => (
                        <option key={f.family} value={f.family}>
                            {f.family} ({f.count})
                        </option>
                    ))}
                </select>
            )}

            {facets && facets.behaviors.length > 0 && (
                <div
                    style={{
                        display: "flex",
                        gap: 3,
                        flexWrap: "wrap",
                        marginBottom: 4,
                    }}
                >
                    <span
                        onClick={() => setBehaviorFilter("")}
                        style={{
                            fontSize: 9,
                            padding: "1px 5px",
                            borderRadius: 3,
                            cursor: "pointer",
                            background: !behaviorFilter
                                ? "var(--accent)"
                                : "var(--bg-tertiary)",
                            color: !behaviorFilter
                                ? "var(--bg-primary)"
                                : "var(--text-muted)",
                        }}
                        title="Measured behaviour across the current sweep (descriptor facet)"
                    >
                        behaviour
                    </span>
                    {facets.behaviors.map((b) => (
                        <span
                            key={b.tag}
                            onClick={() =>
                                setBehaviorFilter(
                                    b.tag === behaviorFilter ? "" : b.tag,
                                )
                            }
                            title={`${b.count} models measured ${b.tag}`}
                            style={{
                                fontSize: 9,
                                padding: "1px 5px",
                                borderRadius: 3,
                                cursor: "pointer",
                                background:
                                    b.tag === behaviorFilter
                                        ? BEHAVIOR_COLORS[b.tag] ||
                                          "var(--accent)"
                                        : "var(--bg-tertiary)",
                                color:
                                    b.tag === behaviorFilter
                                        ? "var(--bg-primary)"
                                        : BEHAVIOR_COLORS[b.tag] ||
                                          "var(--text-muted)",
                            }}
                        >
                            {b.tag} {b.count}
                        </span>
                    ))}
                </div>
            )}

            <div style={{ display: "flex", gap: 3, marginBottom: 4 }}>
                {[
                    { tier: 0, label: "all evidence" },
                    { tier: 2, label: "curated+" },
                    { tier: 3, label: "verified" },
                ].map((o) => (
                    <span
                        key={o.tier}
                        onClick={() => setMinTier(o.tier)}
                        style={{
                            fontSize: 9,
                            padding: "1px 6px",
                            borderRadius: 3,
                            cursor: "pointer",
                            background:
                                minTier === o.tier
                                    ? "var(--accent)"
                                    : "var(--bg-tertiary)",
                            color:
                                minTier === o.tier
                                    ? "var(--bg-primary)"
                                    : "var(--text-muted)",
                        }}
                        title={
                            o.tier === 0
                                ? "Show all models"
                                : o.tier === 2
                                  ? "Tier 2+ — scientifically curated"
                                  : "Tier 3 — engineering-verified (parity + reproducibility)"
                        }
                    >
                        {o.label}
                    </span>
                ))}
            </div>

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
                        onClick={() => setPatternFilter("")}
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
                            onClick={() =>
                                setPatternFilter(p === patternFilter ? "" : p)
                            }
                            style={{
                                fontSize: 9,
                                padding: "1px 5px",
                                borderRadius: 3,
                                cursor: "pointer",
                                background:
                                    p === patternFilter
                                        ? PATTERN_COLORS[p] || "var(--accent)"
                                        : "var(--bg-tertiary)",
                                color:
                                    p === patternFilter
                                        ? "var(--bg-primary)"
                                        : PATTERN_COLORS[p] ||
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
                                        onClick={() => selectModel(m.name)}
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
                                        <span>
                                            {m.name
                                                .replace("Neuron", "")
                                                .replace("Model", "")}
                                        </span>
                                        <span
                                            style={{
                                                display: "flex",
                                                gap: 4,
                                                alignItems: "center",
                                            }}
                                        >
                                            <EvidenceTierBadge
                                                tier={m.tier}
                                                evidenceKind={m.evidence_kind}
                                            />
                                            {m.provenance?.doi && (
                                                <a
                                                    href={`https://doi.org/${m.provenance.doi}`}
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    onClick={(e) =>
                                                        e.stopPropagation()
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
                                                                        ] ||
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
                                                        ] ||
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
                                                            ] ||
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
