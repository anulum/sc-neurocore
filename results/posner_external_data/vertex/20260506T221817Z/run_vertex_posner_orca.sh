#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Vertex ORCA Posner verification runner

set -euo pipefail

: "${ORCA_ARCHIVE_URI:?ORCA_ARCHIVE_URI is required}"
: "${ORCA_ARCHIVE_SHA1:?ORCA_ARCHIVE_SHA1 is required}"
: "${INPUT_XYZ_URI:?INPUT_XYZ_URI is required}"
: "${OUTPUT_URI:?OUTPUT_URI is required}"

NPROCS="${NPROCS:-6}"
MAXCORE_MB="${MAXCORE_MB:-3000}"
WORKDIR="${WORKDIR:-/tmp/sc-neurocore-posner-orca}"
JOB_NAME="${JOB_NAME:-posner_vertex_neutral_opt}"

export PATH="/google-cloud-sdk/bin:/usr/lib/google-cloud-sdk/bin:/usr/local/google-cloud-sdk/bin:${PATH}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_orte_allow_run_as_root=1
export OMPI_MCA_orte_allow_run_as_root_confirm=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe
export PRTE_MCA_prte_allow_run_as_root=1
export PRTE_MCA_prte_allow_run_as_root_confirm=1

GCLOUD_BIN="$(command -v gcloud || true)"
if [[ -z "$GCLOUD_BIN" ]]; then
    echo "gcloud CLI not found in container PATH" >&2
    exit 4
fi

mkdir -p "$WORKDIR"/{input,orca,run,output}

if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates xz-utils openmpi-bin libopenmpi-dev
fi

"$GCLOUD_BIN" storage cp "$ORCA_ARCHIVE_URI" "$WORKDIR/input/orca.tar.xz"
"$GCLOUD_BIN" storage cp "$INPUT_XYZ_URI" "$WORKDIR/run/input.xyz"

actual_sha1="$(sha1sum "$WORKDIR/input/orca.tar.xz" | awk '{print $1}')"
if [[ "$actual_sha1" != "$ORCA_ARCHIVE_SHA1" ]]; then
    echo "ORCA archive SHA1 mismatch: expected $ORCA_ARCHIVE_SHA1, got $actual_sha1" >&2
    exit 2
fi

tar -xJf "$WORKDIR/input/orca.tar.xz" -C "$WORKDIR/orca"
ORCA_BIN="$(find "$WORKDIR/orca" -type f -name orca -perm -111 | head -n 1)"
if [[ -z "$ORCA_BIN" ]]; then
    echo "No executable ORCA binary found after extracting archive" >&2
    exit 3
fi

cat >"$WORKDIR/run/${JOB_NAME}.inp" <<EOF
! B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq
%pal nprocs ${NPROCS} end
%maxcore ${MAXCORE_MB}
* xyzfile 0 1 input.xyz
EOF

cat >"$WORKDIR/run/manifest.json" <<EOF
{
  "schema": "sc-neurocore.vertex-posner-orca.v1",
  "job_name": "${JOB_NAME}",
  "orca_archive_uri": "${ORCA_ARCHIVE_URI}",
  "orca_archive_sha1": "${ORCA_ARCHIVE_SHA1}",
  "input_xyz_uri": "${INPUT_XYZ_URI}",
  "nprocs": ${NPROCS},
  "maxcore_mb": ${MAXCORE_MB},
  "method": "B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq",
  "charge": 0,
  "multiplicity": 1
}
EOF

cleanup_done=0
cleanup() {
    status="${1:-$?}"
    reason="${2:-exit}"
    if [[ "$cleanup_done" -eq 1 ]]; then
        exit "$status"
    fi
    cleanup_done=1
    cp -f "$WORKDIR/run/${JOB_NAME}.inp" "$WORKDIR/output/" 2>/dev/null || true
    cp -f "$WORKDIR/run/manifest.json" "$WORKDIR/output/" 2>/dev/null || true
    find "$WORKDIR/run" -maxdepth 1 -type f \
        \( -name "${JOB_NAME}*" -o -name "input.xyz" \) \
        -exec cp -f {} "$WORKDIR/output/" \; 2>/dev/null || true
    echo "$status" >"$WORKDIR/output/exit_status.txt"
    echo "$reason" >"$WORKDIR/output/exit_reason.txt"
    "$GCLOUD_BIN" storage cp --recursive "$WORKDIR/output" "$OUTPUT_URI/" || true
    exit "$status"
}
trap 'cleanup "$?" exit' EXIT
trap 'cleanup 124 sigterm' TERM
trap 'cleanup 130 sigint' INT

cd "$WORKDIR/run"
"$ORCA_BIN" "${JOB_NAME}.inp" >"${JOB_NAME}.out" 2>&1
if ! grep -q "ORCA TERMINATED NORMALLY" "${JOB_NAME}.out"; then
    echo "ORCA did not terminate normally; see ${JOB_NAME}.out" >&2
    exit 5
fi
if ! grep -q "THE OPTIMIZATION HAS CONVERGED" "${JOB_NAME}.out"; then
    echo "ORCA optimization did not converge; see ${JOB_NAME}.out" >&2
    exit 6
fi
