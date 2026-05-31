#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 ANULUM
#
# This file is part of sc-neurocore.
# See the LICENSE file in the project root for full license text.
# Commercial licensing is available; contact protoscience@anulum.li.
# Launch a local ML350 ORCA Posner optimisation/frequency run.

set -euo pipefail

: "${ORCA_ARCHIVE:?ORCA_ARCHIVE is required}"
: "${INPUT_XYZ:?INPUT_XYZ is required}"

JOB_NAME="${JOB_NAME:-posner_ml350_neutral_opt_r6_seeded}"
RUN_ROOT="${RUN_ROOT:-$HOME/sc-neurocore-orca-runs/$JOB_NAME}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local/sc-neurocore/orca-6.1.1}"
NPROCS="${NPROCS:-6}"
MAXCORE_MB="${MAXCORE_MB:-3500}"
ORCA_ARCHIVE_SHA1="${ORCA_ARCHIVE_SHA1:-98490e09ad999792bd23ed7a06a6799aef01fb5a}"
SEED_GBW="${SEED_GBW:-}"
GEOM_MAXITER="${GEOM_MAXITER:-}"

mkdir -p "$RUN_ROOT"/{input,run,output,logs} "$INSTALL_ROOT"

sha1_actual="$(sha1sum "$ORCA_ARCHIVE" | awk '{print $1}')"
if [[ "$sha1_actual" != "$ORCA_ARCHIVE_SHA1" ]]; then
    echo "ORCA archive SHA1 mismatch: expected $ORCA_ARCHIVE_SHA1, got $sha1_actual" >&2
    exit 2
fi

if [[ ! -x "$INSTALL_ROOT/orca" ]]; then
    marker="$INSTALL_ROOT/.extracting.$$"
    touch "$marker"
    tar -xJf "$ORCA_ARCHIVE" -C "$INSTALL_ROOT" --strip-components=1
    rm -f "$marker"
fi

ORCA_BIN="$INSTALL_ROOT/orca"
if [[ ! -x "$ORCA_BIN" ]]; then
    echo "No executable ORCA binary found at $ORCA_BIN" >&2
    exit 3
fi

cp -f "$INPUT_XYZ" "$RUN_ROOT/run/input.xyz"
archive_copy="$RUN_ROOT/input/orca_6_1_1_linux_x86-64_shared_openmpi418.tar.xz"
if [[ "$(readlink -f "$ORCA_ARCHIVE")" != "$(readlink -m "$archive_copy")" ]]; then
    cp -f "$ORCA_ARCHIVE" "$archive_copy"
fi
if [[ -n "$SEED_GBW" ]]; then
    cp -f "$SEED_GBW" "$RUN_ROOT/run/seed.gbw"
fi

if [[ -f "$RUN_ROOT/run/seed.gbw" ]]; then
    moread=" MOREAD"
    moinp='%moinp "seed.gbw"'
else
    moread=""
    moinp=""
fi

if [[ "$NPROCS" -gt 1 ]]; then
    pal_block="%pal nprocs ${NPROCS} end"
else
    pal_block=""
fi

if [[ -n "$GEOM_MAXITER" ]]; then
    geom_block=$(cat <<EOF
%geom
  MaxIter ${GEOM_MAXITER}
end
EOF
)
else
    geom_block=""
fi

cat >"$RUN_ROOT/run/${JOB_NAME}.inp" <<EOF
! B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq${moread}
${pal_block}
%maxcore ${MAXCORE_MB}
${moinp}
${geom_block}
* xyzfile 0 1 input.xyz
EOF

cat >"$RUN_ROOT/manifest.json" <<EOF
{
  "schema": "sc-neurocore.ml350-posner-orca.v1",
  "job_name": "${JOB_NAME}",
  "run_root": "${RUN_ROOT}",
  "install_root": "${INSTALL_ROOT}",
  "orca_archive_sha1": "${ORCA_ARCHIVE_SHA1}",
  "nprocs": ${NPROCS},
  "maxcore_mb": ${MAXCORE_MB},
  "method": "B3LYP def2-TZVP D3BJ RIJCOSX VeryTightSCF DefGrid3 Opt Freq",
  "charge": 0,
  "multiplicity": 1,
  "geom_maxiter": $(if [[ -n "$GEOM_MAXITER" ]]; then echo "$GEOM_MAXITER"; else echo null; fi),
  "seeded_from_gbw": $(if [[ -f "$RUN_ROOT/run/seed.gbw" ]]; then echo true; else echo false; fi)
}
EOF

export OMP_NUM_THREADS="$NPROCS"
export MKL_NUM_THREADS="$NPROCS"
export OPENBLAS_NUM_THREADS="$NPROCS"
export OMPI_MCA_rmaps_base_oversubscribe=1
export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe

cd "$RUN_ROOT/run"
echo "$$" >"$RUN_ROOT/${JOB_NAME}.pid"
{
    date -Is
    hostname
    "$ORCA_BIN" "${JOB_NAME}.inp"
    status="$?"
    date -Is
    echo "$status" >"$RUN_ROOT/output/exit_status.txt"
    exit "$status"
} >"${JOB_NAME}.out" 2>&1
