#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Prepare the official Python API reference inventory

# Usage: prepare_python_inventory.sh MANIFEST CACHE_DIRECTORY
# Downloads the pinned official documentation archive and verifies both digests.
# Sphinx consumes CACHE_DIRECTORY/python.inv through
# SC_NEUROCORE_PYTHON_REFERENCE_INVENTORY.
set -euo pipefail

# Read a required value from the source-controlled inventory manifest.
manifest_value() {
    local manifest="$1"
    local key="$2"
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$manifest" "$key"
}

# Materialise only the selected inventory after archive and content verification.
main() {
    if [[ "$#" -ne 2 ]]; then
        echo "Usage: $0 MANIFEST CACHE_DIRECTORY" >&2
        return 2
    fi
    local required_command
    for required_command in python3 curl sha256sum tar; do
        if ! command -v "$required_command" >/dev/null; then
            echo "Required command not found: $required_command" >&2
            return 127
        fi
    done
    local manifest="$1"
    local inventory_cache_dir="$2"
    local archive_url archive_sha256 inventory_member inventory_sha256
    archive_url="$(manifest_value "$manifest" archive_url)"
    archive_sha256="$(manifest_value "$manifest" archive_sha256)"
    inventory_member="$(manifest_value "$manifest" inventory_member)"
    inventory_sha256="$(manifest_value "$manifest" inventory_sha256)"
    mkdir -p "$inventory_cache_dir"
    local archive_path="$inventory_cache_dir/python-docs.tar.bz2"
    local inventory_path="$inventory_cache_dir/python.inv"
    if [[ ! -f "$archive_path" ]]; then
        curl --fail --silent --show-error --location \
            --proto '=https' --proto-redir '=https' \
            --connect-timeout 15 --max-time 120 \
            --retry 3 --retry-delay 2 --retry-max-time 180 \
            --output "$archive_path.partial" "$archive_url"
        mv -- "$archive_path.partial" "$archive_path"
    fi
    printf '%s  %s\n' "$archive_sha256" "$archive_path" | sha256sum --check --strict -
    tar --extract --bzip2 --file "$archive_path" --to-stdout -- "$inventory_member" > "$inventory_path.partial"
    printf '%s  %s\n' "$inventory_sha256" "$inventory_path.partial" | sha256sum --check --strict -
    mv -- "$inventory_path.partial" "$inventory_path"
    printf 'Prepared official Python reference inventory: %s\n' "$inventory_path"
}

main "$@"
