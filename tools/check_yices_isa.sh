#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Yices object instruction-set admission

set -euo pipefail
object_root="${1:?Yices object directory required}"
scan_root="${2:?scan output directory required}"
test -d "$object_root"
mkdir -p "$scan_root"
mapfile -d '' -t objects < <(find "$object_root" -type f -name '*.o' -print0 | sort -z)
test "${#objects[@]}" -gt 0
objdump -d -M intel --no-show-raw-insn "${objects[@]}" > "$scan_root/disassembly.txt"
# Reject all VEX-vector instructions as well as BMI and AVX-512 opmask instructions.
# GCC emits rep bsf (decoded as tzcnt) for ctz: nonzero inputs have identical
# results on Sandy Bridge's bsf. Lzcnt instead decodes as bsr with different results.
pattern='^[[:space:]]*[[:xdigit:]]+:[[:space:]]+((v[a-z0-9]+|andn|bextr|blsi|blsmsk|blsr|bzhi|mulx|pdep|pext|rorx|sarx|shlx|shrx|k[a-z0-9]+|lzcnt|adcx|adox|movbe|rdseed|rdrand|sha[a-z0-9]+|gf2p8[a-z0-9]+|clflushopt|clwb)([[:space:]]|$)|.*[[:space:],{]([yz]mm[0-9]+|k[0-7])([[:space:],}]|$))'
if grep -Ein "$pattern" "$scan_root/disassembly.txt" > "$scan_root/unsupported.txt"; then
    cat "$scan_root/unsupported.txt" >&2
    exit 1
else
    test "$?" -eq 1
fi
printf 'Yices own-object ISA scan PASS: %s objects; no AVX vectors, BMI1/2 or AVX-512\n' "${#objects[@]}"
printf 'Yices own-object disassembly SHA256 %s\n' "$(sha256sum "$scan_root/disassembly.txt" | cut -d ' ' -f 1)"
printf '%s\n' 'libgmp.a deliberately excluded: pinned Ubuntu GMP uses CPUID dispatch for CPU-specific routines.'
printf '%s\n' 'Sandy Bridge exclusions also cover LZCNT, ADX, MOVBE, RDSEED, RDRAND, SHA, GFNI, CLFLUSHOPT and CLWB; GCC rep-bsf/ctz is permitted for nonzero inputs.'
