#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generic x86-64 Yices SDK build

set -euo pipefail
tool_prefix="$(realpath -m "${1:?tool prefix required}")"
source_root="$(realpath -m "${2:?source root required}")/yices-build"
recipe="$(realpath "$0")"
export LC_ALL=C TZ=UTC SOURCE_DATE_EPOCH=1752692760
export TMPDIR="$source_root/tmp"
mkdir -p "$source_root/source" "$source_root/deps" "$TMPDIR"

fetch() {
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 "$1" -o "$source_root/$2"
    printf '%s  %s\n' "$3" "$source_root/$2" | sha256sum --check --status
}

fetch https://codeload.github.com/SRI-CSL/yices2/tar.gz/85cf17e44eac76b5d14b297c09fc9bfecf47ef65 \
    source.tar.gz 37ba930fddde2b087e5a95097a9cec5010510346aad8a0987204e6decd220898
fetch https://archive.ubuntu.com/ubuntu/pool/main/g/gmp/libgmp-dev_6.3.0+dfsg-2ubuntu6_amd64.deb \
    gmp-dev.deb ffadf91b116cc2647ef213742ac1cbe80f450af14413c146df6e200f8c3c9a1a
fetch https://archive.ubuntu.com/ubuntu/pool/main/g/gmp/libgmp10_6.3.0+dfsg-2ubuntu6_amd64.deb \
    gmp.deb 649a5ee5f74f0aa1e1b9c4f169eabbae56a79aa5aac51ef733eb9e18700bd8fc
fetch https://archive.ubuntu.com/ubuntu/pool/universe/g/gperf/gperf_3.1-1build1_amd64.deb \
    gperf.deb f6701027304592fe819d219a5ee00d8b56ba4b2a05a7623dd923f1884d595042
tar -xzf "$source_root/source.tar.gz" -C "$source_root/source" --strip-components=1
for package in gmp-dev gmp gperf; do
    dpkg-deb --extract "$source_root/$package.deb" "$source_root/deps"
done
export PATH="$source_root/deps/usr/bin:$PATH"
flags="-O2 -march=x86-64 -mtune=generic -ffile-prefix-map=$source_root=/usr/src/scn-yices -fmacro-prefix-map=$source_root=/usr/src/scn-yices"
(
    cd "$source_root/source"
    autoconf
    CC=/usr/bin/gcc-13 CFLAGS="$flags" \
        CPPFLAGS="-I$source_root/deps/usr/include/x86_64-linux-gnu" \
        LDFLAGS="-L$source_root/deps/usr/lib/x86_64-linux-gnu" \
        ./configure --build=x86_64-pc-linux-gnu \
        --with-static-gmp="$source_root/deps/usr/lib/x86_64-linux-gnu/libgmp.a" \
        --with-static-gmp-include-dir="$source_root/deps/usr/include/x86_64-linux-gnu"
    # Upstream's version generator reads wall-clock date and the enclosing Git tree.
    sed -e 's/YICES_VERSION/2.7.0/g' \
        -e "s/YICES_COMPILATION_DATE/$(date -u -d "@$SOURCE_DATE_EPOCH" +%Y-%m-%d)/g" \
        -e 's/REVISION/85cf17e44eac76b5d14b297c09fc9bfecf47ef65/g' \
        -e 's/YICES_ARCH/x86_64-pc-linux-gnu/g' -e 's/YICES_BUILD_MODE/release/g' \
        src/api/yices_version_template.txt > src/api/yices_release_version.c
    # Command-line CFLAGS also prevents upstream release mode from appending -O3.
    make -j "${YICES_BUILD_JOBS:-2}" static-bin MKVERSION=true CFLAGS="$flags"
)
mkdir -p "$tool_prefix/bin" "$tool_prefix/share/yices-build" \
    "$tool_prefix/share/licenses/yices" "$tool_prefix/share/licenses/gmp" \
    "$tool_prefix/share/licenses/gperf"
bash "$(dirname "$recipe")/check_yices_isa.sh" \
    "$source_root/source/build/x86_64-pc-linux-gnu-release/static_obj" \
    "$source_root/isa-scan" > "$tool_prefix/share/yices-build/isa-scan.txt"
install -m 755 "$source_root/source/build/x86_64-pc-linux-gnu-release/static_bin/yices_smt2" \
    "$tool_prefix/bin/yices-smt2"
cp "$source_root/source/LICENSE.txt" "$tool_prefix/share/licenses/yices/LICENSE"
cp "$source_root/source/README.md" "$tool_prefix/share/licenses/yices/README"
cp "$source_root/deps/usr/share/doc/libgmp10/copyright" "$tool_prefix/share/licenses/gmp/"
cp "$source_root/deps/usr/share/doc/gperf/copyright" "$tool_prefix/share/licenses/gperf/"
cp /usr/share/common-licenses/GPL-3 "$tool_prefix/share/licenses/yices/"
cp /usr/share/common-licenses/LGPL-3 "$tool_prefix/share/licenses/gmp/"
printf '%s\n' '(set-logic QF_BV)' '(declare-fun x () (_ BitVec 1))' \
    '(assert (= x #b1))' '(check-sat)' '(push 1)' '(assert (= x #b0))' \
    '(check-sat)' '(pop 1)' '(exit)' > "$tool_prefix/share/yices-build/smoke.smt2"
timeout 5 "$tool_prefix/bin/yices-smt2" --incremental \
    < "$tool_prefix/share/yices-build/smoke.smt2" > "$tool_prefix/share/yices-build/smoke.out"
test "$(cat "$tool_prefix/share/yices-build/smoke.out")" = "$(printf 'sat\nunsat')"
(
    cd "$tool_prefix"
    sha256sum bin/yices-smt2 share/yices-build/smoke.smt2 share/yices-build/smoke.out \
        share/yices-build/isa-scan.txt \
        > share/yices-build/SHA256SUMS
)
{
    printf '%s\n' 'Yices source commit 85cf17e44eac76b5d14b297c09fc9bfecf47ef65 (yices-2.7.0)'
    printf '%s\n' 'Yices source archive SHA256 37ba930fddde2b087e5a95097a9cec5010510346aad8a0987204e6decd220898'
    printf '%s\n' 'Ubuntu noble libgmp-dev 2:6.3.0+dfsg-2ubuntu6 SHA256 ffadf91b116cc2647ef213742ac1cbe80f450af14413c146df6e200f8c3c9a1a'
    printf '%s\n' 'Ubuntu noble libgmp10 2:6.3.0+dfsg-2ubuntu6 SHA256 649a5ee5f74f0aa1e1b9c4f169eabbae56a79aa5aac51ef733eb9e18700bd8fc'
    printf '%s\n' 'Ubuntu noble gperf 3.1-1build1 SHA256 f6701027304592fe819d219a5ee00d8b56ba4b2a05a7623dd923f1884d595042'
    printf '%s\n' 'Ubuntu archive signature F6ECB3762474EDA9D21B7022871920D1991BC93C'
    printf '%s\n' 'Ubuntu noble InRelease SHA256 cdb2f31d809f589719a53c6ad15f255b27569c4059542ada282aaa21b8e164b0'
    printf '%s\n' 'Ubuntu noble main Packages.xz SHA256 2a6a199e1031a5c279cb346646d594993f35b1c03dd4a82aaa0323980dd92451'
    printf '%s\n' 'Ubuntu noble universe Packages.xz SHA256 ba9057fa1b91438cc8a1d26808d00c85389fe101d0c1496254df97236405599a'
    printf '%s\n' 'Yices source provenance: pinned GitHub commit archive SHA256; no upstream signature verified'
    printf 'Yices CFLAGS %s\nSOURCE_DATE_EPOCH %s\n' "$flags" "$SOURCE_DATE_EPOCH"
    printf 'Yices recipe SHA256 %s\n' "$(sha256sum "$recipe" | cut -d ' ' -f 1)"
    printf 'Yices ISA checker SHA256 %s\n' "$(sha256sum "$(dirname "$recipe")/check_yices_isa.sh" | cut -d ' ' -f 1)"
    /usr/bin/gcc-13 --version | head -1
    autoconf --version | head -1
    sha256sum "$source_root/deps/usr/bin/gperf" \
        "$source_root/deps/usr/lib/x86_64-linux-gnu/libgmp.a" \
        "$source_root/deps/usr/include/x86_64-linux-gnu/gmp.h"
    cat "$tool_prefix/share/yices-build/SHA256SUMS"
    cat "$tool_prefix/share/yices-build/isa-scan.txt"
    printf '%s\n' 'Yices build smoke: SAT then UNSAT, rc 0; exact output checked'
    printf '%s\n' 'Cross-host byte identity requires distribution of this single SDK artifact; independent-build reproducibility is not established.'
} > "$tool_prefix/share/yices-build/PROVENANCE.txt"
