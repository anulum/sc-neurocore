#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Replay the SHD AXI input-commit regression
set -euo pipefail
if (( $# != 1 )); then
    echo "Usage: bash hdl/test_shd_axi.sh NEW_WORK_DIRECTORY" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -- "$1"
cd -- "$1"
work_dir="$PWD"
export TMPDIR="$work_dir"
verilator --version
for layer in 1 2 3; do
    case "$layer" in
        1) rows=128; columns=140 ;;
        2) rows=128; columns=128 ;;
        3) rows=20; columns=128 ;;
    esac
    awk -v count="$((rows * columns))" -v columns="$columns" 'BEGIN {
        for (i=0; i<count; i++) {
            value = ((i*13 + int(i/columns)*7) % 17) - 8;
            printf "%02x\n", (value+256)%256;
        }
    }' > "weights_layer${layer}.hex"
done
for layer in 1 2; do
    if (( layer == 1 )); then channels=140; else channels=128; fi
    awk -v count="$channels" 'BEGIN {
        for (i=0; i<count; i++) printf "%02x\n", (271-(i*7)%31)%256;
    }' > "delays_layer${layer}.hex"
done
cp -- "$repo_root/hdl/fixtures/shd_axi_expected.hex" expected.hex
iverilog -g2012 -I "$repo_root/hdl" -s tb_sc_shd_axi_wrapper -o "$work_dir/test.vvp" \
    "$repo_root/hdl/sc_axonal_delay.v" "$repo_root/hdl/sc_dense_int8_sparse.v" \
    "$repo_root/hdl/sc_vmin_lif_neuron.v" "$repo_root/hdl/sc_shd_top.v" \
    "$repo_root/hdl/sc_shd_axi_wrapper.v" "$repo_root/hdl/tb_sc_shd_axi_wrapper.v"
timeout 120s vvp "$work_dir/test.vvp"

iverilog -g2012 -I "$repo_root/hdl" -s tb_sc_shd_top -o "$work_dir/direct.vvp" \
    "$repo_root/hdl/sc_axonal_delay.v" "$repo_root/hdl/sc_dense_int8_sparse.v" \
    "$repo_root/hdl/sc_vmin_lif_neuron.v" "$repo_root/hdl/sc_shd_top.v" \
    "$repo_root/hdl/tb_sc_shd_top.v"
printf '4096 4096 2048\n' > scales.txt
mapfile -t reference_hex < <(awk '!/^\/\// && NF { print $1 }' expected.hex)
for sample in 0 1 2; do
    case "$sample" in 0) length=6 ;; 1) length=1 ;; 2) length=3 ;; esac
    awk -v sample="$sample" -v time_bins="$length" 'BEGIN {
        limit = sample == 0 ? 6 : 11;
        for (t=0; t<time_bins; t++) {
            for (nibble=34; nibble>=0; nibble--) {
                digit = 0;
                for (bit=0; bit<4; bit++) {
                    channel = 4*nibble + bit;
                    if (sample != 2 && (7*channel+11*t+13*sample)%17 < limit)
                        digit += 2^bit;
                }
                printf "%x", digit;
            }
            printf "\n";
        }
    }' > spikes.hex
    timeout 120s vvp "$work_dir/direct.vvp" "+T=$length"
    count=0
    while read -r class_index actual; do
        [[ "$class_index" =~ ^[0-9]+$ ]] || continue
        expected_value=$((16#${reference_hex[20*sample+class_index]}))
        if (( expected_value >= 2147483648 )); then
            expected_value=$((expected_value - 4294967296))
        fi
        if (( actual != expected_value )); then
            echo "Direct-core sample $sample class $class_index differs from Python" >&2
            exit 1
        fi
        count=$((count+1))
    done < outputs.txt
    test "$count" -eq 20
    cp -- outputs.txt "outputs-direct-$sample.txt"
done
echo "PASS: default free-running core matches all three Python-golden samples"

if ! verilator --cc --exe --build --timing --coverage-line -j 1 \
    --top-module tb_sc_shd_axi_wrapper --Mdir "$work_dir/verilated" \
    -I"$repo_root/hdl" \
    "$repo_root/hdl/sc_axonal_delay.v" "$repo_root/hdl/sc_dense_int8_sparse.v" \
    "$repo_root/hdl/sc_vmin_lif_neuron.v" "$repo_root/hdl/sc_shd_top.v" \
    "$repo_root/hdl/sc_shd_axi_wrapper.v" "$repo_root/hdl/tb_sc_shd_axi_wrapper.v" \
    "$repo_root/hdl/shd_axi_sim_main.cpp" > verilator-build.txt 2>&1; then
    cat verilator-build.txt >&2
    exit 1
fi
timeout 120s "$work_dir/verilated/Vtb_sc_shd_axi_wrapper"
verilator_coverage --write-info coverage.info coverage.dat
# Enforce the AXI transport and logical-time controller coverage in CI.
# Existing cell arithmetic/LUT coverage is reported by the same LCOV file.
awk '
    /^SF:/ {
        source = $0;
        sub(/^.*\//, "", source);
        selected = source == "sc_shd_axi_wrapper.v" || source == "sc_shd_top.v";
    }
    /^DA:/ && selected {
        split(substr($0, 4), fields, ",");
        total[source]++;
        if (fields[2]+0 > 0) hit[source]++;
    }
    END {
        names[1] = "sc_shd_axi_wrapper.v";
        names[2] = "sc_shd_top.v";
        for (i=1; i<=2; i++) {
            name = names[i];
            printf "%s: %d/%d RTL lines covered\n", name, hit[name], total[name];
            if (!total[name] || 100*hit[name] < 95*total[name]) failed = 1;
        }
        exit failed;
    }
' coverage.info
