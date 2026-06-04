// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust benchmark measurement context helpers

use std::fs;
use std::path::Path;
use std::process::Command;

pub fn load_average() -> String {
    fs::read_to_string("/proc/loadavg")
        .map(|content| content.trim().to_string())
        .unwrap_or_else(|_| "unavailable".to_string())
}

pub fn rust_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_string())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn read_proc_field(path: &str, prefix: &str) -> String {
    fs::read_to_string(path)
        .ok()
        .and_then(|content| {
            content.lines().find_map(|line| {
                line.strip_prefix(prefix)
                    .map(|value| value.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unavailable".to_string())
}

fn cpu_affinity() -> String {
    read_proc_field("/proc/self/status", "Cpus_allowed_list:")
}

fn cgroup_relative_path() -> Option<String> {
    fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|content| {
            content.lines().find_map(|line| {
                line.strip_prefix("0::")
                    .map(|value| value.trim_start_matches('/').to_string())
            })
        })
}

fn cgroup_effective_cpuset() -> String {
    let Some(relative) = cgroup_relative_path() else {
        return "unavailable".to_string();
    };
    let path = Path::new("/sys/fs/cgroup")
        .join(relative)
        .join("cpuset.cpus.effective");
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unavailable".to_string())
}

fn cpu_indices() -> Vec<u32> {
    let mut indices = fs::read_dir("/sys/devices/system/cpu")
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter_map(|name| name.strip_prefix("cpu")?.parse::<u32>().ok())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices
}

fn read_trimmed(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn cpu_governor() -> String {
    let mut governors = cpu_indices()
        .into_iter()
        .filter_map(|cpu| {
            read_trimmed(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            ))
        })
        .collect::<Vec<_>>();
    governors.sort();
    governors.dedup();
    if governors.is_empty() {
        "unavailable".to_string()
    } else {
        governors.join(",")
    }
}

fn cpu_frequency_mhz() -> String {
    let mut values = cpu_indices()
        .into_iter()
        .filter_map(|cpu| {
            read_trimmed(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_cur_freq"
            ))
            .or_else(|| {
                read_trimmed(&format!(
                    "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
                ))
            })
        })
        .filter_map(|raw| raw.parse::<f64>().ok())
        .map(|khz| khz / 1000.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return "unavailable".to_string();
    }
    values.sort_by(|a, b| a.total_cmp(b));
    format!(
        "min={:.3},max={:.3},sampled_cpu_count={}",
        values[0],
        values[values.len() - 1],
        values.len()
    )
}

pub fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            value if value.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", value as u32));
            }
            value => escaped.push(value),
        }
    }
    escaped.push('"');
    escaped
}

pub fn measurement_context_json(load_average_before: &str) -> String {
    format!(
        concat!(
            "{{\n",
            "    \"host_load\": \"workstation under concurrent load during capture\",\n",
            "    \"cpu_isolation\": \"runtime cpuset shield when cgroup_effective_cpuset matches benchmark CPUs; kernel-reserved isolated cores were not detected on this workstation\",\n",
            "    \"cpu_affinity\": {cpu_affinity},\n",
            "    \"cgroup_effective_cpuset\": {cgroup_effective_cpuset},\n",
            "    \"runtime_cpuset_shield_claimed\": {runtime_cpuset_shield_claimed},\n",
            "    \"load_average_before\": {load_average_before},\n",
            "    \"load_average_after\": {load_average_after},\n",
            "    \"cpu_governor\": {cpu_governor},\n",
            "    \"cpu_frequency_mhz\": {cpu_frequency_mhz},\n",
            "    \"concurrent_load_status\": \"non-exclusive workstation run; other heavy jobs may have been active\",\n",
            "    \"timing_interpretation\": \"use timing medians as local regression context only, not final throughput claims\",\n",
            "    \"production_rerun_requirement\": \"rerun on reserved isolated cores with recorded affinity, governor, frequency, versions, and host-load evidence before publishing performance claims\"\n",
            "  }}"
        ),
        cpu_affinity = json_string(&cpu_affinity()),
        cgroup_effective_cpuset = json_string(&cgroup_effective_cpuset()),
        runtime_cpuset_shield_claimed = cgroup_effective_cpuset() == cpu_affinity()
            && cgroup_effective_cpuset() != "unavailable",
        load_average_before = json_string(load_average_before),
        load_average_after = json_string(&load_average()),
        cpu_governor = json_string(&cpu_governor()),
        cpu_frequency_mhz = json_string(&cpu_frequency_mhz()),
    )
}
