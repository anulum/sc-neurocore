// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Host-derived time budgets for the engine's performance tests

//! Time budgets for the engine's performance tests, derived from the host that runs them.
//!
//! A performance test used to assert a fixed number of milliseconds. Such a number holds on the machine it was
//! written on and fails on a slower or busier one. Here the budget of a test is
//!
//! `margin * ratio * calibration`
//!
//! - `calibration` is the time this host takes, in this build, for a fixed reference workload. It is measured in the
//!   test's own thread on the same clock as the test, five times just before the work and five times just after.
//!   The larger of the two medians is used, so a host that slows down during the work is not held to the faster
//!   figure.
//! - `ratio` is the test's work over the calibration, measured on the reference hosts. It is read from
//!   `engine/test_host_profiles.toml`, one value for builds with debug assertions (`cargo test`) and one for builds
//!   without them (`cargo test --release`).
//! - `margin` is the policy value in the same file.
//!
//! The clock is the thread's CPU time where the platform has one (`CLOCK_THREAD_CPUTIME_ID` on Unix), so time the
//! thread spends waiting for a core is not counted. Elsewhere it is the wall clock. A failure names the clock, both
//! calibrations, the ratio, the margin and the load average.
//!
//! With the environment variable `SCN_ENGINE_BUDGET_RECORD` set to a file path, each call appends its measurement to
//! that file as one tab-separated line, and a test without a ratio yet is recorded instead of failing. The reference
//! ratios are computed from such records; `engine/test_host_profiles.toml` says how.

#![deny(missing_docs)]

use std::hint::black_box;
use std::io::Write as _;
use std::sync::OnceLock;
use std::time::Instant;

/// The reference profiles, compiled into the test binary from the file that holds them.
const PROFILES_TEXT: &str = include_str!("../../test_host_profiles.toml");

/// The only schema of `engine/test_host_profiles.toml` this module reads.
const SCHEMA_VERSION: i64 = 1;

/// The top-level keys and tables the file may hold.
const TOP_LEVEL_KEYS: [&str; 5] = [
    "schema_version",
    "margin",
    "reference_hosts",
    "measurement",
    "ratio",
];

/// How many calibration runs make one median.
const CALIBRATION_RUNS: usize = 5;

/// How many steps the reference workload takes; about a millisecond on the reference hosts.
const REFERENCE_STEPS: u32 = 20_000;

/// The environment variable that turns on recording.
const RECORD_VARIABLE: &str = "SCN_ENGINE_BUDGET_RECORD";

/// The clock a measurement was taken on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Clock {
    /// The CPU time of the calling thread.
    ThreadCpu,
    /// The monotonic wall clock, where the platform has no thread CPU clock.
    Wall,
}

impl Clock {
    /// The name the record and the failure message use.
    fn name(self) -> &'static str {
        match self {
            Clock::ThreadCpu => "thread-cpu",
            Clock::Wall => "wall",
        }
    }
}

/// One test's reference ratios.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ratio {
    /// The ratio for builds with debug assertions.
    pub(crate) debug: f64,
    /// The ratio for builds without debug assertions.
    pub(crate) release: f64,
}

impl Ratio {
    /// The ratio for the build this code was compiled in.
    fn for_this_build(self) -> f64 {
        if cfg!(debug_assertions) {
            self.debug
        } else {
            self.release
        }
    }
}

/// The parts of `engine/test_host_profiles.toml` a budget needs.
#[derive(Debug, PartialEq)]
pub(crate) struct Profiles {
    /// The policy margin every budget is multiplied by.
    pub(crate) margin: f64,
    /// Each test's ratios, by its name as `cargo test` prints it.
    pub(crate) ratios: Vec<(String, Ratio)>,
}

impl Profiles {
    /// The ratios of the test called `key`, if the file holds them.
    fn ratio(&self, key: &str) -> Option<Ratio> {
        self.ratios
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, ratio)| *ratio)
    }
}

/// Read a positive, finite number, integer or float, naming `what` in a refusal.
fn positive_number(item: &toml_edit::Item, what: &str) -> Result<f64, String> {
    let value = item
        .as_float()
        .or_else(|| item.as_integer().map(|integer| integer as f64))
        .ok_or_else(|| format!("{what} must be a number, got a {}", item.type_name()))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{what} must be positive and finite, got {value}"));
    }
    Ok(value)
}

/// Read one entry of the `[ratio]` table: a table holding a `debug` and a `release` number, and nothing else.
fn parse_ratio(key: &str, item: &toml_edit::Item) -> Result<Ratio, String> {
    let table = item.as_table_like().ok_or_else(|| {
        format!("{key:?} must be set to {{ debug = <number>, release = <number> }}")
    })?;
    if let Some((other, _)) = table
        .iter()
        .find(|(name, _)| *name != "debug" && *name != "release")
    {
        return Err(format!(
            "{key:?} has an unknown field {other:?}; only debug and release are read"
        ));
    }
    let field = |name: &str| {
        table
            .get(name)
            .ok_or_else(|| format!("{key:?} must set both debug and release"))
            .and_then(|value| positive_number(value, &format!("{key:?} {name}")))
    };
    Ok(Ratio {
        debug: field("debug")?,
        release: field("release")?,
    })
}

/// Read the margin and the `[ratio]` table from the text of `engine/test_host_profiles.toml`.
///
/// The text is read as TOML by `toml_edit`. A budget needs only the margin and the ratios; the reference hosts and
/// the measurements are left to the repository's Python test, which checks the whole file. Any other top-level key
/// is refused, so a per-test margin or another policy value cannot enter without a change here.
pub(crate) fn parse_profiles(text: &str) -> Result<Profiles, String> {
    let document: toml_edit::DocumentMut = text
        .parse()
        .map_err(|error| format!("the file is not valid TOML: {error}"))?;
    if let Some((other, _)) = document
        .iter()
        .find(|(name, _)| !TOP_LEVEL_KEYS.contains(name))
    {
        return Err(format!("unknown top-level key {other:?}"));
    }
    match document
        .get("schema_version")
        .and_then(toml_edit::Item::as_integer)
    {
        Some(SCHEMA_VERSION) => {}
        found => {
            return Err(format!(
                "schema_version must be {SCHEMA_VERSION}, got {found:?}"
            ))
        }
    }
    let margin = positive_number(
        document.get("margin").ok_or("the file sets no margin")?,
        "margin",
    )?;
    let table = document
        .get("ratio")
        .and_then(toml_edit::Item::as_table_like)
        .ok_or("the file has no [ratio] table")?;
    let ratios = table
        .iter()
        .map(|(key, item)| parse_ratio(key, item).map(|ratio| (key.to_owned(), ratio)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Profiles { margin, ratios })
}

/// The reference profiles of this build, read once.
fn profiles() -> &'static Profiles {
    static PROFILES: OnceLock<Profiles> = OnceLock::new();
    PROFILES.get_or_init(|| {
        parse_profiles(PROFILES_TEXT)
            .unwrap_or_else(|reason| panic!("engine/test_host_profiles.toml: {reason}"))
    })
}

/// The CPU time of the calling thread in nanoseconds, if the platform has the clock.
#[cfg(unix)]
fn thread_cpu_ns() -> Option<u64> {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid, writable `timespec` owned by this frame, and `CLOCK_THREAD_CPUTIME_ID` is a clock the
    // platform defines; `clock_gettime` writes only through the pointer it is given.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) } != 0 {
        return None;
    }
    let seconds = u64::try_from(now.tv_sec).ok()?;
    let nanos = u64::try_from(now.tv_nsec).ok()?;
    seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
}

/// The CPU time of the calling thread in nanoseconds, if the platform has the clock.
#[cfg(not(unix))]
fn thread_cpu_ns() -> Option<u64> {
    None
}

/// The clock this platform's budgets are measured on.
pub(crate) fn clock() -> Clock {
    if thread_cpu_ns().is_some() {
        Clock::ThreadCpu
    } else {
        Clock::Wall
    }
}

/// The nanoseconds `work` takes on `clock`.
pub(crate) fn time_ns(clock: Clock, work: impl FnOnce()) -> u64 {
    match clock {
        Clock::ThreadCpu => {
            let start =
                thread_cpu_ns().expect("the thread CPU clock was readable when it was chosen");
            work();
            let end =
                thread_cpu_ns().expect("the thread CPU clock was readable when it was chosen");
            end.saturating_sub(start)
        }
        Clock::Wall => {
            let start = Instant::now();
            work();
            u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
        }
    }
}

/// The fixed reference workload: dependent exponentials, each step kept from the optimiser.
fn reference_work() {
    let mut x = black_box(0.5_f64);
    for _ in 0..black_box(REFERENCE_STEPS) {
        x = black_box(0.5 * (-x).exp() + 0.25);
    }
    black_box(x);
}

/// The median of an odd number of timings.
fn median(mut timings: Vec<u64>) -> u64 {
    timings.sort_unstable();
    timings[timings.len() / 2]
}

/// The median time of the reference workload on `clock`, over five runs.
pub(crate) fn calibration_ns(clock: Clock) -> u64 {
    median(
        (0..CALIBRATION_RUNS)
            .map(|_| time_ns(clock, reference_work))
            .collect(),
    )
}

/// The test name `cargo test` prints, from the test's `module_path!()` and function name.
pub(crate) fn test_key(module: &str, name: &str) -> String {
    let path = module.split_once("::").map_or("", |(_, rest)| rest);
    if path.is_empty() {
        name.to_owned()
    } else {
        format!("{path}::{name}")
    }
}

/// One measurement of a test against its budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Measurement {
    /// The clock every figure was taken on.
    pub(crate) clock: Clock,
    /// The nanoseconds the test's work took.
    pub(crate) elapsed_ns: u64,
    /// The median calibration just before the work.
    pub(crate) calibration_before_ns: u64,
    /// The median calibration just after the work.
    pub(crate) calibration_after_ns: u64,
}

/// The first three fields of `/proc/loadavg`, or `unavailable` where the host has none.
fn load_average() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|text| {
            text.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|fields| !fields.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

/// The build profile's name in the record and the message.
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// Hold `measurement` of the test `key` to `margin * ratio * calibration`, with the larger calibration.
///
/// The refusal names the elapsed time, the budget and every term of it, the clock and `load`.
pub(crate) fn check(
    key: &str,
    measurement: Measurement,
    ratio: f64,
    margin: f64,
    load: &str,
) -> Result<(), String> {
    let calibration = measurement
        .calibration_before_ns
        .max(measurement.calibration_after_ns);
    let budget = margin * ratio * calibration as f64;
    if measurement.elapsed_ns as f64 <= budget {
        return Ok(());
    }
    Err(format!(
        "{key} took {} ns of {} time; its budget is {budget:.0} ns = margin {margin} x ratio {ratio} ({}) x \
         calibration {calibration} ns (the larger of {} ns before and {} ns after the work); load average {load}",
        measurement.elapsed_ns,
        measurement.clock.name(),
        build_profile(),
        measurement.calibration_before_ns,
        measurement.calibration_after_ns,
    ))
}

/// Append `measurement` of `key` to the record file, as one tab-separated line.
fn record(path: &str, key: &str, measurement: Measurement, load: &str) {
    let line = format!(
        "{key}\t{}\t{}\t{}\t{}\t{}\t{load}\n",
        build_profile(),
        measurement.clock.name(),
        measurement.elapsed_ns,
        measurement.calibration_before_ns,
        measurement.calibration_after_ns,
    );
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|error| panic!("{RECORD_VARIABLE}={path}: {error}"));
    file.write_all(line.as_bytes())
        .unwrap_or_else(|error| panic!("{RECORD_VARIABLE}={path}: {error}"));
}

/// Run `work` once and hold its time to the budget of the test `name` in the module `module`.
///
/// Call it as `assert_within(module_path!(), "<test function>", || { ... })`, with the test's whole timed workload in
/// the closure.
pub(crate) fn assert_within(module: &str, name: &str, work: impl FnOnce()) {
    let recording = std::env::var(RECORD_VARIABLE)
        .ok()
        .filter(|path| !path.is_empty());
    assert_within_profiles(
        profiles(),
        recording.as_deref(),
        &test_key(module, name),
        work,
    );
}

/// [`assert_within`] against `profiles`, recording to `recording` when it is given.
///
/// A test without a ratio is refused unless it is being recorded.
fn assert_within_profiles(
    profiles: &Profiles,
    recording: Option<&str>,
    key: &str,
    work: impl FnOnce(),
) {
    let clock = clock();
    let calibration_before_ns = calibration_ns(clock);
    let elapsed_ns = time_ns(clock, work);
    let calibration_after_ns = calibration_ns(clock);
    let measurement = Measurement {
        clock,
        elapsed_ns,
        calibration_before_ns,
        calibration_after_ns,
    };
    let load = load_average();
    if let Some(path) = recording {
        record(path, key, measurement, &load);
    }
    let Some(ratio) = profiles.ratio(key) else {
        assert!(
            recording.is_some(),
            "engine/test_host_profiles.toml holds no ratio for {key}; record one with {RECORD_VARIABLE}"
        );
        return;
    };
    if let Err(message) = check(
        key,
        measurement,
        ratio.for_this_build(),
        profiles.margin,
        &load,
    ) {
        panic!("{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = "schema_version = 1\nmargin = 5.0 # policy\n\n[reference_hosts.a]\ncpu = \"x\"\n\n[ratio]\n\
                        \"m::tests::a\" = { debug = 0.5, release = 0.25 }\n\"m::tests::b\" = { release = 2, debug = 3 }\n";

    fn refusal(text: &str) -> String {
        parse_profiles(text).expect_err("the text must be refused")
    }

    #[test]
    fn reads_the_margin_and_every_ratio_and_skips_other_tables() {
        let profiles = parse_profiles(GOOD).expect("the text is well formed");
        assert_eq!(profiles.margin, 5.0);
        assert_eq!(
            profiles.ratios,
            vec![
                (
                    "m::tests::a".to_owned(),
                    Ratio {
                        debug: 0.5,
                        release: 0.25
                    }
                ),
                (
                    "m::tests::b".to_owned(),
                    Ratio {
                        debug: 3.0,
                        release: 2.0
                    }
                ),
            ]
        );
        assert_eq!(
            profiles.ratio("m::tests::b"),
            Some(Ratio {
                debug: 3.0,
                release: 2.0
            })
        );
        assert_eq!(profiles.ratio("m::tests::c"), None);
    }

    #[test]
    fn refuses_a_file_without_its_schema_or_margin_or_ratios() {
        assert!(refusal("margin = 5.0\n").contains("schema_version must be 1, got None"));
        assert!(refusal("schema_version = 2\nmargin = 5.0\n").contains("got Some(2)"));
        assert!(refusal("schema_version = \"1\"\nmargin = 5.0\n").contains("got None"));
        assert_eq!(refusal("schema_version = 1\n"), "the file sets no margin");
        assert_eq!(
            refusal("schema_version = 1\nmargin = 5\n"),
            "the file has no [ratio] table"
        );
    }

    #[test]
    fn refuses_a_margin_that_is_not_a_positive_number() {
        for (value, expected) in [
            ("0", "positive"),
            ("-1", "positive"),
            ("inf", "finite"),
            ("nan", "finite"),
            ("\"five\"", "a number"),
        ] {
            let reason = refusal(&format!("schema_version = 1\nmargin = {value}\n"));
            assert!(reason.contains(expected), "{value}: {reason}");
        }
    }

    #[test]
    fn refuses_an_unknown_top_level_key_and_text_that_is_not_toml() {
        let reason = refusal("schema_version = 1\nmargin = 5\nslack = 2\n");
        assert!(
            reason.contains("unknown top-level key \"slack\""),
            "{reason}"
        );
        let reason = refusal("schema_version = 1\nmargin = 5\n[per_test_margin]\n");
        assert!(reason.contains("\"per_test_margin\""), "{reason}");
        assert!(refusal("schema_version = 1\nmargin\n").contains("not valid TOML"));
    }

    #[test]
    fn refuses_a_malformed_ratio() {
        let head = "schema_version = 1\nmargin = 5\n[ratio]\n";
        for (line, expected) in [
            ("\"m::a\" = 1", "must be set to"),
            ("\"m::a\" = { debug = 1 }", "both debug and release"),
            (
                "\"m::a\" = { debug = 1, release = 1, margin = 9 }",
                "unknown field \"margin\"",
            ),
            ("\"m::a\" = { debug = 0, release = 1 }", "positive"),
            ("\"m::a\" = { debug = \"1\", release = 1 }", "a number"),
            (
                "\"m::a\" = { debug = 1, debug = 2, release = 1 }",
                "not valid TOML",
            ),
            ("m::a = { debug = 1, release = 1 }", "not valid TOML"),
        ] {
            let reason = refusal(&format!("{head}{line}\n"));
            assert!(reason.contains(expected), "{line}: {reason}");
        }
        let twice = format!("{head}\"m::a\" = {{ debug = 1, release = 1 }}\n\"m::a\" = {{ debug = 2, release = 2 }}\n");
        assert!(refusal(&twice).contains("not valid TOML"));
    }

    #[test]
    fn a_ratio_may_be_a_table_of_its_own() {
        let text =
            "schema_version = 1\nmargin = 5\n[ratio.\"m::tests::a\"]\ndebug = 2\nrelease = 0.5\n";
        let profiles = parse_profiles(text).expect("a ratio written as a table is TOML too");
        assert_eq!(
            profiles.ratio("m::tests::a"),
            Some(Ratio {
                debug: 2.0,
                release: 0.5
            })
        );
    }

    #[test]
    fn the_shipped_profiles_parse() {
        let profiles =
            parse_profiles(PROFILES_TEXT).expect("engine/test_host_profiles.toml is well formed");
        assert!(!profiles.ratios.is_empty());
        assert_eq!(profiles, *super::profiles());
    }

    #[test]
    fn the_key_is_the_name_cargo_test_prints() {
        assert_eq!(
            test_key(
                "sc_neurocore_engine::neurons::misc::gap_junction::tests",
                "gap"
            ),
            "neurons::misc::gap_junction::tests::gap"
        );
        assert_eq!(
            test_key(module_path!(), "x"),
            "neurons::host_budget::tests::x"
        );
        assert_eq!(test_key("sc_neurocore_engine", "x"), "x");
    }

    fn measured(elapsed_ns: u64, before: u64, after: u64) -> Measurement {
        Measurement {
            clock: Clock::ThreadCpu,
            elapsed_ns,
            calibration_before_ns: before,
            calibration_after_ns: after,
        }
    }

    #[test]
    fn the_budget_uses_the_larger_calibration() {
        // margin 5 x ratio 2 x calibration 100 = 1000 ns, whichever side holds the 100.
        assert_eq!(
            check("t", measured(1_000, 100, 40), 2.0, 5.0, "0.0"),
            Ok(())
        );
        assert_eq!(
            check("t", measured(1_000, 40, 100), 2.0, 5.0, "0.0"),
            Ok(())
        );
        assert!(check("t", measured(1_001, 100, 40), 2.0, 5.0, "0.0").is_err());
        assert!(check("t", measured(1_001, 40, 100), 2.0, 5.0, "0.0").is_err());
    }

    #[test]
    fn a_refusal_names_every_term_of_the_budget() {
        let message = check(
            "m::tests::slow",
            measured(2_500, 100, 120),
            2.0,
            5.0,
            "3.1 2.0 1.5",
        )
        .expect_err("over budget");
        for part in [
            "m::tests::slow took 2500 ns of thread-cpu time",
            "its budget is 1200 ns = margin 5 x ratio 2",
            "calibration 120 ns (the larger of 100 ns before and 120 ns after the work)",
            "load average 3.1 2.0 1.5",
        ] {
            assert!(message.contains(part), "{part:?} missing from {message:?}");
        }
        assert!(message.contains(&format!("({})", build_profile())));
    }

    #[cfg(unix)]
    #[test]
    fn the_thread_cpu_clock_does_not_count_time_spent_waiting() {
        assert_eq!(clock(), Clock::ThreadCpu);
        let waited = time_ns(Clock::ThreadCpu, || {
            std::thread::sleep(std::time::Duration::from_millis(50))
        });
        assert!(
            waited < 10_000_000,
            "a 50 ms sleep took {waited} ns of thread CPU time"
        );
        let slept = time_ns(Clock::Wall, || {
            std::thread::sleep(std::time::Duration::from_millis(50))
        });
        assert!(
            slept >= 50_000_000,
            "a 50 ms sleep took {slept} ns of wall time"
        );
    }

    #[test]
    fn the_ratio_is_the_one_for_this_build() {
        let ratio = Ratio {
            debug: 3.0,
            release: 0.5,
        };
        let expected = if cfg!(debug_assertions) { 3.0 } else { 0.5 };
        assert_eq!(ratio.for_this_build(), expected);
        assert_eq!(
            build_profile(),
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
    }

    #[test]
    fn the_calibration_is_the_median_of_five_runs() {
        assert_eq!(median(vec![9, 1, 5, 7, 3]), 5);
        assert!(calibration_ns(clock()) > 0);
    }

    /// A record file of this process, named for `what`, removed first.
    fn record_path(what: &str) -> String {
        let path = std::env::temp_dir().join(format!(
            "scn_engine_budget_{what}_{}.tsv",
            std::process::id()
        ));
        let path = path
            .to_str()
            .expect("the temporary path is text")
            .to_owned();
        let _ = std::fs::remove_file(&path);
        path
    }

    /// The lines of a record file, which is then removed.
    fn take_record(path: &str) -> Vec<String> {
        let text = std::fs::read_to_string(path).expect("the record was written");
        std::fs::remove_file(path).expect("the record is removed");
        text.lines().map(str::to_owned).collect()
    }

    #[test]
    fn a_recording_appends_one_tab_separated_line() {
        let path = record_path("lines");
        record(&path, "m::tests::a", measured(7, 3, 4), "0.5 0.4 0.3");
        record(&path, "m::tests::b", measured(8, 5, 6), "0.5 0.4 0.3");
        let profile = build_profile();
        assert_eq!(
            take_record(&path),
            [
                format!("m::tests::a\t{profile}\tthread-cpu\t7\t3\t4\t0.5 0.4 0.3"),
                format!("m::tests::b\t{profile}\tthread-cpu\t8\t5\t6\t0.5 0.4 0.3"),
            ]
        );
    }

    fn one_ratio(key: &str, ratio: f64) -> Profiles {
        Profiles {
            margin: 5.0,
            ratios: vec![(
                key.to_owned(),
                Ratio {
                    debug: ratio,
                    release: ratio,
                },
            )],
        }
    }

    #[test]
    fn work_within_its_budget_passes() {
        // A ratio of 1000 allows 5000 calibrations; one reference run takes one.
        assert_within_profiles(
            &one_ratio("m::tests::a", 1_000.0),
            None,
            "m::tests::a",
            reference_work,
        );
    }

    #[test]
    #[should_panic(expected = "m::tests::a took")]
    fn work_over_its_budget_fails() {
        // A ratio of 0.01 allows a twentieth of a calibration; fifty reference runs take fifty.
        assert_within_profiles(&one_ratio("m::tests::a", 0.01), None, "m::tests::a", || {
            (0..50).for_each(|_| reference_work())
        });
    }

    #[test]
    #[should_panic(
        expected = "holds no ratio for m::tests::b; record one with SCN_ENGINE_BUDGET_RECORD"
    )]
    fn a_test_without_a_ratio_fails() {
        assert_within_profiles(
            &one_ratio("m::tests::a", 1.0),
            None,
            "m::tests::b",
            reference_work,
        );
    }

    #[test]
    fn a_test_without_a_ratio_is_recorded_when_recording() {
        let path = record_path("unrated");
        assert_within_profiles(
            &one_ratio("m::tests::a", 1.0),
            Some(&path),
            "m::tests::b",
            reference_work,
        );
        let lines = take_record(&path);
        assert_eq!(lines.len(), 1);
        let fields: Vec<&str> = lines[0].split('\t').collect();
        assert_eq!(
            fields[..3],
            ["m::tests::b", build_profile(), clock().name()]
        );
        assert_eq!(fields.len(), 7);
        assert!(
            fields[3..6]
                .iter()
                .all(|field| field.parse::<u64>().is_ok()),
            "{fields:?}"
        );
    }

    /// Removes its record file when the test ends, also when it ends in a panic.
    struct RemovedAtEnd(String);

    impl Drop for RemovedAtEnd {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    #[should_panic(expected = "m::tests::a took")]
    fn a_recorded_test_is_still_held_to_its_budget() {
        let path = RemovedAtEnd(record_path("held"));
        assert_within_profiles(
            &one_ratio("m::tests::a", 0.01),
            Some(&path.0),
            "m::tests::a",
            || (0..50).for_each(|_| reference_work()),
        );
    }
}
