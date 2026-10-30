//! Native CLI contract tests.
//!
//! The installed binary is named `openEMS` and takes
//! `openEMS <FDTD_XML_FILE> [<options>...]`. These tests drive the built
//! `openems-cli` binary exactly as a user or the Octave/Python layers would.

use std::path::PathBuf;
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_openems-cli"))
}

fn case_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("openems_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create case dir");
    dir
}

fn write_case(dir: &PathBuf, nr_ts: u32) -> PathBuf {
    let lines = (0..21)
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let xml = format!(
        r#"<openEMS>
            <FDTD NumberOfTimesteps="{nr_ts}" EndCriteria="1e-6"/>
            <ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="1e-3">
                    <XLines>{lines}</XLines>
                    <YLines>{lines}</YLines>
                    <ZLines>{lines}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Number="0" Enabled="1" Frequency="2e9" Delay="0" Type="0" Excite="0,1,0">
                        <Primitives><Box Priority="0"><P1 X="10" Y="10" Z="10"/><P2 X="11" Y="11" Z="11"/></Box></Primitives>
                    </Excitation>
                </Properties>
            </ContinuousStructure>
        </openEMS>"#
    );
    let path = dir.join("case.xml");
    std::fs::write(&path, xml).expect("write case");
    path
}

#[test]
fn no_arguments_shows_usage_and_exits_255() {
    let output = Command::new(binary()).output().expect("run");
    assert_eq!(output.status.code(), Some(255));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage: openEMS <FDTD_XML_FILE>"));
}

#[test]
fn help_exits_zero_and_lists_the_native_options() {
    let output = Command::new(binary())
        .arg("--help")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    for option in [
        "--disable-dumps",
        "--debug-material",
        "--debug-PEC",
        "--engine",
        "--numThreads",
        "--no-simulation",
        "--dump-statistics",
    ] {
        assert!(stdout.contains(option), "missing {option} in help");
    }
}

#[test]
fn the_fdtd_element_sets_the_timestep_budget() {
    let dir = case_dir("nrts");
    let case = write_case(&dir, 6);
    let output = Command::new(binary())
        .current_dir(&dir)
        .arg(&case)
        .arg("--dump-statistics")
        .arg("--numThreads=2")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    let stats = std::fs::read_to_string(dir.join("openEMS_run_stats.txt")).expect("run stats");
    // The FDTD element sets the budget; the end criteria may stop the run
    // earlier on its Nyquist schedule, never later.
    let timesteps: u32 = stats
        .lines()
        .find_map(|line| line.strip_prefix("timesteps = "))
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or_else(|| panic!("unexpected run stats: {stats}"));
    assert!(
        (1..=6).contains(&timesteps),
        "timestep budget not honoured: {timesteps}"
    );
    let setup = std::fs::read_to_string(dir.join("openEMS_stats.txt")).expect("setup stats");
    assert!(setup.contains("cells = "));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn no_simulation_stops_before_the_run() {
    let dir = case_dir("nosim");
    let case = write_case(&dir, 6);
    let output = Command::new(binary())
        .current_dir(&dir)
        .arg(&case)
        .arg("--no-simulation")
        .arg("--dump-statistics")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    assert!(!dir.join("openEMS_run_stats.txt").exists());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn disable_dumps_and_debug_switches_are_accepted() {
    let dir = case_dir("debug");
    let case = write_case(&dir, 4);
    let output = Command::new(binary())
        .current_dir(&dir)
        .arg(&case)
        .arg("--disable-dumps")
        .arg("--debug-CSX")
        .arg("--debug-material")
        .arg("--debug-PEC")
        .arg("--debug-operator")
        .arg("--debug-boxes")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    for file in [
        "debugCSX.xml",
        "material_dump.vtk",
        "PEC_dump.vtp",
        "operator_dump.vtk",
        "box_dump_0.vtk",
    ] {
        assert!(dir.join(file).exists(), "{file} was not written");
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn an_unknown_option_exits_one() {
    let output = Command::new(binary())
        .arg("--definitely-not-an-option")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));
}
