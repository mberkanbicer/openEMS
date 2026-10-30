//! Command-line interface for the openEMS FDTD solver.
//!
//! Mirrors the native `openEMS <FDTD_XML_FILE> [<options>...]` contract:
//! the first positional argument is the FDTD input file, the solver options
//! live in its `<FDTD>` element, and the command line only overrides the
//! execution-level settings (engine, threads, dumps, debug output).
//!
//! Exit codes follow the native entry point: `0` on success, `1` for a parse
//! or setup failure, and `255` (the native `-1`) when no input file is given.

use clap::{ArgAction, Parser};
use openems_core::openems::OpenEMS;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = " Usage: openEMS <FDTD_XML_FILE> [<options>...]";

#[derive(Parser, Debug)]
#[command(
    name = "openEMS",
    disable_help_flag = true,
    disable_version_flag = true,
    disable_help_subcommand = true
)]
struct Args {
    /// FDTD XML input file (native `<FDTD>` + `<ContinuousStructure>`).
    #[arg(value_name = "FDTD_XML_FILE")]
    xml: Option<PathBuf>,

    /// Show this help message and exit.
    #[arg(short = 'h', long = "help", action = ArgAction::SetTrue)]
    help: bool,

    /// Disable all field dumps for a faster simulation.
    #[arg(long = "disable-dumps")]
    disable_dumps: bool,

    /// Dump the material distribution to `material_dump.vtk`.
    #[arg(long = "debug-material")]
    debug_material: bool,

    /// Dump the metal distribution to `PEC_dump.vtk`.
    #[arg(long = "debug-PEC")]
    debug_pec: bool,

    /// Dump the operator to `operator_dump.vtk`.
    #[arg(long = "debug-operator")]
    debug_operator: bool,

    /// Dump the box list to `box_dump_0.vtk`.
    #[arg(long = "debug-boxes")]
    debug_boxes: bool,

    /// Write the CSX geometry to `debugCSX.xml`.
    #[arg(long = "debug-CSX")]
    debug_csx: bool,

    /// Engine selection (accepted for compatibility; the Rust kernel is used).
    #[arg(long = "engine", default_value = "fastest", value_name = "TYPE")]
    engine: String,

    /// Force the number of worker threads (0 = automatic).
    #[arg(long = "numThreads", alias = "num-threads", default_value_t = 0, value_name = "N")]
    num_threads: i32,

    /// Only run preprocessing; do not simulate.
    #[arg(long = "no-simulation")]
    no_simulation: bool,

    /// Write `openEMS_run_stats.txt` and `openEMS_stats.txt`.
    #[arg(long = "dump-statistics")]
    dump_statistics: bool,

    /// Evaluate the end criterion on a fixed Nyquist schedule.
    #[arg(long = "exact-endcriteria")]
    exact_end_criteria: bool,

    /// Show probe discretization information.
    #[arg(long = "showProbeDiscretization", action = ArgAction::Count)]
    show_probe_discretization: u8,

    /// Dump all fields using the native field components.
    #[arg(long = "nativeFieldDumps", action = ArgAction::Count)]
    native_field_dumps: u8,

    /// Dump all fields using the legacy HDF5 layout (repeatable, matching the
    /// native boolean switch).
    #[arg(long = "legacyHDF5Dumps", action = ArgAction::Count)]
    legacy_hdf5_dumps: u8,

    /// Verbose level (repeatable: -v, -vv, -vvv).
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count)]
    verbose: u8,
}

fn welcome() {
    println!("--------------------------------------------------------------------------------");
    println!(" openEMS - Electromagnetic Field Solver (Rust)");
    println!(" Version: {} (Rust migration)", env!("CARGO_PKG_VERSION"));
    println!("--------------------------------------------------------------------------------");
}

fn show_options() {
    println!("{USAGE}");
    println!();
    println!("Options:");
    println!("  -h [ --help ]                 Show this help message and exit");
    println!("  --disable-dumps               Disable all field dumps for faster simulation");
    println!("  --debug-material              Dump material distribution to 'material_dump.vtk'");
    println!("  --debug-PEC                   Dump metal distribution to 'PEC_dump.vtk'");
    println!("  --debug-operator              Dump operator to 'operator_dump.vtk'");
    println!("  --debug-boxes                 Dump boxes to 'box_dump_0.vtk'");
    println!("  --debug-CSX                   Write CSX geometry to 'debugCSX.xml'");
    println!("  --engine arg                  Engine type: fastest, basic, sse, sse-compressed,");
    println!("                                multithreaded (default: fastest)");
    println!("  --numThreads arg              Force the number of worker threads (0 = automatic)");
    println!("  --no-simulation               Only run preprocessing; do not simulate");
    println!("  --dump-statistics             Write 'openEMS_run_stats.txt' and 'openEMS_stats.txt'");
    println!("  --exact-endcriteria           Evaluate the end criterion on a fixed schedule");
    println!("  --showProbeDiscretization     Show probe discretization information");
    println!("  --nativeFieldDumps            Dump all fields using the native components");
    println!("  --legacyHDF5Dumps             Dump all fields using the legacy HDF5 layout");
    println!("  -v [ --verbose ] arg          Verbose level (1-3), also -v, -vv, -vvv");
}

fn main() -> ExitCode {
    welcome();

    let args = match Args::try_parse() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(1);
        }
    };

    if args.help {
        show_options();
        return ExitCode::SUCCESS;
    }
    let Some(xml_path) = args.xml.as_ref() else {
        show_options();
        return ExitCode::from(255);
    };

    let xml = match std::fs::read_to_string(xml_path) {
        Ok(xml) => xml,
        Err(error) => {
            eprintln!("openEMS - could not read '{}': {error}", xml_path.display());
            return ExitCode::from(1);
        }
    };

    let mut sim = OpenEMS::new();
    sim.set_csx_xml(xml);
    sim.set_verbose_level(args.verbose as i32);
    if let Err(error) = sim.set_engine(&args.engine) {
        eprintln!("openEMS - {error}");
        return ExitCode::from(1);
    }
    sim.set_enable_dumps(!args.disable_dumps);
    sim.set_legacy_hdf5_dumps(args.legacy_hdf5_dumps > 0);
    sim.set_native_field_dumps(args.native_field_dumps > 0);
    sim.set_no_simulation(args.no_simulation);
    sim.set_dump_statistics(args.dump_statistics);
    sim.set_exact_end_criteria(args.exact_end_criteria);
    sim.set_debug_dumps(
        args.debug_material,
        args.debug_operator,
        args.debug_boxes,
        args.debug_pec,
        args.debug_csx,
    );

    // The native `fastest`/`multithreaded` engines use all available cores;
    // `basic` keeps the serial kernel. An explicit --numThreads wins.
    let available = std::thread::available_parallelism()
        .map(|value| value.get() as i32)
        .unwrap_or(1);
    let threads = match args.engine.to_ascii_lowercase().as_str() {
        "basic" => 1,
        "sse" | "sse-compressed" => {
            eprintln!(
                "openEMS - the SSE engine is not implemented; using the basic Rust kernel"
            );
            1
        }
        _ => available,
    };
    sim.set_number_of_threads(if args.num_threads > 0 {
        args.num_threads
    } else {
        threads
    });

    if args.show_probe_discretization > 0 {
        if let Some(array) = sim.processing_array.as_ref() {
            println!("openEMS - probe process intervals: {:?}", array.process_intervals());
        } else {
            println!("openEMS - no probes registered");
        }
    }
    if args.native_field_dumps > 0 {
        // Native `Global::optionDesc` prints this on the switch.
        println!("openEMS - dumping all fields using the native field components");
    }

    if !sim.setup_fdtd() {
        eprintln!("openEMS - ParseFDTDSetup failed.");
        return ExitCode::from(1);
    }
    if args.no_simulation {
        return ExitCode::SUCCESS;
    }
    if !sim.run_fdtd() {
        eprintln!("openEMS - RunFDTD failed.");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}
