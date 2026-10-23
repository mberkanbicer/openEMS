//! Command-line interface for the openEMS FDTD solver.
//! Mirrors the C++ `openEMS` application entry point.

use clap::Parser;
use openems_core::openems::OpenEMS;
use openems_core::processing::FieldProbe;
use std::fs;
use std::path::PathBuf;

/// openEMS FDTD solver — command-line interface.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to CSXCAD XML geometry file.
    #[arg(short = 'c', long = "csx")]
    csx: Option<PathBuf>,

    /// Number of time steps to run.
    #[arg(short = 't', long = "timesteps", default_value = "100")]
    timesteps: u32,

    /// End criteria (dB).
    #[arg(long = "end-crit", default_value = "1e-6")]
    end_crit: f64,

    /// Gaussian pulse center frequency (Hz).
    #[arg(long = "f0", default_value = "1e9")]
    f0: f64,

    /// Gaussian pulse cutoff frequency (Hz).
    #[arg(long = "fc", default_value = "5e8")]
    fc: f64,

    /// Boundary condition type for all sides (0=PEC,1=PMC,2=MUR,3=PML).
    #[arg(long = "bc", default_value = "0")]
    bc: i32,

    /// PML size (cells) for all sides.
    #[arg(long = "pml", default_value = "0")]
    pml: u32,

    /// Add a field probe at X,Y,Z (meters), repeatable.
    #[arg(long = "probe", value_delimiter = ',')]
    probe: Option<Vec<f64>>,

    /// Enable verbose output.
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        println!("[openems-cli] Arguments: {:?}", args);
    }

    let mut sim = OpenEMS::new();
    sim.set_number_of_time_steps(args.timesteps);
    sim.set_end_criteria(args.end_crit);
    sim.set_gauss_excite(args.f0, args.fc);

    // Apply boundary condition to all 6 sides.
    for side in 0..6 {
        sim.set_bc_type(side, args.bc);
    }
    for side in 0..6 {
        sim.set_pml_size(side, args.pml);
    }

    // Load CSXCAD geometry if provided.
    if let Some(csx_path) = &args.csx {
        let xml = match fs::read_to_string(csx_path) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[openems-cli] Failed to read CSX file '{}': {}", csx_path.display(), e);
                std::process::exit(1);
            }
        };
        sim.set_csx_xml(xml);
    }

    // Add field probes if specified.
    if let Some(probes) = &args.probe {
        for p in probes.chunks(3) {
            if p.len() == 3 {
                let name = format!("probe_{}_{}_{}", p[0], p[1], p[2]);
                sim.add_processing(Box::new(FieldProbe::new(&name, [p[0], p[1], p[2]])));
            }
        }
    }

    println!("[openems-cli] Setting up FDTD simulation...");
    if !sim.setup_fdtd() {
        eprintln!("[openems-cli] Setup failed.");
        std::process::exit(1);
    }

    println!("[openems-cli] Running FDTD simulation ({} timesteps)...", args.timesteps);
    if !sim.run_fdtd() {
        eprintln!("[openems-cli] Run failed.");
        std::process::exit(1);
    }

    println!("[openems-cli] Simulation complete.");
}
