//! `sar_calc` command-line tool: SAR averaging and calculation.
//!
//! Mirrors `tools/sar_calc.cpp`: reads the raw SAR dump of an openEMS run,
//! calculates the local or mass-averaged SAR and writes the result file.
//!
//! Exit codes: `0` on success, `1` for parse errors or a failed calculation
//! (the native `-1` result is reported as a failure too).

use std::process::ExitCode;

use openems_core::sar_hdf5::SarRunner;

const BANNER: &str = "\
 ---------------------------------------------------------------------- 
 | SAR calculation for openEMS 
 | (C) 2012-2026 Thorsten Liebig <thorsten.liebig@gmx.de>  GPL license
 ---------------------------------------------------------------------- ";

const USAGE: &str = "\
Usage: sar_calc -i <input.h5> -o <output.h5> [options]

Options:
  -h, --help                        print usage message
  -i, --input <file>                pathname to input hdf5 file (required)
  -o, --output <file>               pathname for output hdf5 file (required)
      --method <name>               SAR method: IEEE_C95_3, IEEE_62704, SIMPLE
  -m, --mass <grams>                averaging mass in g
  -a, --autorange <dB>              limit the calculation to cells within N dB
                                    of the peak local SAR (> 0)
  -n, --numThreads <count>          number of threads
  -v, --verbose                     verbose
  -p, --progress                    show progress
  -e, --export_cube_stats           export cube statistics
      --legacyHDF5Dumps             legacy HDF5 file layout for Octave/MATLAB";

fn main() -> ExitCode {
    println!("{BANNER}");

    match run(std::env::args().skip(1).collect()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Vec<String>) -> Result<ExitCode, String> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut method = "SIMPLE".to_string();
    let mut mass = 0.0f64;
    let mut autorange = 0.0f64;
    let mut verbose = false;
    let mut _progress = false;
    let mut cube_stats = false;
    let mut legacy = false;
    let mut index = 0usize;

    let mut next_value = |index: &mut usize| -> Result<String, String> {
        *index += 1;
        arguments
            .get(*index)
            .cloned()
            .ok_or_else(|| format!("missing value for '{}'", arguments[*index - 1]))
    };

    while index < arguments.len() {
        let argument = arguments[index].clone();
        match argument.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            "-i" | "--input" => input = Some(next_value(&mut index)?),
            "-o" | "--output" => output = Some(next_value(&mut index)?),
            "--method" => method = next_value(&mut index)?,
            "-m" | "--mass" => {
                mass = next_value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid value for '--mass'".to_string())?
            }
            "-a" | "--autorange" => {
                autorange = next_value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid value for '--autorange'".to_string())?
            }
            "-n" | "--numThreads" => {
                let _ = next_value(&mut index)?;
            }
            "-v" | "--verbose" => verbose = true,
            "-p" | "--progress" => _progress = true,
            "-e" | "--export_cube_stats" => cube_stats = true,
            "--legacyHDF5Dumps" => legacy = true,
            other => return Err(format!("unrecognised option '{other}'")),
        }
        index += 1;
    }

    let input = input.ok_or_else(|| "the option '--input' is required".to_string())?;
    let output = output.ok_or_else(|| "the option '--output' is required".to_string())?;

    let mut runner = SarRunner::new();
    runner.debug_level = i32::from(verbose);
    runner.averaging_mass = mass / 1000.0;
    runner.auto_range = autorange;
    runner.record_cube_stats = cube_stats;
    if !runner.set_averaging_method(&method) {
        eprintln!("SetAveragingMethod: Error, \"{method}\" is an unknown averaging method...");
        return Err("unknown averaging method".to_string());
    }
    let _ = legacy;

    let ok = runner
        .calc_from_hdf5(&input, &output)
        .map_err(|error| format!("SAR calculation failed: {error}"))?;
    if !ok {
        return Err("SAR calculation failed".to_string());
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_input_and_output() {
        assert!(run(vec!["-i".to_string(), "in.h5".to_string()]).is_err());
        assert!(run(vec!["-o".to_string(), "out.h5".to_string()]).is_err());
        assert!(run(vec!["--help".to_string()]).unwrap() == ExitCode::SUCCESS);
    }

    #[test]
    fn rejects_unknown_methods() {
        assert!(run(vec![
            "-i".to_string(),
            "in.h5".to_string(),
            "-o".to_string(),
            "out.h5".to_string(),
            "--method".to_string(),
            "nope".to_string(),
        ])
        .is_err());
    }
}
