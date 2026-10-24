use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_HDF5");
    println!("cargo:rerun-if-env-changed=HDF5_LIB_DIR");
    if env::var_os("CARGO_FEATURE_HDF5").is_none() {
        return;
    }

    if let Some(directory) = env::var_os("HDF5_LIB_DIR") {
        println!(
            "cargo:rustc-link-search=native={}",
            directory.to_string_lossy()
        );
        return;
    }

    for package in ["hdf5-serial", "hdf5"] {
        let Ok(output) = Command::new("pkg-config")
            .args(["--libs-only-L", package])
            .output()
        else {
            break;
        };
        if output.status.success() {
            for flag in String::from_utf8_lossy(&output.stdout).split_whitespace() {
                if let Some(directory) = flag.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={directory}");
                }
            }
            return;
        }
    }

    panic!("the hdf5 feature requires HDF5 development libraries; install pkg-config metadata or set HDF5_LIB_DIR");
}
