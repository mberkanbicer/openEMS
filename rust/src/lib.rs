//! Rust foundations for the openEMS FDTD solver.
//!
//! This crate mirrors the C++ openEMS architecture:
//! - `constants` — FDTD_FLOAT type and physical constants
//! - `array_nijk` — 4-D field storage (component, i, j, k)
//! - `operator_base` — mesh, disc lines, boundary conditions, material storage
//! - `operator` — equivalent-circuit operator, material averaging, PEC
//! - `engine` — time-stepping loop, voltage/current updates
//! - `extensions` — physics extensions (PML, Mur, excitation, dispersive, etc.)
//! - `excitation` — time-domain source signals
//! - `openems` — main solver orchestrator (ParseFDTDSetup, SetupFDTD, RunFDTD)
//! - `processing` — post-processing (probes, dumps, SAR, NF2FF interface)
//! - `hdf5` — HDF5 field dump writer/reader (native-compatible layout)

pub mod array_nijk;
pub mod conducting_sheet;
pub mod constants;
pub mod engine;
pub mod excitation;
pub mod extensions;
pub mod hdf5;
pub mod openems;
pub mod operator;
pub mod operator_base;
pub mod operator_extension;
pub mod processing;
pub mod processing_integral;

// Re-export the main solver and boundary condition types at the crate root
// so dependents can use `openems_core::{OpenEMS, BC_PEC, ...}`.
pub use openems::{OpenEMS, BC_PEC, BC_PMC, BC_MUR, BC_PML};
