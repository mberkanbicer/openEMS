//! Physical constants and type aliases used throughout the solver.
//! Mirrors `tools/constants.h` in the C++ codebase.

/// FDTD field storage type — matches the C++ `FDTD_FLOAT` which is `float`.
#[allow(non_camel_case_types)]
pub type FDTD_FLOAT = f32;

/// Speed of light in vacuum (m/s).
pub const C0: f64 = 299792458.0;

/// Vacuum permittivity (F/m).
pub const EPS0: f64 = 8.8541878128e-12;

/// Vacuum permeability (H/m).
pub const MU0: f64 = 1.25663706212e-6;

/// Pi constant.
pub const PI: f64 = std::f64::consts::PI;

/// Coordinate system identifiers, mirroring the C++ `CoordinateSystem` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoordinateSystem {
    #[default]
    Undefined = 0,
    Cartesian = 1,
    Cylindrical = 2,
}
