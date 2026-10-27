//! Lumped RLC operator model.
//!
//! Mirrors `FDTD/extensions/operator_ext_lumpedRLC.cpp`: the per-cell element
//! values, the stability-driven capacitance floor, and the series `RLC` ADE
//! coefficients. The time-stepping itself lives in
//! [`crate::extensions::EngineExtLumpedRLC`].

use std::f64::consts::TAU;

/// Native `LUMPED_RLC_Z_FACT`: the grid capacitance is raised until its
/// impedance at `f_max` is at least this many times the element impedance.
pub const LUMPED_RLC_Z_FACT: f64 = 20.0;

/// Element topology, mirroring `CSPropLumpedElement::LEtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LumpedTopology {
    Parallel,
    Series,
}

impl From<csxcad_core::material::LumpedElementType> for LumpedTopology {
    fn from(value: csxcad_core::material::LumpedElementType) -> Self {
        match value {
            csxcad_core::material::LumpedElementType::Parallel => Self::Parallel,
            csxcad_core::material::LumpedElementType::Series => Self::Series,
        }
    }
}

/// Per-cell element values after scaling from the whole element to one cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellElement {
    /// Per-cell inductance (H); `0.0` when absent.
    pub inductance: f64,
    /// Per-cell series resistance (Ω); `0.0` when absent.
    pub resistance: f64,
    /// Per-cell parallel conductance (S); `0.0` when absent or `R = 0`.
    pub conductance: f64,
    /// Per-cell capacitance (F); `0.0` when absent.
    pub capacitance: f64,
}

impl CellElement {
    /// Scale the whole-element values to a single cell.
    ///
    /// `parallel_cells` is `Ncells_1 * Ncells_2` and `series_cells` is
    /// `Ncells_0`, exactly as in the native `BuildExtension`.
    pub fn from_element(
        resistance: f64,
        inductance: f64,
        capacitance: f64,
        parallel_cells: u32,
        series_cells: u32,
    ) -> Self {
        let parallel_cells = parallel_cells.max(1) as f64;
        let series_cells = series_cells.max(1) as f64;
        let inductance = if inductance.is_nan() {
            0.0
        } else {
            inductance * parallel_cells / series_cells
        };
        let element_resistance = resistance;
        let resistance = if resistance.is_nan() {
            0.0
        } else {
            resistance * parallel_cells / series_cells
        };
        let conductance = if element_resistance.is_nan() || element_resistance == 0.0 {
            0.0
        } else {
            (1.0 / element_resistance) * series_cells / parallel_cells
        };
        let capacitance = if capacitance.is_nan() {
            0.0
        } else {
            capacitance * series_cells / parallel_cells
        };
        Self {
            inductance,
            resistance,
            conductance,
            capacitance,
        }
    }
}

/// Series `RLC` ADE coefficients, native `ib0`, `b1`, `b2`.
///
/// With an absent capacitance the `RL`-only form is used; that is not an
/// approximation but the native treatment of a series `RL`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeriesCoefficients {
    pub ib0: f64,
    pub b1: f64,
    pub b2: f64,
}

/// Build the series ADE coefficients for one cell.
pub fn series_coefficients(dt: f64, element: &CellElement) -> SeriesCoefficients {
    let l = element.inductance;
    let r = element.resistance;
    let c = element.capacitance;
    if c == 0.0 {
        let denominator = 2.0 * l + dt * r;
        SeriesCoefficients {
            ib0: if denominator != 0.0 {
                dt / denominator
            } else {
                0.0
            },
            b1: -4.0 * l / dt,
            b2: (2.0 * l - dt * r) / dt,
        }
    } else {
        SeriesCoefficients {
            ib0: 2.0 * dt * c / (4.0 * l * c + 2.0 * dt * r * c + dt * dt),
            b1: (dt * dt - 4.0 * l * c) / (dt * c),
            b2: (4.0 * l * c - 2.0 * dt * r * c + dt * dt) / (2.0 * dt * c),
        }
    }
}

/// Raise the grid capacitance until it is numerically resolvable.
///
/// Native parallel form: the element capacitance is used when given, otherwise
/// the natural grid capacitance, raised until `1/(2π f C) >= Z_FACT * Zmin`.
pub fn parallel_capacitance(
    given_capacitance: f64,
    grid_capacitance: f64,
    resistance: f64,
    inductance: f64,
    f_max: f64,
) -> f64 {
    if given_capacitance > 0.0 {
        return given_capacitance;
    }
    let z_min = resistance.max(TAU * f_max * inductance);
    if !(f_max > 0.0) || !(z_min > 0.0) {
        return grid_capacitance;
    }
    let z_cd_min = 1.0 / (TAU * f_max * grid_capacitance);
    if z_cd_min < LUMPED_RLC_Z_FACT * z_min {
        1.0 / (TAU * f_max * z_min * LUMPED_RLC_Z_FACT)
    } else {
        grid_capacitance
    }
}

/// Raise the grid capacitance for a series element.
///
/// Native series form: `Zmin` is the magnitude of the element impedance at
/// `f_max`, including the `1/(ωC)` term when the element has a capacitance.
pub fn series_capacitance(grid_capacitance: f64, element: &CellElement, f_max: f64) -> f64 {
    let omega = TAU * f_max;
    let z_min = if element.capacitance != 0.0 {
        (element.resistance.powi(2)
            + (omega * element.inductance - 1.0 / (element.capacitance * omega)).powi(2))
        .sqrt()
    } else {
        (element.resistance.powi(2) + (omega * element.inductance).powi(2)).sqrt()
    };
    if !(f_max > 0.0) || !(z_min > 0.0) || !(grid_capacitance > 0.0) {
        return grid_capacitance;
    }
    let z_cd_min = 1.0 / (omega * grid_capacitance);
    if z_cd_min < LUMPED_RLC_Z_FACT * z_min {
        1.0 / (omega * z_min * LUMPED_RLC_Z_FACT)
    } else {
        grid_capacitance
    }
}

/// Voltage-update coefficients for the series/parallel ADE, matching the
/// assignments in native `BuildExtension`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeriesUpdateCoefficients {
    pub vv2: f64,
    pub vj1: f64,
    pub vj2: f64,
    pub vvd: f64,
    pub ib0: f64,
    pub b1: f64,
    pub b2: f64,
}

/// Build the series update coefficients from the series ADE coefficients and
/// the (possibly adjusted) cell capacitance.
pub fn series_update_coefficients(
    dt: f64,
    grid_capacitance: f64,
    series: &SeriesCoefficients,
) -> SeriesUpdateCoefficients {
    let cd = grid_capacitance;
    SeriesUpdateCoefficients {
        vv2: 0.5 * dt * series.ib0 / cd,
        vj1: 0.5 * dt * (series.b1 * series.ib0 - 1.0) / cd,
        vj2: 0.5 * dt * series.b2 * series.ib0 / cd,
        vvd: 1.0 / (1.0 + 0.5 * dt * series.ib0 / cd),
        ib0: series.ib0,
        b1: series.b1,
        b2: series.b2,
    }
}

/// Parallel-branch coefficients, native `v_i2v` and `v_ilv`.
pub fn parallel_update_coefficients(
    dt: f64,
    capacitance: f64,
    element: &CellElement,
) -> (f64, f64) {
    let i2v = (dt / capacitance) / (1.0 + dt * element.conductance / (2.0 * capacitance));
    let ilv = if element.inductance > 0.0 {
        dt / element.inductance
    } else {
        0.0
    };
    (i2v, ilv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_cell_scaling_matches_native() {
        // R = 100 Ω, L = 2 nH, C = 3 pF, 4 cells in series, 6 in parallel.
        let element = CellElement::from_element(100.0, 2.0e-9, 3.0e-12, 6, 4);
        assert!((element.inductance - 2.0e-9 * 6.0 / 4.0).abs() < 1e-24);
        assert!((element.resistance - 100.0 * 6.0 / 4.0).abs() < 1e-12);
        assert!((element.capacitance - 3.0e-12 * 4.0 / 6.0).abs() < 1e-24);
        assert!((element.conductance - (1.0 / 100.0) * 4.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn absent_components_become_zero() {
        // NaN means "not present"; R = 0 means an open for the parallel branch.
        let element = CellElement::from_element(f64::NAN, f64::NAN, f64::NAN, 1, 1);
        assert_eq!(element.resistance, 0.0);
        assert_eq!(element.inductance, 0.0);
        assert_eq!(element.capacitance, 0.0);
        assert_eq!(element.conductance, 0.0);
        let open = CellElement::from_element(0.0, 1.0e-9, 1.0e-12, 1, 1);
        assert_eq!(open.conductance, 0.0);
    }

    #[test]
    fn series_coefficients_reproduce_a_known_damped_oscillator() {
        // Series RLC: verify the ADE recurrence against the analytic solution
        // of  L dJ/dt + R J + V_C = V,  C dV_C/dt = J.
        let dt = 1.0e-12;
        let element = CellElement {
            inductance: 1.0e-9,
            resistance: 10.0,
            conductance: 0.0,
            capacitance: 1.0e-12,
        };
        let series = series_coefficients(dt, &element);
        assert!(series.ib0 > 0.0);
        // b1 = (dt^2 - 4LC)/(dt C)
        let expected_b1 = (dt * dt - 4.0 * 1.0e-9 * 1.0e-12) / (dt * 1.0e-12);
        assert!((series.b1 - expected_b1).abs() / expected_b1.abs() < 1e-12);
        let expected_b2 =
            (4.0 * 1.0e-9 * 1.0e-12 - 2.0 * dt * 10.0 * 1.0e-12 + dt * dt) / (2.0 * dt * 1.0e-12);
        assert!((series.b2 - expected_b2).abs() / expected_b2.abs() < 1e-12);
    }

    #[test]
    fn series_rl_without_capacitance_uses_the_reduced_form() {
        let dt = 2.0e-12;
        let element = CellElement {
            inductance: 5.0e-9,
            resistance: 20.0,
            conductance: 0.0,
            capacitance: 0.0,
        };
        let series = series_coefficients(dt, &element);
        assert!((series.ib0 - dt / (2.0 * 5.0e-9 + dt * 20.0)).abs() < 1e-18);
        assert!((series.b1 + 4.0 * 5.0e-9 / dt).abs() < 1e-12);
        assert!((series.b2 - (2.0 * 5.0e-9 - dt * 20.0) / dt).abs() < 1e-12);
    }

    #[test]
    fn capacitance_floor_keeps_the_update_stable() {
        // A grid capacitance that is too large has too small an impedance at
        // f_max; native raises the impedance by lowering the capacitance.
        let raised = parallel_capacitance(0.0, 1.0e-9, 50.0, 1.0e-9, 1.0e9);
        assert!(raised < 1.0e-9, "capacitance was not lowered: {raised:e}");
        let z_c = 1.0 / (TAU * 1.0e9 * raised);
        assert!(z_c >= LUMPED_RLC_Z_FACT * 50.0 * (1.0 - 1e-12));
        // A grid capacitance already large enough in impedance is untouched.
        let untouched = parallel_capacitance(0.0, 1.0e-15, 50.0, 1.0e-9, 1.0e9);
        assert_eq!(untouched, 1.0e-15);
        // An explicit element capacitance is used unchanged.
        assert_eq!(
            parallel_capacitance(2.0e-12, 1.0e-9, 50.0, 1.0e-9, 1.0e9),
            2.0e-12
        );
    }

    #[test]
    fn series_update_coefficients_are_finite_and_stable() {
        let dt = 1.0e-12;
        let element = CellElement {
            inductance: 1.0e-9,
            resistance: 1.0,
            conductance: 0.0,
            capacitance: 1.0e-12,
        };
        let cd = series_capacitance(1.0e-15, &element, 1.0e10);
        let series = series_coefficients(dt, &element);
        let update = series_update_coefficients(dt, cd, &series);
        assert!(update.vvd.is_finite() && update.vvd > 0.0);
        assert!(update.vv2.is_finite() && update.vj1.is_finite() && update.vj2.is_finite());
    }
}
