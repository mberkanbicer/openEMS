//! Conducting-sheet ADE model.
//!
//! Mirrors the coefficient construction in
//! `FDTD/extensions/operator_ext_conductingsheet.cpp`. The rational
//! approximation table is copied verbatim from `cond_sheet_parameter.h` (it was
//! generated for the native solver and must not be re-derived).

/// Upper band edge of each approximation range, normalized to `w / w0`.
const OMEGA_STOP: [f64; 30] = [
    20.0,
    30.0,
    45.0,
    67.5,
    101.25,
    151.875,
    227.8125,
    341.71875,
    512.578125,
    768.8671875,
    1153.300781,
    1729.951172,
    2594.926758,
    3892.390137,
    5838.585205,
    8757.877808,
    13136.81671,
    19705.22507,
    29557.8376,
    44336.7564,
    66505.1346,
    99757.7019,
    149636.5529,
    224454.8293,
    336682.2439,
    505023.3659,
    757535.0488,
    1136302.573,
    1704453.86,
    2556680.79,
];

const G: [f64; 30] = [
    0.1370843401,
    0.1244327709,
    0.1087425925,
    0.09215450299,
    0.07639249525,
    0.06239206861,
    0.05068701937,
    0.04124161681,
    0.03364925947,
    0.0274825647,
    0.02244156564,
    0.01832317373,
    0.01496080786,
    0.01221545053,
    0.009973874666,
    0.008143635713,
    0.006649251918,
    0.005429092739,
    0.00443283702,
    0.003619397792,
    0.002955227653,
    0.002412935361,
    0.001970156167,
    0.001608629812,
    0.001313423397,
    0.001072422468,
    0.0009031383439,
    0.0007674973627,
    0.0006482579571,
    0.0005450699565,
];

const R1: [f64; 30] = [
    1.302754546,
    1.2992612,
    1.320033314,
    1.41829213,
    1.674326257,
    2.138770494,
    2.769126112,
    3.498802047,
    4.311446869,
    5.266204065,
    6.443980095,
    7.893402541,
    9.667414736,
    11.84010502,
    14.50110421,
    17.76014698,
    21.75163911,
    26.64019322,
    32.6274156,
    39.96021835,
    48.94100192,
    59.94011376,
    73.41109983,
    89.90930992,
    110.1195785,
    134.8631576,
    149.2770459,
    153.5258961,
    148.9373215,
    119.3522908,
];

const L1: [f64; 30] = [
    0.9542567116,
    0.9412051444,
    0.8999655165,
    0.8189903271,
    0.702028543,
    0.5718817624,
    0.4577262324,
    0.3689027719,
    0.3004106661,
    0.2455514861,
    0.2005633225,
    0.1637496639,
    0.1337010005,
    0.1091664663,
    0.0891340699,
    0.07277768839,
    0.05942276018,
    0.04851850861,
    0.03961522618,
    0.03234573021,
    0.02641021614,
    0.02156389712,
    0.0176069076,
    0.01437606898,
    0.0117376269,
    0.009584103516,
    0.008750848841,
    0.008547706786,
    0.00854127488,
    0.008801399365,
];

const R2: [f64; 30] = [
    10.62245057,
    10.21952962,
    10.24755832,
    10.99147713,
    12.70450822,
    15.59830718,
    19.55714986,
    24.32580904,
    29.88733852,
    36.55680626,
    44.75374078,
    54.81575133,
    67.13533424,
    82.22360504,
    100.7029064,
    123.3353165,
    151.0542147,
    185.0027444,
    226.5809553,
    277.5035137,
    339.8704104,
    416.2534638,
    509.8022059,
    624.3729288,
    764.7279941,
    936.5525496,
    1000.0,
    999.9999981,
    999.999984,
    999.9999985,
];

const L2: [f64; 30] = [
    0.7925524826,
    0.6926195458,
    0.5814718257,
    0.4784644339,
    0.3916667501,
    0.3204661897,
    0.2620413998,
    0.2140939121,
    0.1748272154,
    0.1427375388,
    0.1165424396,
    0.09515681664,
    0.07769522035,
    0.06343788358,
    0.05179681795,
    0.04229192785,
    0.03453121766,
    0.02819462457,
    0.0230208182,
    0.01879642336,
    0.01534721999,
    0.01253095815,
    0.01023149158,
    0.008353988465,
    0.006820957973,
    0.005569332359,
    0.004553963932,
    0.003696380632,
    0.002978504727,
    0.002391295771,
];

/// Result of the sheet model for one edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheetCoefficients {
    /// Modified equivalent-circuit voltage coefficients.
    pub vv: f64,
    pub vi: f64,
    /// ADE coefficients for both orders.
    pub v_int_ade: [f64; 2],
    pub v_ext_ade: [f64; 2],
    /// Approximation band index (for diagnostics).
    pub band: usize,
}

/// Build the conducting-sheet ADE coefficients for one edge.
///
/// * `sigma` — sheet conductivity (S/m)
/// * `thickness` — sheet thickness (m)
/// * `dt` — FDTD timestep (s)
/// * `f_max` — highest frequency of interest (Hz)
/// * `width_to_length` — `edge length / node width` along the sheet
/// * `sheet_neighbours` — native `factor` (1 when both sides carry a sheet,
///   2 when the edge is shared with a non-sheet cell)
pub fn sheet_coefficients(
    sigma: f64,
    thickness: f64,
    dt: f64,
    f_max: f64,
    width_to_length: f64,
    factor: f64,
) -> Result<SheetCoefficients, String> {
    if !(sigma > 0.0) || !(thickness > 0.0) {
        return Err("conducting sheet needs a positive conductivity and thickness".to_string());
    }
    if !(f_max >= 0.0)
        || !f_max.is_finite()
        || !(dt > 0.0)
        || !(width_to_length > 0.0)
        || !(factor > 0.0)
    {
        return Err("invalid conducting-sheet input".to_string());
    }

    let g0 = sigma * thickness;
    let w0 = 8.0 / g0 / thickness / crate::constants::MU0;
    let omega_max = f_max * 2.0 * std::f64::consts::PI / w0;
    let band = OMEGA_STOP
        .iter()
        .position(|&limit| limit > omega_max)
        .unwrap_or(OMEGA_STOP.len() - 1);

    let l1 = L1[band] / g0 / w0 * factor;
    let l2 = L2[band] / g0 / w0 * factor;
    let r1 = R1[band] / g0 * factor;
    let r2 = R2[band] / g0 * factor;
    let g = G[band] * g0 / factor;

    let l1 = l1 * width_to_length;
    let l2 = l2 * width_to_length;
    let r1 = r1 * width_to_length;
    let r2 = r2 * width_to_length;
    let g = g / width_to_length;

    let l_min = l1.min(l2);
    // Native: EC_C = dt^2 / 4 * (16/Lmin + 1/L1 + 1/L2), EC_G = G.
    let c = dt * dt / 4.0 * (16.0 / l_min + 1.0 / l1 + 1.0 / l2);
    if !(c > 0.0) || !g.is_finite() {
        return Err("conducting-sheet model produced non-positive coefficients".to_string());
    }

    let denominator = 1.0 + dt * g / (2.0 * c);
    let vv = (1.0 - dt * g / (2.0 * c)) / denominator;
    let vi = (dt / c) / denominator;

    let v_int_ade = [
        (2.0 * l1 - dt * r1) / (2.0 * l1 + dt * r1),
        (2.0 * l2 - dt * r2) / (2.0 * l2 + dt * r2),
    ];
    let v_ext_ade = [
        dt / (l1 + dt * r1 / 2.0) * vi,
        dt / (l2 + dt * r2 / 2.0) * vi,
    ];

    Ok(SheetCoefficients {
        vv,
        vi,
        v_int_ade,
        v_ext_ade,
        band,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_are_finite_and_stable() {
        let result =
            sheet_coefficients(1.0e5, 35.0e-6, 1.0e-12, 10.0e9, 1.0, 1.0).expect("valid sheet");
        assert!(result.vv.is_finite() && result.vv.abs() <= 1.0 + 1e-12);
        assert!(result.vi.is_finite() && result.vi > 0.0);
        for value in result.v_int_ade {
            assert!(value.is_finite() && value.abs() < 1.0);
        }
        for value in result.v_ext_ade {
            assert!(value.is_finite() && value > 0.0);
        }
        assert!(result.band < OMEGA_STOP.len());
    }

    #[test]
    fn a_thicker_or_more_conductive_sheet_lowers_the_loss_coefficient() {
        let thin = sheet_coefficients(1.0e5, 10.0e-6, 1.0e-12, 5.0e9, 1.0, 1.0).unwrap();
        let thick = sheet_coefficients(1.0e5, 50.0e-6, 1.0e-12, 5.0e9, 1.0, 1.0).unwrap();
        assert!(
            thick.vi > thin.vi,
            "thicker sheet should increase vi ({} vs {})",
            thick.vi,
            thin.vi
        );
    }

    #[test]
    fn rejects_invalid_input() {
        assert!(sheet_coefficients(0.0, 35.0e-6, 1.0e-12, 10.0e9, 1.0, 1.0).is_err());
        assert!(sheet_coefficients(1.0e5, 0.0, 1.0e-12, 10.0e9, 1.0, 1.0).is_err());
        assert!(sheet_coefficients(1.0e5, 35.0e-6, 0.0, 10.0e9, 1.0, 1.0).is_err());
        // A zero maximum frequency selects the first band, as in the native model.
        assert!(sheet_coefficients(1.0e5, 35.0e-6, 1.0e-12, 0.0, 1.0, 1.0).is_ok());
    }
}
