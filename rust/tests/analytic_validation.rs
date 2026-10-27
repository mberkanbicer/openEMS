//! Analytic validation cases for the FDTD core.
//!
//! Each test compares the solver against a closed-form electromagnetic result.
//! Tolerances are stated per assertion and are dominated by the discrete
//! sampling of the probe signal, not by the update scheme.

use openems_core::constants::{EPS0, MU0};
use openems_core::engine::Engine;
use openems_core::excitation::Excitation;
use openems_core::extensions::EngineExtExcitation;
use openems_core::operator::Operator;
use openems_core::{BC_PEC, BC_PML};
use std::sync::Arc;

const DELTA: f64 = 1.0e-3;

fn light_speed() -> f64 {
    1.0 / (EPS0 * MU0).sqrt()
}

fn mesh_xml(nx: usize, ny: usize, nz: usize, properties: &str) -> String {
    let lines = |count: usize| {
        (0..count)
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="{DELTA}">
                <XLines>{}</XLines>
                <YLines>{}</YLines>
                <ZLines>{}</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>{properties}</Properties>
        </ContinuousStructure>"#,
        lines(nx),
        lines(ny),
        lines(nz)
    )
}

fn build_engine(
    nx: usize,
    ny: usize,
    nz: usize,
    boundary: [i32; 6],
    pml_size: [u32; 6],
    properties: &str,
) -> (Engine, f64) {
    let mut op = Operator::new();
    op.load_csx(&mesh_xml(nx, ny, nz, properties))
        .expect("mesh should load");
    op.base.m_bc = boundary;
    op.calc_timestep();
    let dt = op.base.d_t;
    op.calc_ec_operator();
    op.apply_electric_bc();
    op.apply_magnetic_bc();
    let extensions = if boundary.contains(&BC_PML) {
        op.build_upml_extensions(pml_size)
            .expect("UPML should build")
    } else {
        Vec::new()
    };
    let mut engine = Engine::new(Box::new(op));
    for extension in extensions {
        engine.add_extension(Box::new(extension));
    }
    (engine, dt)
}

fn add_point_source(
    engine: &mut Engine,
    component: usize,
    pos: [usize; 3],
    signal: Excitation,
    dt: f64,
    max_ts: u32,
) {
    let mut signal = signal;
    signal.build(dt, max_ts).expect("signal should build");
    let mut extension = EngineExtExcitation::new();
    extension.add_voltage_excitation(component, pos, 0, 1.0, Arc::new(signal));
    engine.add_extension(Box::new(extension));
}

/// Add a uniform sheet source over the whole cross-section at `x`.
///
/// With PEC side walls this excites only the TEM mode, so the setup behaves as
/// a parallel-plate transmission line and the closed-form reflection
/// coefficients apply.
fn add_sheet_source(
    engine: &mut Engine,
    component: usize,
    x: usize,
    ny: usize,
    nz: usize,
    signal: Excitation,
    dt: f64,
    max_ts: u32,
) {
    let mut signal = signal;
    signal.build(dt, max_ts).expect("signal should build");
    let signal = Arc::new(signal);
    let mut extension = EngineExtExcitation::new();
    for j in 0..ny {
        for k in 0..nz {
            extension.add_voltage_excitation(component, [x, j, k], 0, 1.0, signal.clone());
        }
    }
    engine.add_extension(Box::new(extension));
}

/// Best delay (in samples, fractional) that aligns `late` onto `early`.
fn correlation_delay(early: &[f64], late: &[f64], max_lag: usize) -> f64 {
    let mut scores = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let mut sum = 0.0;
        let mut count = 0usize;
        for index in 0..early.len() {
            if index + lag >= late.len() {
                break;
            }
            sum += early[index] * late[index + lag];
            count += 1;
        }
        if count > 0 {
            scores.push(sum / count as f64);
        } else {
            scores.push(0.0);
        }
    }
    let best = scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0);
    if best == 0 || best + 1 >= scores.len() {
        return best as f64;
    }
    let left = scores[best - 1];
    let centre = scores[best];
    let right = scores[best + 1];
    let denominator = left - 2.0 * centre + right;
    if denominator == 0.0 {
        return best as f64;
    }
    best as f64 + 0.5 * (left - right) / denominator
}

/// Propagation speed of a radiated pulse in vacuum, measured between two
/// probes: `v = dx / dt_arrival`.
#[test]
fn vacuum_propagation_speed_matches_c() {
    let (nx, ny, nz) = (121usize, 11usize, 11usize);
    let (mut engine, dt) = build_engine(nx, ny, nz, [BC_PML; 6], [4; 6], "");
    let source = [10, 5, 5];
    add_point_source(
        &mut engine,
        2,
        source,
        Excitation::gauss(8.0e9, 8.0e9),
        dt,
        1_200,
    );

    let near = [60usize, 5, 5];
    let far = [100usize, 5, 5];
    let steps = 1_200u32;
    let mut near_samples = Vec::with_capacity(steps as usize);
    let mut far_samples = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        engine.iterate_ts(1);
        near_samples.push(*engine.volt_ptr.get(2, near[0], near[1], near[2]) as f64);
        far_samples.push(*engine.volt_ptr.get(2, far[0], far[1], far[2]) as f64);
    }

    let lag = correlation_delay(&near_samples, &far_samples, 200);
    let distance = (far[0] as f64 - near[0] as f64) * DELTA;
    let measured = distance / (lag * dt);
    let c = light_speed();
    let error = (measured - c).abs() / c;
    println!(
        "wave speed: expected {:.4e} m/s, measured {:.4e} m/s ({:.3} samples), error {:.2}%",
        c,
        measured,
        lag,
        error * 100.0
    );
    assert!(
        error < 0.02,
        "measured wave speed {measured:.4e} m/s vs c={c:.4e} m/s"
    );
}

/// Propagation speed inside a homogeneous dielectric, `v = c / sqrt(eps_r)`.
///
/// This exercises the material scaling of the equivalent-circuit capacitance
/// and the permeability path (vacuum) without needing an amplitude calibration.
#[test]
fn dielectric_propagation_speed_matches_c_over_sqrt_eps() {
    let (nx, ny, nz) = (121usize, 11usize, 11usize);
    let properties = r#"
        <Material Name="dielectric">
            <Property Epsilon="4" Mue="1" Kappa="0" Sigma="0"/>
            <Primitives>
                <Box Priority="1"><P1 X="-5" Y="-5" Z="-5"/><P2 X="130" Y="30" Z="30"/></Box>
            </Primitives>
        </Material>"#;
    let (mut engine, dt) = build_engine(nx, ny, nz, [BC_PML; 6], [4; 6], properties);
    let source = [10, 5, 5];
    add_point_source(
        &mut engine,
        2,
        source,
        Excitation::gauss(4.0e9, 4.0e9),
        dt,
        1_200,
    );

    let near = [60usize, 5, 5];
    let far = [100usize, 5, 5];
    let steps = 1_200u32;
    let mut near_samples = Vec::with_capacity(steps as usize);
    let mut far_samples = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        engine.iterate_ts(1);
        near_samples.push(*engine.volt_ptr.get(2, near[0], near[1], near[2]) as f64);
        far_samples.push(*engine.volt_ptr.get(2, far[0], far[1], far[2]) as f64);
    }

    let lag = correlation_delay(&near_samples, &far_samples, 400);
    let distance = (far[0] as f64 - near[0] as f64) * DELTA;
    let measured = distance / (lag * dt);
    let expected = light_speed() / 2.0;
    let error = (measured - expected).abs() / expected;
    println!(
        "dielectric speed: expected {:.4e} m/s, measured {:.4e} m/s ({:.3} samples), error {:.2}%",
        expected,
        measured,
        lag,
        error * 100.0
    );
    assert!(
        error < 0.02,
        "measured dielectric speed {measured:.4e} m/s vs {expected:.4e} m/s"
    );
}
