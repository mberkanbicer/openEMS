//! Mesh-refinement study for the propagation speed in a homogeneous dielectric.
//!
//! The model is a parallel-plate line filled with `eps_r = 4`, excited by a
//! Gaussian point source. Two probes measure the pulse delay; the speed is
//! compared with `c / sqrt(eps_r)` at h, h/2 and h/4.
//!
//! Run with:
//!   cargo run --release -p openems-core --example dielectric_convergence
//!
//! This is a study, not a CI gate; the short analytic test stays in
//! `tests/analytic_validation.rs`.

use openems_core::constants::{EPS0, MU0};
use openems_core::engine::Engine;
use openems_core::excitation::Excitation;
use openems_core::extensions::EngineExtExcitation;
use openems_core::operator::Operator;
use openems_core::{BC_PEC, BC_PML, BC_PMC};
use std::sync::Arc;

const DELTA: f64 = 1.0e-3;

fn light_speed() -> f64 {
    1.0 / (EPS0 * MU0).sqrt()
}

fn mesh_xml(nx: usize, ny: usize, nz: usize, delta: f64, properties: &str) -> String {
    let lines = |count: usize| {
        (0..count)
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="{delta}">
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
    delta: f64,
    boundary: [i32; 6],
    pml_x: u32,
    properties: &str,
) -> (Engine, f64) {
    let mut op = Operator::new();
    op.load_csx(&mesh_xml(nx, ny, nz, delta, properties))
        .expect("mesh should load");
    op.base.m_bc = boundary;
    op.calc_timestep();
    let dt = op.base.d_t;
    op.calc_ec_operator();
    op.apply_electric_bc();
    op.apply_magnetic_bc();
    let extensions = op.build_upml_extensions([pml_x, pml_x, 0, 0, 0, 0]).expect("UPML");
    let mut engine = Engine::new(Box::new(op));
    for extension in extensions {
        engine.add_extension(Box::new(extension));
    }
    (engine, dt)
}

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
        scores.push(if count > 0 { sum / count as f64 } else { 0.0 });
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

fn run_level(scale: usize) -> f64 {
    // Physical domain 120 x 10 x 10 mm; `scale` cells per mm.
    let delta = DELTA / scale as f64;
    let (nx, ny, nz) = (120 * scale + 1, 10 * scale + 1, 10 * scale + 1);
    let unit = DELTA / delta; // drawing units per millimetre
    let properties = format!(
        r#"
        <Material Name="dielectric">
            <Property Epsilon="4" Mue="1" Kappa="0" Sigma="0"/>
            <Primitives>
                <Box Priority="1"><P1 X="{}" Y="{}" Z="{}"/><P2 X="{}" Y="{}" Z="{}"/></Box>
            </Primitives>
        </Material>"#,
        -5.0 * unit,
        -5.0 * unit,
        -5.0 * unit,
        130.0 * unit,
        30.0 * unit,
        30.0 * unit,
    );
    // Parallel-plate line: PEC at z, PMC at y, PML at the x ends. A sheet
    // source across the cross-section drives a plane wave, so the delay is a
    // clean 1-D observable rather than a radiated spherical wave.
    let (mut engine, dt) = build_engine(
        nx,
        ny,
        nz,
        delta,
        [BC_PML, BC_PML, BC_PMC, BC_PMC, BC_PEC, BC_PEC],
        4 * scale as u32,
        &properties,
    );
    let source_x = (10.0 * unit) as usize;
    let steps = 1_200 * scale as u32;
    add_sheet_source(
        &mut engine,
        2,
        source_x,
        ny,
        nz,
        Excitation::gauss(4.0e9, 4.0e9),
        dt,
        steps,
    );

    let mid_y = ny / 2;
    let mid_z = nz / 2;
    let near = [(30.0 * unit) as usize, mid_y, mid_z];
    let far = [(90.0 * unit) as usize, mid_y, mid_z];
    let mut near_samples = Vec::with_capacity(steps as usize);
    let mut far_samples = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        engine.iterate_ts(1);
        near_samples.push(*engine.volt_ptr.get(2, near[0], near[1], near[2]) as f64);
        far_samples.push(*engine.volt_ptr.get(2, far[0], far[1], far[2]) as f64);
    }
    let lag = correlation_delay(&near_samples, &far_samples, 400 * scale);
    let distance = (far[0] as f64 - near[0] as f64) * delta;
    let measured = distance / (lag * dt);
    let expected = light_speed() / 2.0;
    let error = (measured - expected).abs() / expected;
    println!(
        "scale {scale}: mesh {nx}x{ny}x{nz}, dt {dt:.3e} s, measured {measured:.6e} m/s, \
         error {:.4e}",
        error
    );
    error
}

fn main() {
    let expected = light_speed() / 2.0;
    println!("analytic dielectric speed (eps_r=4): {expected:.6e} m/s");
    let errors: Vec<f64> = [1usize, 2, 4].into_iter().map(run_level).collect();
    for pair in 1..errors.len() {
        let order = (errors[pair - 1] / errors[pair]).log2();
        println!(
            "observed order {}->{}: {order:.2}",
            1 << (pair - 1),
            1 << pair
        );
    }
}
