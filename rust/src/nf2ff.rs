//! Near-field to far-field (NF2FF) transform.
//!
//! Ports `nf2ff/nf2ff_calc.cpp`: the tangential E and H fields on a closed
//! surface are converted into equivalent currents, integrated against the
//! radiation direction, and evaluated at a far-field radius.
//!
//! The module works on frequency-domain fields provided by the caller, so it
//! can be fed either from the running DFT dumps or from an HDF5 file.

use std::f64::consts::PI;

use crate::constants::C0;
use crate::processing::ComplexPoint;

/// Mirror boundary type for the symmetry planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MirrorType {
    #[default]
    Off,
    Pec,
    Pmc,
}

/// A complex vector field on the `(n, i, j, k)` grid.
#[derive(Clone, Debug)]
pub struct ComplexField {
    data: Vec<ComplexPoint>,
    extent: [usize; 4],
    stride: [usize; 4],
}

impl ComplexField {
    /// Allocate a field with the given `(i, j, k)` extent.
    pub fn new(extent: [usize; 3]) -> Self {
        let extent = [3, extent[0], extent[1], extent[2]];
        let stride = [
            extent[1] * extent[2] * extent[3],
            extent[2] * extent[3],
            extent[3],
            1,
        ];
        let size = extent[0] * extent[1] * extent[2] * extent[3];
        Self {
            data: vec![ComplexPoint::default(); size],
            extent,
            stride,
        }
    }

    /// Field extent `[n, i, j, k]`.
    pub fn extent(&self) -> [usize; 4] {
        self.extent
    }

    fn index(&self, n: usize, i: usize, j: usize, k: usize) -> usize {
        self.stride[0] * n + self.stride[1] * i + self.stride[2] * j + self.stride[3] * k
    }

    /// Read one component at a grid position.
    pub fn get(&self, n: usize, i: usize, j: usize, k: usize) -> ComplexPoint {
        self.data[self.index(n, i, j, k)]
    }

    /// Write one component at a grid position.
    pub fn set(&mut self, n: usize, i: usize, j: usize, k: usize, value: ComplexPoint) {
        let index = self.index(n, i, j, k);
        self.data[index] = value;
    }

    /// Scale every component by `factor`.
    pub fn scale(&mut self, factor: [f64; 3]) {
        for n in 0..3 {
            let scale = factor[n];
            let start = n * self.stride[0];
            let end = start + self.stride[0];
            for value in &mut self.data[start..end] {
                value.re *= scale;
                value.im *= scale;
            }
        }
    }
}

fn mul(a: ComplexPoint, b: ComplexPoint) -> ComplexPoint {
    ComplexPoint {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

fn sub(a: ComplexPoint, b: ComplexPoint) -> ComplexPoint {
    ComplexPoint {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

fn conj(a: ComplexPoint) -> ComplexPoint {
    ComplexPoint {
        re: a.re,
        im: -a.im,
    }
}

fn scale(a: ComplexPoint, s: f64) -> ComplexPoint {
    ComplexPoint {
        re: a.re * s,
        im: a.im * s,
    }
}

fn add_assign(a: &mut ComplexPoint, b: ComplexPoint) {
    a.re += b.re;
    a.im += b.im;
}

/// Near-to-far-field transform.
pub struct Nf2ffCalc {
    pub frequency: f64,
    pub radius: f64,
    pub permittivity: f64,
    pub permeability: f64,
    pub center: [f64; 3],
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub mirror_type: [MirrorType; 3],
    pub mirror_pos: [f64; 3],
    /// Far-field components, indexed `[theta][phi]`.
    pub e_theta: Vec<Vec<ComplexPoint>>,
    pub e_phi: Vec<Vec<ComplexPoint>>,
    pub h_theta: Vec<Vec<ComplexPoint>>,
    pub h_phi: Vec<Vec<ComplexPoint>>,
    /// Radiation density `|E|^2 / (2 Z)` in W/m^2.
    pub p_rad: Vec<Vec<f64>>,
    /// Total radiated power in W.
    pub radiated_power: f64,
    /// Maximum directivity (unitless).
    pub max_directivity: f64,
}

impl Nf2ffCalc {
    /// Create a transform at `frequency` for the given angles (radians).
    pub fn new(frequency: f64, theta: Vec<f64>, phi: Vec<f64>, center: [f64; 3]) -> Self {
        let num_theta = theta.len();
        let num_phi = phi.len();
        Self {
            frequency,
            radius: 1.0,
            permittivity: 1.0,
            permeability: 1.0,
            center,
            theta,
            phi,
            mirror_type: [MirrorType::Off; 3],
            mirror_pos: [0.0; 3],
            e_theta: vec![vec![ComplexPoint::default(); num_phi]; num_theta],
            e_phi: vec![vec![ComplexPoint::default(); num_phi]; num_theta],
            h_theta: vec![vec![ComplexPoint::default(); num_phi]; num_theta],
            h_phi: vec![vec![ComplexPoint::default(); num_phi]; num_theta],
            p_rad: vec![vec![0.0; num_phi]; num_theta],
            radiated_power: 0.0,
            max_directivity: 0.0,
        }
    }

    /// Enable a mirror plane (`PEC` or `PMC`) at `position` along `axis`.
    pub fn set_mirror(&mut self, mirror: MirrorType, axis: usize, position: f64) {
        if axis < 3 && mirror != MirrorType::Off {
            self.mirror_type[axis] = mirror;
            self.mirror_pos[axis] = position;
        }
    }

    /// Normal direction of a plane, or `None` when the plane is not flat.
    fn normal_dir(num_lines: [usize; 3]) -> Option<usize> {
        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            if num_lines[n] == 1 && num_lines[n_p] > 2 && num_lines[n_pp] > 2 {
                return Some(n);
            }
        }
        None
    }

    fn edge_lengths(lines: &[f64]) -> Vec<f64> {
        let count = lines.len();
        let mut lengths = vec![0.0; count];
        for index in 1..count.saturating_sub(1) {
            lengths[index] = 0.5 * (lines[index + 1] - lines[index - 1]).abs();
        }
        if count >= 2 {
            lengths[0] = 0.5 * (lines[1] - lines[0]).abs();
            lengths[count - 1] = 0.5 * (lines[count - 1] - lines[count - 2]).abs();
        }
        lengths
    }

    /// Add one frequency-domain plane. `lines` are the mesh lines in metres.
    pub fn add_plane(
        &mut self,
        lines: [&[f64]; 3],
        num_lines: [usize; 3],
        e_field: &ComplexField,
        h_field: &ComplexField,
        cylindrical: bool,
    ) -> bool {
        if !self.add_single_plane(lines, num_lines, e_field, h_field, cylindrical) {
            return false;
        }
        let mut mirrored = false;
        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            if self.mirror_type[n] != MirrorType::Off
                && self.mirror_type[n_p] == MirrorType::Off
                && self.mirror_type[n_pp] == MirrorType::Off
            {
                self.add_mirror_plane(n, lines, num_lines, e_field, h_field, cylindrical);
                mirrored = true;
                break;
            } else if self.mirror_type[n] == MirrorType::Off
                && self.mirror_type[n_p] != MirrorType::Off
                && self.mirror_type[n_pp] != MirrorType::Off
            {
                self.add_mirror_plane(n_p, lines, num_lines, e_field, h_field, cylindrical);
                self.add_mirror_plane(n_pp, lines, num_lines, e_field, h_field, cylindrical);
                self.add_mirror_plane(n_p, lines, num_lines, e_field, h_field, cylindrical);
                mirrored = true;
                break;
            }
        }
        if self
            .mirror_type
            .iter()
            .all(|mirror| *mirror != MirrorType::Off)
        {
            for axis in [0usize, 1, 0, 2, 0, 1, 0] {
                self.add_mirror_plane(axis, lines, num_lines, e_field, h_field, cylindrical);
            }
            mirrored = true;
        }
        if mirrored {
            // A mirrored copy must not be reused for the next plane.
            return true;
        }
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn add_mirror_plane(
        &mut self,
        axis: usize,
        lines: [&[f64]; 3],
        num_lines: [usize; 3],
        e_field: &ComplexField,
        h_field: &ComplexField,
        cylindrical: bool,
    ) {
        let e_factor = match self.mirror_type[axis] {
            MirrorType::Pec => {
                let n_p = (axis + 1) % 3;
                let n_pp = (axis + 2) % 3;
                let mut factor = [1.0; 3];
                factor[n_p] = -1.0;
                factor[n_pp] = -1.0;
                factor
            }
            MirrorType::Pmc => {
                let mut factor = [1.0; 3];
                factor[axis] = -1.0;
                factor
            }
            MirrorType::Off => [1.0; 3],
        };
        let h_factor = match self.mirror_type[axis] {
            MirrorType::Pec => {
                let mut factor = [1.0; 3];
                factor[axis] = -1.0;
                factor
            }
            MirrorType::Pmc => {
                let n_p = (axis + 1) % 3;
                let n_pp = (axis + 2) % 3;
                let mut factor = [1.0; 3];
                factor[n_p] = -1.0;
                factor[n_pp] = -1.0;
                factor
            }
            MirrorType::Off => [1.0; 3],
        };
        let mirror = 2.0 * self.mirror_pos[axis];
        let mut mirrored_lines = [Vec::new(), Vec::new(), Vec::new()];
        for n in 0..3 {
            mirrored_lines[n] = lines[n].to_vec();
        }
        for value in &mut mirrored_lines[axis] {
            *value = mirror - *value;
        }
        let mut e_copy = e_field.clone();
        let mut h_copy = h_field.clone();
        e_copy.scale(e_factor);
        h_copy.scale(h_factor);
        let refs: [&[f64]; 3] = [&mirrored_lines[0], &mirrored_lines[1], &mirrored_lines[2]];
        self.add_single_plane(refs, num_lines, &e_copy, &h_copy, cylindrical);
    }

    fn add_single_plane(
        &mut self,
        lines: [&[f64]; 3],
        num_lines: [usize; 3],
        e_field: &ComplexField,
        h_field: &ComplexField,
        cylindrical: bool,
    ) -> bool {
        let Some(ny) = Self::normal_dir(num_lines) else {
            eprintln!("[openEMS] NF2FF: cannot determine the plane normal direction");
            return false;
        };
        let n_p = (ny + 1) % 3;
        let n_pp = (ny + 2) % 3;
        let norm_sign = if lines[ny][0] >= self.center[ny] {
            1.0
        } else {
            -1.0
        };
        let mut norm_dir = [0.0f64; 3];
        norm_dir[ny] = norm_sign;

        let mut edge_p = Self::edge_lengths(lines[n_p]);
        let mut edge_pp = Self::edge_lengths(lines[n_pp]);
        if cylindrical {
            if ny == 0 {
                let radius = lines[0][0];
                for value in &mut edge_p {
                    *value *= radius;
                }
            } else if ny == 2 {
                for index in 1..num_lines[n_p].saturating_sub(1) {
                    edge_p[index] *= lines[n_p][index];
                }
                if num_lines[n_p] > 0 {
                    edge_p[0] *= lines[n_p][0] + 0.5 * edge_p[0];
                    let last = num_lines[n_p] - 1;
                    edge_p[last] *= lines[n_p][last] - 0.5 * edge_p[last];
                }
            }
        }

        // Equivalent surface currents and the radiated power through the plane.
        let mut js = ComplexField::new(num_lines);
        let mut ms = ComplexField::new(num_lines);
        let mut pos = [0usize; 3];
        pos[ny] = 0;
        for pos_p in 0..num_lines[n_p] {
            pos[n_p] = pos_p;
            for pos_pp in 0..num_lines[n_pp] {
                pos[n_pp] = pos_pp;
                let [i, j, k] = pos;
                let h0 = h_field.get(0, i, j, k);
                let h1 = h_field.get(1, i, j, k);
                let h2 = h_field.get(2, i, j, k);
                let e0 = e_field.get(0, i, j, k);
                let e1 = e_field.get(1, i, j, k);
                let e2 = e_field.get(2, i, j, k);
                let mut current_j = [
                    ComplexPoint {
                        re: norm_dir[1] * h2.re - norm_dir[2] * h1.re,
                        im: norm_dir[1] * h2.im - norm_dir[2] * h1.im,
                    },
                    ComplexPoint {
                        re: norm_dir[2] * h0.re - norm_dir[0] * h2.re,
                        im: norm_dir[2] * h0.im - norm_dir[0] * h2.im,
                    },
                    ComplexPoint {
                        re: norm_dir[0] * h1.re - norm_dir[1] * h0.re,
                        im: norm_dir[0] * h1.im - norm_dir[1] * h0.im,
                    },
                ];
                let mut current_m = [
                    ComplexPoint {
                        re: norm_dir[2] * e1.re - norm_dir[1] * e2.re,
                        im: norm_dir[2] * e1.im - norm_dir[1] * e2.im,
                    },
                    ComplexPoint {
                        re: norm_dir[0] * e2.re - norm_dir[2] * e0.re,
                        im: norm_dir[0] * e2.im - norm_dir[2] * e0.im,
                    },
                    ComplexPoint {
                        re: norm_dir[1] * e0.re - norm_dir[0] * e1.re,
                        im: norm_dir[1] * e0.im - norm_dir[0] * e1.im,
                    },
                ];
                if cylindrical {
                    let angle = lines[1][j];
                    let (sin, cos) = angle.sin_cos();
                    let jx = current_j[0];
                    let jy = current_j[1];
                    current_j[0] = ComplexPoint {
                        re: jx.re * cos - jy.re * sin,
                        im: jx.im * cos - jy.im * sin,
                    };
                    current_j[1] = ComplexPoint {
                        re: jx.re * sin + jy.re * cos,
                        im: jx.im * sin + jy.im * cos,
                    };
                    let mx = current_m[0];
                    let my = current_m[1];
                    current_m[0] = ComplexPoint {
                        re: mx.re * cos - my.re * sin,
                        im: mx.im * cos - my.im * sin,
                    };
                    current_m[1] = ComplexPoint {
                        re: mx.re * sin + my.re * cos,
                        im: mx.im * sin + my.im * cos,
                    };
                }
                js.set(0, i, j, k, current_j[0]);
                js.set(1, i, j, k, current_j[1]);
                js.set(2, i, j, k, current_j[2]);
                ms.set(0, i, j, k, current_m[0]);
                ms.set(1, i, j, k, current_m[1]);
                ms.set(2, i, j, k, current_m[2]);

                let area = edge_p[pos_p] * edge_pp[pos_pp];
                let power = mul(e_field.get(n_p, i, j, k), conj(h_field.get(n_pp, i, j, k)));
                let power2 = mul(e_field.get(n_pp, i, j, k), conj(h_field.get(n_p, i, j, k)));
                let difference = sub(power, power2);
                self.radiated_power += 0.5 * area * difference.re * norm_dir[ny];
            }
        }

        // Radiation integrals.
        let wavenumber = std::f64::consts::TAU * self.frequency / C0
            * (self.permittivity * self.permeability).sqrt();
        let mut center = self.center;
        if cylindrical {
            center[0] = self.center[0] * self.center[1].cos();
            center[1] = self.center[0] * self.center[1].sin();
        }
        let num_theta = self.theta.len();
        let num_phi = self.phi.len();
        let mut n_t = vec![vec![ComplexPoint::default(); num_phi]; num_theta];
        let mut n_phi_acc = vec![vec![ComplexPoint::default(); num_phi]; num_theta];
        let mut l_t = vec![vec![ComplexPoint::default(); num_phi]; num_theta];
        let mut l_phi_acc = vec![vec![ComplexPoint::default(); num_phi]; num_theta];

        for (ti, &theta) in self.theta.iter().enumerate() {
            let (sin_t, cos_t) = theta.sin_cos();
            for (pi, &phi) in self.phi.iter().enumerate() {
                let (sin_p, cos_p) = phi.sin_cos();
                let cos_t_cos_p = cos_t * cos_p;
                let cos_t_sin_p = cos_t * sin_p;
                let cos_p_sin_t = cos_p * sin_t;
                let sin_t_sin_p = sin_p * sin_t;
                let mut pos = [0usize; 3];
                pos[ny] = 0;
                for pos_p in 0..num_lines[n_p] {
                    pos[n_p] = pos_p;
                    for pos_pp in 0..num_lines[n_pp] {
                        pos[n_pp] = pos_pp;
                        let [i, j, k] = pos;
                        let (x, y, z) = if cylindrical {
                            (
                                lines[0][i] * lines[1][j].cos(),
                                lines[0][i] * lines[1][j].sin(),
                                lines[2][k],
                            )
                        } else {
                            (lines[0][i], lines[1][j], lines[2][k])
                        };
                        let r_cos_psi = (x - center[0]) * cos_p_sin_t
                            + (y - center[1]) * sin_t_sin_p
                            + (z - center[2]) * cos_t;
                        let phase = wavenumber * r_cos_psi;
                        let exponential = ComplexPoint {
                            re: phase.cos(),
                            im: phase.sin(),
                        };
                        let area = edge_p[pos_p] * edge_pp[pos_pp];
                        let current_js =
                            [js.get(0, i, j, k), js.get(1, i, j, k), js.get(2, i, j, k)];
                        let current_ms =
                            [ms.get(0, i, j, k), ms.get(1, i, j, k), ms.get(2, i, j, k)];
                        let n_theta = ComplexPoint {
                            re: current_js[0].re * cos_t_cos_p + current_js[1].re * cos_t_sin_p
                                - current_js[2].re * sin_t,
                            im: current_js[0].im * cos_t_cos_p + current_js[1].im * cos_t_sin_p
                                - current_js[2].im * sin_t,
                        };
                        let n_phi = ComplexPoint {
                            re: current_js[1].re * cos_p - current_js[0].re * sin_p,
                            im: current_js[1].im * cos_p - current_js[0].im * sin_p,
                        };
                        let l_theta = ComplexPoint {
                            re: current_ms[0].re * cos_t_cos_p + current_ms[1].re * cos_t_sin_p
                                - current_ms[2].re * sin_t,
                            im: current_ms[0].im * cos_t_cos_p + current_ms[1].im * cos_t_sin_p
                                - current_ms[2].im * sin_t,
                        };
                        let l_phi = ComplexPoint {
                            re: current_ms[1].re * cos_p - current_ms[0].re * sin_p,
                            im: current_ms[1].im * cos_p - current_ms[0].im * sin_p,
                        };
                        let weight = scale(exponential, area);
                        add_assign(&mut n_t[ti][pi], mul(weight, n_theta));
                        add_assign(&mut n_phi_acc[ti][pi], mul(weight, n_phi));
                        add_assign(&mut l_t[ti][pi], mul(weight, l_theta));
                        add_assign(&mut l_phi_acc[ti][pi], mul(weight, l_phi));
                    }
                }
            }
        }

        // Equations 8.23a/b and 8.24a/b.
        let factor_magnitude = wavenumber / 4.0 / PI / self.radius;
        let (sin_r, cos_r) = (-wavenumber * self.radius).sin_cos();
        // factor = i k/(4 pi r) exp(-i k r)
        let factor = ComplexPoint {
            re: -factor_magnitude * sin_r,
            im: factor_magnitude * cos_r,
        };
        let z_wave = crate::constants::Z0 * (self.permeability / self.permittivity).sqrt();
        let mut p_max = 0.0f64;
        for ti in 0..num_theta {
            for pi in 0..num_phi {
                let n_theta = n_t[ti][pi];
                let n_phi = n_phi_acc[ti][pi];
                let l_theta = l_t[ti][pi];
                let l_phi = l_phi_acc[ti][pi];

                let term_e_theta = ComplexPoint {
                    re: l_phi.re + z_wave * n_theta.re,
                    im: l_phi.im + z_wave * n_theta.im,
                };
                self.e_theta[ti][pi] = sub(self.e_theta[ti][pi], mul(factor, term_e_theta));
                let term_e_phi = ComplexPoint {
                    re: l_theta.re - z_wave * n_phi.re,
                    im: l_theta.im - z_wave * n_phi.im,
                };
                self.e_phi[ti][pi] = add(self.e_phi[ti][pi], mul(factor, term_e_phi));
                let term_h_theta = ComplexPoint {
                    re: n_phi.re - l_theta.re / z_wave,
                    im: n_phi.im - l_theta.im / z_wave,
                };
                self.h_theta[ti][pi] = add(self.h_theta[ti][pi], mul(factor, term_h_theta));
                let term_h_phi = ComplexPoint {
                    re: n_theta.re + l_phi.re / z_wave,
                    im: n_theta.im + l_phi.im / z_wave,
                };
                self.h_phi[ti][pi] = sub(self.h_phi[ti][pi], mul(factor, term_h_phi));

                let magnitude = self.e_theta[ti][pi].magnitude().powi(2)
                    + self.e_phi[ti][pi].magnitude().powi(2);
                let density = magnitude / (2.0 * z_wave);
                self.p_rad[ti][pi] = density;
                if density > p_max {
                    p_max = density;
                }
            }
        }
        if self.radiated_power != 0.0 {
            self.max_directivity =
                p_max * (4.0 * PI * self.radius * self.radius / self.radiated_power);
        }
        true
    }
}

fn add(a: ComplexPoint, b: ComplexPoint) -> ComplexPoint {
    ComplexPoint {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_plane_matches_the_analytic_radiation_integral() {
        let frequency = 1.0e9f64;
        let radius = 1.0f64;
        let lines_values = vec![0.0, 1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3];
        let num_lines = [1usize, 5, 5];
        // Plane normal to +x, wave travelling in +x: E = z, H = -y/Z0.
        let mut e_field = ComplexField::new(num_lines);
        let mut h_field = ComplexField::new(num_lines);
        let z0 = crate::constants::Z0;
        for j in 0..5 {
            for k in 0..5 {
                e_field.set(2, 0, j, k, ComplexPoint { re: 1.0, im: 0.0 });
                h_field.set(
                    1,
                    0,
                    j,
                    k,
                    ComplexPoint {
                        re: -1.0 / z0,
                        im: 0.0,
                    },
                );
            }
        }
        let mut calc = Nf2ffCalc::new(
            frequency,
            vec![std::f64::consts::FRAC_PI_2],
            vec![0.0],
            [0.0; 3],
        );
        calc.radius = radius;
        assert!(calc.add_plane(
            [&[0.0], &lines_values, &lines_values],
            num_lines,
            &e_field,
            &h_field,
            false
        ));

        // Area of the surface cell weights: sum of dual edge lengths squared.
        let edge = [0.5e-3, 1.0e-3, 1.0e-3, 1.0e-3, 0.5e-3];
        let sum: f64 = edge.iter().sum();
        let area = sum * sum;
        let expected_power = 0.5 * area / z0;
        assert!(
            (calc.radiated_power - expected_power).abs() / expected_power < 1.0e-12,
            "radiated power {} vs {}",
            calc.radiated_power,
            expected_power
        );

        let k = std::f64::consts::TAU * frequency / C0;
        let factor = k / (4.0 * PI * radius);
        let expected_e_theta = 2.0 * factor * area;
        let measured = calc.e_theta[0][0].magnitude();
        assert!(
            (measured - expected_e_theta).abs() / expected_e_theta < 1.0e-9,
            "E_theta {measured:e} vs {expected_e_theta:e}"
        );
        // A +r wave has H_phi = -E_theta / Z0 and E_phi = H_theta = 0.
        assert!(calc.e_phi[0][0].magnitude() < 1.0e-12);
        assert!(calc.h_theta[0][0].magnitude() < 1.0e-12);
        assert!((calc.h_phi[0][0].magnitude() - measured / z0).abs() / (measured / z0) < 1.0e-9);
        // Directivity of the sampled aperture.
        let expected_directivity = calc.p_rad[0][0] * 4.0 * PI * radius * radius / expected_power;
        assert!(
            (calc.max_directivity - expected_directivity).abs() / expected_directivity < 1.0e-9
        );
    }

    #[test]
    fn a_non_flat_box_is_rejected() {
        let mut calc = Nf2ffCalc::new(1.0e9, vec![0.0], vec![0.0], [0.0; 3]);
        let field = ComplexField::new([3, 3, 3]);
        assert!(!calc.add_plane(
            [&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]],
            [3, 3, 3],
            &field,
            &field,
            false
        ));
    }
}
