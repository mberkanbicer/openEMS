//! Equivalent-circuit operator: EC coefficients, material averaging, PEC.
//! Mirrors `FDTD/operator.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::{C0, EPS0, FDTD_FLOAT, MU0, Z0};
use crate::excitation::Excitation;
use crate::extensions::{
    ConductingSheetCoeffs, EngineExtConductingSheet, EngineExtDebyeMaterial, EngineExtExcitation,
    EngineExtLorentzMaterial, EngineExtLumpedRLC, EngineExtMurABC, EngineExtSteadyState,
    EngineExtTFSF, EngineExtUPML, LumpedRLCCoeffs,
};
use crate::operator_base::{GridPath, OperatorBase};
use csxcad_core::grid::{transform_coordinates, CoordinateSystem};
use csxcad_core::material::{ConductingSheetProperties, ExcitationProperties, MaterialField};
use csxcad_core::model::{Primitive, PropertyType};
use std::collections::HashMap;
use std::sync::Arc;

/// Per-cell material properties for material averaging.
/// Mirrors the C++ material storage per Yee cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellMaterial {
    pub eps_r: f64,
    pub mue_r: f64,
    pub kappa: f64,
    pub sigma: f64,
    /// True when this cell is a perfect electric conductor (Metal property).
    /// Such cells have all EC coefficients zeroed (no E/H field inside).
    pub is_pec: bool,
}

/// Discretisation metric of the operator mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MeshType {
    /// Cartesian `(x, y, z)` mesh.
    #[default]
    Cartesian,
    /// Cylindrical `(rho, alpha, z)` mesh. `closed_alpha` marks a mesh that
    /// spans the full `2*pi` with a duplicated boundary plane (native
    /// `Operator_Cylinder`).
    Cylindrical { closed_alpha: bool },
}

/// Geometry of one mesh: edge lengths, face areas and cell volumes.
///
/// Mirrors the native `GetEdgeLength` / `GetEdgeArea` / `GetCellVolume`
/// family, including the `Operator_Cylinder` overrides: an alpha edge carries
/// the factor `rho`, and a z face is the annulus segment
/// `dalpha/2 * (r2^2 - r1^2)`.
#[derive(Clone, Copy)]
pub struct MeshMetric<'a> {
    pub lines: &'a [Option<Vec<f64>>; 3],
    pub counts: [u32; 3],
    pub grid_delta: f64,
    pub mesh: MeshType,
}

impl<'a> MeshMetric<'a> {
    pub fn new(
        lines: &'a [Option<Vec<f64>>; 3],
        counts: [u32; 3],
        grid_delta: f64,
        mesh: MeshType,
    ) -> Self {
        Self {
            lines,
            counts,
            grid_delta,
            mesh,
        }
    }

    pub fn is_cylindrical(&self) -> bool {
        matches!(self.mesh, MeshType::Cylindrical { .. })
    }

    /// Mesh line in drawing units (native `GetDiscLine`).
    pub fn line(&self, axis: usize, pos: usize, dual: bool) -> f64 {
        disc_line_value(&self.lines[axis], self.counts[axis] as usize, pos, dual)
    }

    /// Cartesian edge length in metres (native base `GetEdgeLength`).
    fn base_length(&self, n: usize, pos: [usize; 3], dual: bool) -> f64 {
        edge_length_value(self.lines, self.counts, n, pos, dual, self.grid_delta)
    }

    /// Native `GetEdgeLength`: an alpha edge is scaled by `rho`.
    pub fn edge_length(&self, n: usize, pos: [usize; 3], dual: bool) -> f64 {
        let mut length = self.base_length(n, pos, dual);
        if n == 1 && self.is_cylindrical() {
            length *= self.line(0, pos[0], dual);
        }
        length
    }

    /// Native `Operator_Cylinder::GetNodeWidth`: the base length across the
    /// node, with the `rho` factor taken from the requested mesh.
    pub fn node_width(&self, n: usize, pos: [usize; 3], dual: bool) -> f64 {
        let mut width = self.base_length(n, pos, !dual);
        if n == 1 && self.is_cylindrical() {
            width *= self.line(0, pos[0], dual);
        }
        width
    }

    /// Native `Operator_Cylinder::GetNodeArea`: a z face is the annulus
    /// segment between `r1` and `r2`.
    pub fn node_area(&self, n: usize, pos: [usize; 3], dual: bool) -> f64 {
        if n == 2 && self.is_cylindrical() {
            let da = disc_delta_value(&self.lines[1], self.counts[1] as usize, pos[1], dual);
            let (r1, r2) = if dual {
                let r1 = self.line(0, pos[0], false) * self.grid_delta;
                (r1, r1 + self.base_length(0, pos, false))
            } else {
                let r2 = self.line(0, pos[0], true) * self.grid_delta;
                (r2 - self.base_length(0, pos, true), r2)
            };
            if r1 <= 0.0 {
                0.5 * da * r2 * r2
            } else {
                0.5 * da * (r2 * r2 - r1 * r1)
            }
        } else {
            self.node_width((n + 1) % 3, pos, dual) * self.node_width((n + 2) % 3, pos, dual)
        }
    }

    /// Native `Operator_Cylinder::GetEdgeArea`: the rho face is spanned by the
    /// alpha and z edge lengths, every other face is the node area.
    pub fn edge_area(&self, n: usize, pos: [usize; 3], dual: bool) -> f64 {
        if n == 0 && self.is_cylindrical() {
            self.edge_length(1, pos, !dual) * self.edge_length(2, pos, !dual)
        } else {
            self.node_area(n, pos, dual)
        }
    }

    /// Native `Operator_Cylinder::GetCellVolume`.
    pub fn cell_volume(&self, pos: [usize; 3], dual: bool) -> f64 {
        self.edge_area(2, pos, dual) * self.edge_length(2, pos, dual)
    }
}

/// Anisotropic material values used while resolving a mesh edge.
#[derive(Clone, Copy, Debug)]
struct AnisoMaterial {
    eps: [f64; 3],
    mue: [f64; 3],
    kappa: [f64; 3],
    sigma: [f64; 3],
    is_pec: bool,
}

impl AnisoMaterial {
    fn vacuum() -> Self {
        Self {
            eps: [1.0; 3],
            mue: [1.0; 3],
            kappa: [0.0; 3],
            sigma: [0.0; 3],
            is_pec: false,
        }
    }

    fn pec() -> Self {
        Self {
            eps: [1.0; 3],
            mue: [1.0; 3],
            kappa: [0.0; 3],
            sigma: [0.0; 3],
            is_pec: true,
        }
    }
}

/// The basic FDTD operator.
/// Computes equivalent-circuit coefficients (vv, vi, ii, iv) for every Yee cell.
pub struct Operator {
    pub base: crate::operator_base::OperatorBaseImpl,
    pub vv_ptr: Option<FieldArray>,
    pub vi_ptr: Option<FieldArray>,
    pub ii_ptr: Option<FieldArray>,
    pub iv_ptr: Option<FieldArray>,
    /// Equivalent-circuit capacitance per component and cell (`EC_C`).
    /// Kept alongside the update coefficients because extensions (lumped RLC,
    /// conducting sheet) rewrite the cell capacitance before recomputing
    /// `vv`/`vi`, exactly like native `Operator::EC_C`.
    pub ec_c: Vec<FDTD_FLOAT>,
    /// Equivalent-circuit conductance per component and cell (`EC_G`).
    pub ec_g: Vec<FDTD_FLOAT>,
    /// Equivalent-circuit inductance per component and cell (`EC_L`).
    pub ec_l: Vec<FDTD_FLOAT>,
    /// Equivalent-circuit resistance per component and cell (`EC_R`).
    pub ec_r: Vec<FDTD_FLOAT>,
    /// Per-cell material grid, indexed as [n][i][j][k] flattened.
    /// When Some, overrides the background material for EC coefficient computation.
    pub material_grid: Option<Vec<CellMaterial>>,
    /// Parsed CSXCAD structure, retained for extension setup (excitation, Mur).
    pub structure: Option<csxcad_core::model::Structure>,
    /// Discretisation metric of the mesh (Cartesian or cylindrical).
    pub mesh_type: MeshType,
    /// A cylindrical mesh that includes the `rho = 0` axis (native
    /// `CC_R0_included`).
    pub r0_included: bool,
    /// Per-z metal flag for the `rho = 0` axis (native `GetPropertyByCoordPriority`).
    pub r0_axis_metal: Vec<bool>,
    /// Per-z `(vv, vi)` update coefficients of the `rho = 0` Ez edge.
    pub r0_pairs: Vec<(FDTD_FLOAT, FDTD_FLOAT)>,
    /// Native `CSXCAD` drawing tolerance used by the material lookup: a mesh
    /// point within this distance of a primitive's surface lies on it.
    pub drawing_tolerance: f64,
}

impl Operator {
    /// Create a new operator.
    pub fn new() -> Self {
        Self {
            base: crate::operator_base::OperatorBaseImpl::new(),
            vv_ptr: None,
            vi_ptr: None,
            ii_ptr: None,
            iv_ptr: None,
            ec_c: Vec::new(),
            ec_g: Vec::new(),
            ec_l: Vec::new(),
            ec_r: Vec::new(),
            material_grid: None,
            structure: None,
            mesh_type: MeshType::Cartesian,
            r0_included: false,
            r0_axis_metal: Vec::new(),
            r0_pairs: Vec::new(),
            drawing_tolerance: 0.0,
        }
    }

    /// Geometry helper for the current mesh.
    pub fn metric(&self) -> MeshMetric<'_> {
        MeshMetric::new(
            &self.base.disc_lines,
            self.base.num_lines,
            self.base.grid_delta,
            self.mesh_type,
        )
    }

    /// True for a cylindrical `(rho, alpha, z)` mesh.
    pub fn is_cylindrical(&self) -> bool {
        matches!(self.mesh_type, MeshType::Cylindrical { .. })
    }

    /// Build the azimuthal seam extension of a cylindrical mesh.
    ///
    /// Mirrors `Operator_Ext_Cylinder::CreateEngineExtention`, including the
    /// `rho = 0` axis coefficients computed by [`Self::build_r0_coefficients`].
    pub fn build_cylinder_extension(&self) -> Option<crate::extensions::EngineExtCylinder> {
        if !self.is_cylindrical() {
            return None;
        }
        let mut extension = crate::extensions::EngineExtCylinder::new([
            self.base.num_lines[0] as usize,
            self.base.num_lines[1] as usize,
            self.base.num_lines[2] as usize,
        ]);
        extension.set_closed_alpha(self.closed_alpha());
        extension.set_r0_included(self.r0_included);
        if let Some(pairs) = self.r0_coefficients() {
            for (k, (vv, vi)) in pairs.iter().enumerate() {
                extension.set_r0_coeffs(k, *vv, *vi);
            }
        }
        Some(extension)
    }

    /// Native `Operator_Ext_Cylinder::BuildExtension` (the `rho = 0` axis).
    ///
    /// At the axis every azimuthal sector shares a single `Ez` edge, so their
    /// capacitances and conductances are summed into one pair of update
    /// coefficients. `E_alpha` is forced to zero and `Ez` is copied to all
    /// azimuths by the engine extension. Must run after `apply_electric_bc`,
    /// exactly like native (`ApplyElectricBC` precedes the extension setup).
    pub fn build_r0_coefficients(&mut self) {
        if !self.r0_included || !self.is_cylindrical() {
            return;
        }
        let counts = self.base.num_lines;
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        if nx == 0 || ny == 0 || nz == 0 {
            return;
        }
        let dt = self.base.d_t;
        let metric = MeshMetric::new(
            &self.base.disc_lines,
            counts,
            self.base.grid_delta,
            self.mesh_type,
        );
        let material_grid = self.material_grid.as_ref();
        let bg_eps_r = self.base.m_bg_eps_r;
        let bg_kappa = self.base.m_bg_kappa;

        let mut pairs = Vec::with_capacity(nz);
        for k in 0..nz {
            let mut capacity = 0.0f64;
            let mut conductance = 0.0f64;
            // Native loops `pos[1] < GetNumberOfLines(1,true) - 2`, one term
            // per azimuthal sector.
            for j in 0..ny.saturating_sub(2) {
                let (eps_r, kappa) = match material_grid {
                    Some(grid) => {
                        let index = Self::ec_index(2, 0, j, k, counts);
                        match grid.get(index) {
                            Some(material) => (material.eps_r, material.kappa),
                            None => (bg_eps_r, bg_kappa),
                        }
                    }
                    None => (bg_eps_r, bg_kappa),
                };
                let area = metric.edge_area(2, [0, j, k], false);
                let delta = metric.edge_length(2, [0, j, k], false);
                if delta > 0.0 {
                    capacity += eps_r * EPS0 * area / delta;
                    conductance += kappa * area / delta;
                }
            }

            // Native `SetVV(2,0,0,k,1)`: the regular update only carries the
            // old value, the extension applies the summed coefficients.
            if let Some(vv) = self.vv_ptr.as_mut() {
                *vv.get_mut(2, 0, 0, k) = 1.0;
            }

            // Share the summed C/G over every azimuthal copy of the axis edge.
            let index = Self::ec_index(2, 0, 0, k, counts);
            for i in 0..ny {
                let index = Self::ec_index(2, 0, i, k, counts);
                if index < self.ec_c.len() {
                    self.ec_c[index] = capacity as FDTD_FLOAT;
                    self.ec_g[index] = conductance as FDTD_FLOAT;
                }
            }

            let metal = self.r0_axis_metal.get(k).copied().unwrap_or(false);
            if metal || capacity <= 0.0 {
                // Native zeroes only the axis entry itself (metal axis = PEC).
                if let Some(vv) = self.vv_ptr.as_mut() {
                    *vv.get_mut(2, 0, 0, k) = 0.0;
                }
                if index < self.ec_c.len() {
                    self.ec_c[index] = 0.0;
                    self.ec_g[index] = 0.0;
                }
                pairs.push((0.0, 0.0));
                continue;
            }
            let denom = 1.0 + dt * conductance / (2.0 * capacity);
            let vv_r0 = ((1.0 - dt * conductance / (2.0 * capacity)) / denom) as FDTD_FLOAT;
            let vi_r0 = ((dt / capacity) / denom) as FDTD_FLOAT;
            pairs.push((vv_r0, vi_r0));
        }
        self.r0_pairs = pairs;
    }

    /// Per-z `(vv, vi)` pairs produced by [`Self::build_r0_coefficients`].
    pub fn r0_coefficients(&self) -> Option<&[(FDTD_FLOAT, FDTD_FLOAT)]> {
        if self.r0_pairs.is_empty() {
            None
        } else {
            Some(&self.r0_pairs)
        }
    }

    /// True for a full `2*pi` azimuthal mesh with a duplicated seam plane.
    pub fn closed_alpha(&self) -> bool {
        matches!(self.mesh_type, MeshType::Cylindrical { closed_alpha: true })
    }

    /// Number of alpha lines in the requested mesh.
    ///
    /// Mirrors native `Operator_Cylinder::GetNumberOfLines`
    /// (`FDTD/operator_cylinder.cpp:110-120`): with `full_mesh == false` a
    /// closed azimuth hides the two seam lines (`numLines[1] - 2`, i.e. the
    /// number of cells), with `full_mesh == true` all lines are reported.
    fn alpha_line_count(&self, full_mesh: bool) -> u32 {
        if !full_mesh && self.closed_alpha() {
            return self.base.num_lines[1].saturating_sub(2);
        }
        self.base.num_lines[1]
    }

    /// Index into the flat per-component EC arrays.
    pub fn ec_index(n: usize, i: usize, j: usize, k: usize, counts: [u32; 3]) -> usize {
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        n * nx * ny * nz + i * ny * nz + j * nz + k
    }

    /// Allocate the EC coefficient field arrays and the per-cell material grid.
    /// Mirrors C++ `Init_EC`.
    pub fn init_data_storage(&mut self, num_lines: [usize; 3]) {
        let extent = [num_lines[0], num_lines[1], num_lines[2]];
        self.base.num_lines = [
            num_lines[0] as u32,
            num_lines[1] as u32,
            num_lines[2] as u32,
        ];
        self.vv_ptr = Some(FieldArray::new("vv", extent));
        self.vi_ptr = Some(FieldArray::new("vi", extent));
        self.ii_ptr = Some(FieldArray::new("ii", extent));
        self.iv_ptr = Some(FieldArray::new("iv", extent));
        // Native keeps the four per-cell EC values; extensions may rewrite the
        // capacitance/conductance and must be able to recompute the update
        // coefficients for a single edge (`Calc_ECOperatorPos`).
        let num_cells = num_lines[0] * num_lines[1] * num_lines[2];
        self.ec_c = vec![0.0; 3 * num_cells];
        self.ec_g = vec![0.0; 3 * num_cells];
        self.ec_l = vec![0.0; 3 * num_cells];
        self.ec_r = vec![0.0; 3 * num_cells];
        // Allocate per-cell material grid (one entry per cell and direction,
        // initialized from the background material). Indexed as [n][i][j][k].
        let background = CellMaterial {
            eps_r: self.base.m_bg_eps_r,
            mue_r: self.base.m_bg_mue_r,
            kappa: self.base.m_bg_kappa,
            sigma: self.base.m_bg_sigma,
            is_pec: false,
        };
        self.material_grid = Some(vec![background; 3 * num_cells]);
    }

    /// Set the material properties for a specific cell.
    pub fn set_cell_material(
        &mut self,
        n: usize,
        i: usize,
        j: usize,
        k: usize,
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
    ) {
        let Some(grid) = &mut self.material_grid else {
            return;
        };
        let nx = self.base.num_lines[0] as usize;
        let ny = self.base.num_lines[1] as usize;
        let nz = self.base.num_lines[2] as usize;
        if i >= nx || j >= ny || k >= nz {
            return;
        }
        let index = n * nx * ny * nz + i * ny * nz + j * nz + k;
        if index < grid.len() {
            grid[index] = CellMaterial {
                eps_r,
                mue_r,
                kappa,
                sigma,
                is_pec: false,
            };
        }
    }

    /// Apply a material to a rectangular box region of cells.
    /// Coordinates are cell indices (inclusive start, exclusive stop).
    pub fn apply_material_box(
        &mut self,
        start: [usize; 3],
        stop: [usize; 3],
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
    ) {
        let nx = self.base.num_lines[0] as usize;
        let ny = self.base.num_lines[1] as usize;
        let nz = self.base.num_lines[2] as usize;
        let x0 = start[0].min(nx).saturating_sub(1);
        let x1 = stop[0].min(nx);
        let y0 = start[1].min(ny).saturating_sub(1);
        let y1 = stop[1].min(ny);
        let z0 = start[2].min(nz).saturating_sub(1);
        let z1 = stop[2].min(nz);
        for n in 0..3 {
            for i in x0..x1 {
                for j in y0..y1 {
                    for k in z0..z1 {
                        self.set_cell_material(n, i, j, k, eps_r, mue_r, kappa, sigma);
                    }
                }
            }
        }
    }

    /// Load a CSXCAD XML geometry and wire its grid, materials, boundary
    /// conditions and primitives into this operator.
    ///
    /// Mirrors the C++ `SetupFDTD` flow that reads the CSX structure, builds
    /// the FDTD grid, applies materials and boundary conditions, and stores
    /// per-cell material data for EC coefficient computation.
    pub fn load_csx(&mut self, xml: &str) -> Result<(), String> {
        let mut structure = csxcad_core::model::Structure::from_xml_str(xml)
            .map_err(|e| format!("CSXCAD XML parse error: {}", e))?;

        // --- Grid ---
        // Keep the native representation: discLines hold drawing units and
        // grid_delta is the unit in metres. Non-uniform spacing is preserved.
        let mut num_lines = [0u32; 3];
        let mut grid_lines: [Vec<f64>; 3] = Default::default();
        for n in 0..3usize {
            let count = structure.grid_mut().line_count(n);
            if count < 2 {
                return Err("CSXCAD grid has insufficient lines".to_string());
            }
            num_lines[n] = count as u32;
            grid_lines[n] = structure.grid_mut().lines(n).unwrap_or_default().to_vec();
        }
        let delta_unit = structure.grid().delta_unit();
        if !delta_unit.is_finite() || delta_unit <= 0.0 {
            return Err("CSXCAD grid delta unit must be positive".to_string());
        }

        // --- Mesh metric (Cartesian or cylindrical) ---
        // Native `Operator_Cylinder::SetupCSXGrid`: a mesh spanning the full
        // `2*pi` is periodic, so the last line is pinned to `2*pi` and one
        // padding line is appended for the dual-mesh operations at the seam.
        let mut mesh_type = MeshType::Cartesian;
        let mut r0_included = false;
        if structure.coordinate_system() == csxcad_core::grid::CoordinateSystem::Cylindrical {
            if grid_lines[0].first().copied().unwrap_or(0.0) < 0.0 {
                return Err(
                    "cylindrical mesh: negative rho coordinates are not allowed".to_string()
                );
            }
            let first = grid_lines[1].first().copied().unwrap_or(0.0);
            let span = grid_lines[1].last().copied().unwrap_or(0.0) - first;
            if span > 2.0 * std::f64::consts::PI + 1e-6 {
                return Err("cylindrical mesh: the alpha range must not exceed 2*pi".to_string());
            }
            let closed_alpha = (span - 2.0 * std::f64::consts::PI).abs() < 1e-6;
            if closed_alpha {
                let second = grid_lines[1][1];
                let last = grid_lines[1].len() - 1;
                let padded = first + 2.0 * std::f64::consts::PI;
                grid_lines[1][last] = padded;
                grid_lines[1].push(padded + (second - first));
                structure.grid_mut().set_line(1, last, padded);
                structure.grid_mut().add_line(1, padded + (second - first));
                num_lines[1] = grid_lines[1].len() as u32;
            }
            // Native `Operator_Cylinder::SetupCSXGrid`: `rho = 0` on a closed
            // azimuth is the axis, which needs the dedicated `Ez` update
            // (`Operator_Ext_Cylinder`, implemented in `build_r0_coefficients`).
            r0_included = grid_lines[0][0] == 0.0 && closed_alpha;
            mesh_type = MeshType::Cylindrical { closed_alpha };
        }
        self.mesh_type = mesh_type;
        self.r0_included = r0_included;
        self.base.grid_delta = delta_unit;
        self.base.num_lines = num_lines;
        self.base.disc_lines = [
            Some(grid_lines[0].clone()),
            Some(grid_lines[1].clone()),
            Some(grid_lines[2].clone()),
        ];

        // --- Background material ---
        let bg = structure.background_material();
        self.base.set_background_material(
            bg.value(csxcad_core::material::MaterialField::Epsilon),
            bg.value(csxcad_core::material::MaterialField::Mue),
            bg.value(csxcad_core::material::MaterialField::Kappa),
            bg.value(csxcad_core::material::MaterialField::Sigma),
            self.base.m_bg_density,
        );

        // --- Boundary conditions ---
        let mut bc = [0i32; 6];
        for prop in 0..structure.property_count() {
            let property = structure
                .property_by_index(prop)
                .ok_or_else(|| "invalid property index".to_string())?;
            if property.property_type() == csxcad_core::model::PropertyType::AbsorbingBoundary {
                if let Some(ab) = property.absorbing_boundary() {
                    let bt = ab.boundary_type();
                    let v = match bt {
                        csxcad_core::material::AbsorbingBoundaryType::MurFirstOrder
                        | csxcad_core::material::AbsorbingBoundaryType::MurFirstOrderSuperAbsorption => 2,
                        csxcad_core::material::AbsorbingBoundaryType::Other(3) => 3,
                        _ => continue,
                    };
                    let sign = if ab.normal_sign_positive() { 0 } else { 1 };
                    // Boundary faces: [x-, x+, y-, y+, z-, z+]
                    // sign 0 = positive-direction face (x+, y+, z+)
                    // sign 1 = negative-direction face (x-, y-, z-)
                    let face = match sign {
                        0 => [1, 3, 5],
                        1 => [0, 2, 4],
                        _ => [0, 0, 0],
                    };
                    // Determine which axis this boundary applies to from its
                    // primitives (boxes). For now, apply to all faces of the
                    // matching sign; the C++ code uses primitive geometry to
                    // determine the exact face.
                    for &f in &face {
                        bc[f] = v;
                    }
                }
            }
        }
        self.base.set_boundary_condition(bc);

        // --- Per-edge materials from Material/Metal properties ---
        // Native openEMS resolves the material at each Yee edge by coordinate
        // priority and averages it over the surrounding quarter cells
        // (`Operator::AverageMatQuarterCell`): epsilon/kappa as an area-weighted
        // arithmetic mean of four transverse quarter cells, mue/sigma as a
        // width-weighted harmonic mean along the edge. PEC is decided from the
        // single edge coordinate (`Operator::CalcPEC`).
        let num_lines_usize: [usize; 3] = num_lines.map(|v| v as usize);
        let ncells = num_lines_usize[0] * num_lines_usize[1] * num_lines_usize[2];
        let background = CellMaterial {
            eps_r: self.base.m_bg_eps_r,
            mue_r: self.base.m_bg_mue_r,
            kappa: self.base.m_bg_kappa,
            sigma: self.base.m_bg_sigma,
            is_pec: false,
        };

        // Native `Operator::SetCSXGrid`: a mesh line closer to a primitive's
        // surface than 1e-3 of the smallest mesh step lies on it. The smallest
        // value wins, e.g. for a multi grid.
        let mut min_step = f64::INFINITY;
        for axis in 0..3usize {
            for pair in grid_lines[axis].windows(2) {
                let step = (pair[1] - pair[0]).abs() * delta_unit;
                if step > 0.0 && step < min_step {
                    min_step = step;
                }
            }
        }
        let tolerance = if min_step.is_finite() {
            1.0e-3 * min_step
        } else {
            0.0
        };
        if tolerance > 0.0 && (self.drawing_tolerance <= 0.0 || tolerance < self.drawing_tolerance) {
            self.drawing_tolerance = tolerance;
        }
        structure.set_drawing_tolerance(match structure.drawing_tolerance() {
            0.0 => self.drawing_tolerance,
            current => current.min(self.drawing_tolerance),
        });
        let drawing_tolerance = self.drawing_tolerance;

        let params = structure.parameters();
        struct Candidate<'a> {
            priority: i32,
            material: AnisoMaterial,
            primitive: &'a Primitive,
        }
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for prop in 0..structure.property_count() {
            let property = structure
                .property_by_index(prop)
                .ok_or_else(|| "invalid property index".to_string())?;
            match property.property_type() {
                PropertyType::Material
                | PropertyType::DispersiveMaterial
                | PropertyType::LorentzMaterial
                | PropertyType::DebyeMaterial
                | PropertyType::DiscreteMaterial => {
                    let mat = property
                        .material()
                        .ok_or_else(|| "material property missing material data".to_string())?;
                    let mut material = AnisoMaterial::vacuum();
                    for component in 0..3 {
                        material.eps[component] = mat
                            .value(MaterialField::Epsilon, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(1.0);
                        material.mue[component] = mat
                            .value(MaterialField::Mue, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(1.0);
                        material.kappa[component] = mat
                            .value(MaterialField::Kappa, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(0.0);
                        material.sigma[component] = mat
                            .value(MaterialField::Sigma, component, Some(params))
                            .ok()
                            .flatten()
                            .unwrap_or(0.0);
                    }
                    for primitive in property.primitives() {
                        candidates.push(Candidate {
                            priority: primitive.priority(),
                            material,
                            primitive,
                        });
                    }
                }
                PropertyType::Metal => {
                    for primitive in property.primitives() {
                        candidates.push(Candidate {
                            priority: primitive.priority(),
                            material: AnisoMaterial::pec(),
                            primitive,
                        });
                    }
                }
                _ => {}
            }
        }

        // Highest-priority containing primitive at `coordinate`. Metals are
        // returned as `is_pec`; native `GetMaterial` treats them as background
        // for epsilon/mue averaging, which the caller mirrors.
        let sample = |coordinate: [f64; 3]| -> Option<AnisoMaterial> {
            let mut winner: Option<&AnisoMaterial> = None;
            let mut winner_priority = i32::MIN;
            for candidate in &candidates {
                if winner.is_some() && candidate.priority <= winner_priority {
                    continue;
                }
                if candidate
                    .primitive
                    .is_inside_with_tolerance(coordinate, Some(params), drawing_tolerance)
                    .unwrap_or(false)
                {
                    winner = Some(&candidate.material);
                    winner_priority = candidate.priority;
                }
            }
            winner.copied()
        };
        let bg_eps = self.base.m_bg_eps_r;
        let bg_mue = self.base.m_bg_mue_r;
        let bg_kappa = self.base.m_bg_kappa;
        let bg_sigma = self.base.m_bg_sigma;
        let disc_lines = self.base.disc_lines.clone();
        let counts = self.base.num_lines;

        // native `Operator_Cylinder::GetYeeCoords`: on a closed azimuth every
        // sampled coordinate wraps back into `[alpha0, alpha0 + 2*pi)`, so the
        // seam and padding cells resolve the material inside the ring instead
        // of falling back to the background
        let closed_mesh = matches!(mesh_type, MeshType::Cylindrical { closed_alpha: true });
        let wrap_alpha = |coordinate: &mut [f64; 3]| {
            if !closed_mesh {
                return;
            }
            let alpha0 = disc_lines[1].as_ref().and_then(|l| l.first()).copied();
            let Some(alpha0) = alpha0 else {
                return;
            };
            if coordinate[1] >= alpha0 + std::f64::consts::TAU {
                coordinate[1] -= std::f64::consts::TAU;
            } else if coordinate[1] < alpha0 {
                coordinate[1] += std::f64::consts::TAU;
            }
        };

        let mut material_grid = vec![background; 3 * ncells];
        for n in 0..3usize {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for i in 0..num_lines_usize[0] {
                for j in 0..num_lines_usize[1] {
                    for k in 0..num_lines_usize[2] {
                        let pos = [i, j, k];
                        let base_coordinate = [
                            disc_lines[0].as_ref().map(|l| l[i]).unwrap_or(0.0),
                            disc_lines[1].as_ref().map(|l| l[j]).unwrap_or(0.0),
                            disc_lines[2].as_ref().map(|l| l[k]).unwrap_or(0.0),
                        ];
                        let delta = raw_disc_delta(&disc_lines, counts, n, pos[n] as i64);
                        let delta_p = raw_disc_delta(&disc_lines, counts, n_p, pos[n_p] as i64);
                        let delta_pp = raw_disc_delta(&disc_lines, counts, n_pp, pos[n_pp] as i64);
                        let delta_m = raw_disc_delta(&disc_lines, counts, n, pos[n] as i64 - 1);
                        let delta_p_m =
                            raw_disc_delta(&disc_lines, counts, n_p, pos[n_p] as i64 - 1);
                        let delta_pp_m =
                            raw_disc_delta(&disc_lines, counts, n_pp, pos[n_pp] as i64 - 1);

                        // PEC: single Yee-edge coordinate (native `CalcPEC`).
                        let edge_coordinate = yee_coordinates(&disc_lines, counts, n, pos, false);
                        let mut edge_coordinate = edge_coordinate;
                        wrap_alpha(&mut edge_coordinate);
                        let is_pec = sample(edge_coordinate)
                            .map(|material| material.is_pec)
                            .unwrap_or(false);

                        // epsilon / kappa: four transverse quarter cells.
                        let mut eps_sum = 0.0;
                        let mut kappa_sum = 0.0;
                        let mut area_sum = 0.0;
                        for (positive_p, positive_pp) in
                            [(true, true), (false, true), (true, false), (false, false)]
                        {
                            let mut loc = pos;
                            if !positive_p {
                                if loc[n_p] == 0 {
                                    continue;
                                }
                                loc[n_p] -= 1;
                            }
                            if !positive_pp {
                                if loc[n_pp] == 0 {
                                    continue;
                                }
                                loc[n_pp] -= 1;
                            }
                            let mut coordinate = base_coordinate;
                            coordinate[n] = base_coordinate[n] + delta * 0.5;
                            coordinate[n_p] = base_coordinate[n_p]
                                + if positive_p {
                                    delta_p * 0.25
                                } else {
                                    -delta_p_m * 0.25
                                };
                            coordinate[n_pp] = base_coordinate[n_pp]
                                + if positive_pp {
                                    delta_pp * 0.25
                                } else {
                                    -delta_pp_m * 0.25
                                };
                            wrap_alpha(&mut coordinate);
                            // native `AverageMatQuarterCell` weights the
                            // quarter cells by `GetNodeArea(ny, loc, true)`;
                            // the rho-face `edge_area` branch would apply the
                            // primary rho factor and vanish on the rho = 0 axis
                            let area = MeshMetric::new(&disc_lines, counts, 1.0, mesh_type)
                                .node_area(n, loc, true);
                            if area <= 0.0 {
                                continue;
                            }
                            let material = sample(coordinate)
                                .filter(|material| !material.is_pec)
                                .unwrap_or(AnisoMaterial {
                                    eps: [bg_eps; 3],
                                    mue: [bg_mue; 3],
                                    kappa: [bg_kappa; 3],
                                    sigma: [bg_sigma; 3],
                                    is_pec: false,
                                });
                            eps_sum += material.eps[n] * area;
                            kappa_sum += material.kappa[n] * area;
                            area_sum += area;
                        }
                        // Metals contribute the background material, as in native.
                        let (eps_r, kappa) = if area_sum > 0.0 {
                            (eps_sum / area_sum, kappa_sum / area_sum)
                        } else {
                            (bg_eps, bg_kappa)
                        };

                        // mue / sigma: two longitudinal nodes, harmonic mean.
                        let mut inv_mue = 0.0;
                        let mut sigma_sum = 0.0;
                        let mut length = 0.0;
                        for lower in [true, false] {
                            let mut loc = pos;
                            if lower {
                                if loc[n] == 0 {
                                    continue;
                                }
                                loc[n] -= 1;
                            }
                            let width = MeshMetric::new(&disc_lines, counts, 1.0, mesh_type)
                                .edge_length(n, loc, false);
                            if width <= 0.0 {
                                continue;
                            }
                            let mut coordinate = base_coordinate;
                            coordinate[n] = base_coordinate[n]
                                + if lower { -delta_m * 0.25 } else { delta * 0.25 };
                            coordinate[n_p] = base_coordinate[n_p] + delta_p * 0.5;
                            coordinate[n_pp] = base_coordinate[n_pp] + delta_pp * 0.5;
                            wrap_alpha(&mut coordinate);
                            let material = sample(coordinate)
                                .filter(|material| !material.is_pec)
                                .unwrap_or(AnisoMaterial {
                                    eps: [bg_eps; 3],
                                    mue: [bg_mue; 3],
                                    kappa: [bg_kappa; 3],
                                    sigma: [bg_sigma; 3],
                                    is_pec: false,
                                });
                            let mue = material.mue[n].max(f64::MIN_POSITIVE);
                            inv_mue += width / mue;
                            let sigma = material.sigma[n];
                            if sigma != 0.0 {
                                sigma_sum += width / sigma;
                            } else {
                                sigma_sum = 0.0;
                            }
                            length += width;
                        }
                        let mue_r = if inv_mue > 0.0 {
                            length / inv_mue
                        } else {
                            bg_mue
                        };
                        let sigma = if sigma_sum != 0.0 {
                            length / sigma_sum
                        } else {
                            bg_sigma
                        };

                        let index = n * ncells
                            + i * num_lines_usize[1] * num_lines_usize[2]
                            + j * num_lines_usize[2]
                            + k;
                        material_grid[index] = CellMaterial {
                            eps_r,
                            mue_r,
                            kappa,
                            sigma,
                            is_pec,
                        };
                    }
                }
            }
        }

        // Native `Operator_Ext_Cylinder::BuildExtension`: a metal property
        // covering the axis turns the `rho = 0` Ez edge into PEC. The lookup
        // must honour property priorities, so it reuses the same sampler.
        let r0_axis_metal = if r0_included {
            let nz = num_lines_usize[2];
            (0..nz)
                .map(|k| {
                    let coordinate = [
                        0.0,
                        0.0,
                        disc_lines[2].as_ref().map(|l| l[k]).unwrap_or(0.0),
                    ];
                    sample(coordinate)
                        .map(|material| material.is_pec)
                        .unwrap_or(false)
                })
                .collect()
        } else {
            Vec::new()
        };
        self.r0_axis_metal = r0_axis_metal;

        self.init_data_storage(num_lines_usize);
        self.material_grid = Some(material_grid);
        self.structure = Some(structure);

        Ok(())
    }

    /// Compute all EC coefficients (vv, vi, ii, iv) for every cell.
    /// Mirrors C++ `Calc_ECOperator`.
    /// Uses per-cell material from `material_grid` when available, falling back
    /// to the background material.
    pub fn calc_ec_operator(&mut self) {
        let counts = self.base.num_lines;
        let grid_delta = self.base.grid_delta;
        let disc_lines = &self.base.disc_lines;
        let metric = MeshMetric::new(disc_lines, counts, grid_delta, self.mesh_type);
        let material_grid = self.material_grid.as_ref();

        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        let dt = self.base.d_t as FDTD_FLOAT;

        let bg_eps_r = self.base.m_bg_eps_r;
        let bg_mue_r = self.base.m_bg_mue_r;
        let bg_kappa = self.base.m_bg_kappa;
        let bg_sigma = self.base.m_bg_sigma;

        let vv = self.vv_ptr.as_mut().expect("vv_ptr not allocated");
        let vi = self.vi_ptr.as_mut().expect("vi_ptr not allocated");
        let ii = self.ii_ptr.as_mut().expect("ii_ptr not allocated");
        let iv = self.iv_ptr.as_mut().expect("iv_ptr not allocated");

        for n in 0..3 {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        // Determine effective material for this cell.
                        let (eps_r, mue_r, kappa, sigma, is_pec) = if let Some(grid) = material_grid
                        {
                            let nx_lines = counts[0] as usize;
                            let ny_lines = counts[1] as usize;
                            let nz_lines = counts[2] as usize;
                            let index = n * nx_lines * ny_lines * nz_lines
                                + i * ny_lines * nz_lines
                                + j * nz_lines
                                + k;
                            if index < grid.len() {
                                let m = grid[index];
                                (m.eps_r, m.mue_r, m.kappa, m.sigma, m.is_pec)
                            } else {
                                (bg_eps_r, bg_mue_r, bg_kappa, bg_sigma, false)
                            }
                        } else {
                            (bg_eps_r, bg_mue_r, bg_kappa, bg_sigma, false)
                        };

                        // Per-direction edge geometry from the real mesh.
                        let pos = [i, j, k];
                        let area = metric.edge_area(n, pos, false);
                        let delta = metric.edge_length(n, pos, false);
                        let area_dual = metric.edge_area(n, pos, true);
                        let delta_dual = metric.edge_length(n, pos, true);

                        let c = if delta > 0.0 {
                            (eps_r * EPS0 * area / delta) as FDTD_FLOAT
                        } else {
                            0.0
                        };
                        let g = if delta > 0.0 {
                            (kappa * area / delta) as FDTD_FLOAT
                        } else {
                            0.0
                        };
                        let l = if delta_dual > 0.0 {
                            (mue_r * MU0 * area_dual / delta_dual) as FDTD_FLOAT
                        } else {
                            0.0
                        };
                        let r = if delta_dual > 0.0 {
                            (sigma * area_dual / delta_dual) as FDTD_FLOAT
                        } else {
                            0.0
                        };

                        let index = Self::ec_index(n, i, j, k, counts);
                        self.ec_c[index] = c;
                        self.ec_g[index] = g;
                        self.ec_l[index] = l;
                        self.ec_r[index] = r;
                        // PEC edges inside a metal primitive have no electric
                        // update, matching native `Calc_PEC` (vv/vi only).
                        if is_pec {
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                        }

                        // Voltage coefficients: vv, vi
                        if c > 0.0 && !is_pec {
                            let denom = 1.0 + dt * g / (2.0 * c);
                            *vv.get_mut(n, i, j, k) = (1.0 - dt * g / (2.0 * c)) / denom;
                            *vi.get_mut(n, i, j, k) = (dt / c) / denom;
                        } else {
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                        }

                        // Current coefficients: ii, iv
                        if l > 0.0 {
                            let denom = 1.0 + dt * r / (2.0 * l);
                            *ii.get_mut(n, i, j, k) = (1.0 - dt * r / (2.0 * l)) / denom;
                            *iv.get_mut(n, i, j, k) = (dt / l) / denom;
                        } else {
                            *ii.get_mut(n, i, j, k) = 0.0;
                            *iv.get_mut(n, i, j, k) = 0.0;
                        }
                    }
                }
            }
        }

        // Native `CalcECOperator` calls `CalcPEC_Curves` after the containment
        // pass: a metal `Curve` is not a volume, so the material grid never
        // marks its edges as PEC.
        self.calc_pec_curves();
    }

    /// Mark the edges along a metal `Curve` as perfect conductors.
    ///
    /// Mirrors `Operator::CalcPEC_Curves` (`FDTD/operator.cpp:2064-2097`): for
    /// every METAL property, every `Curve` primitive is walked point by point
    /// with [`Self::find_path`], and each edge on the path gets `VV = VI = 0`.
    /// The primitive is marked used when a path is non-empty. Like the native
    /// code, a `Wire` is not a curve (`CSPrimitives::ToCurve()` matches
    /// `Type == CURVE` only), and the native `m_Nr_PEC` statistics counter has
    /// no Rust consumer.
    fn calc_pec_curves(&mut self) {
        let Some(structure) = self.structure.as_ref() else {
            return;
        };
        let mesh_system = if self.is_cylindrical() {
            csxcad_core::grid::CoordinateSystem::Cylindrical
        } else {
            csxcad_core::grid::CoordinateSystem::Cartesian
        };
        let params = structure.parameters();
        let mut paths: Vec<(usize, usize, Vec<(usize, [usize; 3])>)> = Vec::new();
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            if property.property_type() != csxcad_core::model::PropertyType::Metal {
                continue;
            }
            for (primitive_index, primitive) in property.primitives().iter().enumerate() {
                let Some(curve) = primitive.curve_shape() else {
                    continue;
                };
                let point_system = if primitive.coordinate_system()
                    == csxcad_core::grid::CoordinateSystem::Undefined
                {
                    primitive.mesh_type()
                } else {
                    primitive.coordinate_system()
                };
                let point = |at: usize| -> Option<[f64; 3]> {
                    let coordinates = curve.point(at, Some(params)).ok()??;
                    let cartesian = csxcad_core::grid::transform_coordinates(
                        coordinates,
                        point_system,
                        csxcad_core::grid::CoordinateSystem::Cartesian,
                    );
                    let cartesian = primitive
                        .transform()
                        .map_or(cartesian, |transform| transform.transform(cartesian));
                    Some(csxcad_core::grid::transform_coordinates(
                        cartesian,
                        csxcad_core::grid::CoordinateSystem::Cartesian,
                        mesh_system,
                    ))
                };
                let mut edges = Vec::new();
                for at in 1..curve.point_count() {
                    let (Some(p1), Some(p2)) = (point(at - 1), point(at)) else {
                        continue;
                    };
                    let path = self.find_path(p1, p2);
                    for (edge, direction) in path.dir.iter().enumerate() {
                        edges.push((
                            *direction as usize,
                            [
                                path.pos_path[0][edge] as usize,
                                path.pos_path[1][edge] as usize,
                                path.pos_path[2][edge] as usize,
                            ],
                        ));
                    }
                }
                if !edges.is_empty() {
                    paths.push((index, primitive_index, edges));
                }
            }
        }
        if paths.is_empty() {
            return;
        }

        if let (Some(vv), Some(vi)) = (self.vv_ptr.as_mut(), self.vi_ptr.as_mut()) {
            for (_, _, edges) in &paths {
                for &(n, pos) in edges {
                    *vv.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
                    *vi.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
                }
            }
        }
        if let Some(structure) = self.structure.as_mut() {
            for (property_index, primitive_index, _) in &paths {
                if let Some(property) = structure.property_by_index_mut(*property_index) {
                    if let Some(primitive) = property.primitive_mut(*primitive_index) {
                        primitive.set_used(true);
                    }
                }
            }
        }
    }

    /// Recompute the update coefficients for one edge from its (possibly
    /// rewritten) `EC_C`/`EC_G`/`EC_L`/`EC_R`. Mirrors native
    /// `Operator::Calc_ECOperatorPos`.
    pub fn calc_ec_operator_pos(&mut self, n: usize, pos: [usize; 3]) {
        let counts = self.base.num_lines;
        let index = Self::ec_index(n, pos[0], pos[1], pos[2], counts);
        if index >= self.ec_c.len() {
            return;
        }
        let dt = self.base.d_t;
        let c = self.ec_c[index] as f64;
        let g = self.ec_g[index] as f64;
        let l = self.ec_l[index] as f64;
        let r = self.ec_r[index] as f64;

        let (vv, vi) = if c > 0.0 {
            let denominator = 1.0 + dt * g / 2.0 / c;
            (
                ((1.0 - dt * g / 2.0 / c) / denominator) as FDTD_FLOAT,
                ((dt / c) / denominator) as FDTD_FLOAT,
            )
        } else {
            (0.0, 0.0)
        };
        let (ii, iv) = if l > 0.0 {
            let denominator = 1.0 + dt * r / 2.0 / l;
            (
                ((1.0 - dt * r / 2.0 / l) / denominator) as FDTD_FLOAT,
                ((dt / l) / denominator) as FDTD_FLOAT,
            )
        } else {
            (0.0, 0.0)
        };

        if let Some(array) = self.vv_ptr.as_mut() {
            *array.get_mut(n, pos[0], pos[1], pos[2]) = vv;
        }
        if let Some(array) = self.vi_ptr.as_mut() {
            *array.get_mut(n, pos[0], pos[1], pos[2]) = vi;
        }
        if let Some(array) = self.ii_ptr.as_mut() {
            *array.get_mut(n, pos[0], pos[1], pos[2]) = ii;
        }
        if let Some(array) = self.iv_ptr.as_mut() {
            *array.get_mut(n, pos[0], pos[1], pos[2]) = iv;
        }
    }

    /// Compute the FDTD timestep from the EC coefficients.
    /// Mirrors C++ `CalcTimestep`: method 3 selects `Rennings_2` (Var3, the
    /// largest stable step for rectilinear meshes), everything else Var1.
    pub fn calc_timestep(&mut self) -> f64 {
        // Native `Operator_Cylinder::CalcTimestep`: a mesh that includes the
        // rho = 0 singularity forces the conservative `Rennings_1` step.
        let r0_singular = self.is_cylindrical()
            && self.base.disc_lines[0]
                .as_ref()
                .and_then(|lines| lines.first())
                .copied()
                == Some(0.0);
        if self.base.m_time_step_var == 3 && !r0_singular {
            return self.calc_timestep_var3();
        }
        self.calc_timestep_var1()
    }

    /// CFL estimate `Rennings_1`, C++ `CalcTimestep_Var1`.
    pub fn calc_timestep_var1(&mut self) -> f64 {
        let counts = self.base.num_lines;
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;

        let mut min_dt = f64::INFINITY;

        for n in 0..3usize {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let pos = [i, j, k];
                        let pos_pm = reflected_pos(pos, shift_rel([0; 3], n_p, -1), counts);
                        let pos_ppm = reflected_pos(pos, shift_rel([0; 3], n_pp, -1), counts);

                        let l_a = self.get_ec_l(n_p as i32, pos[0], pos[1], pos[2]);
                        let l_b = self.get_ec_l(n_p as i32, pos_ppm[0], pos_ppm[1], pos_ppm[2]);
                        let l_c = self.get_ec_l(n_pp as i32, pos[0], pos[1], pos[2]);
                        let l_d = self.get_ec_l(n_pp as i32, pos_pm[0], pos_pm[1], pos_pm[2]);
                        let ec_c = self.get_ec_c(n as i32, i, j, k);

                        if ec_c <= 0.0 || l_a <= 0.0 || l_b <= 0.0 || l_c <= 0.0 || l_d <= 0.0 {
                            continue;
                        }
                        let sum_inv_l = 4.0 / l_a + 4.0 / l_b + 4.0 / l_c + 4.0 / l_d;
                        let dt = 2.0 / (sum_inv_l / ec_c).sqrt();
                        if dt > 0.0 && dt < min_dt {
                            min_dt = dt;
                        }
                    }
                }
            }
        }

        if min_dt.is_finite() {
            self.base.d_t = min_dt;
        }
        min_dt
    }

    /// CFL estimate `Rennings_2`, C++ `CalcTimestep_Var3`.
    ///
    /// Uses the equivalent-circuit C and L coefficients of the surrounding
    /// edges and their shared nodes, which gives a larger (still stable) step
    /// than Var1 on rectilinear meshes.
    pub fn calc_timestep_var3(&mut self) -> f64 {
        let counts = self.base.num_lines;
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;

        let mut min_dt = f64::INFINITY;
        let c_at = |operator: &Self, n: usize, pos: [usize; 3]| {
            operator.get_ec_c(n as i32, pos[0], pos[1], pos[2])
        };
        let l_at = |operator: &Self, n: usize, pos: [usize; 3]| {
            operator.get_ec_l(n as i32, pos[0], pos[1], pos[2])
        };
        let inverse_product = |l: f64, c: f64| -> Option<f64> {
            if l > 0.0 && c > 0.0 {
                Some(1.0 / (l * c))
            } else {
                None
            }
        };

        for n in 0..3usize {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let pos = [i, j, k];
                        // Native keeps the shift as a *relative* offset and
                        // reflects against the base cell in `AdrOp::GetPos`
                        // (`tools/AdrOp.cpp:119-160`, `SetReflection2Cell`:
                        // offset 1). Reflecting an already mirrored absolute
                        // coordinate gave a different neighbour at the domain
                        // edge, which is the graded-mesh Var3 difference.
                        let rel = [0i32; 3];
                        let rel_pm = shift_rel(rel, n_p, -1);
                        let rel_pp_mm = shift_rel(rel_pm, n_pp, -1);

                        let pos_pm = reflected_pos(pos, rel_pm, counts);
                        let pos_pp_mm = reflected_pos(pos, rel_pp_mm, counts);

                        let c_here = c_at(self, n, pos);
                        let mut terms: [Option<f64>; 8] = [None; 8];
                        terms[0] = inverse_product(
                            l_at(self, n_pp, pos),
                            c_at(self, n, reflected_pos(pos, shift_rel(rel, n_p, 1), counts)),
                        );
                        terms[1] = inverse_product(l_at(self, n_pp, pos), c_here);
                        terms[2] = inverse_product(
                            l_at(self, n_p, pos),
                            c_at(self, n, reflected_pos(pos, shift_rel(rel, n_pp, 1), counts)),
                        );
                        terms[3] = inverse_product(l_at(self, n_p, pos), c_here);
                        terms[4] = inverse_product(
                            l_at(self, n_pp, pos_pm),
                            c_at(
                                self,
                                n,
                                reflected_pos(pos, shift_rel(rel_pm, n_p, 1), counts),
                            ),
                        );
                        terms[5] = inverse_product(l_at(self, n_pp, pos_pm), c_at(self, n, pos_pm));
                        terms[6] = inverse_product(
                            l_at(self, n_p, pos_pp_mm),
                            c_at(
                                self,
                                n,
                                reflected_pos(pos, shift_rel(rel_pp_mm, n_pp, 1), counts),
                            ),
                        );
                        terms[7] =
                            inverse_product(l_at(self, n_p, pos_pp_mm), c_at(self, n, pos_pp_mm));
                        let Some(wqp) = sum_terms(&terms) else {
                            continue;
                        };

                        let wt_1 = [
                            inverse_product(l_at(self, n_pp, pos), c_at(self, n_p, pos)),
                            inverse_product(l_at(self, n_pp, pos_pm), c_at(self, n_p, pos)),
                            inverse_product(l_at(self, n_p, pos), c_at(self, n_pp, pos)),
                            inverse_product(
                                l_at(
                                    self,
                                    n_p,
                                    reflected_pos(pos, shift_rel(rel, n_pp, -1), counts),
                                ),
                                c_at(self, n_pp, pos),
                            ),
                        ];
                        let Some(wt1) = weighted_term(&wt_1) else {
                            continue;
                        };

                        let shifted_n = reflected_pos(pos, shift_rel(rel, n, 1), counts);
                        let wt_2 = [
                            inverse_product(l_at(self, n_pp, pos), c_at(self, n_p, shifted_n)),
                            inverse_product(l_at(self, n_pp, pos_pm), c_at(self, n_p, shifted_n)),
                            inverse_product(l_at(self, n_p, pos), c_at(self, n_pp, shifted_n)),
                            inverse_product(
                                l_at(
                                    self,
                                    n_p,
                                    reflected_pos(pos, shift_rel(rel, n_pp, -1), counts),
                                ),
                                c_at(self, n_pp, shifted_n),
                            ),
                        ];
                        let Some(wt2) = weighted_term(&wt_2) else {
                            continue;
                        };

                        let total = wqp + wt1 + wt2;
                        if total <= 0.0 || !total.is_finite() {
                            continue;
                        }
                        let dt = 2.0 / total.sqrt();
                        if dt > 0.0 && dt < min_dt {
                            min_dt = dt;
                        }
                    }
                }
            }
        }

        if min_dt.is_finite() {
            self.base.d_t = min_dt;
        }
        min_dt
    }

    /// Apply electric boundary conditions (PEC): zero the tangential voltage
    /// coefficients on PEC faces. The normal component is untouched, matching
    /// native `ApplyElectricBC`.
    pub fn apply_electric_bc(&mut self) {
        let vv = self.vv_ptr.as_mut().expect("vv_ptr not allocated");
        let vi = self.vi_ptr.as_mut().expect("vi_ptr not allocated");
        let bc = self.base.m_bc;

        let nx = vv.ni();
        let ny = vv.nj();
        let nz = vv.nk();

        // bc[0]=x-min, bc[1]=x-max, bc[2]=y-min, bc[3]=y-max, bc[4]=z-min, bc[5]=z-max
        // PEC = 0. Only the two tangential components are forced to zero.
        for &(face, i) in &[(0usize, 0usize), (1usize, nx.saturating_sub(1))] {
            if bc[face] != 0 {
                continue;
            }
            for n in [1usize, 2] {
                for j in 0..ny {
                    for k in 0..nz {
                        *vv.get_mut(n, i, j, k) = 0.0;
                        *vi.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
        for &(face, j) in &[(2usize, 0usize), (3usize, ny.saturating_sub(1))] {
            if bc[face] != 0 {
                continue;
            }
            for n in [0usize, 2] {
                for i in 0..nx {
                    for k in 0..nz {
                        *vv.get_mut(n, i, j, k) = 0.0;
                        *vi.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
        for &(face, k) in &[(4usize, 0usize), (5usize, nz.saturating_sub(1))] {
            if bc[face] != 0 {
                continue;
            }
            for n in [0usize, 1] {
                for i in 0..nx {
                    for j in 0..ny {
                        *vv.get_mut(n, i, j, k) = 0.0;
                        *vi.get_mut(n, i, j, k) = 0.0;
                    }
                }
            }
        }
    }

    /// Modify EC voltage coefficients at a specific mesh position.
    /// Used by extensions like conducting sheet to adjust vv/vi coefficients.
    pub fn set_ec_voltage_coeffs(
        &mut self,
        n: usize,
        i: usize,
        j: usize,
        k: usize,
        vv: FDTD_FLOAT,
        vi: FDTD_FLOAT,
    ) {
        *self
            .vv_ptr
            .as_mut()
            .expect("vv_ptr not allocated")
            .get_mut(n, i, j, k) = vv;
        *self
            .vi_ptr
            .as_mut()
            .expect("vi_ptr not allocated")
            .get_mut(n, i, j, k) = vi;
    }

    /// Apply magnetic boundary conditions (PMC), mirroring native
    /// `Operator::ApplyMagneticBC` exactly:
    ///
    /// * a PMC face on the low side clears all three current components on the
    ///   face line,
    /// * a PMC face on the high side clears the two tangential components on
    ///   the line *inside* the outermost one (`numLines - 2`),
    /// * the outermost current line of every axis is always cleared because it
    ///   lies outside the iterated FDTD domain.
    ///
    /// The unconditional last-line pass is part of native's loop body, so it
    /// runs whenever the PMC mask is built (i.e. always).
    pub fn apply_magnetic_bc(&mut self) {
        let bc = self.base.m_bc;
        let counts = self.base.num_lines;

        let ii = self.ii_ptr.as_mut().expect("ii_ptr not allocated");
        let iv = self.iv_ptr.as_mut().expect("iv_ptr not allocated");
        let dims = [ii.ni(), ii.nj(), ii.nk()];

        for n in 0..3 {
            let np = (n + 1) % 3;
            let npp = (n + 2) % 3;
            let lines = counts[n] as usize;
            if lines == 0 {
                continue;
            }
            let last = lines - 1;
            for a in 0..dims[np] {
                for b in 0..dims[npp] {
                    let mut pos = [0usize; 3];
                    pos[np] = a;
                    pos[npp] = b;

                    if bc[2 * n] == 1 {
                        pos[n] = 0;
                        for component in [n, np, npp] {
                            Self::clear_current_line(ii, iv, component, pos);
                        }
                    }
                    if bc[2 * n + 1] == 1 && last >= 1 {
                        pos[n] = last - 1;
                        for component in [np, npp] {
                            Self::clear_current_line(ii, iv, component, pos);
                        }
                    }
                    pos[n] = last;
                    for component in [n, np, npp] {
                        Self::clear_current_line(ii, iv, component, pos);
                    }
                }
            }
        }
    }

    /// Clear the `ii`/`iv` update coefficients of one current line.
    fn clear_current_line(ii: &mut FieldArray, iv: &mut FieldArray, n: usize, pos: [usize; 3]) {
        *ii.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
        *iv.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
    }

    // ----- private helpers -----

    /// Snap one coordinate to a mesh line index.
    ///
    /// Mirrors native `Operator::SnapToMeshLine` (`FDTD/operator.cpp:253-282`)
    /// with the `Operator_Cylinder` override in front of it
    /// (`FDTD/operator_cylinder.cpp:277-282`): an azimuth is folded into the
    /// alpha range with `FitToAlphaRange` before the base snap runs. The other
    /// directions, and every direction of a Cartesian mesh, are untouched.
    ///
    /// Returns the index and whether the coordinate was inside the mesh.
    pub fn snap_to_mesh_line(&self, axis: usize, coord: f64, dual: bool) -> (u32, bool) {
        let coord = if axis == 1 && self.is_cylindrical() {
            // operator_cylinder.cpp:279-280 — `FitToAlphaRange(coord)` with the
            // default `fullMesh = false`.
            self.fit_to_alpha_range(coord, false)
        } else {
            coord
        };
        let counts = self.base.num_lines;
        let n = counts[axis] as usize;
        if n == 0 {
            return (0, false);
        }
        let line = |pos: usize, dual_mesh: bool| {
            disc_line_value(&self.base.disc_lines[axis], n, pos, dual_mesh)
        };
        if coord < line(0, false) {
            return (0, false);
        }
        if coord > line(n - 1, false) {
            return ((n - 1) as u32, false);
        }
        if !dual {
            for pos in 0..n {
                if coord <= line(pos, true) {
                    return (pos as u32, true);
                }
            }
        } else {
            for pos in 1..n {
                if coord <= line(pos, false) {
                    return ((pos - 1) as u32, true);
                }
            }
        }
        (0, true)
    }

    /// Snap a coordinate triple to the mesh. Mirrors native
    /// `Operator::SnapToMesh`; `full_mesh` is accepted for call compatibility
    /// and, like native `Operator::GetNumberOfLines`, does not change the line
    /// count.
    pub fn snap_to_mesh_native(
        &self,
        coord: [f64; 3],
        dual: bool,
        _full_mesh: bool,
    ) -> ([u32; 3], [bool; 3]) {
        let mut index = [0u32; 3];
        let mut inside = [false; 3];
        for axis in 0..3 {
            let (snapped, is_inside) = self.snap_to_mesh_line(axis, coord[axis], dual);
            index[axis] = snapped;
            inside[axis] = is_inside;
        }
        (index, inside)
    }

    /// Snap a box to the mesh. Mirrors native `Operator::SnapBox2Mesh` for the
    /// three supported snap methods:
    ///
    /// * `0` — keep the snapped indices,
    /// * `1` — expand the box outward by one line where that keeps the snapped
    ///   coordinate inside the original box,
    /// * `2` — shrink the box inward by one line.
    ///
    /// A negative result means the box lies outside the domain (`-2`) or that
    /// the snap method is unknown (`-1`).
    pub fn snap_box_to_mesh_native(
        &self,
        start: [f64; 3],
        stop: [f64; 3],
        dual: bool,
        full_mesh: bool,
        snap_method: i32,
    ) -> (i32, [u32; 3], [u32; 3], [bool; 3], [bool; 3]) {
        // Native `Operator_Cylinder::SnapBox2Mesh`
        // (`FDTD/operator_cylinder.cpp:283-304`): the azimuth span is
        // re-centred on the wrapped midpoint and clamped into the mesh range
        // before the base snap runs. Without it a seam-crossing or full-circle
        // box folds to an empty span.
        let (start, stop) = if self.is_cylindrical() {
            let count = self.alpha_line_count(full_mesh) as usize;
            if count > 0 {
                let lines = self.base.num_lines[1] as usize;
                let a_min = disc_line_value(&self.base.disc_lines[1], lines, 0, false);
                let a_max = disc_line_value(&self.base.disc_lines[1], lines, count - 1, false);
                let size = stop[1] - start[1];
                let centre = self.fit_to_alpha_range(0.5 * (start[1] + stop[1]), false);
                let raw_start = centre - size * 0.5;
                let a_start = raw_start.clamp(a_min, a_max);
                let a_stop = (raw_start + size).clamp(a_min, a_max);
                let mut start = start;
                let mut stop = stop;
                start[1] = a_start;
                stop[1] = a_stop;
                (start, stop)
            } else {
                (start, stop)
            }
        } else {
            (start, stop)
        };
        let start = start;
        let stop = stop;
        let mut lower = [0.0f64; 3];
        let mut upper = [0.0f64; 3];
        for axis in 0..3 {
            lower[axis] = start[axis].min(stop[axis]);
            upper[axis] = start[axis].max(stop[axis]);
            let counts = self.base.num_lines;
            let n = counts[axis] as usize;
            let minimum = disc_line_value(&self.base.disc_lines[axis], n, 0, false);
            let maximum = disc_line_value(&self.base.disc_lines[axis], n, n - 1, false);
            if (lower[axis] < minimum && upper[axis] < minimum)
                || (lower[axis] > maximum && upper[axis] > maximum)
            {
                return (-2, [0; 3], [0; 3], [false; 3], [false; 3]);
            }
        }
        let (mut ui_start, start_in) = self.snap_to_mesh_native(lower, dual, full_mesh);
        let (mut ui_stop, stop_in) = self.snap_to_mesh_native(upper, dual, full_mesh);

        let mut dimension = 0i32;
        match snap_method {
            0 => {
                for axis in 0..3 {
                    if ui_stop[axis] > ui_start[axis] {
                        dimension += 1;
                    }
                }
            }
            1 | 2 => {
                let counts = self.base.num_lines;
                for axis in 0..3 {
                    let n = counts[axis] as usize;
                    if ui_stop[axis] > ui_start[axis] {
                        let line = |pos: u32| {
                            disc_line_value(&self.base.disc_lines[axis], n, pos as usize, dual)
                        };
                        if snap_method == 1 {
                            if line(ui_start[axis]) > lower[axis] && ui_start[axis] > 0 {
                                ui_start[axis] -= 1;
                            }
                            if line(ui_stop[axis]) < upper[axis] && (ui_stop[axis] as usize) < n - 1
                            {
                                ui_stop[axis] += 1;
                            }
                        } else {
                            if line(ui_start[axis]) < lower[axis]
                                && (ui_start[axis] as usize) < n - 1
                            {
                                ui_start[axis] += 1;
                            }
                            if line(ui_stop[axis]) > upper[axis] && ui_stop[axis] > 0 {
                                ui_stop[axis] -= 1;
                            }
                        }
                    }
                    if ui_stop[axis] > ui_start[axis] {
                        dimension += 1;
                    }
                }
            }
            _ => return (-1, ui_start, ui_stop, start_in, stop_in),
        }
        (dimension, ui_start, ui_stop, start_in, stop_in)
    }

    /// Snap a start/stop coordinate pair along a line onto the mesh.
    ///
    /// The shared entry point for native `SnapLine2Mesh`: the base version
    /// (`FDTD/operator.cpp:361-389`) plus, on a cylindrical mesh, the
    /// `Operator_Cylinder` override (`FDTD/operator_cylinder.cpp:307-317`) that
    /// re-anchors a stop index which the base snap folded over the azimuth
    /// seam. `None` is the native `-1` return: both endpoints of at least one
    /// axis lie on the same side outside the mesh, i.e. the line cannot be
    /// snapped at all. The native positive return value only reports which
    /// endpoints were clamped and is not used by `FindPath`, so it is not
    /// modelled.
    pub fn snap_line_to_mesh_native(
        &self,
        start: [f64; 3],
        stop: [f64; 3],
        dual: bool,
        full_mesh: bool,
    ) -> Option<([u32; 3], [u32; 3])> {
        let mut snapped = self.snap_line_to_mesh_base(start, stop, dual, full_mesh)?;
        if self.is_cylindrical() {
            // operator_cylinder.cpp:311-314. `GetNumberOfLines(1, fullMesh)` is
            // the full line count when `fullMesh` is set, which is the only way
            // `FindPath` calls this; the pad line is dropped from the closed
            // count, then `-1` moves from the line to the cell index.
            let last_alpha = self.alpha_line_count(full_mesh) - 1 - u32::from(self.closed_alpha());
            if stop[1] > start[1] && snapped.1[1] < snapped.0[1] && snapped.1[1] == 0 {
                snapped.1[1] = last_alpha;
            }
            if stop[1] < start[1] && snapped.1[1] > snapped.0[1] && snapped.1[1] == last_alpha {
                snapped.1[1] = 0;
            }
        }
        Some(snapped)
    }

    /// Native base `Operator::SnapLine2Mesh` (`FDTD/operator.cpp:361-389`),
    /// without the cylindrical seam remap.
    fn snap_line_to_mesh_base(
        &self,
        start: [f64; 3],
        stop: [f64; 3],
        dual: bool,
        full_mesh: bool,
    ) -> Option<([u32; 3], [u32; 3])> {
        let (ui_start, _) = self.snap_to_mesh_native(start, dual, full_mesh);
        let (ui_stop, _) = self.snap_to_mesh_native(stop, dual, full_mesh);

        let counts = self.base.num_lines;
        for axis in 0..3 {
            let n = counts[axis] as usize;
            if n == 0 {
                return None;
            }
            // Native `GetNumberOfLines(n, true)` (operator.cpp:372-373): the
            // base operator ignores the flag and returns `numLines[n]`.
            let lower = disc_line_value(&self.base.disc_lines[axis], n, 0, false);
            let upper = disc_line_value(&self.base.disc_lines[axis], n, n - 1, false);
            if start[axis] < lower && stop[axis] < lower {
                return None;
            }
            if start[axis] > upper && stop[axis] > upper {
                return None;
            }
        }

        Some((ui_start, ui_stop))
    }

    /// Distance of `p` to the line `start`-`stop` and the relative foot
    /// position along it.
    ///
    /// Mirrors native `Point_Line_Distance` (`CSXCAD/src/CSPrimitives.cpp:36-52`)
    /// for the operator mesh: for a cylindrical mesh all three points are
    /// converted to Cartesian coordinates first (`TransformCoordSystem`,
    /// `CSXCAD/src/ParameterCoord.cpp:22-62`), so the distance and the foot
    /// parameter are the genuine cylindrical-mesh metric rather than the naive
    /// `(rho, alpha, z)` box metric.
    pub fn point_line_distance(&self, p: [f64; 3], start: [f64; 3], stop: [f64; 3]) -> (f64, f64) {
        let to_cartesian = |c: [f64; 3]| {
            if self.is_cylindrical() {
                let [rho, alpha, z] = c;
                [rho * alpha.cos(), rho * alpha.sin(), z]
            } else {
                c
            }
        };
        let p = to_cartesian(p);
        let start = to_cartesian(start);
        let stop = to_cartesian(stop);

        let dir = [stop[0] - start[0], stop[1] - start[1], stop[2] - start[2]];
        // Native `pow(dir[n],2)` — the same correctly rounded square.
        let length_squared = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        let mut foot =
            (p[0] - start[0]) * dir[0] + (p[1] - start[1]) * dir[1] + (p[2] - start[2]) * dir[2];
        foot /= length_squared;

        let foot_point = [
            start[0] + foot * dir[0],
            start[1] + foot * dir[1],
            start[2] + foot * dir[2],
        ];
        let delta = [
            p[0] - foot_point[0],
            p[1] - foot_point[1],
            p[2] - foot_point[2],
        ];
        let dist = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        (foot, dist)
    }

    /// Path of mesh edges from `start` to `stop`.
    ///
    /// The shared entry point for native `FindPath`: a Cartesian mesh (and any
    /// non-cylindrical operator) walks the base implementation, a cylindrical
    /// mesh walks `Operator_Cylinder::FindPath` (`FDTD/operator_cylinder.cpp:320-475`),
    /// which folds the azimuth, swaps decreasing endpoints, and stitches a
    /// segment that crosses the closed-azimuth seam. Both curve consumers —
    /// the excitation curve branch and `calc_pec_curves` — call this method and
    /// therefore take the override automatically, exactly like the native
    /// virtual call.
    pub fn find_path(&self, start: [f64; 3], stop: [f64; 3]) -> GridPath {
        if self.is_cylindrical() {
            self.find_path_cylindrical(start, stop)
        } else {
            self.find_path_base(start, stop)
        }
    }

    /// Fold an azimuth into the alpha range of the mesh.
    ///
    /// Mirrors native `Operator_Cylinder::FitToAlphaRange`
    /// (`FDTD/operator_cylinder.cpp:226-250`) verbatim, including the two
    /// `while` loops, the "cannot happen" tail and the mesh range it folds
    /// into: `GetDiscLine(1, GetNumberOfLines(1, fullMesh) - 1)`. On a closed
    /// azimuth with `full_mesh == false` that upper bound is one cell short of
    /// `2*pi`, so a coordinate can be returned unchanged even though it lies
    /// above the bound (native `operator_cylinder.cpp:240-247` returns the
    /// original value then) — the callers rely on that.
    pub fn fit_to_alpha_range(&self, coord: f64, full_mesh: bool) -> f64 {
        if !self.is_cylindrical() {
            return coord;
        }
        let metric = self.metric();
        let count = self.alpha_line_count(full_mesh) as usize;
        if count == 0 {
            return coord;
        }
        let min = metric.line(1, 0, false);
        let max = metric.line(1, count - 1, false);
        let mut a_coord = coord;
        if (a_coord >= min) && (a_coord <= max) {
            return a_coord;
        }
        while a_coord < min {
            a_coord += 2.0 * std::f64::consts::PI;
            if a_coord > max {
                return a_coord - 2.0 * std::f64::consts::PI;
            }
            if a_coord > min {
                return a_coord;
            }
        }
        while a_coord > max {
            a_coord -= 2.0 * std::f64::consts::PI;
            if a_coord < min {
                return a_coord + 2.0 * std::f64::consts::PI;
            }
            if a_coord < max {
                return a_coord;
            }
        }
        // this cannot happen
        a_coord
    }

    /// Cylindrical `FindPath`. Mirrors native `Operator_Cylinder::FindPath`
    /// (`FDTD/operator_cylinder.cpp:320-475`).
    fn find_path_cylindrical(&self, start: [f64; 3], stop: [f64; 3]) -> GridPath {
        let empty = || GridPath {
            pos_path: [Vec::new(), Vec::new(), Vec::new()],
            dir: Vec::new(),
        };
        let counts = self.base.num_lines;
        let metric = self.metric();
        let mesh_system = CoordinateSystem::Cylindrical;

        let mut l_start = start;
        let mut l_stop = stop;

        // operator_cylinder.cpp:331-337: fold the azimuth difference into
        // +/-pi, so a span above half a turn is walked the short way round.
        while (l_stop[1] - l_start[1]).abs() > std::f64::consts::PI {
            if l_stop[1] > l_start[1] {
                l_stop[1] -= 2.0 * std::f64::consts::PI;
            } else {
                l_stop[1] += 2.0 * std::f64::consts::PI;
            }
        }

        // operator_cylinder.cpp:339-348: the greedy walk only moves forward, so
        // a decreasing azimuth swaps the two endpoints component by component.
        if l_start[1] > l_stop[1] {
            for n in 0..3 {
                std::mem::swap(&mut l_start[n], &mut l_stop[n]);
            }
        }

        // operator_cylinder.cpp:350-358
        let a_start = self.fit_to_alpha_range(l_start[1], false);
        let a_stop = self.fit_to_alpha_range(l_stop[1], false);
        if a_stop >= a_start {
            l_start[1] = a_start;
            l_stop[1] = a_stop;
            // Native `Operator_Multithread::FindPath` — the base walk, whose
            // `SnapLine2Mesh` is still the virtual cylindrical one.
            return self.find_path_base(l_start, l_stop);
        }

        // operator_cylinder.cpp:362-369: only a line whose two endpoints lie on
        // the same side of an axis fails; the bounds are the full mesh.
        for n in 0..3 {
            let lower = metric.line(n, 0, false);
            let upper = metric.line(n, counts[n] as usize - 1, false);
            if (l_start[n] < lower) && (l_stop[n] < lower) {
                return empty(); // lower bound violation
            }
            if (l_start[n] > upper) && (l_stop[n] > upper) {
                return empty(); // upper bound violation
            }
        }

        // operator_cylinder.cpp:374-465: the segment steps over the alpha
        // bounds, so both ends are walked separately and stitched together.
        // The seam plane sits at the last alpha line that is not the pad.
        let seam_index = counts[1] as usize - 1 - usize::from(self.closed_alpha());
        let seam_alpha = metric.line(1, seam_index, false);

        // operator_cylinder.cpp:381-403: plane through the alpha = max corner.
        let mut p0 = [
            metric.line(0, 0, false),
            metric.line(1, 0, false),
            metric.line(2, 0, false),
        ];
        let mut p1 = p0;
        let mut p2 = p0;
        p0[1] = seam_alpha;
        p1[1] = seam_alpha;
        p2[1] = seam_alpha;
        p1[0] = metric.line(0, counts[0] as usize - 1, false);
        p2[2] = metric.line(2, counts[2] as usize - 1, false);
        let p0 = transform_coordinates(p0, mesh_system, CoordinateSystem::Cartesian);
        let p1 = transform_coordinates(p1, mesh_system, CoordinateSystem::Cartesian);
        let p2 = transform_coordinates(p2, mesh_system, CoordinateSystem::Cartesian);
        let c_start = transform_coordinates(l_start, mesh_system, CoordinateSystem::Cartesian);
        let c_stop = transform_coordinates(l_stop, mesh_system, CoordinateSystem::Cartesian);

        let (mut ret, mut intersect) = line_plane_intersection(p0, p1, p2, c_start, c_stop);
        if ret < 0 {
            eprintln!("FindPath: Error, unable to calculate intersection, this should not happen!");
            return empty(); // return empty path;
        }

        let mut path1 = empty();
        let mut path2 = empty();

        if ret == 0 {
            // operator_cylinder.cpp:410-418
            intersect = transform_coordinates(intersect, CoordinateSystem::Cartesian, mesh_system);
            intersect[1] = seam_alpha;
            l_start[1] = self.fit_to_alpha_range(l_start[1], false);
            // Native calls `Operator::FindPath` here, i.e. the base walk.
            path1 = self.find_path_base(l_start, intersect);
        }

        if !self.closed_alpha() {
            // operator_cylinder.cpp:420-440: the same plane at the alpha
            // minimum. `ret` and `intersect` are overwritten, so on a closed
            // azimuth the values of the upper bound above are reused below —
            // that is native behaviour, not an omission.
            let q0 = [
                metric.line(0, 0, false),
                metric.line(1, 0, false),
                metric.line(2, 0, false),
            ];
            let mut q1 = q0;
            let mut q2 = q0;
            q1[0] = metric.line(0, counts[0] as usize - 1, false);
            q2[2] = metric.line(2, counts[2] as usize - 1, false);
            let q0 = transform_coordinates(q0, mesh_system, CoordinateSystem::Cartesian);
            let q1 = transform_coordinates(q1, mesh_system, CoordinateSystem::Cartesian);
            let q2 = transform_coordinates(q2, mesh_system, CoordinateSystem::Cartesian);
            let c_start = transform_coordinates(l_start, mesh_system, CoordinateSystem::Cartesian);
            let c_stop = transform_coordinates(l_stop, mesh_system, CoordinateSystem::Cartesian);

            let (lower_ret, lower_intersect) = line_plane_intersection(q0, q1, q2, c_start, c_stop);
            ret = lower_ret;
            intersect =
                transform_coordinates(lower_intersect, CoordinateSystem::Cartesian, mesh_system);
        }

        if ret == 0 {
            // operator_cylinder.cpp:442-449
            intersect[1] = metric.line(1, 0, false);
            l_stop[1] = self.fit_to_alpha_range(l_stop[1], false);
            path2 = self.find_path_base(intersect, l_stop);
        }

        // operator_cylinder.cpp:451-465: path1 then path2, in that order.
        let mut path = empty();
        for t in 0..path1.dir.len() {
            path.pos_path[0].push(path1.pos_path[0][t]);
            path.pos_path[1].push(path1.pos_path[1][t]);
            path.pos_path[2].push(path1.pos_path[2][t]);
            path.dir.push(path1.dir[t]);
        }
        for t in 0..path2.dir.len() {
            path.pos_path[0].push(path2.pos_path[0][t]);
            path.pos_path[1].push(path2.pos_path[1][t]);
            path.pos_path[2].push(path2.pos_path[2][t]);
            path.dir.push(path2.dir[t]);
        }

        // operator_cylinder.cpp:467-472: on a closed azimuth the radial and z
        // edges of the second half still sit on alpha line 0; move them to the
        // duplicated seam plane so both halves meet.
        if self.closed_alpha() {
            for t in 0..path.dir.len() {
                if ((path.dir[t] == 0) || (path.dir[t] == 2)) && (path.pos_path[1][t] == 0) {
                    path.pos_path[1][t] = counts[1] - 2;
                }
            }
        }

        path
    }

    /// Native base `Operator::FindPath` (`FDTD/operator.cpp:391-497`).
    ///
    /// `SnapLine2Mesh` with the main mesh (`dualMesh=false`, `fullMesh=true`),
    /// then a greedy walk over the six neighbouring nodes that only accepts a
    /// foot parameter strictly larger than the current one (`foot > currFoot`)
    /// and, among those, the first node with the smallest distance
    /// (`dist < minDist`, checked in the native order: axis 0 lower, axis 0
    /// upper, axis 1 lower, ...). The walk stops once no neighbour moves the
    /// foot past `stopFoot`, and the remaining axis-aligned edges to the
    /// snapped stop index are appended afterwards.
    ///
    /// An unsnappable line or a walk that leaves the domain yields the empty
    /// path, exactly like the native early returns (`operator.cpp:397-398` and
    /// `465-468`).
    fn find_path_base(&self, start: [f64; 3], stop: [f64; 3]) -> GridPath {
        let empty = || GridPath {
            pos_path: [Vec::new(), Vec::new(), Vec::new()],
            dir: Vec::new(),
        };

        // operator.cpp:396-398
        let Some((ui_start, ui_stop)) = self.snap_line_to_mesh_native(start, stop, false, true)
        else {
            return empty();
        };

        let counts = self.base.num_lines;
        let metric = self.metric();
        // Native indexes `discLines[n][pos]` directly (operator.cpp:403-404);
        // `MeshMetric::line` with `dual = false` is the same value.
        let line = |axis: usize, pos: u32| metric.line(axis, pos as usize, false);

        let mut curr_pos = ui_start;
        let mesh_start = [
            line(0, ui_start[0]),
            line(1, ui_start[1]),
            line(2, ui_start[2]),
        ];
        let mesh_stop = [
            line(0, ui_stop[0]),
            line(1, ui_stop[1]),
            line(2, ui_stop[2]),
        ];

        let (start_foot, _) = self.point_line_distance(mesh_start, start, stop);
        let (stop_foot, _) = self.point_line_distance(mesh_stop, start, stop);
        let mut curr_foot = start_foot;
        let mut min_foot = start_foot;
        let mut up_dir = false;
        let mut min_dir = 0usize;
        let mut path = empty();

        // operator.cpp:416-474
        while min_foot < stop_foot {
            let mut min_dist = 1e300f64;
            for n in 0..3 {
                // The probe point is rebuilt from `currPos` for every axis, so
                // the lower/upper probes never leak into each other (native
                // operator.cpp:421-423).
                let mut p = [
                    line(0, curr_pos[0]),
                    line(1, curr_pos[1]),
                    line(2, curr_pos[2]),
                ];
                if curr_pos[n] >= 1 {
                    p[n] = line(n, curr_pos[n] - 1);
                    let (foot, dist) = self.point_line_distance(p, start, stop);
                    if foot > curr_foot && dist < min_dist {
                        min_foot = foot;
                        min_dist = dist;
                        min_dir = n;
                        up_dir = false;
                    }
                }
                if (curr_pos[n] + 1) < counts[n] {
                    p[n] = line(n, curr_pos[n] + 1);
                    let (foot, dist) = self.point_line_distance(p, start, stop);
                    if foot > curr_foot && dist < min_dist {
                        min_foot = foot;
                        min_dist = dist;
                        min_dir = n;
                        up_dir = true;
                    }
                }
            }

            let mut min_pos = curr_pos;
            if up_dir {
                curr_pos[min_dir] += 1;
            } else {
                // Native decrements an unsigned index here; when no neighbour
                // qualified the sign flag is stale, so the wrap and the domain
                // check below reproduce the native error return.
                curr_pos[min_dir] = curr_pos[min_dir].wrapping_sub(1);
                min_pos[min_dir] = min_pos[min_dir].wrapping_sub(1);
            }
            for n in 0..3 {
                if curr_pos[n] >= counts[n] {
                    // operator.cpp:462-468
                    eprintln!(
                        "FindPath: Error, path went out of simulation domain, skipping path!"
                    );
                    return empty();
                }
            }
            path.pos_path[0].push(min_pos[0]);
            path.pos_path[1].push(min_pos[1]);
            path.pos_path[2].push(min_pos[2]);
            curr_foot = min_foot;
            path.dir.push(min_dir as u16);
        }

        // operator.cpp:476-494: close the missing edges between the last walk
        // position and the snapped stop index.
        for n in 0..3 {
            if curr_pos[n] > ui_stop[n] {
                curr_pos[n] -= 1;
                path.pos_path[0].push(curr_pos[0]);
                path.pos_path[1].push(curr_pos[1]);
                path.pos_path[2].push(curr_pos[2]);
                path.dir.push(n as u16);
            } else if curr_pos[n] < ui_stop[n] {
                path.pos_path[0].push(curr_pos[0]);
                path.pos_path[1].push(curr_pos[1]);
                path.pos_path[2].push(curr_pos[2]);
                path.dir.push(n as u16);
            }
        }

        path
    }

    fn get_ec_c(&self, n: i32, i: usize, j: usize, k: usize) -> f64 {
        let dir = n.rem_euclid(3) as usize;
        let pos = [i, j, k];
        let metric = self.metric();
        let area = metric.edge_area(dir, pos, false);
        let delta = metric.edge_length(dir, pos, false);
        if delta <= 0.0 {
            return 0.0;
        }
        let eps_r = if let Some(grid) = &self.material_grid {
            let nx = self.base.num_lines[0] as usize;
            let ny = self.base.num_lines[1] as usize;
            let nz = self.base.num_lines[2] as usize;
            let index = dir * nx * ny * nz + i * ny * nz + j * nz + k;
            grid.get(index).map_or(self.base.m_bg_eps_r, |m| m.eps_r)
        } else {
            self.base.m_bg_eps_r
        };
        eps_r * EPS0 * area / delta
    }

    fn get_ec_l(&self, n: i32, i: usize, j: usize, k: usize) -> f64 {
        let dir = n.rem_euclid(3) as usize;
        let pos = [i, j, k];
        let metric = self.metric();
        let area = metric.edge_area(dir, pos, true);
        let delta = metric.edge_length(dir, pos, true);
        if delta <= 0.0 {
            return 0.0;
        }
        let mue_r = if let Some(grid) = &self.material_grid {
            let nx = self.base.num_lines[0] as usize;
            let ny = self.base.num_lines[1] as usize;
            let nz = self.base.num_lines[2] as usize;
            let index = dir * nx * ny * nz + i * ny * nz + j * nz + k;
            grid.get(index).map_or(self.base.m_bg_mue_r, |m| m.mue_r)
        } else {
            self.base.m_bg_mue_r
        };
        mue_r * MU0 * area / delta
    }

    /// Compute EC[4] = [C, G, L, R] for direction `ny` at position `pos`.
    /// Mirrors C++ `Calc_ECPos`.
    /// Kept for future CSXCAD integration; currently unused (background-material only).
    #[allow(dead_code)]
    fn calc_ec_pos(&self, ny: i32, pos: [u32; 3]) -> [f64; 4] {
        let eff = self.calc_eff_mat_pos(ny, pos);
        let eps_r = eff[0];
        let kappa = eff[1];
        let mue_r = eff[2];
        let sigma = eff[3];

        let area = self.get_edge_area(ny, [pos[0], pos[1], pos[2]], false);
        let delta = self.get_edge_length(ny, [pos[0], pos[1], pos[2]], false);

        let c = eps_r * EPS0 * area / delta;
        let g = kappa * area / delta;
        let l = mue_r * MU0 * area / delta;
        let r = sigma * area / delta;

        [c, g, l, r]
    }

    /// Compute effective material [epsR, kappa, mueR, sigma] at position.
    /// Mirrors C++ `Calc_EffMatPos` / `AverageMatQuarterCell`.
    /// Checks per-cell material grid first, falls back to background material.
    fn calc_eff_mat_pos(&self, ny: i32, pos: [u32; 3]) -> [f64; 4] {
        if let Some(grid) = &self.material_grid {
            let num_lines = self.base.num_lines;
            let nx = num_lines[0] as usize;
            let ny_lines = num_lines[1] as usize;
            let nz = num_lines[2] as usize;
            let i = pos[0] as usize;
            let j = pos[1] as usize;
            let k = pos[2] as usize;
            if i < nx && j < ny_lines && k < nz {
                let dir = ny.rem_euclid(3) as usize;
                let index = dir * nx * ny_lines * nz + i * ny_lines * nz + j * nz + k;
                if index < grid.len() {
                    let m = grid[index];
                    return [m.eps_r, m.kappa, m.mue_r, m.sigma];
                }
            }
        }
        [
            self.base.m_bg_eps_r,
            self.base.m_bg_kappa,
            self.base.m_bg_mue_r,
            self.base.m_bg_sigma,
        ]
    }

    /// Material (eps_r, mue_r) for a component at a mesh position, falling back
    /// to the background material.
    fn material_value(&self, component: usize, pos: [usize; 3]) -> (f64, f64) {
        if let Some(grid) = &self.material_grid {
            let counts = self.base.num_lines;
            let nx = counts[0] as usize;
            let ny = counts[1] as usize;
            let nz = counts[2] as usize;
            let index = component * nx * ny * nz + pos[0] * ny * nz + pos[1] * nz + pos[2];
            if let Some(material) = grid.get(index) {
                return (material.eps_r, material.mue_r);
            }
        }
        (self.base.m_bg_eps_r, self.base.m_bg_mue_r)
    }

    /// Build first-order Mur ABC faces from the boundary conditions.
    /// Mirrors the coefficient computation in `Operator_Ext_Mur_ABC`.
    pub fn build_mur_extension(&self, phase_velocity: [f64; 6]) -> EngineExtMurABC {
        let mut extension = EngineExtMurABC::new();
        let counts = self.base.num_lines;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return extension;
        }
        for face in 0..6usize {
            if self.base.m_bc[face] != 2 {
                continue; // BC_MUR
            }
            let dir = face / 2;
            let top = face % 2 == 1;
            let line = if top {
                counts[dir].saturating_sub(1) as usize
            } else {
                0
            };
            let shift = if top {
                counts[dir].saturating_sub(2) as usize
            } else {
                1
            };
            let n_p = (dir + 1) % 3;
            let n_pp = (dir + 2) % 3;
            let dims_p = counts[n_p] as usize;
            let dims_pp = counts[n_pp] as usize;
            let mut pos = [0usize; 3];
            pos[dir] = line;
            let delta = self.metric().edge_length(dir, pos, false).abs();
            if delta <= 0.0 {
                continue;
            }

            let mut coeff_p = vec![0.0f32; dims_p * dims_pp];
            let mut coeff_pp = vec![0.0f32; dims_p * dims_pp];
            for ip in 0..dims_p {
                for ipp in 0..dims_pp {
                    let idx = ip * dims_pp + ipp;
                    let mut cell = [0usize; 3];
                    cell[dir] = shift.min(counts[dir].saturating_sub(1) as usize);
                    cell[n_p] = ip;
                    cell[n_pp] = ipp;
                    let (eps_p, mue_p) = self.material_value(n_p, cell);
                    let (eps_pp, mue_pp) = self.material_value(n_pp, cell);
                    let c_p = if phase_velocity[face] > 0.0 {
                        phase_velocity[face] * dt
                    } else {
                        C0 * dt / (eps_p * mue_p).sqrt()
                    };
                    let c_pp = if phase_velocity[face] > 0.0 {
                        phase_velocity[face] * dt
                    } else {
                        C0 * dt / (eps_pp * mue_pp).sqrt()
                    };
                    coeff_p[idx] = ((c_p - delta) / (c_p + delta)) as FDTD_FLOAT;
                    coeff_pp[idx] = ((c_pp - delta) / (c_pp + delta)) as FDTD_FLOAT;
                }
            }
            extension.add_face(
                face, dir, top, line, shift, dims_p, dims_pp, coeff_p, coeff_pp,
            );
        }
        extension
    }

    /// Build the excitation extension by mapping CSXCAD Excitation properties
    /// onto Yee edges. Mirrors `Operator_Ext_Excitation::BuildExtension`.
    pub fn build_excitation_extension(&mut self, signal: &Excitation) -> EngineExtExcitation {
        let mut extension = EngineExtExcitation::new();
        let counts = self.base.num_lines;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return extension;
        }

        let mut hard_voltage: Vec<[usize; 4]> = Vec::new();
        let mut hard_current: Vec<[usize; 4]> = Vec::new();
        {
            let Some(structure) = self.structure.as_ref() else {
                return extension;
            };
            let params = structure.parameters();
            let mut candidates: Vec<(
                i32,
                &ExcitationProperties,
                &Primitive,
                csxcad_core::grid::CoordinateSystem,
            )> = Vec::new();
            for index in 0..structure.property_count() {
                let Some(property) = structure.property_by_index(index) else {
                    continue;
                };
                if let Some(excitation) = property.excitation() {
                    let coordinate_system = property.coordinate_input_type();
                    for primitive in property.primitives() {
                        candidates.push((
                            primitive.priority(),
                            excitation,
                            primitive,
                            coordinate_system,
                        ));
                    }
                }
            }
            if candidates.is_empty() {
                return extension;
            }

            let shared = Arc::new(signal.clone());
            let disc_lines = &self.base.disc_lines;
            let grid_delta = self.base.grid_delta;
            let metric = MeshMetric::new(disc_lines, counts, grid_delta, self.mesh_type);

            for n in 0..3usize {
                for i in 0..counts[0] as usize {
                    for j in 0..counts[1] as usize {
                        for k in 0..counts[2] as usize {
                            for dual in [false, true] {
                                if dual
                                    && (i + 1 >= counts[0] as usize
                                        || j + 1 >= counts[1] as usize
                                        || k + 1 >= counts[2] as usize)
                                {
                                    continue;
                                }
                                let coordinate =
                                    yee_coordinates(disc_lines, counts, n, [i, j, k], dual);
                                let mut winner: Option<&ExcitationProperties> = None;
                                let mut winner_priority = i32::MIN;
                                let mut winner_coordinate_system =
                                    csxcad_core::grid::CoordinateSystem::Cartesian;
                                for (priority, excitation, primitive, coordinate_system) in
                                    &candidates
                                {
                                    if winner.is_some() && *priority <= winner_priority {
                                        continue;
                                    }
                                    if primitive
                                        .is_inside(coordinate, Some(params))
                                        .unwrap_or(false)
                                    {
                                        winner = Some(*excitation);
                                        winner_priority = *priority;
                                        winner_coordinate_system = *coordinate_system;
                                    }
                                }
                                let Some(excitation) = winner else {
                                    continue;
                                };
                                let coordinate_system = winner_coordinate_system;
                                if !excitation.is_enabled()
                                    || excitation.active_direction(n) != Some(true)
                                {
                                    continue;
                                }
                                let excitation_type = excitation.excitation_type();
                                let is_voltage = matches!(excitation_type, 0 | 1);
                                let is_current = matches!(excitation_type, 2 | 3);
                                if dual != is_current {
                                    continue;
                                }
                                // Native `GetWeightedExcitation`: the weight
                                // function is evaluated at the Yee vertex that
                                // the operator exposes (metres, as returned by
                                // `GetYeeCoords`), including the origin shift.
                                let vertex = coordinate;
                                let weight = excitation
                                    .weighted_excitation_weight(
                                        n,
                                        Some(params),
                                        vertex,
                                        coordinate_system,
                                    )
                                    .unwrap_or(1.0);
                                let base_amp = excitation
                                    .excitation(n, Some(params))
                                    .ok()
                                    .flatten()
                                    .unwrap_or(0.0)
                                    * weight;
                                if base_amp == 0.0 {
                                    continue;
                                }
                                let length = metric.edge_length(n, [i, j, k], dual);
                                let amp = (base_amp * length) as FDTD_FLOAT;
                                if amp == 0.0 {
                                    continue;
                                }
                                let delay = (excitation.delay(Some(params)).unwrap_or(0.0) / dt)
                                    .max(0.0) as u32;
                                if is_voltage {
                                    extension.add_voltage_excitation(
                                        n,
                                        [i, j, k],
                                        delay,
                                        amp,
                                        shared.clone(),
                                    );
                                    if excitation_type == 1 {
                                        hard_voltage.push([n, i, j, k]);
                                    }
                                } else if is_current {
                                    extension.add_current_excitation(
                                        n,
                                        [i, j, k],
                                        delay,
                                        amp,
                                        shared.clone(),
                                    );
                                    if excitation_type == 3 {
                                        hard_current.push([n, i, j, k]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Special treatment for primitives of type curve (treated as wires):
        // `FDTD/extensions/operator_ext_excitation.cpp:235-288`. A curve is a
        // one-dimensional object, so the containment test above never selects
        // an excitation edge; instead every segment between two consecutive
        // curve points is walked with `FindPath` and each edge of the path is
        // excited in its own axis direction.
        let mut used_curves: Vec<(usize, usize)> = Vec::new();
        {
            let Some(structure) = self.structure.as_ref() else {
                return extension;
            };
            let params = structure.parameters();
            let shared = Arc::new(signal.clone());
            let metric = MeshMetric::new(
                &self.base.disc_lines,
                counts,
                self.base.grid_delta,
                self.mesh_type,
            );
            // Native `m_Op->m_MeshType`: the coordinate system the curve points
            // are resolved into before `FindPath` (operator_ext_excitation.cpp:251).
            let mesh_system = if self.is_cylindrical() {
                csxcad_core::grid::CoordinateSystem::Cylindrical
            } else {
                csxcad_core::grid::CoordinateSystem::Cartesian
            };

            for index in 0..structure.property_count() {
                let Some(property) = structure.property_by_index(index) else {
                    continue;
                };
                let Some(excitation) = property.excitation() else {
                    continue;
                };
                let coordinate_system = property.coordinate_input_type();
                for (primitive_index, primitive) in property.primitives().iter().enumerate() {
                    // Native `prim->ToCurve()` (`CSXCAD/src/CSPrimitives.h:163`)
                    // matches only a primitive whose CSXCAD type is `CURVE`; a
                    // `WIRE` keeps `Type == WIRE` (`CSPrimWire.cpp:30`) and is
                    // covered by the containment path above instead.
                    let Some(curve) = primitive.curve_shape() else {
                        continue;
                    };
                    // Native `CSPrimCurve::GetPoint(i, p, m_MeshType)`: the
                    // stored points live in the primitive coordinate system
                    // (`m_PrimCoordSystem`, falling back to the mesh type) and
                    // are reported in the operator mesh type, with the forward
                    // primitive transform applied (`CSPrimitives::TransformCoords`).
                    let point_system = if primitive.coordinate_system()
                        == csxcad_core::grid::CoordinateSystem::Undefined
                    {
                        primitive.mesh_type()
                    } else {
                        primitive.coordinate_system()
                    };
                    let mut points = Vec::with_capacity(curve.point_count());
                    for point_index in 0..curve.point_count() {
                        let Ok(Some(mut point)) = curve.point(point_index, Some(params)) else {
                            continue;
                        };
                        point = transform_coordinates(point, point_system, mesh_system);
                        if let Some(transform) = primitive.transform() {
                            point = transform_coordinates(
                                point,
                                mesh_system,
                                csxcad_core::grid::CoordinateSystem::Cartesian,
                            );
                            point = transform.transform(point);
                            point = transform_coordinates(
                                point,
                                csxcad_core::grid::CoordinateSystem::Cartesian,
                                mesh_system,
                            );
                        }
                        points.push(point);
                    }

                    let mut curve_used = false;
                    for segment in 1..points.len() {
                        let path = self.find_path(points[segment - 1], points[segment]);
                        if !path.dir.is_empty() {
                            curve_used = true;
                        }
                        for step in 0..path.dir.len() {
                            let n = path.dir[step] as usize;
                            let pos = [
                                path.pos_path[0][step] as usize,
                                path.pos_path[1][step] as usize,
                                path.pos_path[2][step] as usize,
                            ];
                            // Native `GetYeeCoords(n, pos, volt_coord, false)`.
                            let vertex =
                                yee_coordinates(&self.base.disc_lines, counts, n, pos, false);
                            // Note: native applies no `GetEnabled()` check and
                            // no current-excitation handling in this curve
                            // branch, so a disabled curve excitation still
                            // excites and `Type="2"/"3"` curves stay silent.
                            if excitation.active_direction(n) != Some(true)
                                || pos[n] >= counts[n] as usize - 1
                            {
                                continue;
                            }
                            let excitation_type = excitation.excitation_type();
                            if !matches!(excitation_type, 0 | 1) {
                                continue;
                            }
                            // Native `GetWeightedExcitation(n, volt_coord)`:
                            // weight and amplitude, then the non-dual edge
                            // length (operator_ext_excitation.cpp:267).
                            let weight = excitation
                                .weighted_excitation_weight(
                                    n,
                                    Some(params),
                                    vertex,
                                    coordinate_system,
                                )
                                .unwrap_or(1.0);
                            let base_amp = excitation
                                .excitation(n, Some(params))
                                .ok()
                                .flatten()
                                .unwrap_or(0.0)
                                * weight;
                            let amp = (base_amp * metric.edge_length(n, pos, false)) as FDTD_FLOAT;
                            if amp != 0.0 {
                                let delay = (excitation.delay(Some(params)).unwrap_or(0.0) / dt)
                                    .max(0.0) as u32;
                                extension.add_voltage_excitation(
                                    n,
                                    pos,
                                    delay,
                                    amp,
                                    shared.clone(),
                                );
                            }
                            if excitation_type == 1 {
                                hard_voltage.push([n, pos[0], pos[1], pos[2]]);
                            }
                        }
                    }
                    if curve_used {
                        // Native `prim->SetPrimitiveUsed(true)`.
                        used_curves.push((index, primitive_index));
                    }
                }
            }
        }
        if let Some(structure) = self.structure.as_mut() {
            for (property_index, primitive_index) in used_curves {
                if let Some(primitive) = structure
                    .property_by_index_mut(property_index)
                    .and_then(|property| property.primitive_mut(primitive_index))
                {
                    primitive.set_used(true);
                }
            }
        }

        if let (Some(vv), Some(vi)) = (self.vv_ptr.as_mut(), self.vi_ptr.as_mut()) {
            for [n, i, j, k] in &hard_voltage {
                *vv.get_mut(*n, *i, *j, *k) = 0.0;
                *vi.get_mut(*n, *i, *j, *k) = 0.0;
            }
        }
        if let (Some(ii), Some(iv)) = (self.ii_ptr.as_mut(), self.iv_ptr.as_mut()) {
            for [n, i, j, k] in &hard_current {
                *ii.get_mut(*n, *i, *j, *k) = 0.0;
                *iv.get_mut(*n, *i, *j, *k) = 0.0;
            }
        }

        extension
    }

    /// Build the conducting-sheet extension from the CSX `ConductingSheet`
    /// properties.
    ///
    /// Mirrors `Operator_Ext_ConductingSheet::BuildExtension`: every Yee edge
    /// inside a two-dimensional sheet primitive gets the ADE model, edges inside
    /// a boundary layer or a non-2D primitive fall back to PEC, and edges sharing
    /// their width with a non-sheet cell get the native `factor = 2`.
    pub fn build_conducting_sheet_extension(
        &mut self,
        f_max: f64,
        boundary_size: [u32; 6],
    ) -> Result<Option<EngineExtConductingSheet>, String> {
        let Some(structure) = self.structure.as_ref() else {
            return Ok(None);
        };
        let params = structure.parameters();

        struct SheetCandidate<'a> {
            priority: i32,
            properties: &'a ConductingSheetProperties,
            primitive: &'a Primitive,
        }
        let mut candidates: Vec<SheetCandidate<'_>> = Vec::new();
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            let Some(sheet) = property.conducting_sheet() else {
                continue;
            };
            for primitive in property.primitives() {
                candidates.push(SheetCandidate {
                    priority: primitive.priority(),
                    properties: sheet,
                    primitive,
                });
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        let sheet_at = |coordinate: [f64; 3]| -> Option<&SheetCandidate<'_>> {
            let mut winner: Option<&SheetCandidate<'_>> = None;
            let mut winner_priority = i32::MIN;
            for candidate in &candidates {
                if winner.is_some() && candidate.priority <= winner_priority {
                    continue;
                }
                if candidate
                    .primitive
                    .is_inside(coordinate, Some(params))
                    .unwrap_or(false)
                {
                    winner = Some(candidate);
                    winner_priority = candidate.priority;
                }
            }
            winner
        };

        let counts = self.base.num_lines;
        let disc_lines = self.base.disc_lines.clone();
        let grid_delta = self.base.grid_delta;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("conducting sheet requires a positive timestep".to_string());
        }
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;

        // Upper triangular bookkeeping: for every position, which edge
        // directions carry a sheet and along which tangential direction.
        let mut tan_dir: HashMap<[usize; 3], [i8; 3]> = HashMap::new();
        let mut active: Vec<[usize; 3]> = Vec::new();
        let mut vv = self
            .vv_ptr
            .as_mut()
            .ok_or("vv coefficients not allocated")?;
        let mut vi = self
            .vi_ptr
            .as_mut()
            .ok_or("vi coefficients not allocated")?;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let pos = [i, j, k];
                    let disabled = (0..3).any(|axis| {
                        pos[axis] <= boundary_size[2 * axis] as usize
                            || pos[axis]
                                >= counts[axis] as usize - boundary_size[2 * axis + 1] as usize - 1
                    });
                    let mut local = [-1i8; 3];
                    let mut any = false;
                    for n in 0..3 {
                        let coordinate = yee_coordinates(&disc_lines, counts, n, pos, false);
                        let Some(candidate) = sheet_at(coordinate) else {
                            continue;
                        };
                        let sigma = candidate
                            .properties
                            .conductivity(Some(params))
                            .unwrap_or(0.0);
                        let thickness = candidate.properties.thickness(Some(params)).unwrap_or(0.0);
                        let bounds = candidate
                            .primitive
                            .bounding_box(Some(params), false)
                            .unwrap_or([0.0; 6]);
                        let dimension = candidate.primitive.dimension(Some(params)).unwrap_or(0);
                        if disabled || dimension != 2 || sigma <= 0.0 || thickness <= 0.0 {
                            // Native fallback: this edge becomes PEC.
                            *vv.get_mut(n, i, j, k) = 0.0;
                            *vi.get_mut(n, i, j, k) = 0.0;
                            local[n] = -1;
                            continue;
                        }
                        let n_p = (n + 1) % 3;
                        let n_pp = (n + 2) % 3;
                        let mut direction = -1i8;
                        if bounds[2 * n_p] != bounds[2 * n_p + 1] {
                            direction = n_p as i8;
                        }
                        if bounds[2 * n_pp] != bounds[2 * n_pp + 1] {
                            direction = n_pp as i8;
                        }
                        if direction >= 0 {
                            local[n] = direction;
                            any = true;
                        }
                    }
                    if any {
                        tan_dir.insert(pos, local);
                        active.push(pos);
                    }
                }
            }
        }

        if active.is_empty() {
            return Ok(None);
        }

        let mut extension = EngineExtConductingSheet::new(active.len());
        for (index, &pos) in active.iter().enumerate() {
            let directions = tan_dir.get(&pos).copied().unwrap_or([-1; 3]);
            let mut coefficients = ConductingSheetCoeffs {
                v_int_ade: [[0.0; 3]; 2],
                v_ext_ade: [[0.0; 3]; 2],
                vv: [0.0; 3],
                vi: [0.0; 3],
            };
            for n in 0..3 {
                let coordinate = yee_coordinates(&disc_lines, counts, n, pos, false);
                let Some(candidate) = sheet_at(coordinate) else {
                    continue;
                };
                let direction = directions[n];
                if direction < 0 {
                    continue;
                }
                let t_dir = direction as usize;
                let sigma = candidate
                    .properties
                    .conductivity(Some(params))
                    .unwrap_or(0.0);
                let thickness = candidate.properties.thickness(Some(params)).unwrap_or(0.0);
                let metric = MeshMetric::new(&disc_lines, counts, grid_delta, self.mesh_type);
                let edge_length = metric.edge_length(n, pos, false);
                let node_width = metric.edge_length(t_dir, pos, true);
                if edge_length <= 0.0 || node_width <= 0.0 {
                    continue;
                }
                // Native `factor`: 2 when the neighbouring cell along the
                // tangential direction does not also carry a sheet.
                let mut factor = 1.0;
                let current = tan_dir.get(&pos).map(|map| map[t_dir]).unwrap_or(-1);
                if current < 0 {
                    factor = 2.0;
                }
                let mut neighbour = pos;
                if neighbour[t_dir] > 0 {
                    neighbour[t_dir] -= 1;
                    let previous = tan_dir.get(&neighbour).map(|map| map[t_dir]).unwrap_or(-1);
                    if previous < 0 {
                        factor = 2.0;
                    }
                } else {
                    factor = 2.0;
                }

                let model = crate::conducting_sheet::sheet_coefficients(
                    sigma,
                    thickness,
                    dt,
                    f_max,
                    edge_length / node_width,
                    factor,
                )?;
                coefficients.vv[n] = model.vv as f32;
                coefficients.vi[n] = model.vi as f32;
                for order in 0..2 {
                    coefficients.v_int_ade[order][n] = model.v_int_ade[order] as f32;
                    coefficients.v_ext_ade[order][n] = model.v_ext_ade[order] as f32;
                }
                *vv.get_mut(n, pos[0], pos[1], pos[2]) = model.vv as f32;
                *vi.get_mut(n, pos[0], pos[1], pos[2]) = model.vi as f32;
            }
            extension.set_position(index, pos, &coefficients);
        }
        Ok(Some(extension))
    }

    /// Build the Lorentz (Drude–Lorentz) dispersive extension.
    ///
    /// Mirrors `Operator_Ext_LorentzMaterial::BuildExtension`: per dispersion
    /// order and per edge, the plasma frequency builds the Drude inductance /
    /// capacitance, the relaxation time the series resistance, and the Lorentz
    /// pole frequency the resonant element. Cells active at different orders
    /// share one position list; per-order coefficients are zero where inactive.
    pub fn build_lorentz_extension(&mut self) -> Result<Option<EngineExtLorentzMaterial>, String> {
        let Some(structure) = self.structure.as_ref() else {
            return Ok(None);
        };
        let params = structure.parameters().clone();
        let coordinate_system = structure.coordinate_system();
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("Lorentz materials need a positive timestep".to_string());
        }

        let mut pole_count = 0usize;
        for index in 0..structure.property_count() {
            if let Some(property) = structure.property_by_index(index) {
                if let Some(lorentz) = property.lorentz_material() {
                    pole_count = pole_count.max(lorentz.order());
                }
            }
        }
        if pole_count == 0 {
            return Ok(None);
        }

        struct Candidate<'a> {
            priority: i32,
            lorentz: &'a csxcad_core::material::LorentzMaterialProperties,
            isotropic: bool,
            primitive: &'a Primitive,
        }
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            let Some(lorentz) = property.lorentz_material() else {
                continue;
            };
            let isotropic = property.material().map_or(true, |m| m.is_isotropic());
            for primitive in property.primitives() {
                candidates.push(Candidate {
                    priority: primitive.priority(),
                    lorentz,
                    isotropic,
                    primitive,
                });
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        let material_at = |coordinate: [f64; 3]| -> Option<&Candidate<'_>> {
            let mut winner: Option<&Candidate<'_>> = None;
            let mut winner_priority = i32::MIN;
            for candidate in &candidates {
                if winner.is_some() && candidate.priority <= winner_priority {
                    continue;
                }
                if candidate
                    .primitive
                    .is_inside(coordinate, Some(&params))
                    .unwrap_or(false)
                {
                    winner = Some(candidate);
                    winner_priority = candidate.priority;
                }
            }
            winner
        };
        let parameter = |candidate: &Candidate<'_>,
                         parameter: csxcad_core::material::LorentzParameter,
                         order: usize,
                         component: usize,
                         coordinate: [f64; 3]|
         -> f64 {
            candidate
                .lorentz
                .weighted_value(
                    parameter,
                    order,
                    if candidate.isotropic { 0 } else { component },
                    Some(&params),
                    coordinate,
                    coordinate_system,
                )
                .ok()
                .flatten()
                .unwrap_or(0.0)
        };

        // Union of all active cells across orders, preserving first-seen order.
        let mut union: Vec<[usize; 3]> = Vec::new();
        let mut union_index: std::collections::HashMap<[usize; 3], usize> =
            std::collections::HashMap::new();
        struct OrderData {
            positions: Vec<(usize, Coefficients)>,
        }
        struct Coefficients {
            v_int: [FDTD_FLOAT; 3],
            v_ext: [FDTD_FLOAT; 3],
            v_lor: [FDTD_FLOAT; 3],
            i_int: [FDTD_FLOAT; 3],
            i_ext: [FDTD_FLOAT; 3],
            i_lor: [FDTD_FLOAT; 3],
        }
        let counts = self.base.num_lines;
        let mut orders: Vec<OrderData> = Vec::with_capacity(pole_count);

        for order in 0..pole_count {
            let mut data = OrderData {
                positions: Vec::new(),
            };
            for i in 0..counts[0] as usize {
                for j in 0..counts[1] as usize {
                    for k in 0..counts[2] as usize {
                        let pos = [i, j, k];
                        let mut coefficients = Coefficients {
                            v_int: [1.0; 3],
                            v_ext: [0.0; 3],
                            v_lor: [0.0; 3],
                            i_int: [1.0; 3],
                            i_ext: [0.0; 3],
                            i_lor: [0.0; 3],
                        };
                        let mut active = false;
                        for n in 0..3 {
                            // ----- electric side -----
                            let vi = self.vi_at(n, pos);
                            let coordinate =
                                yee_coordinates(&self.base.disc_lines, counts, n, pos, false);
                            let index = Self::ec_index(n, i, j, k, counts);
                            if vi != 0.0 {
                                if let Some(candidate) = material_at(coordinate) {
                                    let w_plasma = parameter(
                                        candidate,
                                        csxcad_core::material::LorentzParameter::EpsilonPlasmaFrequency,
                                        order,
                                        n,
                                        coordinate,
                                    ) * std::f64::consts::TAU;
                                    let mut l_d = 0.0;
                                    let mut r_d = 0.0;
                                    let mut c_l = 0.0;
                                    if w_plasma > 0.0 {
                                        let ec_c = *self.ec_c.get(index).unwrap_or(&0.0) as f64;
                                        if ec_c > 0.0 {
                                            l_d = 1.0 / (w_plasma * w_plasma * ec_c);
                                            active = true;
                                        }
                                    }
                                    let t_relax = parameter(
                                        candidate,
                                        csxcad_core::material::LorentzParameter::EpsilonRelaxTime,
                                        order,
                                        n,
                                        coordinate,
                                    );
                                    if t_relax > 0.0 && l_d > 0.0 {
                                        r_d = l_d / t_relax;
                                    }
                                    let w_lor = parameter(
                                        candidate,
                                        csxcad_core::material::LorentzParameter::EpsilonLorentzPoleFrequency,
                                        order,
                                        n,
                                        coordinate,
                                    ) * std::f64::consts::TAU;
                                    if w_lor > 0.0 && l_d > 0.0 {
                                        c_l = 1.0 / (w_lor * w_lor * l_d);
                                    }
                                    if l_d > 0.0 {
                                        coefficients.v_int[n] = ((2.0 * l_d - dt * r_d)
                                            / (2.0 * l_d + dt * r_d))
                                            as FDTD_FLOAT;
                                        coefficients.v_ext[n] =
                                            (dt / (l_d + dt * r_d / 2.0) * vi as f64) as FDTD_FLOAT;
                                    }
                                    if c_l > 0.0 {
                                        coefficients.v_lor[n] =
                                            (dt / c_l / vi as f64) as FDTD_FLOAT;
                                    }
                                }
                            }

                            // ----- magnetic side -----
                            let iv = self.iv_at(n, pos);
                            let coordinate =
                                yee_coordinates(&self.base.disc_lines, counts, n, pos, true);
                            if iv != 0.0 {
                                if let Some(candidate) = material_at(coordinate) {
                                    let w_plasma = parameter(
                                        candidate,
                                        csxcad_core::material::LorentzParameter::MuePlasmaFrequency,
                                        order,
                                        n,
                                        coordinate,
                                    ) * std::f64::consts::TAU;
                                    let mut c_d = 0.0;
                                    let mut g_d = 0.0;
                                    let mut l_l = 0.0;
                                    if w_plasma > 0.0 {
                                        let ec_l = *self.ec_l.get(index).unwrap_or(&0.0) as f64;
                                        if ec_l > 0.0 {
                                            c_d = 1.0 / (w_plasma * w_plasma * ec_l);
                                            active = true;
                                        }
                                    }
                                    let t_relax = parameter(
                                        candidate,
                                        csxcad_core::material::LorentzParameter::MueRelaxTime,
                                        order,
                                        n,
                                        coordinate,
                                    );
                                    if t_relax > 0.0 && c_d > 0.0 {
                                        g_d = c_d / t_relax;
                                    }
                                    let w_lor = parameter(
                                        candidate,
                                        csxcad_core::material::LorentzParameter::MueLorentzPoleFrequency,
                                        order,
                                        n,
                                        coordinate,
                                    ) * std::f64::consts::TAU;
                                    if w_lor > 0.0 && c_d > 0.0 {
                                        l_l = 1.0 / (w_lor * w_lor * c_d);
                                    }
                                    if c_d > 0.0 {
                                        coefficients.i_int[n] = ((2.0 * c_d - dt * g_d)
                                            / (2.0 * c_d + dt * g_d))
                                            as FDTD_FLOAT;
                                        coefficients.i_ext[n] =
                                            (dt / (c_d + dt * g_d / 2.0) * iv as f64) as FDTD_FLOAT;
                                    }
                                    if l_l > 0.0 {
                                        coefficients.i_lor[n] =
                                            (dt / l_l / iv as f64) as FDTD_FLOAT;
                                    }
                                }
                            }
                        }
                        if !active {
                            continue;
                        }
                        let position_index = *union_index.entry(pos).or_insert_with(|| {
                            union.push(pos);
                            union.len() - 1
                        });
                        data.positions.push((position_index, coefficients));
                    }
                }
            }
            orders.push(data);
        }

        if union.is_empty() {
            return Ok(None);
        }

        let mut extension = EngineExtLorentzMaterial::new(pole_count, union.len());
        for (index, &pos) in union.iter().enumerate() {
            extension.set_position(index, pos);
        }
        for (order, data) in orders.iter().enumerate() {
            let mut voltage_on = false;
            let mut current_on = false;
            for (position_index, coefficients) in &data.positions {
                extension.set_order_coefficients(
                    *position_index,
                    order,
                    &coefficients.v_int,
                    &coefficients.v_ext,
                    &coefficients.v_lor,
                    &coefficients.i_int,
                    &coefficients.i_ext,
                    &coefficients.i_lor,
                );
                if coefficients.v_int != [1.0; 3] || coefficients.v_ext != [0.0; 3] {
                    voltage_on = true;
                }
                if coefficients.i_int != [1.0; 3] || coefficients.i_ext != [0.0; 3] {
                    current_on = true;
                }
            }
            extension.set_order_active(order, voltage_on, current_on);
        }
        Ok(Some(extension))
    }

    /// Build the Debye dispersive extension from the CSX `DebyeMaterial`
    /// properties.
    ///
    /// Mirrors `Operator_Ext_DebyeMaterial::BuildExtension`: the maximum pole
    /// count comes from the properties, each edge samples the highest-priority
    /// containing material, and a pole whose relaxation time is shorter than
    /// three timesteps is skipped (the trapezoidal update is not stable).
    pub fn build_debye_extension(&mut self) -> Result<Option<EngineExtDebyeMaterial>, String> {
        let Some(structure) = self.structure.as_ref() else {
            return Ok(None);
        };
        let params = structure.parameters().clone();
        let coordinate_system = structure.coordinate_system();
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("Debye materials need a positive timestep".to_string());
        }

        let mut pole_count = 0usize;
        for index in 0..structure.property_count() {
            if let Some(property) = structure.property_by_index(index) {
                if let Some(debye) = property.debye_material() {
                    pole_count = pole_count.max(debye.order());
                }
            }
        }
        if pole_count == 0 {
            return Ok(None);
        }

        struct Candidate<'a> {
            priority: i32,
            debye: &'a csxcad_core::material::DebyeMaterialProperties,
            isotropic: bool,
            primitive: &'a Primitive,
        }
        let mut candidates: Vec<Candidate<'_>> = Vec::new();
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            let Some(debye) = property.debye_material() else {
                continue;
            };
            for primitive in property.primitives() {
                candidates.push(Candidate {
                    priority: primitive.priority(),
                    debye,
                    isotropic: property.material().map_or(true, |m| m.is_isotropic()),
                    primitive,
                });
            }
        }
        if candidates.is_empty() {
            return Ok(None);
        }
        let debye_at = |coordinate: [f64; 3]| -> Option<&Candidate<'_>> {
            let mut winner: Option<&Candidate<'_>> = None;
            let mut winner_priority = i32::MIN;
            for candidate in &candidates {
                if winner.is_some() && candidate.priority <= winner_priority {
                    continue;
                }
                if candidate
                    .primitive
                    .is_inside(coordinate, Some(&params))
                    .unwrap_or(false)
                {
                    winner = Some(candidate);
                    winner_priority = candidate.priority;
                }
            }
            winner
        };

        const DEBYE_MIN_TAU_PER_DT: f64 = 3.0;
        let counts = self.base.num_lines;
        let mut warn_once = true;
        let mut positions: Vec<[usize; 3]> = Vec::new();
        let mut relax: Vec<[Vec<f64>; 3]> = Vec::new();
        let mut drive: Vec<[Vec<f64>; 3]> = Vec::new();
        let mut solve: Vec<[f64; 3]> = Vec::new();

        for i in 0..counts[0] as usize {
            for j in 0..counts[1] as usize {
                for k in 0..counts[2] as usize {
                    let pos = [i, j, k];
                    let mut any = false;
                    let mut cell_relax: [Vec<f64>; 3] = [
                        vec![0.0; pole_count],
                        vec![0.0; pole_count],
                        vec![0.0; pole_count],
                    ];
                    let mut cell_drive: [Vec<f64>; 3] = [
                        vec![0.0; pole_count],
                        vec![0.0; pole_count],
                        vec![0.0; pole_count],
                    ];
                    let mut cell_solve = [1.0f64; 3];
                    for n in 0..3 {
                        let vi = self.vi_at(n, pos);
                        if vi == 0.0 {
                            continue;
                        }
                        let coordinate =
                            yee_coordinates(&self.base.disc_lines, counts, n, pos, false);
                        let Some(candidate) = debye_at(coordinate) else {
                            continue;
                        };
                        let metric = self.metric();
                        let area = metric.edge_area(n, pos, false);
                        let length = metric.edge_length(n, pos, false);
                        if length <= 0.0 {
                            continue;
                        }
                        let area_per_length = area / length;
                        let order = candidate.debye.order().min(pole_count);
                        // Native `CSPropMaterial::GetValue` pins the component
                        // to 0 for isotropic properties.
                        let component = if candidate.isotropic { 0 } else { n };
                        let mut sum = 0.0f64;
                        for pole in 0..order {
                            let delta_eps = candidate
                                .debye
                                .weighted_epsilon_delta(
                                    pole,
                                    component,
                                    Some(&params),
                                    coordinate,
                                    coordinate_system,
                                )
                                .ok()
                                .flatten()
                                .unwrap_or(0.0);
                            let tau = candidate
                                .debye
                                .weighted_epsilon_relax_time(
                                    pole,
                                    component,
                                    Some(&params),
                                    coordinate,
                                    coordinate_system,
                                )
                                .ok()
                                .flatten()
                                .unwrap_or(0.0);
                            if delta_eps <= 0.0 || tau <= 0.0 {
                                continue;
                            }
                            if tau < DEBYE_MIN_TAU_PER_DT * dt {
                                if warn_once {
                                    warn_once = false;
                                    eprintln!(
                                        "[openEMS] Debye: relaxation time {tau:e}s needs at least \
                                         3 timesteps (dt={dt:e}s); the pole is skipped"
                                    );
                                }
                                continue;
                            }
                            let c_l = EPS0 * delta_eps * area_per_length;
                            cell_relax[n][pole] = (2.0 * tau - dt) / (2.0 * tau + dt);
                            cell_drive[n][pole] = c_l * vi as f64 / (2.0 * tau + dt);
                            sum += cell_drive[n][pole];
                            any = true;
                        }
                        cell_solve[n] = 1.0 / (1.0 + sum);
                    }
                    if !any {
                        continue;
                    }
                    positions.push(pos);
                    relax.push(cell_relax);
                    drive.push(cell_drive);
                    solve.push(cell_solve);
                }
            }
        }

        if positions.is_empty() {
            return Ok(None);
        }
        let mut extension = EngineExtDebyeMaterial::new(pole_count, positions.len());
        for index in 0..positions.len() {
            extension.set_position(index, positions[index]);
            for n in 0..3 {
                let relax_f32: Vec<FDTD_FLOAT> = relax[index][n]
                    .iter()
                    .map(|&value| value as FDTD_FLOAT)
                    .collect();
                let drive_f32: Vec<FDTD_FLOAT> = drive[index][n]
                    .iter()
                    .map(|&value| value as FDTD_FLOAT)
                    .collect();
                extension.set_position_component(
                    index,
                    n,
                    &relax_f32,
                    &drive_f32,
                    solve[index][n] as FDTD_FLOAT,
                );
            }
        }
        Ok(Some(extension))
    }

    /// Numerical phase velocity along `prop_dir` for a plane wave at
    /// `frequency`. Mirrors native `Operator::CalcNumericPhaseVelocity`.
    pub fn calc_numeric_phase_velocity(
        &self,
        start: [usize; 3],
        stop: [usize; 3],
        prop_dir: [f64; 3],
        frequency: f64,
    ) -> f64 {
        let mut average = [0.0f64; 3];
        let c0 = C0 / (self.base.m_bg_eps_r * self.base.m_bg_mue_r).sqrt();
        for n in 0..3 {
            let distance = (disc_line_value(
                &self.base.disc_lines[n],
                self.base.num_lines[n] as usize,
                start[n],
                false,
            ) - disc_line_value(
                &self.base.disc_lines[n],
                self.base.num_lines[n] as usize,
                stop[n],
                false,
            ))
            .abs()
                * self.base.grid_delta;
            let cells = (stop[n] as f64 - start[n] as f64).abs();
            average[n] = if cells > 0.0 { distance / cells } else { 0.0 };
        }

        // A Cartesian propagation direction has an analytic solution.
        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            if prop_dir[n].abs() == 1.0 && prop_dir[n_p] == 0.0 && prop_dir[n_pp] == 0.0 {
                if average[0] <= 0.0 {
                    return c0;
                }
                let kx = (average[0] / c0 / self.base.d_t
                    * (std::f64::consts::TAU * frequency * self.base.d_t / 2.0).sin())
                .asin()
                    * 2.0
                    / average[0];
                if kx != 0.0 {
                    return std::f64::consts::TAU * frequency / kx;
                }
                return c0;
            }
        }

        // Otherwise solve the dispersion relation with Newton's method.
        let k0 = std::f64::consts::TAU * frequency / c0;
        let mut k = k0;
        let rhs =
            (std::f64::consts::TAU * frequency * self.base.d_t / 2.0).sin() / c0 / self.base.d_t;
        let rhs = rhs * rhs;
        let mut old_phase_velocity = 0.0f64;
        let mut phase_velocity = c0;
        let tolerance = 1.0e-6;
        let mut iterations = 0;
        while (old_phase_velocity - phase_velocity).abs() > tolerance {
            iterations += 1;
            old_phase_velocity = phase_velocity;
            let mut f = 0.0;
            let mut df = 0.0;
            for n in 0..3 {
                if average[n] == 0.0 {
                    continue;
                }
                let half = prop_dir[n] * k * average[n] / 2.0;
                f += (half.sin() / average[n]).powi(2);
                df += prop_dir[n] * half.sin() * half.cos() / average[n];
            }
            f -= rhs;
            if df != 0.0 {
                k -= f / df;
            }
            if k < k0 {
                k = k0;
            }
            phase_velocity = std::f64::consts::TAU * frequency / k;
            if iterations > 99 {
                eprintln!("[openEMS] CalcNumericPhaseVelocity: Newton iteration did not converge");
                break;
            }
        }
        phase_velocity
    }

    /// Build the TFSF extension from a plane-wave CSX excitation.
    ///
    /// Mirrors `Operator_Ext_TFSF::BuildExtension`: exactly one box primitive,
    /// a valid propagation direction and polarization, and the native delay /
    /// amplitude layout on all six faces.
    pub fn build_tfsf_extension(
        &mut self,
        signal: &Excitation,
    ) -> Result<Option<EngineExtTFSF>, String> {
        let Some(structure) = self.structure.as_ref() else {
            return Ok(None);
        };
        let params = structure.parameters().clone();
        let mut candidate: Option<(Primitive, f64, [f64; 3], [f64; 3])> = None;
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            let Some(excitation) = property.excitation() else {
                continue;
            };
            if excitation.excitation_type() != 10 || !excitation.is_enabled() {
                continue;
            }
            if property.primitives().len() != 1 {
                eprintln!(
                    "[openEMS] TFSF: plane wave excitation must have exactly one primitive; skipped"
                );
                continue;
            }
            let primitive = &property.primitives()[0];
            if !matches!(
                primitive.shape(),
                csxcad_core::model::PrimitiveShape::Box(_)
            ) {
                eprintln!("[openEMS] TFSF: plane wave primitive is not a box; skipped");
                continue;
            }
            if candidate.is_some() {
                eprintln!("[openEMS] TFSF: more than one plane wave excitation; using the first");
                break;
            }
            let frequency = excitation.frequency(Some(&params)).unwrap_or(0.0);
            let mut direction = [0.0f64; 3];
            for component in 0..3 {
                direction[component] = excitation
                    .propagation_direction(component, Some(&params))
                    .ok()
                    .flatten()
                    .unwrap_or(0.0);
            }
            let mut amplitude = [0.0f64; 3];
            for component in 0..3 {
                amplitude[component] = excitation
                    .excitation(component, Some(&params))
                    .ok()
                    .flatten()
                    .unwrap_or(0.0);
            }
            candidate = Some((primitive.clone(), frequency, direction, amplitude));
        }
        let Some((primitive, frequency, mut prop_dir, mut e_amp)) = candidate else {
            return Ok(None);
        };

        let norm =
            (prop_dir[0] * prop_dir[0] + prop_dir[1] * prop_dir[1] + prop_dir[2] * prop_dir[2])
                .sqrt();
        if norm == 0.0 {
            eprintln!("[openEMS] TFSF: plane wave propagation direction is zero; disabled");
            return Ok(None);
        }
        for component in 0..3 {
            prop_dir[component] /= norm;
        }

        let bounds = primitive
            .bounding_box(Some(&params), false)
            .map_err(|error| format!("TFSF box: {error}"))?;
        let start_coordinate = [bounds[0], bounds[2], bounds[4]];
        let stop_coordinate = [bounds[1], bounds[3], bounds[5]];
        let (dimension, start, stop, _start_in, _stop_in) =
            self.snap_box_to_mesh_native(start_coordinate, stop_coordinate, false, true, 0);
        if dimension != 3 {
            eprintln!(
                "[openEMS] TFSF: plane wave box dimension is {dimension}, expected 3; disabled"
            );
            return Ok(None);
        }
        let start = start.map(|value| value as usize);
        let stop = stop.map(|value| value as usize);

        let ref_index = (self.base.m_bg_eps_r * self.base.m_bg_mue_r).sqrt();
        let phase_velocity = if frequency <= 0.0 {
            C0 / ref_index
        } else {
            self.calc_numeric_phase_velocity(start, stop, prop_dir, frequency)
        };
        let phase_velocity = if phase_velocity < 0.0
            || phase_velocity > C0 / ref_index
            || !phase_velocity.is_finite()
        {
            eprintln!("[openEMS] TFSF: invalid phase velocity, resetting to c0");
            C0 / ref_index
        } else {
            phase_velocity
        };

        let dot = e_amp[0] * prop_dir[0] + e_amp[1] * prop_dir[1] + e_amp[2] * prop_dir[2];
        let e_norm = e_amp[0] * e_amp[0] + e_amp[1] * e_amp[1] + e_amp[2] * e_amp[2];
        if e_norm == 0.0 {
            eprintln!("[openEMS] TFSF: plane wave polarization is zero; disabled");
            return Ok(None);
        }
        // Native computes acos(E·d / |E|²); preserved for compatibility.
        let cosine = (dot / e_norm).clamp(-1.0, 1.0);
        let degrees = cosine.acos() / std::f64::consts::PI * 180.0;
        if degrees == 0.0 {
            eprintln!("[openEMS] TFSF: propagation and polarization are parallel; disabled");
            return Ok(None);
        }
        if (degrees - 90.0).abs() > 1.0e-9 {
            for component in 0..3 {
                e_amp[component] -= prop_dir[component] * dot;
            }
            eprintln!(
                "[openEMS] TFSF: polarization is not orthogonal; projected to ({}, {}, {})",
                e_amp[0], e_amp[1], e_amp[2]
            );
        }

        let mut h_amp = [0.0f64; 3];
        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            h_amp[n] = (prop_dir[n_p] * e_amp[n_pp] - prop_dir[n_pp] * e_amp[n_p])
                / (Z0 * (self.base.m_bg_mue_r / self.base.m_bg_eps_r).sqrt());
        }

        let mut inc_low = [true; 3];
        let mut active_dir = [[false; 2]; 3];
        let mut origin = [0.0f64; 3];
        let mut num_lines = [0usize; 3];
        let total_lines = self.base.num_lines;
        for n in 0..3 {
            num_lines[n] = stop[n] - start[n] + 1;
            inc_low[n] = prop_dir[n] >= 0.0;
            active_dir[n][0] = start[n] != 0;
            active_dir[n][1] = stop[n] != total_lines[n] as usize - 1;
            let ui_origin = if inc_low[n] {
                start[n].saturating_sub(1)
            } else {
                stop[n] + 1
            };
            origin[n] = disc_line_value(
                &self.base.disc_lines[n],
                total_lines[n] as usize,
                ui_origin,
                false,
            );
        }

        let mut extension = EngineExtTFSF::new();
        extension.set_region(start, stop);
        extension.set_signal(signal.clone());
        for n in 0..3 {
            for face in 0..2 {
                if active_dir[n][face] {
                    extension.activate_face(n, face as i32);
                }
            }
        }
        extension.allocate();

        let unit = self.base.grid_delta;
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("TFSF requires a positive timestep".to_string());
        }
        let mut max_delay = 0u32;
        let mut pos = [0usize; 3];

        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            if !active_dir[n][0] && !active_dir[n][1] {
                continue;
            }
            let mut ui_pos = 0usize;
            for first in 0..num_lines[n_p] {
                pos[n_p] = start[n_p] + first;
                for second in 0..num_lines[n_pp] {
                    pos[n_pp] = start[n_pp] + second;

                    // ----- current updates -----
                    if active_dir[n][0] {
                        pos[n] = start[n];
                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_p,
                            pos,
                            false,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt;
                        max_delay = max_delay.max(delay as u32);
                        extension.curr_delay[n][0][1][ui_pos] = delay.floor() as u32;
                        extension.curr_delay_delta[n][0][1][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.curr_amp[n][0][1][ui_pos] =
                            (e_amp[n_p] * self.metric().edge_length(n_p, pos, false)) as FDTD_FLOAT;

                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_pp,
                            pos,
                            false,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt;
                        max_delay = max_delay.max(delay as u32);
                        extension.curr_delay[n][0][0][ui_pos] = delay.floor() as u32;
                        extension.curr_delay_delta[n][0][0][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.curr_amp[n][0][0][ui_pos] = (e_amp[n_pp]
                            * self.metric().edge_length(n_pp, pos, false))
                            as FDTD_FLOAT;

                        if pos[n] > 0 {
                            pos[n] -= 1;
                        }
                        let iv_p = self.iv_at(n_p, pos);
                        let iv_pp = self.iv_at(n_pp, pos);
                        extension.curr_amp[n][0][0][ui_pos] *= iv_p;
                        extension.curr_amp[n][0][1][ui_pos] *= iv_pp;
                    }

                    if active_dir[n][1] {
                        pos[n] = stop[n];
                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_p,
                            pos,
                            false,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt;
                        max_delay = max_delay.max(delay as u32);
                        extension.curr_delay[n][1][1][ui_pos] = delay.floor() as u32;
                        extension.curr_delay_delta[n][1][1][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.curr_amp[n][1][1][ui_pos] =
                            (e_amp[n_p] * self.metric().edge_length(n_p, pos, false)) as FDTD_FLOAT;

                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_pp,
                            pos,
                            false,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt;
                        max_delay = max_delay.max(delay as u32);
                        extension.curr_delay[n][1][0][ui_pos] = delay.floor() as u32;
                        extension.curr_delay_delta[n][1][0][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.curr_amp[n][1][0][ui_pos] = (e_amp[n_pp]
                            * self.metric().edge_length(n_pp, pos, false))
                            as FDTD_FLOAT;

                        let iv_p = self.iv_at(n_p, pos);
                        let iv_pp = self.iv_at(n_pp, pos);
                        extension.curr_amp[n][1][0][ui_pos] *= iv_p;
                        extension.curr_amp[n][1][1][ui_pos] *= iv_pp;
                    }

                    if active_dir[n][0] {
                        extension.curr_amp[n][0][0][ui_pos] *= -1.0;
                    }
                    if active_dir[n][1] {
                        extension.curr_amp[n][1][1][ui_pos] *= -1.0;
                    }
                    if pos[n_p] == stop[n_p] {
                        if active_dir[n][0] {
                            extension.curr_amp[n][0][1][ui_pos] = 0.0;
                        }
                        if active_dir[n][1] {
                            extension.curr_amp[n][1][1][ui_pos] = 0.0;
                        }
                    }
                    if pos[n_pp] == stop[n_pp] {
                        if active_dir[n][0] {
                            extension.curr_amp[n][0][0][ui_pos] = 0.0;
                        }
                        if active_dir[n][1] {
                            extension.curr_amp[n][1][0][ui_pos] = 0.0;
                        }
                    }

                    // ----- voltage updates -----
                    if active_dir[n][0] {
                        pos[n] = start[n].saturating_sub(1);
                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_p,
                            pos,
                            true,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt + 1.0;
                        max_delay = max_delay.max(delay as u32);
                        extension.volt_delay[n][0][1][ui_pos] = delay.floor() as u32;
                        extension.volt_delay_delta[n][0][1][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.volt_amp[n][0][1][ui_pos] =
                            (h_amp[n_p] * self.metric().edge_length(n_p, pos, true)) as FDTD_FLOAT;

                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_pp,
                            pos,
                            true,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt + 1.0;
                        max_delay = max_delay.max(delay as u32);
                        extension.volt_delay[n][0][0][ui_pos] = delay.floor() as u32;
                        extension.volt_delay_delta[n][0][0][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.volt_amp[n][0][0][ui_pos] = (h_amp[n_pp]
                            * self.metric().edge_length(n_pp, pos, true))
                            as FDTD_FLOAT;

                        pos[n] += 1;
                        let vi_p = self.vi_at(n_p, pos);
                        let vi_pp = self.vi_at(n_pp, pos);
                        extension.volt_amp[n][0][0][ui_pos] *= vi_p;
                        extension.volt_amp[n][0][1][ui_pos] *= vi_pp;
                    }

                    if active_dir[n][1] {
                        pos[n] = stop[n];
                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_p,
                            pos,
                            true,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt + 1.0;
                        max_delay = max_delay.max(delay as u32);
                        extension.volt_delay[n][1][1][ui_pos] = delay.floor() as u32;
                        extension.volt_delay_delta[n][1][1][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.volt_amp[n][1][1][ui_pos] =
                            (h_amp[n_p] * self.metric().edge_length(n_p, pos, true)) as FDTD_FLOAT;

                        let coordinate = yee_coordinates(
                            &self.base.disc_lines,
                            self.base.num_lines,
                            n_pp,
                            pos,
                            true,
                        );
                        let distance = (0..3)
                            .map(|axis| ((coordinate[axis] - origin[axis]) * prop_dir[axis]).abs())
                            .sum::<f64>();
                        let delay = distance * unit / phase_velocity / dt + 1.0;
                        max_delay = max_delay.max(delay as u32);
                        extension.volt_delay[n][1][0][ui_pos] = delay.floor() as u32;
                        extension.volt_delay_delta[n][1][0][ui_pos] =
                            (delay - delay.floor()) as FDTD_FLOAT;
                        extension.volt_amp[n][1][0][ui_pos] = (h_amp[n_pp]
                            * self.metric().edge_length(n_pp, pos, true))
                            as FDTD_FLOAT;

                        let vi_p = self.vi_at(n_p, pos);
                        let vi_pp = self.vi_at(n_pp, pos);
                        extension.volt_amp[n][1][0][ui_pos] *= vi_p;
                        extension.volt_amp[n][1][1][ui_pos] *= vi_pp;
                    }

                    if active_dir[n][1] {
                        extension.volt_amp[n][1][0][ui_pos] *= -1.0;
                    }
                    if active_dir[n][0] {
                        extension.volt_amp[n][0][1][ui_pos] *= -1.0;
                    }
                    if pos[n_p] == stop[n_p] {
                        if active_dir[n][0] {
                            extension.volt_amp[n][0][0][ui_pos] = 0.0;
                        }
                        if active_dir[n][1] {
                            extension.volt_amp[n][1][0][ui_pos] = 0.0;
                        }
                    }
                    if pos[n_pp] == stop[n_pp] {
                        if active_dir[n][0] {
                            extension.volt_amp[n][0][1][ui_pos] = 0.0;
                        }
                        if active_dir[n][1] {
                            extension.volt_amp[n][1][1][ui_pos] = 0.0;
                        }
                    }

                    ui_pos += 1;
                }
            }
        }

        extension.set_max_delay(max_delay + 1);
        Ok(Some(extension))
    }

    fn iv_at(&self, component: usize, pos: [usize; 3]) -> FDTD_FLOAT {
        self.iv_ptr
            .as_ref()
            .map(|array| *array.get(component, pos[0], pos[1], pos[2]))
            .unwrap_or(1.0)
    }

    fn vi_at(&self, component: usize, pos: [usize; 3]) -> FDTD_FLOAT {
        self.vi_ptr
            .as_ref()
            .map(|array| *array.get(component, pos[0], pos[1], pos[2]))
            .unwrap_or(1.0)
    }

    /// Build the lumped `RLC` extension from the CSX `LumpedElement`
    /// properties.
    ///
    /// Mirrors `Operator_Ext_LumpedRLC::BuildExtension`: only series elements
    /// and parallel elements with an inductance are handled here (a parallel
    /// `RC` is a plain lossy edge and needs no ADE); the box is snapped to the
    /// mesh, the element is scaled to one cell, the EC capacitance/conductance
    /// are rewritten and the single-edge update coefficients are recomputed.
    pub fn build_lumped_rlc_extension(
        &mut self,
        f_max: f64,
    ) -> Result<Option<EngineExtLumpedRLC>, String> {
        let Some(structure) = self.structure.as_ref() else {
            return Ok(None);
        };
        let params = structure.parameters().clone();
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("lumped RLC needs a positive timestep".to_string());
        }

        struct ElementSpec {
            priority: i32,
            topology: crate::lumped_rlc::LumpedTopology,
            direction: usize,
            resistance: f64,
            inductance: f64,
            capacitance: f64,
            caps: bool,
            boxes: Vec<(i32, Primitive)>,
        }

        let mut specs: Vec<ElementSpec> = Vec::new();
        for index in 0..structure.property_count() {
            let Some(property) = structure.property_by_index(index) else {
                continue;
            };
            let Some(lumped) = property.lumped_element() else {
                continue;
            };
            let topology = crate::lumped_rlc::LumpedTopology::from(lumped.element_type());
            let direction = lumped.direction();
            let mut resistance = lumped.resistance(Some(&params)).unwrap_or(f64::NAN);
            let mut inductance = lumped.inductance(Some(&params)).unwrap_or(f64::NAN);
            let mut capacitance = lumped.capacitance(Some(&params)).unwrap_or(f64::NAN);

            // Native warnings: NaN means "absent"; negative values are clamped
            // to zero because they are not physical.
            if capacitance < 0.0 {
                eprintln!(
                    "[openEMS] Warning: lumped element '{}' has C < 0; clamped to 0",
                    property.name()
                );
                capacitance = 0.0;
            }
            if resistance < 0.0 {
                eprintln!(
                    "[openEMS] Warning: lumped element '{}' has R < 0; clamped to 0",
                    property.name()
                );
                resistance = 0.0;
            }
            if inductance < 0.0 {
                eprintln!(
                    "[openEMS] Warning: lumped element '{}' has L < 0; clamped to 0",
                    property.name()
                );
                inductance = 0.0;
            }

            // Same gate as native `IsLElumpedRLC`.
            let is_lumped_rlc = match topology {
                crate::lumped_rlc::LumpedTopology::Parallel => inductance > 0.0,
                crate::lumped_rlc::LumpedTopology::Series => true,
            };
            if !is_lumped_rlc {
                continue;
            }
            if direction > 2 {
                eprintln!(
                    "[openEMS] Warning: lumped element '{}' has an invalid direction; skipped",
                    property.name()
                );
                continue;
            }

            let mut boxes = Vec::new();
            for primitive in property.primitives() {
                if matches!(
                    primitive.shape(),
                    csxcad_core::model::PrimitiveShape::Box(_)
                ) {
                    boxes.push((primitive.priority(), primitive.clone()));
                } else {
                    eprintln!(
                        "[openEMS] Warning: lumped element '{}' contains a non-box primitive; skipped",
                        property.name()
                    );
                }
            }
            if boxes.is_empty() {
                continue;
            }
            let priority = property
                .primitives()
                .iter()
                .map(|primitive| primitive.priority())
                .max()
                .unwrap_or(0);
            specs.push(ElementSpec {
                priority,
                topology,
                direction: direction as usize,
                resistance,
                inductance,
                capacitance,
                caps: lumped.caps(),
                boxes,
            });
        }

        if specs.is_empty() {
            return Ok(None);
        }

        let mut positions: Vec<[usize; 3]> = Vec::new();
        let mut coefficients: Vec<LumpedRLCCoeffs> = Vec::new();

        for spec in specs {
            let dir = spec.direction;
            let dir_p1 = (dir + 1) % 3;
            let dir_p2 = (dir + 2) % 3;

            for (_priority, primitive) in &spec.boxes {
                let bounds = match primitive.bounding_box(Some(&params), false) {
                    Ok(bounds) => bounds,
                    Err(_) => continue,
                };
                // CSX bounding boxes are interleaved: [min_x, max_x, min_y, max_y, min_z, max_z].
                let start = [bounds[0], bounds[2], bounds[4]];
                let stop = [bounds[1], bounds[3], bounds[5]];
                let (dimension, ui_start, ui_stop, _start_in, _stop_in) =
                    self.snap_box_to_mesh_native(start, stop, false, true, 0);
                if dimension <= 0 {
                    if dimension >= -1 {
                        eprintln!(
                            "[openEMS] Warning: lumped RLC box snapping failed (dimension {dimension}); skipped"
                        );
                    }
                    continue;
                }
                if ui_start[dir] == ui_stop[dir] {
                    eprintln!("[openEMS] Warning: lumped RLC has zero snapped length; skipped");
                    continue;
                }

                let cells_0 = ui_stop[dir] - ui_start[dir];
                let cells_1 = ui_stop[dir_p1] - ui_start[dir_p1] + 1;
                let cells_2 = ui_stop[dir_p2] - ui_start[dir_p2] + 1;
                let parallel_cells = cells_1 * cells_2;
                let element = crate::lumped_rlc::CellElement::from_element(
                    spec.resistance,
                    spec.inductance,
                    spec.capacitance,
                    parallel_cells,
                    cells_0,
                );

                let series = if spec.topology == crate::lumped_rlc::LumpedTopology::Series {
                    Some(crate::lumped_rlc::series_coefficients(dt, &element))
                } else {
                    None
                };

                let mut pos = [0usize; 3];
                for along in ui_start[dir]..ui_stop[dir] {
                    pos[dir] = along as usize;
                    for first in ui_start[dir_p1]..=ui_stop[dir_p1] {
                        pos[dir_p1] = first as usize;
                        for second in ui_start[dir_p2]..=ui_stop[dir_p2] {
                            pos[dir_p2] = second as usize;
                            let index =
                                Self::ec_index(dir, pos[0], pos[1], pos[2], self.base.num_lines);
                            if index >= self.ec_c.len() {
                                continue;
                            }

                            let mut coeffs = LumpedRLCCoeffs {
                                direction: dir,
                                ..Default::default()
                            };
                            match spec.topology {
                                crate::lumped_rlc::LumpedTopology::Parallel => {
                                    let grid_capacitance = self.ec_c[index] as f64;
                                    let capacitance = crate::lumped_rlc::parallel_capacitance(
                                        element.capacitance,
                                        grid_capacitance,
                                        element.resistance,
                                        element.inductance,
                                        f_max,
                                    );
                                    self.ec_c[index] = capacitance as FDTD_FLOAT;
                                    if element.resistance >= 0.0 {
                                        self.ec_g[index] = element.conductance as FDTD_FLOAT;
                                    }
                                    let (i2v, ilv) =
                                        crate::lumped_rlc::parallel_update_coefficients(
                                            dt,
                                            capacitance,
                                            &element,
                                        );
                                    coeffs.i2v = i2v as FDTD_FLOAT;
                                    coeffs.ilv = ilv as FDTD_FLOAT;
                                    coeffs.vvd = 1.0;
                                }
                                crate::lumped_rlc::LumpedTopology::Series => {
                                    self.ec_g[index] = 0.0;
                                    let grid_capacitance = self.ec_c[index] as f64;
                                    let capacitance = crate::lumped_rlc::series_capacitance(
                                        grid_capacitance,
                                        &element,
                                        f_max,
                                    );
                                    self.ec_c[index] = capacitance as FDTD_FLOAT;
                                    let series = series.expect("series coefficients built above");
                                    let update = crate::lumped_rlc::series_update_coefficients(
                                        dt,
                                        capacitance,
                                        &series,
                                    );
                                    coeffs.vv2 = update.vv2 as FDTD_FLOAT;
                                    coeffs.vj1 = update.vj1 as FDTD_FLOAT;
                                    coeffs.vj2 = update.vj2 as FDTD_FLOAT;
                                    coeffs.vvd = update.vvd as FDTD_FLOAT;
                                    coeffs.ib0 = update.ib0 as FDTD_FLOAT;
                                    coeffs.b1 = update.b1 as FDTD_FLOAT;
                                    coeffs.b2 = update.b2 as FDTD_FLOAT;
                                }
                            }
                            self.calc_ec_operator_pos(dir, pos);
                            positions.push(pos);
                            coefficients.push(coeffs);
                        }
                    }
                }

                // Metallic caps at both ends, as in native `GetCaps()`.
                if spec.caps {
                    for first in ui_start[dir_p1]..=ui_stop[dir_p1] {
                        for second in ui_start[dir_p2]..=ui_stop[dir_p2] {
                            let mut cap_pos = [0usize; 3];
                            cap_pos[dir_p1] = first as usize;
                            cap_pos[dir_p2] = second as usize;
                            for plane in [ui_start[dir], ui_stop[dir]] {
                                cap_pos[dir] = plane as usize;
                                for transverse in [dir_p1, dir_p2] {
                                    if cap_pos[transverse] < ui_stop[transverse] as usize {
                                        if let Some(vv) = self.vv_ptr.as_mut() {
                                            *vv.get_mut(
                                                transverse, cap_pos[0], cap_pos[1], cap_pos[2],
                                            ) = 0.0;
                                        }
                                        if let Some(vi) = self.vi_ptr.as_mut() {
                                            *vi.get_mut(
                                                transverse, cap_pos[0], cap_pos[1], cap_pos[2],
                                            ) = 0.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if positions.is_empty() {
            return Ok(None);
        }
        let mut extension = EngineExtLumpedRLC::new(positions.len());
        for index in 0..positions.len() {
            extension.set_position(index, positions[index], &coefficients[index]);
        }
        Ok(Some(extension))
    }

    /// Build the steady-state detection extension with the native probe layout:
    /// the centre plus the quarter and three-quarter positions along each axis,
    /// all three E components.
    ///
    /// Mirrors the `Operator_Ext_SteadyState` setup in `openEMS::SetupFDTD`.
    pub fn build_steady_state_extension(&self, period_ts: u32) -> Option<EngineExtSteadyState> {
        if period_ts == 0 {
            return None;
        }
        let counts = self.base.num_lines;
        let num_lines = [counts[0] as usize, counts[1] as usize, counts[2] as usize];
        let energy_lines = [
            self.get_number_of_lines(0, false) as usize,
            self.get_number_of_lines(1, false) as usize,
            self.get_number_of_lines(2, false) as usize,
        ];
        let center = [num_lines[0] / 2, num_lines[1] / 2, num_lines[2] / 2];
        let mut extension = EngineExtSteadyState::with_num_lines(period_ts as usize, energy_lines);
        for component in 0..3 {
            extension.add_e_probe(center, component);
        }
        for axis in 0..3 {
            for position in [num_lines[axis] / 4, num_lines[axis] * 3 / 4] {
                let mut probe = center;
                probe[axis] = position;
                for component in 0..3 {
                    extension.add_e_probe(probe, component);
                }
            }
        }
        Some(extension)
    }

    /// Build UPML regions and their engine extensions, modifying the main
    /// operator coefficients as `Operator_Ext_UPML::BuildExtension` does.
    pub fn build_upml_extensions(
        &mut self,
        pml_size: [u32; 6],
    ) -> Result<Vec<EngineExtUPML>, String> {
        let counts = self.base.num_lines;
        let bc = self.base.m_bc;
        let sizes = pml_size;
        for dir in 0..3 {
            if sizes[2 * dir] as usize + sizes[2 * dir + 1] as usize >= counts[dir] as usize {
                return Err(format!("PML size is too large in direction {dir}"));
            }
        }
        let dt = self.base.d_t;
        if dt <= 0.0 {
            return Err("UPML requires a positive timestep".to_string());
        }

        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        let lo = |face: usize| sizes[face] as usize;
        let mut regions: Vec<([usize; 3], [usize; 3])> = Vec::new();
        if bc[0] == 3 {
            regions.push(([0, 0, 0], [lo(0), ny - 1, nz - 1]));
        }
        if bc[1] == 3 {
            regions.push(([nx - 1 - lo(1), 0, 0], [nx - 1, ny - 1, nz - 1]));
        }
        let x_lo = if bc[0] == 3 { lo(0) + 1 } else { 0 };
        let x_hi = if bc[1] == 3 {
            nx.saturating_sub(2 + lo(1))
        } else {
            nx - 1
        };
        if bc[2] == 3 {
            regions.push(([x_lo, 0, 0], [x_hi, lo(2), nz - 1]));
        }
        if bc[3] == 3 {
            regions.push(([x_lo, ny - 1 - lo(3), 0], [x_hi, ny - 1, nz - 1]));
        }
        let y_lo = if bc[2] == 3 { lo(2) + 1 } else { 0 };
        let y_hi = if bc[3] == 3 {
            ny.saturating_sub(2 + lo(3))
        } else {
            ny - 1
        };
        if bc[4] == 3 {
            regions.push(([x_lo, y_lo, 0], [x_hi, y_hi, lo(4)]));
        }
        if bc[5] == 3 {
            regions.push(([x_lo, y_lo, nz - 1 - lo(5)], [x_hi, y_hi, nz - 1]));
        }
        if regions.is_empty() {
            return Err("No PML boundary is configured".to_string());
        }

        let disc_lines = &self.base.disc_lines;
        let grid_delta = self.base.grid_delta;
        let metric = MeshMetric::new(disc_lines, counts, grid_delta, self.mesh_type);
        let material_grid = self.material_grid.as_deref();
        let background = (
            self.base.m_bg_eps_r,
            self.base.m_bg_mue_r,
            self.base.m_bg_kappa,
        );

        let vv = self
            .vv_ptr
            .as_mut()
            .ok_or("vv coefficients not allocated")?;
        let vi = self
            .vi_ptr
            .as_mut()
            .ok_or("vi coefficients not allocated")?;
        let ii = self
            .ii_ptr
            .as_mut()
            .ok_or("ii coefficients not allocated")?;
        let iv = self
            .iv_ptr
            .as_mut()
            .ok_or("iv coefficients not allocated")?;

        let mut extensions = Vec::new();
        let debug_upml = std::env::var_os("OPENEMS_DEBUG_UPML").is_some();
        for (start, stop) in regions {
            if start.iter().zip(stop.iter()).any(|(low, high)| low > high) {
                continue;
            }
            let local = [
                stop[0] - start[0] + 1,
                stop[1] - start[1] + 1,
                stop[2] - start[2] + 1,
            ];
            if debug_upml {
                eprintln!(
                    "[debug] UPML region {start:?}..{stop:?} ({} cells), bc={bc:?} sizes={sizes:?}",
                    local[0] * local[1] * local[2]
                );
            }
            let mut enabled_cells = 0usize;
            let mut disabled_cells = 0usize;
            let mut extension = EngineExtUPML::new(start, local);
            for l0 in 0..local[0] {
                for l1 in 0..local[1] {
                    for l2 in 0..local[2] {
                        let pos = [start[0] + l0, start[1] + l1, start[2] + l2];
                        for n in 0..3usize {
                            let (eps_r, mue_r, kappa) =
                                material_full_value(material_grid, counts, background, n, pos);
                            let (kappa_v, kappa_i) = upml_grading_kappa(&metric, bc, sizes, n, pos);
                            let n_p = (n + 1) % 3;
                            let n_pp = (n + 2) % 3;
                            let edge_len = metric.edge_length(n, pos, false);
                            let edge_area = metric.edge_area(n, pos, false);
                            let dual_len = metric.edge_length(n, pos, true);
                            let dual_area = metric.edge_area(n, pos, true);

                            let main_vv = *vv.get(n, pos[0], pos[1], pos[2]);
                            let main_vi = *vi.get(n, pos[0], pos[1], pos[2]);
                            let kappa_v_sum = kappa_v[0] + kappa_v[1] + kappa_v[2];
                            if kappa_v_sum != 0.0
                                && kappa < 1.0e3
                                && main_vv + main_vi != 0.0
                                && edge_area > 0.0
                            {
                                enabled_cells += 1;
                                let denom = 2.0 * EPS0 + kappa_v[n_p] * dt;
                                let vv_new = ((2.0 * EPS0 - kappa_v[n_p] * dt) / denom) as f32;
                                let vi_new =
                                    ((2.0 * EPS0 * dt) / denom * edge_len / edge_area) as f32;
                                *vv.get_mut(n, pos[0], pos[1], pos[2]) = vv_new;
                                *vi.get_mut(n, pos[0], pos[1], pos[2]) = vi_new;
                                let denom_pp = 2.0 * EPS0 + kappa_v[n_pp] * dt;
                                let eps_scale = if eps_r > 0.0 { eps_r * EPS0 } else { EPS0 };
                                *extension.vv[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_v[n_pp] * dt) / denom_pp) as f32;
                                *extension.vvfn[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 + kappa_v[n] * dt) / denom_pp / eps_scale) as f32;
                                *extension.vvfo[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_v[n] * dt) / denom_pp / eps_scale) as f32;
                            } else {
                                disabled_cells += 1;
                                *extension.vv[n].get_mut(0, l0, l1, l2) = main_vv;
                                *vv.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
                                *extension.vvfo[n].get_mut(0, l0, l1, l2) = 0.0;
                                *extension.vvfn[n].get_mut(0, l0, l1, l2) = 1.0;
                            }

                            let main_ii = *ii.get(n, pos[0], pos[1], pos[2]);
                            let main_iv = *iv.get(n, pos[0], pos[1], pos[2]);
                            let kappa_i_sum = kappa_i[0] + kappa_i[1] + kappa_i[2];
                            if kappa_i_sum != 0.0 && main_ii + main_iv != 0.0 && dual_area > 0.0 {
                                let denom = 2.0 * EPS0 + kappa_i[n_p] * dt;
                                let ii_new = ((2.0 * EPS0 - kappa_i[n_p] * dt) / denom) as f32;
                                let iv_new =
                                    ((2.0 * EPS0 * dt) / denom * dual_len / dual_area) as f32;
                                *ii.get_mut(n, pos[0], pos[1], pos[2]) = ii_new;
                                *iv.get_mut(n, pos[0], pos[1], pos[2]) = iv_new;
                                let denom_pp = 2.0 * EPS0 + kappa_i[n_pp] * dt;
                                let mue_scale = if mue_r > 0.0 { mue_r * MU0 } else { MU0 };
                                *extension.ii[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_i[n_pp] * dt) / denom_pp) as f32;
                                *extension.iifn[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 + kappa_i[n] * dt) / denom_pp / mue_scale) as f32;
                                *extension.iifo[n].get_mut(0, l0, l1, l2) =
                                    ((2.0 * EPS0 - kappa_i[n] * dt) / denom_pp / mue_scale) as f32;
                            } else {
                                *extension.ii[n].get_mut(0, l0, l1, l2) = main_ii;
                                *ii.get_mut(n, pos[0], pos[1], pos[2]) = 0.0;
                                *extension.iifo[n].get_mut(0, l0, l1, l2) = 0.0;
                                *extension.iifn[n].get_mut(0, l0, l1, l2) = 1.0;
                            }
                        }
                    }
                }
            }
            extensions.push(extension);
            if debug_upml {
                eprintln!(
                    "[debug] UPML region {start:?}..{stop:?}: {enabled_cells} enabled, {disabled_cells} disabled component cells"
                );
            }
        }
        Ok(extensions)
    }
}

/// Material (eps_r, mue_r, kappa) for a component at a mesh position.
fn material_full_value(
    grid: Option<&[CellMaterial]>,
    counts: [u32; 3],
    background: (f64, f64, f64),
    component: usize,
    pos: [usize; 3],
) -> (f64, f64, f64) {
    if let Some(grid) = grid {
        let nx = counts[0] as usize;
        let ny = counts[1] as usize;
        let nz = counts[2] as usize;
        let index = component * nx * ny * nz + pos[0] * ny * nz + pos[1] * nz + pos[2];
        if let Some(material) = grid.get(index) {
            return (material.eps_r, material.mue_r, material.kappa);
        }
    }
    background
}

/// Default UPML grading profile from the native `Operator_Ext_UPML`.
fn default_upml_grading(depth: f64, dl: f64, width: f64) -> f64 {
    if depth <= 0.0 || dl <= 0.0 || width <= 0.0 {
        return 0.0;
    }
    let a = -(1.0e-6_f64).ln();
    let b = 2.5_f64.ln();
    let denominator = 2.0 * dl * Z0 * ((2.5_f64).powf(width / dl) - 1.0);
    if denominator == 0.0 {
        return 0.0;
    }
    a * b / denominator * (2.5_f64).powf(depth / dl)
}

/// Grading conductivities for one edge direction at a global position.
fn upml_grading_kappa(
    metric: &MeshMetric<'_>,
    bc: [i32; 6],
    sizes: [u32; 6],
    n_edge: usize,
    pos: [usize; 3],
) -> ([f64; 3], [f64; 3]) {
    let mut kappa_v = [0.0f64; 3];
    let mut kappa_i = [0.0f64; 3];
    let counts = metric.counts;
    let line = |dir: usize, index: usize| metric.line(dir, index, false) * metric.grid_delta;
    let edge = |dir: usize| metric.edge_length(dir, pos, false);

    for dir in 0..3 {
        let size_lo = sizes[2 * dir] as usize;
        let size_hi = sizes[2 * dir + 1] as usize;
        let last = counts[dir] as usize - 1;
        let lower = bc[2 * dir] == 3 && size_lo > 0 && pos[dir] <= size_lo;
        let upper = bc[2 * dir + 1] == 3 && size_hi > 0 && pos[dir] + size_hi >= last;

        if lower {
            let mut width = (line(dir, size_lo) - line(dir, 0)).abs();
            let cells = size_lo as f64;
            let mut depth = width - (line(dir, pos[dir]) - line(dir, 0)).abs();
            // Native `CalcGradingKappa`: an azimuthal PML is wider by `rho`.
            if metric.is_cylindrical() && dir == 1 {
                let rho = metric.line(0, pos[0], false);
                width *= rho;
                depth *= rho;
            }
            let dl = width / cells;
            if dir == n_edge {
                depth -= edge(dir) / 2.0;
            }
            kappa_v[dir] = default_upml_grading(depth, dl, width);
            if dir == n_edge {
                depth += edge(dir) / 2.0;
            }
            if dir != n_edge {
                depth -= edge(dir) / 2.0;
            }
            kappa_i[dir] = default_upml_grading(depth, dl, width);
        } else if upper {
            let mut width = (line(dir, last) - line(dir, last - size_hi)).abs();
            let cells = size_hi as f64;
            let mut depth = width - (line(dir, last) - line(dir, pos[dir])).abs();
            if metric.is_cylindrical() && dir == 1 {
                let rho = metric.line(0, pos[0], false);
                width *= rho;
                depth *= rho;
            }
            let dl = width / cells;
            if dir == n_edge {
                depth += edge(dir) / 2.0;
            }
            kappa_v[dir] = default_upml_grading(depth, dl, width);
            if dir == n_edge {
                depth -= edge(dir) / 2.0;
            }
            if dir != n_edge {
                depth += edge(dir) / 2.0;
            }
            if depth > width {
                depth = 0.0;
            }
            kappa_i[dir] = default_upml_grading(depth, dl, width);
        }
    }
    (kappa_v, kappa_i)
}

/// Disc line value in drawing units. Matches native `GetDiscLine`.
/// Falls back to a uniform unit-spaced grid when no disc lines were loaded.
fn disc_line_value(lines: &Option<Vec<f64>>, count: usize, pos: usize, dual: bool) -> f64 {
    let raw = lines.as_deref().unwrap_or(&[]);
    if raw.len() >= 2 && pos < count && pos < raw.len() {
        if !dual {
            raw[pos]
        } else if pos + 1 < count && pos + 1 < raw.len() {
            0.5 * (raw[pos] + raw[pos + 1])
        } else if pos > 0 {
            raw[pos] + 0.5 * (raw[pos] - raw[pos - 1])
        } else {
            raw[pos]
        }
    } else if dual {
        pos as f64 + 0.5
    } else {
        pos as f64
    }
}

/// Disc line delta in drawing units. Matches native `GetDiscDelta`.
fn disc_delta_value(lines: &Option<Vec<f64>>, count: usize, pos: usize, dual: bool) -> f64 {
    let raw = lines.as_deref().unwrap_or(&[]);
    if raw.len() >= 2 && pos < count {
        if !dual {
            if pos + 1 < count && pos + 1 < raw.len() {
                raw[pos + 1] - raw[pos]
            } else if pos > 0 {
                raw[pos] - raw[pos - 1]
            } else {
                0.0
            }
        } else if pos > 0 {
            disc_line_value(lines, count, pos, true) - disc_line_value(lines, count, pos - 1, true)
        } else if raw.len() >= 2 {
            raw[1] - raw[0]
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// Reflect a node index across the mesh boundary, matching `AdrOp::GetPos`
/// with `SetReflection2Cell` (`uiTypeOffset = 1`).
/// Add a relative shift along one axis.
fn shift_rel(mut rel: [i32; 3], axis: usize, delta: i32) -> [i32; 3] {
    rel[axis] += delta;
    rel
}

/// Apply a relative shift to `base` with the native cell reflection.
///
/// Mirrors `AdrOp::GetPos` with `SetReflection2Cell()` (`uiTypeOffset = 1`,
/// `tools/AdrOp.cpp:119-160`, `:423-433`): the shift is reflected against the
/// base index, so `-1` at index 0 stays at 0 and `+1` at the last index stays
/// at the last index.
fn reflected_pos(base: [usize; 3], rel: [i32; 3], counts: [u32; 3]) -> [usize; 3] {
    let mut out = base;
    for axis in 0..3 {
        let max = counts[axis] as i32 - 1;
        let mut r = rel[axis];
        if base[axis] as i32 + r < 0 {
            r = -2 * base[axis] as i32 - r - 1;
        }
        if base[axis] as i32 + r > max {
            r = 2 * (max - base[axis] as i32) - r + 1;
        }
        out[axis] = (base[axis] as i32 + r).clamp(0, max) as usize;
    }
    out
}

/// Sum `1/(L*C)` terms; `None` when any coefficient is non-positive.
fn sum_terms(terms: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    for term in terms {
        sum += (*term)?;
    }
    Some(sum)
}

/// `sum - 2*min` over four `1/(L*C)` terms, as in `CalcTimestep_Var3`.
fn weighted_term(terms: &[Option<f64>; 4]) -> Option<f64> {
    let mut sum = 0.0;
    let mut minimum = f64::INFINITY;
    for term in terms {
        let value = (*term)?;
        sum += value;
        minimum = minimum.min(value);
    }
    Some(sum - 2.0 * minimum)
}

/// Raw (non-dual) mesh delta for a node index, matching native
/// `Operator::GetRawDiscDelta`: the spacing after `index`, mirrored at the ends.
fn raw_disc_delta(lines: &[Option<Vec<f64>>; 3], counts: [u32; 3], axis: usize, index: i64) -> f64 {
    let Some(values) = lines.get(axis).and_then(|entry| entry.as_ref()) else {
        return 0.0;
    };
    let count = (counts.get(axis).copied().unwrap_or(0) as usize).min(values.len());
    if count < 2 {
        return 0.0;
    }
    if index < 0 {
        return values[0] - values[1];
    }
    if index >= count as i64 - 1 {
        return values[count - 2] - values[count - 1];
    }
    values[index as usize + 1] - values[index as usize]
}

/// Yee coordinate in drawing units for an edge. Matches native `GetYeeCoords`:
/// along the edge direction use the requested mesh, across it use the dual one.
fn yee_coordinates(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    n: usize,
    pos: [usize; 3],
    dual: bool,
) -> [f64; 3] {
    let mut coordinates = [0.0; 3];
    for d in 0..3 {
        // Native `GetYeeCoords`: transverse axes use the requested mesh
        // (`dualMesh`), the edge axis uses the opposite one.
        let use_dual = if d == n { !dual } else { dual };
        coordinates[d] = disc_line_value(&lines[d], counts[d] as usize, pos[d], use_dual);
    }
    coordinates
}

/// Edge length in metres. Matches native `GetEdgeLength`.
fn edge_length_value(
    lines: &[Option<Vec<f64>>; 3],
    counts: [u32; 3],
    n: usize,
    pos: [usize; 3],
    dual: bool,
    grid_delta: f64,
) -> f64 {
    if n >= 3 || pos[n] >= counts[n] as usize {
        return 0.0;
    }
    disc_delta_value(&lines[n], counts[n] as usize, pos[n], dual) * grid_delta
}

/// 3x3 determinant in the row-major layout of native `Determinant`
/// (`tools/useful.cpp:272-275`).
fn determinant3(mat: &[f64; 9]) -> f64 {
    mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7]
        - mat[2] * mat[4] * mat[6]
        - mat[1] * mat[3] * mat[8]
        - mat[0] * mat[5] * mat[7]
}

/// 3x3 inverse in the row-major layout of native `Invert`
/// (`tools/useful.cpp:277-290`). The caller has already rejected a vanishing
/// determinant, so this is only reached for the values native divides by.
fn invert3(mat: &[f64; 9]) -> [f64; 9] {
    let det = determinant3(mat);
    [
        (mat[4] * mat[8] - mat[5] * mat[7]) / det,
        (mat[2] * mat[7] - mat[1] * mat[8]) / det,
        (mat[1] * mat[5] - mat[2] * mat[4]) / det,
        (mat[5] * mat[6] - mat[3] * mat[8]) / det,
        (mat[0] * mat[8] - mat[2] * mat[6]) / det,
        (mat[2] * mat[3] - mat[0] * mat[5]) / det,
        (mat[3] * mat[7] - mat[4] * mat[6]) / det,
        (mat[1] * mat[6] - mat[0] * mat[7]) / det,
        (mat[0] * mat[4] - mat[1] * mat[3]) / det,
    ]
}

/// Intersection of the segment `l_start`-`l_stop` with the parallelogram
/// `p0`, `p1`, `p2`.
///
/// Mirrors native `LinePlaneIntersection` (`tools/useful.cpp:292-328`):
/// returns `-1` for a degenerate matrix, `1` when the intersection lies outside
/// the parallelogram (`u`/`v` outside `[0, 1]`), `2` when it lies outside the
/// segment (`t` outside `[0, 1]`) and `0` for a real hit. The returned point is
/// always `l_start*(1-t) + l_stop*t`, i.e. the native `is_point`, which is only
/// meaningful for a `0` return. The native `dist` out-parameter is `t` and has
/// no other consumer.
fn line_plane_intersection(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    l_start: [f64; 3],
    l_stop: [f64; 3],
) -> (i32, [f64; 3]) {
    let mut is_point = [0.0; 3];
    let mut mat = [0.0; 9];
    for n in 0..3 {
        mat[3 * n] = l_start[n] - l_stop[n];
        mat[3 * n + 1] = p1[n] - p0[n];
        mat[3 * n + 2] = p2[n] - p0[n];
    }
    let det = determinant3(&mat);
    if det.abs() < 1e-50 {
        return (-1, is_point);
    }
    let inv_mat = invert3(&mat);

    let mut t = 0.0;
    let mut u = 0.0;
    let mut v = 0.0;
    for n in 0..3 {
        t += inv_mat[n] * (l_start[n] - p0[n]);
        u += inv_mat[3 + n] * (l_start[n] - p0[n]);
        v += inv_mat[6 + n] * (l_start[n] - p0[n]);
    }
    for n in 0..3 {
        is_point[n] = l_start[n] * (1.0 - t) + l_stop[n] * t;
    }

    if (u < 0.0) || (u > 1.0) || (v < 0.0) || (v > 1.0) {
        return (1, is_point);
    }
    if (t < 0.0) || (t > 1.0) {
        return (2, is_point);
    }
    (0, is_point)
}

impl Default for Operator {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorBase for Operator {
    fn mesh_type(&self) -> MeshType {
        self.mesh_type
    }

    fn closed_alpha(&self) -> bool {
        Operator::closed_alpha(self)
    }

    fn material_grid_data(&self) -> Option<&[CellMaterial]> {
        self.material_grid.as_deref()
    }

    fn set_geometry_csx(&mut self, geo: *mut std::ffi::c_void) -> bool {
        self.base.set_geometry_csx(geo)
    }

    fn get_geometry_csx(&self) -> *mut std::ffi::c_void {
        self.base.get_geometry_csx()
    }

    fn get_timestep(&self) -> f64 {
        self.base.get_timestep()
    }

    fn get_number_cells(&self) -> f64 {
        self.base.num_lines[0] as f64
            * self.base.num_lines[1] as f64
            * self.base.num_lines[2] as f64
    }

    fn get_number_of_nyquist_timesteps(&self) -> u32 {
        self.base.get_number_of_nyquist_timesteps()
    }

    fn get_number_of_lines(&self, ny: i32, full: bool) -> u32 {
        let axis = ny as usize;
        // native `Operator_Cylinder::GetNumberOfLines`: a closed azimuth hides
        // the duplicated seam lines from post-processing and the energy estimate
        if axis == 1 {
            return self.alpha_line_count(full);
        }
        self.base.num_lines[axis]
    }

    fn get_grid_delta(&self) -> f64 {
        self.base.grid_delta
    }

    fn get_disc_line(&self, n: i32, pos: u32, dual_mesh: bool) -> f64 {
        if !(0..3).contains(&n) {
            return 0.0;
        }
        let dir = n as usize;
        if pos >= self.base.num_lines[dir] {
            return 0.0;
        }
        disc_line_value(
            &self.base.disc_lines[dir],
            self.base.num_lines[dir] as usize,
            pos as usize,
            dual_mesh,
        )
    }

    fn get_disc_delta(&self, n: i32, pos: u32, dual_mesh: bool) -> f64 {
        if !(0..3).contains(&n) {
            return 0.0;
        }
        let dir = n as usize;
        if pos >= self.base.num_lines[dir] {
            return 0.0;
        }
        disc_delta_value(
            &self.base.disc_lines[dir],
            self.base.num_lines[dir] as usize,
            pos as usize,
            dual_mesh,
        )
    }

    fn get_node_width(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        let dir = ny as usize;
        edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            dir,
            [pos[0] as usize, pos[1] as usize, pos[2] as usize],
            dual_mesh,
            self.base.grid_delta,
        )
    }

    fn get_node_area(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        let dir = ny as usize;
        let ui = [pos[0] as usize, pos[1] as usize, pos[2] as usize];
        let n_p = (dir + 1) % 3;
        let n_pp = (dir + 2) % 3;
        edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            n_p,
            ui,
            dual_mesh,
            self.base.grid_delta,
        ) * edge_length_value(
            &self.base.disc_lines,
            self.base.num_lines,
            n_pp,
            ui,
            dual_mesh,
            self.base.grid_delta,
        )
    }

    fn get_edge_length(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        self.metric().edge_length(
            ny as usize,
            [pos[0] as usize, pos[1] as usize, pos[2] as usize],
            dual_mesh,
        )
    }

    fn get_edge_area(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64 {
        if !(0..3).contains(&ny) {
            return 0.0;
        }
        self.metric().edge_area(
            ny as usize,
            [pos[0] as usize, pos[1] as usize, pos[2] as usize],
            dual_mesh,
        )
    }

    fn get_cell_volume(&self, pos: [u32; 3], dual_mesh: bool) -> f64 {
        let ui = [pos[0] as usize, pos[1] as usize, pos[2] as usize];
        self.metric().cell_volume(ui, dual_mesh)
    }

    fn get_vv(&self) -> Option<&FieldArray> {
        self.vv_ptr.as_ref()
    }

    fn get_vi(&self) -> Option<&FieldArray> {
        self.vi_ptr.as_ref()
    }

    fn get_ii(&self) -> Option<&FieldArray> {
        self.ii_ptr.as_ref()
    }

    fn get_iv(&self) -> Option<&FieldArray> {
        self.iv_ptr.as_ref()
    }

    fn snap_to_mesh(
        &self,
        coord: [f64; 3],
        uicoord: &mut [u32; 3],
        dual_mesh: bool,
        full_mesh: bool,
        inside: Option<&mut bool>,
    ) -> bool {
        let (index, axis_inside) = self.snap_to_mesh_native(coord, dual_mesh, full_mesh);
        *uicoord = index;
        let all_inside = axis_inside.iter().all(|&value| value);
        if let Some(inside) = inside {
            *inside = all_inside;
        }
        all_inside
    }

    fn snap_box_to_mesh(
        &self,
        start: [f64; 3],
        stop: [f64; 3],
        ui_start: &mut [u32; 3],
        ui_stop: &mut [u32; 3],
        dual_mesh: bool,
        full_mesh: bool,
        snap_method: i32,
        b_start_in: Option<&mut bool>,
        b_stop_in: Option<&mut bool>,
    ) -> i32 {
        let (dimension, snapped_start, snapped_stop, start_in, stop_in) =
            self.snap_box_to_mesh_native(start, stop, dual_mesh, full_mesh, snap_method);
        *ui_start = snapped_start;
        *ui_stop = snapped_stop;
        if let Some(b_start_in) = b_start_in {
            *b_start_in = start_in.iter().all(|&value| value);
        }
        if let Some(b_stop_in) = b_stop_in {
            *b_stop_in = stop_in.iter().all(|&value| value);
        }
        dimension
    }

    fn get_primitives_bbox(
        &self,
        _pos_x: i32,
        _pos_y: i32,
        _pos_z: i32,
        _type_: i32,
    ) -> Vec<*mut std::ffi::c_void> {
        // No CSXCAD geometry loaded in the operator yet; return empty.
        // When CSXCAD structures are loaded, this should return pointers to
        // primitives whose bounding boxes overlap the query region.
        Vec::new()
    }

    fn set_boundary_condition(&mut self, bcs: [i32; 6]) {
        self.base.set_boundary_condition(bcs);
    }

    fn set_material_store_flags(&mut self, _type_: i32, _val: bool) {
        // Material storage for post-processing is not yet wired; no-op.
    }

    fn cleanup_material_storage(&mut self) {
        // No material storage to clean up yet.
    }

    fn get_disc_material(&self, _type_: i32, _ny: i32, _pos: [u32; 3]) -> f64 {
        0.0
    }

    fn get_cell_center_material_avg_coord_uint(&self, pos: [u32; 3], coord: &mut [f64; 3]) -> bool {
        let grid_delta = self.base.grid_delta;
        if grid_delta <= 0.0 {
            return false;
        }
        for n in 0..3 {
            coord[n] = pos[n] as f64 * grid_delta;
        }
        true
    }

    fn get_cell_center_material_avg_coord_int(&self, pos: [i32; 3], coord: &mut [f64; 3]) -> bool {
        let grid_delta = self.base.grid_delta;
        if grid_delta <= 0.0 {
            return false;
        }
        for n in 0..3 {
            coord[n] = pos[n] as f64 * grid_delta;
        }
        true
    }

    fn set_background_material(
        &mut self,
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
        density: f64,
    ) {
        self.base
            .set_background_material(eps_r, mue_r, kappa, sigma, density);
    }

    fn get_background_eps_r(&self) -> f64 {
        self.base.get_background_eps_r()
    }

    fn set_background_eps_r(&mut self, val: f64) {
        self.base.set_background_eps_r(val);
    }

    fn get_background_mue_r(&self) -> f64 {
        self.base.get_background_mue_r()
    }

    fn set_background_mue_r(&mut self, val: f64) {
        self.base.set_background_mue_r(val);
    }

    fn get_background_kappa(&self) -> f64 {
        self.base.get_background_kappa()
    }

    fn set_background_kappa(&mut self, val: f64) {
        self.base.set_background_kappa(val);
    }

    fn get_background_sigma(&self) -> f64 {
        self.base.get_background_sigma()
    }

    fn set_background_sigma(&mut self, val: f64) {
        self.base.set_background_sigma(val);
    }

    fn get_background_density(&self) -> f64 {
        self.base.get_background_density()
    }

    fn set_background_density(&mut self, val: f64) {
        self.base.set_background_density(val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let op = Operator::new();
        assert_eq!(op.get_timestep(), 0.0);
        assert_eq!(op.get_background_eps_r(), 1.0);
    }

    /// A minimal but valid CSXCAD XML geometry: a 10×10×10 mm uniform grid,
    /// vacuum background, and a small PEC box.
    const SAMPLE_CSX: &str = r#"<ContinuousStructure CoordSystem="0">
        <RectilinearGrid>
            <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
            <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
            <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
        </RectilinearGrid>
        <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
        <Properties>
            <Metal Name="pec">
                <Primitives>
                    <Box Priority="0">
                        <P1 X="3" Y="3" Z="3"/>
                        <P2 X="5" Y="5" Z="5"/>
                    </Box>
                </Primitives>
            </Metal>
            <Material Name="substrate">
                <Property Epsilon="4.4,4.4,4.4" Mue="1,1,1" Kappa="0,0,0" Sigma="0,0,0" Density="0"/>
                <Primitives>
                    <Box Priority="0">
                        <P1 X="0" Y="0" Z="0"/>
                        <P2 X="10" Y="10" Z="2"/>
                    </Box>
                </Primitives>
            </Material>
            <AbsorbingBC Name="boundary">
                <Primitives>
                    <Box Priority="0">
                        <P1 X="0" Y="0" Z="0"/>
                        <P2 X="10" Y="10" Z="10"/>
                    </Box>
                </Primitives>
            </AbsorbingBC>
        </Properties>
    </ContinuousStructure>"#;

    #[test]
    fn test_load_csx_grid_and_background() {
        let mut op = Operator::new();
        let result = op.load_csx(SAMPLE_CSX);
        if let Err(ref e) = result {
            eprintln!("load_csx error: {}", e);
        }
        assert!(result.is_ok(), "load_csx failed");
        assert_eq!(op.base.num_lines, [11, 11, 11]);
        assert!((op.base.grid_delta - 1.0).abs() < 1e-9);
        assert!((op.base.get_background_eps_r() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_the_operator_sets_the_drawing_tolerance_from_the_mesh() {
        let mut op = Operator::new();
        op.load_csx(SAMPLE_CSX).expect("load_csx");
        // Native `Operator::SetCSXGrid`: 1e-3 of the smallest mesh step
        // (SAMPLE_CSX uses 1.0 drawing units with a unit delta).
        let expected = 1.0e-3 * op.base.grid_delta;
        assert!(
            (op.drawing_tolerance - expected).abs() <= 1.0e-12,
            "tolerance {} vs expected {}",
            op.drawing_tolerance,
            expected
        );
        assert!(op
            .structure
            .as_ref()
            .expect("structure")
            .drawing_tolerance()
            > 0.0);
    }

    #[test]
    fn test_load_csx_material_grid_populated() {
        let mut op = Operator::new();
        let result = op.load_csx(SAMPLE_CSX);
        if let Err(ref e) = result {
            eprintln!("load_csx error: {}", e);
        }
        assert!(result.is_ok(), "load_csx failed");
        // material_grid must be Some after load_csx.
        assert!(op.material_grid.is_some());
        let grid = op.material_grid.as_ref().unwrap();
        let nx = op.base.num_lines[0] as usize;
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        assert_eq!(grid.len(), 3 * nx * ny * nz);
        // Cell (4,4,4) is inside the PEC box in every direction.
        let pec_cell = 4 * ny * nz + 4 * nz + 4;
        // Cell (4,4,1) is inside the substrate box in every direction.
        let sub_cell = 4 * ny * nz + 4 * nz + 1;
        for n in 0..3 {
            let pec_index = n * nx * ny * nz + pec_cell;
            let sub_index = n * nx * ny * nz + sub_cell;
            assert!(
                grid[pec_index].is_pec,
                "cell (4,4,4) should be PEC in dir {n}"
            );
            assert!(
                (grid[sub_index].eps_r - 4.4).abs() < 1e-9,
                "cell (4,4,1) should have eps_r=4.4 in dir {n}"
            );
        }
    }

    #[test]
    fn ec_coefficients_use_native_conductivity_units() {
        let mut op = Operator::new();
        op.base.grid_delta = 1e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_background_material(4.0, 2.0, 0.01, 0.02, 0.0);
        op.init_data_storage([3, 3, 3]);
        op.calc_ec_operator();

        let dt = 1.0e-12;
        let delta = 1.0e-3;
        let c = 4.0 * EPS0 * delta;
        let g = 0.01 * delta;
        let l = 2.0 * MU0 * delta;
        let r = 0.02 * delta;
        let v_denom = 1.0 + dt * g / (2.0 * c);
        let i_denom = 1.0 + dt * r / (2.0 * l);
        let expected_vv = (1.0 - dt * g / (2.0 * c)) / v_denom;
        let expected_vi = (dt / c) / v_denom;
        let expected_ii = (1.0 - dt * r / (2.0 * l)) / i_denom;
        let expected_iv = (dt / l) / i_denom;

        let vv = op.vv_ptr.as_ref().unwrap();
        let vi = op.vi_ptr.as_ref().unwrap();
        let ii = op.ii_ptr.as_ref().unwrap();
        let iv = op.iv_ptr.as_ref().unwrap();
        for n in 0..3 {
            for (i, j, k) in [(0, 0, 0), (1, 1, 1), (2, 2, 2)] {
                let actual_vv = *vv.get(n, i, j, k) as f64;
                let actual_vi = *vi.get(n, i, j, k) as f64;
                let actual_ii = *ii.get(n, i, j, k) as f64;
                let actual_iv = *iv.get(n, i, j, k) as f64;
                assert!(
                    (actual_vv - expected_vv).abs() <= expected_vv.abs() * 1.0e-5,
                    "vv mismatch at ({n},{i},{j},{k}): {actual_vv:e} vs {expected_vv:e}"
                );
                assert!(
                    (actual_vi - expected_vi).abs() <= expected_vi.abs() * 1.0e-5,
                    "vi mismatch at ({n},{i},{j},{k}): {actual_vi:e} vs {expected_vi:e}"
                );
                assert!(
                    (actual_ii - expected_ii).abs() <= expected_ii.abs() * 1.0e-5,
                    "ii mismatch at ({n},{i},{j},{k}): {actual_ii:e} vs {expected_ii:e}"
                );
                assert!(
                    (actual_iv - expected_iv).abs() <= expected_iv.abs() * 1.0e-5,
                    "iv mismatch at ({n},{i},{j},{k}): {actual_iv:e} vs {expected_iv:e}"
                );
            }
        }
    }

    #[test]
    fn test_load_csx_invalid_xml() {
        let mut op = Operator::new();
        assert!(op.load_csx("not valid xml").is_err());
    }

    #[test]
    fn pec_boundary_zeroes_only_tangential_components() {
        let mut op = Operator::new();
        op.base.grid_delta = 1e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_boundary_condition([0; 6]);
        op.init_data_storage([5, 5, 5]);
        op.calc_ec_operator();
        op.apply_electric_bc();

        let vv = op.vv_ptr.as_ref().unwrap();
        // x-min face at (0,2,2): Ey and Ez are tangential, Ex is normal.
        assert_eq!(*vv.get(1, 0, 2, 2), 0.0);
        assert_eq!(*vv.get(2, 0, 2, 2), 0.0);
        assert!(*vv.get(0, 0, 2, 2) > 0.0);
        // y-min face at (2,0,2): Ex and Ez are tangential, Ey is normal.
        assert_eq!(*vv.get(0, 2, 0, 2), 0.0);
        assert_eq!(*vv.get(2, 2, 0, 2), 0.0);
        assert!(*vv.get(1, 2, 0, 2) > 0.0);
        // z-min face at (2,2,0): Ex and Ey are tangential, Ez is normal.
        assert_eq!(*vv.get(0, 2, 2, 0), 0.0);
        assert_eq!(*vv.get(1, 2, 2, 0), 0.0);
        assert!(*vv.get(2, 2, 2, 0) > 0.0);
    }

    #[test]
    fn pmc_boundary_matches_the_native_magnetic_bc() {
        let mut op = Operator::new();
        op.base.grid_delta = 1e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_boundary_condition([1; 6]);
        op.init_data_storage([5, 5, 5]);
        op.calc_ec_operator();
        op.apply_magnetic_bc();

        let ii = op.ii_ptr.as_ref().unwrap();
        // Native `ApplyMagneticBC` clears *all three* current components on a
        // low PMC face, and the first inner line keeps its coefficients.
        for n in 0..3 {
            assert_eq!(*ii.get(n, 0, 2, 2), 0.0);
            assert!(*ii.get(n, 1, 2, 2) > 0.0);
        }
        // On the high face only the tangential components are cleared, and on
        // the line inside the outermost one (numLines - 2 = 3).
        assert_eq!(*ii.get(1, 3, 2, 2), 0.0);
        assert_eq!(*ii.get(2, 3, 2, 2), 0.0);
        assert!(*ii.get(0, 3, 2, 2) > 0.0);
        // The outermost line always lies outside the iterated domain.
        for n in 0..3 {
            assert_eq!(*ii.get(n, 4, 2, 2), 0.0);
            assert_eq!(*ii.get(n, 2, 4, 2), 0.0);
            assert_eq!(*ii.get(n, 2, 2, 4), 0.0);
        }
        // Low faces of the other axes follow the same rule.
        for n in 0..3 {
            assert_eq!(*ii.get(n, 2, 0, 2), 0.0);
            assert_eq!(*ii.get(n, 2, 2, 0), 0.0);
        }
    }

    #[test]
    fn material_priority_and_containment_select_the_right_edge() {
        // low box covers the whole domain (eps=2, priority 5 is below high box),
        // high box covers 4..6 (eps=4), sphere around (5,5,5) has priority 10.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="1">
                <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
                <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
                <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Material Name="low">
                    <Property Epsilon="2" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="1"><P1 X="0" Y="0" Z="0"/><P2 X="10" Y="10" Z="10"/></Box>
                    </Primitives>
                </Material>
                <Material Name="high">
                    <Property Epsilon="4" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="5"><P1 X="4" Y="4" Z="4"/><P2 X="6" Y="6" Z="6"/></Box>
                    </Primitives>
                </Material>
                <Material Name="sphere">
                    <Property Epsilon="6" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Sphere Priority="10" Radius="1"><Center X="5" Y="5" Z="5"/></Sphere>
                    </Primitives>
                </Material>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("priority CSX should load");
        let grid = op.material_grid.as_ref().expect("material grid");
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        let at = |i: usize, j: usize, k: usize| {
            let index = i * ny * nz + j * nz + k;
            grid[index].eps_r
        };
        // Outside the sphere and high box: low-priority box.
        assert!((at(3, 3, 3) - 2.0).abs() < 1e-12, "eps={}", at(3, 3, 3));
        // The high-priority box corner averages with the surrounding low box:
        // only one of the four quarter cells lies inside the box.
        assert!((at(4, 4, 4) - 2.5).abs() < 1e-12, "eps={}", at(4, 4, 4));
        // Inside the sphere all four quarter cells are in the sphere.
        assert!((at(5, 5, 5) - 6.0).abs() < 1e-12, "eps={}", at(5, 5, 5));
    }

    #[test]
    fn quarter_cell_averaging_matches_the_analytic_half_space() {
        // Half space eps=4 for x >= 5, background eps=1 elsewhere, uniform mesh.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="1">
                <XLines>0,1,2,3,4,5,6,7,8,9,10</XLines>
                <YLines>0,1,2,3,4,5,6,7,8,9,10</YLines>
                <ZLines>0,1,2,3,4,5,6,7,8,9,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Material Name="halfspace">
                    <Property Epsilon="4" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="1"><P1 X="5" Y="-10" Z="-10"/><P2 X="10" Y="10" Z="10"/></Box>
                    </Primitives>
                </Material>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("half-space CSX should load");
        let grid = op.material_grid.as_ref().expect("material grid");
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        // E-y edge at node (5,5,5): transverse x samples at 4.75 and 5.25,
        // so exactly two of the four quarter cells are in eps=4.
        let index = 1 * (op.base.num_lines[0] as usize) * ny * nz + 5 * ny * nz + 5 * nz + 5;
        assert!(
            (grid[index].eps_r - 2.5).abs() < 1e-12,
            "eps={}",
            grid[index].eps_r
        );
        // Far inside the half space all four quarter cells are eps=4.
        let inside = 1 * (op.base.num_lines[0] as usize) * ny * nz + 7 * ny * nz + 5 * nz + 5;
        assert!(
            (grid[inside].eps_r - 4.0).abs() < 1e-12,
            "eps={}",
            grid[inside].eps_r
        );
        // Far outside stays at the background.
        let outside = 1 * (op.base.num_lines[0] as usize) * ny * nz + 2 * ny * nz + 5 * nz + 5;
        assert!(
            (grid[outside].eps_r - 1.0).abs() < 1e-12,
            "eps={}",
            grid[outside].eps_r
        );
    }

    #[test]
    fn metal_edges_zero_the_electric_update() {
        let mut op = Operator::new();
        op.load_csx(SAMPLE_CSX).expect("sample CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        let vv = op.vv_ptr.as_ref().unwrap();
        // Cell (4,4,4) is inside the PEC box in every direction.
        for n in 0..3 {
            assert_eq!(*vv.get(n, 4, 4, 4), 0.0, "PEC vv should be zero in dir {n}");
        }
        // A vacuum edge outside the metal keeps its update coefficient.
        assert!(*vv.get(0, 8, 8, 8) > 0.0);
    }

    #[test]
    fn mur_extension_builds_six_faces() {
        let mut op = Operator::new();
        op.base.grid_delta = 1.0e-3;
        op.base.d_t = 1.9e-12;
        op.base.set_boundary_condition([2; 6]);
        op.init_data_storage([6, 6, 6]);
        let extension = op.build_mur_extension([0.0; 6]);
        assert_eq!(extension.faces.len(), 6);
        for face in &extension.faces {
            assert_eq!(face.coeff_p.len(), face.dims_p * face.dims_pp);
            assert_eq!(face.coeff_pp.len(), face.dims_p * face.dims_pp);
            assert!(face
                .coeff_p
                .iter()
                .all(|coefficient| coefficient.abs() < 1.0));
            assert!(face
                .coeff_pp
                .iter()
                .all(|coefficient| coefficient.abs() < 1.0));
        }
    }

    #[test]
    fn mur_uses_the_local_face_delta_on_a_graded_mesh() {
        // DeltaUnit 1 mm; graded X (1, 2, 3 mm), Y (2 mm), Z (5 mm). Native
        // `GetDiscDelta` repeats the previous delta at the last node, so the
        // x-max face also sees 3 mm. A hard-coded or global delta would give
        // the same coefficient on every face.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,3,6</XLines>
                <YLines>0,2,4</YLines>
                <ZLines>0,5,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties></Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("graded CSX should load");
        let dt = 1.0e-12;
        op.base.d_t = dt;
        op.base.set_boundary_condition([2; 6]);
        op.init_data_storage([4, 3, 3]);

        let extension = op.build_mur_extension([0.0; 6]);
        assert_eq!(extension.faces.len(), 6);

        let c_dt = C0 * dt;
        for face in &extension.faces {
            let delta = match (face.dir, face.top) {
                (0, false) => 1.0e-3,
                (0, true) => 3.0e-3,
                (1, _) => 2.0e-3,
                (2, _) => 5.0e-3,
                _ => unreachable!("face direction {0}", face.dir),
            };
            let expected = ((c_dt - delta) / (c_dt + delta)) as f32;
            for coefficient in face.coeff_p.iter().chain(face.coeff_pp.iter()) {
                assert!(
                    (coefficient - expected).abs() < 1.0e-6,
                    "face {} (dir {}, top {}) coefficient {coefficient} vs {expected} for delta {delta}",
                    face.face,
                    face.dir,
                    face.top
                );
            }
        }

        // The two x faces must differ: the same coefficient everywhere would
        // mean the delta was not taken from the mesh.
        let x_min = extension
            .faces
            .iter()
            .find(|f| f.dir == 0 && !f.top)
            .unwrap();
        let x_max = extension
            .faces
            .iter()
            .find(|f| f.dir == 0 && f.top)
            .unwrap();
        assert!(
            (x_min.coeff_p[0] - x_max.coeff_p[0]).abs() > 0.2,
            "the graded x faces must use different deltas: {} vs {}",
            x_min.coeff_p[0],
            x_max.coeff_p[0]
        );
    }

    #[test]
    fn upml_builds_six_regions_with_finite_coefficients() {
        let mut op = Operator::new();
        op.base.grid_delta = 1.0e-3;
        op.base.d_t = 1.0e-12;
        op.base.set_boundary_condition([3; 6]);
        op.init_data_storage([10, 10, 10]);
        op.calc_ec_operator();
        let extensions = op
            .build_upml_extensions([2; 6])
            .expect("UPML should build on a 10^3 grid");
        assert_eq!(extensions.len(), 6, "one region per PML face");
        let mut non_trivial = false;
        for extension in &extensions {
            assert_eq!(extension.vv.len(), 3);
            let [nx, ny, nz] = extension.num_lines;
            for comp in 0..3 {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let vv = *extension.vv[comp].get(0, i, j, k);
                            let vvfn = *extension.vvfn[comp].get(0, i, j, k);
                            let ii = *extension.ii[comp].get(0, i, j, k);
                            assert!(vv.is_finite() && vvfn.is_finite() && ii.is_finite());
                            if (vvfn - 1.0).abs() > 1e-12 {
                                non_trivial = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(non_trivial, "UPML coefficients were never activated");
    }

    #[test]
    fn nonuniform_mesh_edge_lengths_and_ec_coefficients() {
        // DeltaUnit 1 mm; graded X (1,2,3), Y (2,2), Z (5,5) in drawing units.
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,3,6</XLines>
                <YLines>0,2,4</YLines>
                <ZLines>0,5,10</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties></Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("non-uniform CSX should load");
        assert_eq!(op.base.num_lines, [4, 3, 3]);
        assert!((op.base.grid_delta - 1.0e-3).abs() < 1e-15);

        // Edge lengths in metres follow the real line spacing.
        assert!((op.get_edge_length(0, [0, 0, 0], false) - 1.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(0, [1, 0, 0], false) - 2.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(0, [2, 0, 0], false) - 3.0e-3).abs() < 1e-15);
        // The last node repeats the previous delta (native GetDiscDelta).
        assert!((op.get_edge_length(0, [3, 0, 0], false) - 3.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(1, [0, 0, 0], false) - 2.0e-3).abs() < 1e-15);
        assert!((op.get_edge_length(2, [0, 0, 0], false) - 5.0e-3).abs() < 1e-15);
        assert!((op.get_edge_area(0, [0, 0, 0], false) - 2.0e-3 * 5.0e-3).abs() < 1e-18);

        // EC_C uses area/length, so vi_y / vi_x equals c_x / c_y for vacuum.
        op.base.d_t = 1.0e-15;
        op.calc_ec_operator();
        let vi = op.vi_ptr.as_ref().unwrap();
        let vi_x = *vi.get(0, 0, 0, 0) as f64;
        let vi_y = *vi.get(1, 0, 0, 0) as f64;
        assert!(vi_x > 0.0 && vi_y > 0.0);
        let ratio = vi_y / vi_x;
        assert!(
            (ratio - 4.0).abs() < 1.0e-4,
            "vi_y/vi_x should be c_x/c_y = 4.0, got {ratio}"
        );

        // A non-uniform mesh must still produce a positive CFL timestep.
        let dt = op.calc_timestep();
        assert!(dt > 0.0 && dt.is_finite(), "CFL timestep = {dt}");
    }
}

#[cfg(test)]
mod timestep_tests {
    use super::*;

    fn uniform_operator(delta_unit: f64, lines: usize) -> Operator {
        let values: Vec<String> = (0..lines).map(|index| index.to_string()).collect();
        let joined = values.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="{delta_unit}">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("uniform CSX should load");
        op
    }

    #[test]
    fn var3_matches_native_on_uniform_and_graded_meshes() {
        // The pinned native binary prints 1.92583e-12 s for the uniform 1 mm
        // mesh (4 cells) and 3.03659e-12 s for the graded mesh, both with
        // `Rennings_2`. Native computes the EC arrays first and the step
        // afterwards (`FDTD/operator.cpp:986-1010`).
        let cases = [
            ("0.001", "0,1,2,3,4", "0,1,2,3,4", "0,1,2,3,4", 1.92583e-12),
            ("0.001", "0,1,3,6", "0,2,4", "0,5,10", 3.03659e-12),
            ("0.0005", "0,1,2,3,4", "0,1,2,3,4", "0,1,2,3,4", 9.62917e-13),
        ];
        for (unit, x, y, z, native) in cases {
            let xml = format!(
                r#"<ContinuousStructure CoordSystem="0">
                    <RectilinearGrid DeltaUnit="{unit}">
                        <XLines>{x}</XLines>
                        <YLines>{y}</YLines>
                        <ZLines>{z}</ZLines>
                    </RectilinearGrid>
                    <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                    <Properties></Properties>
                </ContinuousStructure>"#
            );
            let mut op = Operator::new();
            op.load_csx(&xml).expect("mesh should load");
            op.base.m_time_step_var = 3;
            op.calc_ec_operator();
            let dt = op.calc_timestep();
            assert!(
                (dt - native).abs() / native < 1.0e-5,
                "Var3 dt {dt:e} vs native {native:e} for {x} / {y} / {z}"
            );
        }
    }

    #[test]
    fn var1_and_var3_match_the_uniform_mesh_analytic_values() {
        // Native `Operator::Operator()` starts on Rennings_2 (Var3); the
        // engine and the XML path already force it, so the bare library
        // default must match too.
        assert_eq!(Operator::new().base.m_time_step_var, 3);
        let delta = 1.0e-3;
        let mut op = uniform_operator(delta, 5);
        let c = 1.0 / (EPS0 * MU0).sqrt();

        let var1 = op.calc_timestep_var1();
        let expected_var1 = delta / (2.0 * c);
        assert!(
            (var1 - expected_var1).abs() / expected_var1 < 1.0e-6,
            "Var1 {var1:e} vs analytic {expected_var1:e}"
        );

        op.base.m_time_step_var = 3;
        let var3 = op.calc_timestep();
        let expected_var3 = delta / (3.0f64.sqrt() * c);
        assert!(
            (var3 - expected_var3).abs() / expected_var3 < 1.0e-6,
            "Var3 {var3:e} vs analytic {expected_var3:e}"
        );
        assert!(var3 > var1, "Var3 {var3:e} should exceed Var1 {var1:e}");
    }

    #[test]
    fn var3_is_larger_than_var1_on_a_nonuniform_mesh() {
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,1.5,3,6</XLines>
                <YLines>0,0.6,1.2,2.5,5</YLines>
                <ZLines>0,2,2.4,2.8,4</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties></Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("non-uniform CSX should load");
        let var1 = op.calc_timestep_var1();
        op.base.m_time_step_var = 3;
        let var3 = op.calc_timestep();
        assert!(var1 > 0.0 && var3 > 0.0);
        assert!(
            var3 >= var1,
            "Var3 {var3:e} should not be smaller than Var1 {var1:e}"
        );
    }
}

#[cfg(test)]
mod anisotropy_tests {
    use super::*;

    #[test]
    fn anisotropic_epsilon_scales_each_edge_direction() {
        let xml = r#"<ContinuousStructure CoordSystem="0">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,2,3,4</XLines>
                <YLines>0,1,2,3,4</YLines>
                <ZLines>0,1,2,3,4</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Material Name="aniso">
                    <Property Epsilon="2,4,6" Mue="1" Kappa="0" Sigma="0"/>
                    <Primitives>
                        <Box Priority="1"><P1 X="-1" Y="-1" Z="-1"/><P2 X="10" Y="10" Z="10"/></Box>
                    </Primitives>
                </Material>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml).expect("anisotropic CSX should load");
        let grid = op.material_grid.as_ref().expect("material grid");
        let ny = op.base.num_lines[1] as usize;
        let nz = op.base.num_lines[2] as usize;
        for n in 0..3 {
            let index = n * (op.base.num_lines[0] as usize) * ny * nz + 2 * ny * nz + 2 * nz + 2;
            let expected = [2.0, 4.0, 6.0][n];
            assert!(
                (grid[index].eps_r - expected).abs() < 1e-12,
                "direction {n}: eps {} vs {expected}",
                grid[index].eps_r
            );
        }

        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        let vi = op.vi_ptr.as_ref().unwrap();
        let vx = *vi.get(0, 2, 2, 2);
        let vy = *vi.get(1, 2, 2, 2);
        let vz = *vi.get(2, 2, 2, 2);
        // vi = dt / C and C is proportional to eps, so vi_x:vi_y:vi_z = 1/2:1/4:1/6.
        assert!((vx / vy - 2.0).abs() < 1e-5, "vi_x/vi_y = {}", vx / vy);
        assert!((vy / vz - 1.5).abs() < 1e-5, "vi_y/vi_z = {}", vy / vz);
    }
}

#[cfg(test)]
mod steady_state_layout_tests {
    use super::*;

    #[test]
    fn steady_state_extension_uses_the_native_probe_layout() {
        let lines: Vec<String> = (0..9).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties></Properties>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("mesh should load");
        assert!(op.build_steady_state_extension(0).is_none());

        let extension = op
            .build_steady_state_extension(10)
            .expect("periodic signals get detection");
        // Centre plus quarter and three-quarter positions on each axis,
        // all three components: 3 + 3 * 2 * 3 = 21 probes.
        assert_eq!(extension.probe_positions().len(), 21);
        assert_eq!(extension.probe_components().len(), 21);
        // Centre of a 9-line mesh is index 4.
        assert_eq!(extension.probe_positions()[0], [4, 4, 4]);
        assert_eq!(extension.probe_components()[0], 0);
        // First offset probe is x at 2 (quarter) for all components.
        assert_eq!(extension.probe_positions()[3], [2, 4, 4]);
        // The next group is x at 6 (three quarters).
        assert_eq!(extension.probe_positions()[6], [6, 4, 4]);
        // y quarter probe.
        assert_eq!(extension.probe_positions()[9], [4, 2, 4]);
        // z three-quarter probe (last group, first component).
        assert_eq!(extension.probe_positions()[18], [4, 4, 6]);
    }
}

#[cfg(test)]
mod conducting_sheet_tests {
    use super::*;

    fn mesh_with_sheet(box_bounds: &str) -> String {
        let lines: Vec<String> = (0..11).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <ConductingSheet Name="sheet" Conductivity="1e5" Thickness="35e-6">
                        <Primitives>{box_bounds}</Primitives>
                    </ConductingSheet>
                </Properties>
            </ContinuousStructure>"#
        )
    }

    #[test]
    fn two_dimensional_sheet_gets_the_ade_model() {
        let xml = mesh_with_sheet(
            r#"<Box Priority="1"><P1 X="5" Y="-1" Z="-1"/><P2 X="5" Y="12" Z="12"/></Box>"#,
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("sheet CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();

        let extension = op
            .build_conducting_sheet_extension(10.0e9, [0; 6])
            .expect("sheet model should build")
            .expect("the sheet must be active");
        assert!(extension.num_positions > 0);
        // Tangential E edges on the sheet plane are modified; the normal
        // component keeps its vacuum update coefficient.
        let vv = op.vv_ptr.as_ref().unwrap();
        let tangential = *vv.get(2, 5, 5, 5);
        let normal = *vv.get(0, 5, 5, 5);
        assert!(
            tangential < 0.999_9,
            "tangential vv should include the sheet loss, got {tangential}"
        );
        assert!(
            normal > 0.999,
            "the normal edge is not on the sheet, got {normal}"
        );
        // The extension carries finite ADE coefficients.
        for order in 0..2 {
            let coefficient = extension.v_int_ade[order][2][0];
            assert!(coefficient.is_finite() && coefficient.abs() < 1.0);
        }
    }

    #[test]
    fn volume_primitive_falls_back_to_pec_and_is_not_registered() {
        let xml = mesh_with_sheet(
            r#"<Box Priority="1"><P1 X="4" Y="0" Z="0"/><P2 X="6" Y="10" Z="10"/></Box>"#,
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("sheet CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        let extension = op
            .build_conducting_sheet_extension(10.0e9, [0; 6])
            .expect("the model must not error");
        eprintln!(
            "3d extension positions: {:?}",
            extension.as_ref().map(|e| e.num_positions)
        );
        assert!(extension.is_none(), "3-D primitives are not sheets");
        let vv = op.vv_ptr.as_ref().unwrap();
        assert_eq!(*vv.get(2, 5, 5, 5), 0.0, "fallback edges become PEC");
    }

    #[test]
    fn boundary_layer_edges_fall_back_to_pec() {
        let xml = mesh_with_sheet(
            r#"<Box Priority="1"><P1 X="5" Y="-1" Z="-1"/><P2 X="5" Y="12" Z="12"/></Box>"#,
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("sheet CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        // A 5-cell boundary layer covers x = 5 (0..5 and 5..10).
        let extension = op
            .build_conducting_sheet_extension(10.0e9, [5, 5, 0, 0, 0, 0])
            .expect("the model must not error");
        eprintln!(
            "layer extension positions: {:?}",
            extension.as_ref().map(|e| e.num_positions)
        );
        assert!(extension.is_none(), "edges inside the layer are disabled");
        let vv = op.vv_ptr.as_ref().unwrap();
        assert_eq!(*vv.get(2, 5, 5, 5), 0.0);
    }
}

#[cfg(test)]
mod yee_convention_tests {
    use super::*;

    fn lines(values: [&[f64]; 3]) -> ([Option<Vec<f64>>; 3], [u32; 3]) {
        (
            [
                Some(values[0].to_vec()),
                Some(values[1].to_vec()),
                Some(values[2].to_vec()),
            ],
            [
                values[0].len() as u32,
                values[1].len() as u32,
                values[2].len() as u32,
            ],
        )
    }

    #[test]
    fn yee_coordinates_use_nodes_transverse_and_dual_along_the_edge() {
        // Non-uniform mesh so node and dual positions are distinguishable.
        let x: &[f64] = &[0.0, 1.0, 3.0];
        let y: &[f64] = &[0.0, 3.0, 7.0];
        let z: &[f64] = &[0.0, 5.0, 11.0];
        let (disc, counts) = lines([x, y, z]);

        // E_y at (1,1,1): x and z are nodes, y is the dual midpoint.
        let coordinate = yee_coordinates(&disc, counts, 1, [1, 1, 1], false);
        assert_eq!(coordinate, [1.0, 5.0, 5.0]);

        // The dual request swaps along/transverse.
        let dual = yee_coordinates(&disc, counts, 1, [1, 1, 1], true);
        assert_eq!(dual, [2.0, 3.0, 8.0]);
    }

    #[test]
    fn capacitance_uses_the_dual_area_on_a_nonuniform_mesh() {
        let x: &[f64] = &[0.0, 1.0, 3.0];
        let y: &[f64] = &[0.0, 3.0, 7.0];
        let z: &[f64] = &[0.0, 5.0, 11.0];
        let (disc, counts) = lines([x, y, z]);
        let mut op = Operator::new();
        op.base.num_lines = counts;
        op.base.disc_lines = disc;
        op.base.grid_delta = 1.0e-3;

        // Native `Calc_ECPos` at the interior position (1,1,1):
        // C = eps * dual_y * dual_z / primary_x.
        let dual_y = 3.5e-3;
        let dual_z = 5.5e-3;
        let primary_x = 2.0e-3;
        let expected = EPS0 * dual_y * dual_z / primary_x;
        let measured = op.get_ec_c(0, 1, 1, 1);
        assert!(
            (measured - expected).abs() / expected < 1e-12,
            "C {measured:e} vs {expected:e}"
        );

        // L = mue * primary_y * primary_z / dual_x.
        let primary_y = 4.0e-3;
        let primary_z = 6.0e-3;
        let dual_x = 1.5e-3;
        let expected_l = MU0 * primary_y * primary_z / dual_x;
        let measured_l = op.get_ec_l(0, 1, 1, 1);
        assert!(
            (measured_l - expected_l).abs() / expected_l < 1e-12,
            "L {measured_l:e} vs {expected_l:e}"
        );
    }
}

#[cfg(test)]
mod lumped_rlc_tests {
    use super::*;

    fn mesh(with_element: &str) -> String {
        let lines: Vec<String> = (0..21).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>{with_element}</Properties>
            </ContinuousStructure>"#
        )
    }

    fn operator(xml: &str) -> Operator {
        let mut op = Operator::new();
        op.load_csx(xml).expect("CSX should load");
        op.base.d_t = 1.0e-12;
        op.calc_ec_operator();
        op
    }

    #[test]
    fn series_element_rewrites_the_ec_capacitance_on_every_cell() {
        let xml = mesh(
            r#"<LumpedElement Name="series" Direction="0" LEtype="1" R="50" L="1e-9" C="1e-12" Caps="0">
                   <Primitives><Box Priority="1"><P1 X="2" Y="5" Z="5"/><P2 X="12" Y="6" Z="6"/></Box></Primitives>
               </LumpedElement>"#,
        );
        let mut op = operator(&xml);
        let extension = op
            .build_lumped_rlc_extension(10.0e9)
            .expect("the model must build")
            .expect("the series element must be active");
        // 10 cells along x, 2 transverse nodes in y and z: 40 edges.\n        assert_eq!(extension.num_positions, 40);
        for index in 0..extension.num_positions {
            assert_eq!(extension.coeffs[index].direction, 0);
            // The series ADE must have a finite update multiplier.
            let vvd = extension.coeffs[index].vvd;
            assert!(vvd.is_finite() && vvd > 0.0 && vvd <= 1.0, "vvd={vvd}");
            // Conductance is zeroed for a series element.
            let pos = extension.positions[index];
            let ec_index = Operator::ec_index(0, pos[0], pos[1], pos[2], op.base.num_lines);
            assert_eq!(op.ec_g[ec_index], 0.0);
            assert!(op.ec_c[ec_index] > 0.0);
            // vv/vi were recomputed for the rewritten capacitance.
            let vv = *op.vv_ptr.as_ref().unwrap().get(0, pos[0], pos[1], pos[2]);
            assert!(vv.is_finite() && vv.abs() <= 1.0);
        }
    }

    #[test]
    fn parallel_element_with_inductance_gets_the_parallel_branch() {
        let xml = mesh(
            r#"<LumpedElement Name="parallel" Direction="1" LEtype="0" R="1e6" L="2e-9" C="1e-12" Caps="0">
                   <Primitives><Box Priority="1"><P1 X="5" Y="2" Z="5"/><P2 X="6" Y="12" Z="6"/></Box></Primitives>
               </LumpedElement>"#,
        );
        let mut op = operator(&xml);
        let extension = op
            .build_lumped_rlc_extension(10.0e9)
            .expect("the model must build")
            .expect("the parallel element must be active");
        // 10 cells along x, 2 transverse nodes in y and z: 40 edges.\n        assert_eq!(extension.num_positions, 40);
        for index in 0..extension.num_positions {
            let coeffs = &extension.coeffs[index];
            assert_eq!(coeffs.direction, 1);
            assert!(coeffs.ilv > 0.0, "parallel inductance coefficient is zero");
            assert!(coeffs.i2v > 0.0);
            assert_eq!(coeffs.vvd, 1.0);
        }
    }

    #[test]
    fn a_parallel_rc_without_inductance_is_left_to_the_plain_engine() {
        let xml = mesh(
            r#"<LumpedElement Name="rc" Direction="2" LEtype="0" R="1e3" C="1e-12">
                   <Primitives><Box Priority="1"><P1 X="5" Y="5" Z="2"/><P2 X="6" Y="6" Z="12"/></Box></Primitives>
               </LumpedElement>"#,
        );
        let mut op = operator(&xml);
        assert!(op
            .build_lumped_rlc_extension(10.0e9)
            .expect("the model must not error")
            .is_none());
    }

    #[test]
    fn invalid_direction_and_zero_length_are_skipped() {
        let invalid_direction = mesh(
            r#"<LumpedElement Name="bad" Direction="5" LEtype="1" R="50" L="1e-9" C="1e-12">
                   <Primitives><Box Priority="1"><P1 X="2" Y="5" Z="5"/><P2 X="12" Y="6" Z="6"/></Box></Primitives>
               </LumpedElement>"#,
        );
        let mut op = operator(&invalid_direction);
        assert!(op.build_lumped_rlc_extension(10.0e9).unwrap().is_none());

        let zero_length = mesh(
            r#"<LumpedElement Name="flat" Direction="0" LEtype="1" R="50" L="1e-9">
                   <Primitives><Box Priority="1"><P1 X="5" Y="5" Z="5"/><P2 X="5" Y="6" Z="6"/></Box></Primitives>
               </LumpedElement>"#,
        );
        let mut op = operator(&zero_length);
        assert!(op.build_lumped_rlc_extension(10.0e9).unwrap().is_none());
    }

    #[test]
    fn metallic_caps_short_both_element_ends() {
        let xml = mesh(
            r#"<LumpedElement Name="caps" Direction="0" LEtype="1" R="50" L="1e-9" Caps="1">
                   <Primitives><Box Priority="1"><P1 X="5" Y="5" Z="5"/><P2 X="10" Y="6" Z="6"/></Box></Primitives>
               </LumpedElement>"#,
        );
        let mut op = operator(&xml);
        op.build_lumped_rlc_extension(10.0e9)
            .unwrap()
            .expect("active");
        let vv = op.vv_ptr.as_ref().unwrap();
        // Transverse edges on both end planes are shorted.
        for plane in [5usize, 10] {
            assert_eq!(*vv.get(1, plane, 5, 5), 0.0, "y edge at x={plane}");
            assert_eq!(*vv.get(2, plane, 5, 5), 0.0, "z edge at x={plane}");
        }
        // An edge just inside the element is untouched.
        assert!(*vv.get(1, 7, 5, 5) > 0.0);
    }
}

#[cfg(test)]
mod tfsf_tests {
    use super::*;

    fn plane_wave_mesh() -> String {
        let lines: Vec<String> = (0..21).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Name="plane" Number="0" Enabled="1" Type="10" Frequency="1e10" Excite="0,1,0" PropDir="1,0,0">
                        <Primitives><Box Priority="0"><P1 X="5" Y="0" Z="0"/><P2 X="15" Y="20" Z="20"/></Box></Primitives>
                    </Excitation>
                </Properties>
            </ContinuousStructure>"#
        )
    }

    #[test]
    fn plane_wave_builds_delays_and_amplitudes_on_the_open_faces() {
        let mut op = Operator::new();
        op.load_csx(&plane_wave_mesh()).expect("CSX should load");
        op.calc_timestep();
        op.calc_ec_operator();
        let signal = crate::excitation::Excitation::gauss(1.0e10, 1.0e10);
        let extension = op
            .build_tfsf_extension(&signal)
            .expect("TFSF must build")
            .expect("a plane wave must be active");

        assert_eq!(extension.start, [5, 0, 0]);
        assert_eq!(extension.stop, [15, 20, 20]);
        assert_eq!(extension.num_lines, [11, 21, 21]);
        // Only the two x faces are open; the box touches the y/z boundaries.
        assert!(extension.active_dir[0][0] && extension.active_dir[0][1]);
        assert!(!extension.active_dir[1][0] && !extension.active_dir[1][1]);
        assert!(!extension.active_dir[2][0] && !extension.active_dir[2][1]);

        // First current sample on the lower x face: origin is one line before
        // the box, the Yee coordinate adds half a cell in the edge direction.
        let velocity =
            op.calc_numeric_phase_velocity([5, 0, 0], [15, 20, 20], [1.0, 0.0, 0.0], 1.0e10);
        let distance = 1.0e-3;
        let expected_delay = distance / velocity / op.base.d_t;
        let delay = extension.curr_delay[0][0][1][0];
        assert_eq!(delay, expected_delay.floor() as u32);
        assert!(
            (extension.curr_delay_delta[0][0][1][0]
                - (expected_delay - expected_delay.floor()) as FDTD_FLOAT)
                .abs()
                < 1.0e-5
        );
        assert!(extension.volt_amp[0][0][0][0].abs() > 0.0);
        assert!(extension.max_delay >= expected_delay as u32);
        assert!(velocity > 0.0 && velocity <= C0);
    }

    #[test]
    fn a_parallel_polarization_is_disabled() {
        let xml = plane_wave_mesh().replace("Excite=\"0,1,0\"", "Excite=\"1,0,0\"");
        let mut op = Operator::new();
        op.load_csx(&xml).expect("CSX should load");
        op.calc_timestep();
        let signal = crate::excitation::Excitation::gauss(1.0e10, 1.0e10);
        assert!(op.build_tfsf_extension(&signal).unwrap().is_none());
    }

    #[test]
    fn numeric_phase_velocity_is_below_c_for_a_cartesian_axis() {
        let mut op = Operator::new();
        op.load_csx(&plane_wave_mesh()).expect("CSX should load");
        op.calc_timestep();
        let velocity =
            op.calc_numeric_phase_velocity([5, 5, 5], [15, 15, 15], [1.0, 0.0, 0.0], 1.0e10);
        // Numerical dispersion slows the wave down, but only slightly at 15
        // cells per wavelength.
        assert!(velocity > 0.9 * C0 && velocity <= C0, "v={velocity:e}");
    }
}

#[cfg(test)]
mod debye_tests {
    use super::*;
    use crate::extensions::EngineExtension;

    fn debye_mesh(epsilon_delta: f64, relax_time: f64) -> String {
        let lines: Vec<String> = (0..5).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <DebyeMaterial Name="debye" Isotropy="1">
                        <Property Epsilon="1" Mue="1" Sigma="0" Kappa="0" EpsilonDelta="{epsilon_delta}" EpsilonRelaxTime="{relax_time}"/>
                        <Primitives><Box Priority="1"><P1 X="-1" Y="-1" Z="-1"/><P2 X="5" Y="5" Z="5"/></Box></Primitives>
                    </DebyeMaterial>
                </Properties>
            </ContinuousStructure>"#
        )
    }

    #[test]
    fn debye_pole_relaxes_to_the_static_permittivity() {
        let relax_time = 1.0e-10f64;
        let epsilon_delta = 2.0f64;
        let mut op = Operator::new();
        op.load_csx(&debye_mesh(epsilon_delta, relax_time))
            .expect("CSX should load");
        op.calc_timestep();
        op.calc_ec_operator();
        // The base permittivity of the Debye property must reach the operator.
        assert!((op.material_grid.as_ref().unwrap()[0].eps_r - 1.0).abs() < 1e-12);
        let mut extension = op
            .build_debye_extension()
            .expect("Debye must build")
            .expect("the pole is active");
        assert!(!extension.positions.is_empty());
        assert_eq!(extension.num_poles, 1);
        assert_eq!(extension.num_positions, 125);

        let index = extension
            .positions
            .iter()
            .position(|&pos| pos == [2, 2, 2])
            .expect("centre cell present");
        let dt = op.base.d_t;
        assert!(relax_time > 3.0 * dt);
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        let curr = FieldArray::new("curr", [5, 5, 5]);
        let steps = (30.0 * relax_time / dt) as usize;
        let mut history = Vec::with_capacity(steps);
        for _ in 0..steps {
            // Pre reads the field left by the previous Apply2, exactly as
            // the engine does.
            extension.do_pre_voltage_updates(&mut volt, &curr);
            // Fixed raw voltage: the main update would write this value.
            *volt.get_mut(0, 2, 2, 2) = 1.0;
            extension.do_post_voltage_updates(&mut volt, &curr);
            extension.apply2_voltages(&mut volt, &curr);
            history.push(extension.volt_pole_ade[0][0][index] as f64);
        }

        // The pole state obeys W[n+1] = a W[n] + d V[n] with
        // a = (2 tau - dt)/(2 tau + dt) and d = C_L vi/(2 tau + dt).
        let relax = extension.v_relax_ade[0][0][index] as f64;
        let drive = extension.v_drive_ade[0][0][index] as f64;
        let expected_relax = (2.0 * relax_time - dt) / (2.0 * relax_time + dt);
        assert!(
            (relax - expected_relax).abs() < 1.0e-6,
            "relax coefficient {relax:e} vs analytic {expected_relax:e}"
        );

        // Steady pole state for a constant drive. The pole is advanced
        // twice per step (pre with V^n, post with V^(n+1)), so the
        // trapezoidal recurrence is W[n+1] = a W[n] + d (V^n + V^(n+1))
        // and the steady state is 2 d / (1 - a) = delta_eps / eps_inf.
        let steady = history[steps - 1];
        let expected_steady = 2.0 * drive / (1.0 - relax);
        assert!(
            (steady - expected_steady).abs() / expected_steady < 1.0e-6,
            "steady pole state {steady:e} vs analytic {expected_steady:e}"
        );
        // `2 d / (1 - a) = delta_eps` holds in exact arithmetic. The native
        // coefficient arrays are `FDTD_FLOAT` (f32), so at Var3 the identity
        // is only accurate to a few 1e-7 relative; assert the stored values
        // are the correctly rounded analytic coefficients instead of
        // loosening the identity bound.
        let metric = op.metric();
        let pos = [2usize, 2, 2];
        let area_per_length =
            metric.edge_area(0, pos, false) / metric.edge_length(0, pos, false);
        let c_l = EPS0 * epsilon_delta * area_per_length;
        let vi = f64::from(op.vi_at(0, pos));
        let expected_drive = c_l * vi / (2.0 * relax_time + dt);
        assert_eq!(relax, f64::from(expected_relax as f32));
        assert_eq!(drive, f64::from(expected_drive as f32));

        // Exponential approach with the pole time constant.
        // Keep the fit away from the f32 noise floor of the pole state.
        let first = steps / 10;
        let last = steps / 3;
        let mut sum_t = 0.0;
        let mut sum_y = 0.0;
        let mut sum_tt = 0.0;
        let mut sum_ty = 0.0;
        let mut count = 0.0;
        for step in first..last {
            let value = (history[step] - steady).abs();
            if value <= 1.0e-30 {
                continue;
            }
            let t = step as f64 * dt;
            let y = value.ln();
            sum_t += t;
            sum_y += y;
            sum_tt += t * t;
            sum_ty += t * y;
            count += 1.0;
        }
        let denominator = count * sum_tt - sum_t * sum_t;
        let slope = (count * sum_ty - sum_t * sum_y) / denominator;
        let measured_tau = -1.0 / slope;
        // The exact discrete time constant follows from the coupled
        // field/pole system
        //   W[n+1] = a W[n] + d (V[n] + V[n+1]),
        //   V[n+1] = (V_raw - (W[n+1] - W[n])) / (1 + d),
        // whose slow eigenvalue is only ~3 % away from the continuum tau.
        let solve_coefficient = 1.0 / (1.0 + drive);
        let matrix = [
            [
                relax + drive * (1.0 - relax) * solve_coefficient,
                drive * solve_coefficient,
            ],
            [
                (1.0 - relax) * solve_coefficient,
                -drive * solve_coefficient,
            ],
        ];
        let trace = matrix[0][0] + matrix[1][1];
        let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        let discriminant = trace * trace - 4.0 * determinant;
        let eigenvalue = (trace + discriminant.sqrt()) / 2.0;
        let discrete_tau = -dt / eigenvalue.ln();
        assert!(
            (measured_tau - discrete_tau).abs() / discrete_tau < 0.01,
            "relaxation time {measured_tau:e} vs discrete {discrete_tau:e}"
        );
        assert!(
            (discrete_tau - relax_time).abs() / relax_time < 0.05,
            "discrete time constant {discrete_tau:e} vs continuum {relax_time:e}"
        );
        assert_eq!(extension.positions[index], [2, 2, 2]);
        // The implicit solve must be the inverse of one plus the summed
        // drive, and the operator must have stored the Debye base material.
        assert!((extension.v_solve_ade[0][index] as f64 - 1.0 / (1.0 + drive)).abs() < 1.0e-6);
        assert!((op.material_grid.as_ref().unwrap()[0].eps_r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_short_relaxation_time_is_rejected() {
        // tau < 3 dt makes the trapezoidal pole update unstable.
        let mut op = Operator::new();
        op.load_csx(&debye_mesh(2.0, 1.0e-15))
            .expect("CSX should load");
        op.calc_timestep();
        op.calc_ec_operator();
        assert!(op.build_debye_extension().unwrap().is_none());
    }
}

#[cfg(test)]
mod lorentz_tests {
    use super::*;
    use crate::extensions::EngineExtension;

    fn lorentz_mesh(plasma: f64, pole: f64, relax: f64) -> String {
        let lines: Vec<String> = (0..5).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <LorentzMaterial Name="lor" Isotropy="1">
                        <Property Epsilon="1" Mue="1" Sigma="0" Kappa="0"
                                  EpsilonPlasmaFrequency="{plasma}"
                                  EpsilonLorPoleFrequency="{pole}"
                                  EpsilonRelaxTime="{relax}"/>
                        <Primitives><Box Priority="1"><P1 X="-1" Y="-1" Z="-1"/><P2 X="5" Y="5" Z="5"/></Box></Primitives>
                    </LorentzMaterial>
                </Properties>
            </ContinuousStructure>"#
        )
    }

    #[test]
    fn lorentz_pole_rings_at_the_pole_frequency() {
        let plasma = 5.0e9f64;
        let pole = 10.0e9f64;
        let relax = 1.0e-9f64;
        let mut op = Operator::new();
        op.load_csx(&lorentz_mesh(plasma, pole, relax))
            .expect("CSX should load");
        op.calc_timestep();
        op.calc_ec_operator();
        let mut extension = op
            .build_lorentz_extension()
            .expect("Lorentz must build")
            .expect("the pole is active");
        assert_eq!(extension.num_orders, 1);
        assert!(extension.volt_ade_on[0]);
        let index = extension
            .positions
            .iter()
            .position(|&pos| pos == [2, 2, 2])
            .expect("centre cell present");
        assert!(extension.v_ext_ade[0][0][index] != 0.0);
        assert!(extension.v_lor_ade[0][0][index] != 0.0);

        let dt = op.base.d_t;
        let mut volt = FieldArray::new("volt", [5, 5, 5]);
        let curr = FieldArray::new("curr", [5, 5, 5]);
        let steps = (60.0 / (pole * dt)) as usize;
        let mut history = Vec::with_capacity(steps);
        for _ in 0..steps {
            extension.do_pre_voltage_updates(&mut volt, &curr);
            *volt.get_mut(0, 2, 2, 2) = 1.0;
            extension.apply2_voltages(&mut volt, &curr);
            history.push(extension.volt_ade[0][0][index] as f64);
        }

        // Count upward zero crossings of the polarization ring.
        let start = steps / 5;
        let mut crossings = Vec::new();
        for step in start + 1..steps {
            if history[step - 1] <= 0.0 && history[step] > 0.0 {
                crossings.push(step);
            }
        }
        assert!(
            crossings.len() >= 5,
            "not enough oscillations: {crossings:?}"
        );
        let period =
            (crossings.last().unwrap() - crossings[0]) as f64 / (crossings.len() - 1) as f64;
        let measured = 1.0 / (period * dt);
        let relative = (measured - pole).abs() / pole;
        assert!(
            relative < 0.05,
            "Lorentz ring {measured:e} Hz vs pole {pole:e} Hz (relative {relative:e})"
        );
    }

    #[test]
    fn lorentz_coefficients_follow_the_native_drude_lorentz_model() {
        let plasma = 5.0e9f64;
        let pole = 10.0e9f64;
        let relax = 1.0e-9f64;
        let mut op = Operator::new();
        op.load_csx(&lorentz_mesh(plasma, pole, relax))
            .expect("CSX should load");
        op.calc_timestep();
        op.calc_ec_operator();
        let extension = op
            .build_lorentz_extension()
            .expect("Lorentz must build")
            .expect("the pole is active");
        let index = extension
            .positions
            .iter()
            .position(|&pos| pos == [2, 2, 2])
            .expect("centre cell present");

        // Re-derive one component from the native formulas.
        let dt = op.base.d_t;
        let component = 0;
        let ec_index = Operator::ec_index(component, 2, 2, 2, op.base.num_lines);
        let ec_c = op.ec_c[ec_index] as f64;
        let vi = *op.vi_ptr.as_ref().unwrap().get(component, 2, 2, 2) as f64;
        let w_plasma = std::f64::consts::TAU * plasma;
        let w_lor = std::f64::consts::TAU * pole;
        let l_d = 1.0 / (w_plasma * w_plasma * ec_c);
        let r_d = l_d / relax;
        let c_l = 1.0 / (w_lor * w_lor * l_d);
        let v_int = (2.0 * l_d - dt * r_d) / (2.0 * l_d + dt * r_d);
        let v_ext = dt / (l_d + dt * r_d / 2.0) * vi;
        let v_lor = dt / c_l / vi;
        assert!((extension.v_int_ade[0][component][index] as f64 - v_int).abs() < 1e-6);
        assert!(
            (extension.v_ext_ade[0][component][index] as f64 - v_ext).abs() / v_ext.abs() < 1e-5
        );
        assert!(
            (extension.v_lor_ade[0][component][index] as f64 - v_lor).abs() / v_lor.abs() < 1e-5
        );
    }
}

#[cfg(test)]
mod coordinate_system_tests {
    use super::*;

    fn cartesian_mesh(coordinate_system: u32) -> String {
        let lines: Vec<String> = (0..5).map(|index| index.to_string()).collect();
        let joined = lines.join(",");
        format!(
            r#"<ContinuousStructure CoordSystem="{coordinate_system}">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties/>
            </ContinuousStructure>"#
        )
    }

    /// Cylindrical model with 1 mm rho/z cells and the given alpha lines.
    fn cylindrical_mesh(rho: &str, alpha: &str) -> String {
        format!(
            r#"<ContinuousStructure CoordSystem="1">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{rho}</XLines>
                    <YLines>{alpha}</YLines>
                    <ZLines>0,1,2,3,4</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties/>
            </ContinuousStructure>"#
        )
    }

    #[test]
    fn cylindrical_full_turn_is_accepted_and_padded() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "1,2,3,4,5",
            "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
        ))
        .expect("a cylindrical CSX must load");
        assert!(op.is_cylindrical());
        assert!(op.closed_alpha());
        // Native `SetupCSXGrid` pins the last line to `2*pi` and appends one
        // padding line for the dual-mesh operations at the seam.
        let alpha = op.base.disc_lines[1].as_ref().unwrap();
        assert_eq!(op.base.num_lines[1], 6);
        assert!((alpha[4] - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert!((alpha[5] - (2.0 * std::f64::consts::PI + alpha[1])).abs() < 1e-12);
        // A Cartesian model still loads.
        let mut op = Operator::new();
        op.load_csx(&cartesian_mesh(0))
            .expect("Cartesian CSX must load");
        assert!(!op.is_cylindrical());
    }

    #[test]
    fn cylindrical_metric_uses_rho_and_annulus_areas() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "1,2,3,4,5",
            "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
        ))
        .expect("cylindrical CSX must load");
        let metric = op.metric();
        // 1 mm cells, alpha step pi/2; rho line 1 is at 2 mm.
        let rho_step = 1.0e-3;
        let rho = 2.0e-3;
        let dalpha = 0.5 * std::f64::consts::PI;
        let length = metric.edge_length(1, [1, 0, 0], false);
        assert!((length - dalpha * rho).abs() < 1e-15, "length {length}");
        // A z face at rho index 1 is the annulus between 1.5 mm and 2.5 mm.
        let area = metric.edge_area(2, [1, 0, 0], false);
        let expected = 0.5 * dalpha * ((2.5e-3_f64).powi(2) - (1.5e-3_f64).powi(2));
        assert!((area - expected).abs() / expected < 1e-12, "area {area}");
        let volume = metric.cell_volume([1, 0, 0], false);
        assert!((volume - expected * rho_step).abs() / volume < 1e-12);
        // The Cartesian metric stays a plain box.
        let mut cart = Operator::new();
        cart.load_csx(&cartesian_mesh(0))
            .expect("Cartesian CSX must load");
        let metric = cart.metric();
        assert!((metric.edge_length(1, [1, 0, 0], false) - 1.0e-3).abs() < 1e-15);
        assert!((metric.edge_area(2, [1, 0, 0], false) - 1.0e-6).abs() < 1e-15);
    }

    #[test]
    fn cylindrical_open_alpha_is_not_padded() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh("1,2,3,4,5", "0,0.5,1.0,1.5"))
            .expect("open cylindrical CSX must load");
        assert!(op.is_cylindrical());
        assert!(!op.closed_alpha());
        assert_eq!(op.base.num_lines[1], 4);
    }

    /// Native `Operator::CalcTimestep_Var1` runs on the material-resolved
    /// EC values, so a dielectric mesh takes a `sqrt(eps_r)` larger step than
    /// the vacuum estimate. The native solver reports 8.12933e-12 s for this
    /// exact mesh (eps_r = 80, 16 sectors, rho = 0 axis).
    #[test]
    fn cylindrical_r0_timestep_includes_the_dielectric() {
        let xml = r#"<ContinuousStructure CoordSystem="1"><RectilinearGrid DeltaUnit="0.001" CoordSystem="1"><XLines Qty="11">0,2,4,6,8,10,12,14,16,18,20</XLines><YLines Qty="17">-3.141592653589793,-2.748893571891069,-2.356194490192345,-1.9634954084936207,-1.5707963267948966,-1.1780972450961724,-0.7853981633974483,-0.39269908169872414,0,0.39269908169872414,0.7853981633974483,1.1780972450961724,1.5707963267948966,1.9634954084936207,2.356194490192345,2.748893571891069,3.141592653589793</YLines><ZLines Qty="11">0,2,4,6,8,10,12,14,16,18,20</ZLines></RectilinearGrid><BackgroundMaterial Epsilon="1" Mue="1" Kappa="0" Sigma="0"/><Properties><Material ID="0" Name="muscle" Isotropy="1"><Primitives><Box Priority="0"><P1 X="0" Y="-3.141592653589793" Z="0"/><P2 X="20" Y="3.141592653589793" Z="20"/></Box></Primitives><Property Epsilon="80,1,1" Mue="1,1,1" Kappa="1,0,0" Sigma="0,0,0" Density="1000"/><Weight Epsilon="1,1,1" Mue="1,1,1" Kappa="1,1,1" Sigma="1,1,1" Density="1"/></Material></Properties></ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml)
            .expect("the lossy cylindrical CSX must load");
        assert!(op.r0_included);
        let dt = op.calc_timestep();
        // the material must be resolved into the cell grid: the z-face
        // capacitance is eps_r * eps0 * annulus / dz with the dual rho bounds
        // 1 mm .. 3 mm around rho line 1
        let eps = op.get_ec_c(2, 1, 4, 4);
        let expected_c = 80.0
            * crate::constants::EPS0
            * (0.5 * (std::f64::consts::FRAC_PI_8) * ((3.0e-3f64).powi(2) - (1.0e-3_f64).powi(2)))
            / 2.0e-3;
        assert!(
            (eps - expected_c).abs() / expected_c < 1e-9,
            "ec_c = {eps} vs the eps_r = 80 value {expected_c}"
        );
        assert!(
            (dt - 8.12933e-12).abs() / 8.12933e-12 < 0.01,
            "dt {dt} differs from the native 8.12933e-12"
        );
    }

    #[test]
    fn closed_cylindrical_post_processing_hides_the_seam_lines() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4",
            "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
        ))
        .expect("closed cylindrical mesh");
        let full = op.get_number_of_lines(1, true);
        let post = op.get_number_of_lines(1, false);
        assert_eq!(post, full - 2, "the seam plane is duplicated");
        // the other axes are unaffected
        for axis in [0, 2] {
            assert_eq!(
                op.get_number_of_lines(axis, false),
                op.get_number_of_lines(axis, true)
            );
        }
    }

    #[test]
    fn cylindrical_negative_rho_is_rejected() {
        let mut op = Operator::new();
        let error = op
            .load_csx(&cylindrical_mesh(
                "-1,0,1,2,3",
                "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
            ))
            .expect_err("negative rho must be rejected");
        assert!(error.contains("negative rho"), "{error}");
    }

    /// Native `Operator_Cylinder::SetupCSXGrid`: `rho = 0` is only the axis on a
    /// closed azimuth; with an open azimuth it is an ordinary PEC boundary.
    #[test]
    fn cylindrical_rho_zero_axis_is_accepted() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4",
            "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
        ))
        .expect("a closed mesh with the rho = 0 axis must load");
        assert!(op.is_cylindrical());
        assert!(op.closed_alpha());
        assert!(op.r0_included);

        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh("0,1,2,3,4", "0,0.5,1.0,1.5"))
            .expect("an open mesh starting at rho = 0 must load");
        assert!(op.is_cylindrical());
        assert!(!op.closed_alpha());
        assert!(
            !op.r0_included,
            "an open azimuth has no special axis update"
        );
    }

    /// Native `Operator_Cylinder::CalcTimestep` forces the conservative
    /// `Rennings_1` step for a mesh that includes the `rho = 0` singularity.
    #[test]
    fn cylindrical_r0_uses_the_conservative_timestep() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4",
            "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
        ))
        .expect("closed r0 mesh");
        op.base.m_time_step_var = 3;
        let dt = op.calc_timestep();
        assert!(dt > 0.0 && dt.is_finite(), "dt {dt}");
        // The conservative step must not exceed the plain 3-D CFL estimate.
        let cfl = 1.0 / (C0 * (2.0e6_f64).sqrt());
        assert!(dt <= cfl, "dt {dt} exceeds the CFL bound {cfl}");

        // The same mesh without the axis keeps the Rennings_2 step, which is
        // larger than Rennings_1.
        let mut shifted = Operator::new();
        shifted
            .load_csx(&cylindrical_mesh(
                "1,2,3,4,5",
                "0,1.5707963267948966,3.141592653589793,4.71238898038469,6.283185307179586",
            ))
            .expect("shifted mesh");
        shifted.base.m_time_step_var = 3;
        let dt_shifted = shifted.calc_timestep();
        assert!(
            dt_shifted > dt,
            "Rennings_2 {dt_shifted} vs Rennings_1 {dt}"
        );
    }

    // ----- native `Operator::FindPath` (curve / wire paths) -----

    /// Build a uniform Cartesian mesh with `lines` unit-spaced lines per axis.
    fn find_path_mesh(delta_unit: f64, lines: usize) -> Operator {
        let values: Vec<String> = (0..lines).map(|index| index.to_string()).collect();
        let joined = values.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="{delta_unit}">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties/>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("uniform CSX should load");
        op
    }

    fn path_edges(path: &GridPath) -> Vec<(usize, [u32; 3])> {
        (0..path.dir.len())
            .map(|step| {
                (
                    path.dir[step] as usize,
                    [
                        path.pos_path[0][step],
                        path.pos_path[1][step],
                        path.pos_path[2][step],
                    ],
                )
            })
            .collect()
    }

    /// A straight diagonal from node (2,2,2) to node (4,4,4) on a unit mesh.
    ///
    /// The expected staircase is derived by hand from `FDTD/operator.cpp:391-497`:
    /// the line direction is (2,2,2), so at every node the three forward
    /// neighbours have the identical foot parameter and the identical distance
    /// to the line. `dist < minDist` (operator.cpp:428/440) is a strict
    /// comparison and the axes are probed in the order x-lower, x-upper,
    /// y-lower, y-upper, z-lower, z-upper, so the tie is always resolved
    /// towards the lowest axis index, except where a step lands exactly on the
    /// line (distance 0 beats the 0.8165 of the other two):
    ///
    /// | node after the step | edge taken | foot |
    /// | --- | --- | --- |
    /// | (3,2,2) | x at (2,2,2) | 1/6 |
    /// | (3,3,2) | y at (3,2,2) | 1/3 |
    /// | (3,3,3) | z at (3,3,2) | 1/2 |
    /// | (4,3,3) | x at (3,3,3) | 2/3 |
    /// | (4,4,3) | y at (4,3,3) | 5/6 |
    /// | (4,4,4) | z at (4,4,3) | 1 |
    ///
    /// The walk ends exactly on the snapped stop node, so the closure loop adds
    /// nothing.
    #[test]
    fn find_path_diagonal_ties_resolve_towards_the_lowest_axis() {
        let op = find_path_mesh(1e-3, 11);
        let path = op.find_path([2.0, 2.0, 2.0], [4.0, 4.0, 4.0]);
        assert_eq!(
            path_edges(&path),
            vec![
                (0, [2, 2, 2]),
                (1, [3, 2, 2]),
                (2, [3, 3, 2]),
                (0, [3, 3, 3]),
                (1, [4, 3, 3]),
                (2, [4, 4, 3]),
            ]
        );
    }

    /// An axis-aligned segment on the same mesh: one edge per mesh interval, all
    /// of them in the line direction and never a detour.
    #[test]
    fn find_path_follows_an_axis_aligned_segment() {
        let op = find_path_mesh(1e-3, 11);
        let path = op.find_path([2.0, 5.0, 5.0], [5.0, 5.0, 5.0]);
        assert_eq!(
            path_edges(&path),
            vec![(0, [2, 5, 5]), (0, [3, 5, 5]), (0, [4, 5, 5])]
        );
    }

    /// A line that lies completely outside the mesh on one axis is rejected by
    /// `SnapLine2Mesh` (`operator.cpp:370-373`): the native `-1` return becomes
    /// the empty path that `FindPath` returns at `operator.cpp:397-398`.
    #[test]
    fn find_path_outside_the_domain_is_empty() {
        let op = find_path_mesh(1e-3, 5);
        for (start, stop) in [
            ([-2.0, 1.0, 1.0], [-0.5, 3.0, 1.0]),
            ([1.0, 9.0, 1.0], [3.0, 12.0, 1.0]),
        ] {
            let path = op.find_path(start, stop);
            assert!(path.dir.is_empty(), "{start:?} -> {stop:?} must not snap");
            assert!(path.pos_path.iter().all(Vec::is_empty));
        }
    }

    /// A curve segment on a cylindrical `(rho, alpha, z)` mesh.
    ///
    /// The mesh has unit rho steps, alpha lines at 0, 30, 60 and 90 degrees, and
    /// the segment runs at constant rho = 4 from 0 to 90 degrees. The native
    /// metric is the *Cartesian* distance to the chord from (4,0) to (0,4)
    /// (`Point_Line_Distance` transforms to Cartesian, CSPrimitives.cpp:36-52),
    /// so the hand-derived walk (foot in parentheses) is
    ///
    /// | node after the step | edge taken | foot |
    /// | --- | --- | --- |
    /// | (3,0,3) | rho 3->4 at alpha 0 | 0.125 |
    /// | (3,1,3) | alpha 0->30 at rho 3 | 0.3627 |
    /// | (3,2,3) | alpha 30->60 at rho 3 | 0.6373 |
    /// | (3,3,3) | alpha 60->90 at rho 3 | 0.875 |
    /// | (4,3,3) | rho 3->4 at alpha 90 | 1 |
    ///
    /// The first step goes *inwards*: (3,0) is 0.7071 away from the chord while
    /// (4,30 deg) is 1.0353 away. A naive `(rho, alpha, z)` metric would instead
    /// walk three alpha edges at rho = 4, so this case pins the Cartesian
    /// transform inside the distance metric.
    #[test]
    fn find_path_on_a_cylindrical_mesh_uses_the_cartesian_metric() {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4,5,6",
            "0,0.5235987755982988,1.0471975511965976,1.5707963267948966",
        ))
        .expect("cylindrical mesh");
        let path = op.find_path([4.0, 0.0, 3.0], [4.0, std::f64::consts::FRAC_PI_2, 3.0]);
        assert_eq!(
            path_edges(&path),
            vec![
                (0, [3, 0, 3]),
                (1, [3, 0, 3]),
                (1, [3, 1, 3]),
                (1, [3, 2, 3]),
                (0, [3, 3, 3]),
            ]
        );

        // The cylindrical metric itself: the midpoint (4, 45 deg) is
        // (2*sqrt(2), 2*sqrt(2)) in Cartesian, so its distance to the chord
        // (4,0)-(0,4) is 4 - 2*sqrt(2) = 1.1716; the naive (rho, alpha) metric
        // would give pi/4 = 0.7854 instead.
        let (foot, dist) = op.point_line_distance(
            [4.0, std::f64::consts::FRAC_PI_4, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, std::f64::consts::FRAC_PI_2, 0.0],
        );
        assert!((foot - 0.5).abs() < 1e-12, "foot {foot}");
        let cartesian_dist = 4.0 - 2.0 * 2.0f64.sqrt();
        assert!(
            (dist - cartesian_dist).abs() < 1e-12,
            "dist {dist} is not the Cartesian chord distance {cartesian_dist}"
        );
        assert!(
            (dist - std::f64::consts::FRAC_PI_4).abs() > 0.3,
            "dist {dist} must not be the naive (rho, alpha) distance"
        );
    }

    /// Exact alpha line values, so the snapped coordinates and the mesh lines
    /// are bit-identical.
    const ALPHA_90: f64 = std::f64::consts::FRAC_PI_2;
    const ALPHA_300: f64 = 5.235987755982989;
    const ALPHA_360: f64 = std::f64::consts::TAU;

    /// A quarter-turn open azimuth: 0/30/60/90 deg, rho 0..6 mm, z 0..4 mm.
    fn quarter_turn_mesh() -> Operator {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4,5,6",
            "0,0.5235987755982988,1.0471975511965976,1.5707963267948966",
        ))
        .expect("the open quarter-turn mesh must load");
        op
    }

    /// A three-quarter-turn open azimuth: 0/90/180/270 deg.
    fn three_quarter_mesh() -> Operator {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4,5,6",
            "0,1.5707963267948966,3.141592653589793,4.71238898038469",
        ))
        .expect("the open three-quarter mesh must load");
        op
    }

    /// A closed azimuth with six sectors: alpha lines at 0, 60, ..., 360 deg
    /// plus the native pad line at 420 deg, i.e. `numLines[1] = 8`.
    fn closed_six_sector_mesh() -> Operator {
        let mut op = Operator::new();
        op.load_csx(&cylindrical_mesh(
            "0,1,2,3,4,5,6",
            "0,1.0471975511965976,2.0943951023931953,3.141592653589793,4.1887902047863905,5.235987755982989,6.283185307179586",
        ))
        .expect("the closed six-sector mesh must load");
        assert!(op.closed_alpha());
        assert_eq!(op.base.num_lines[1], 8);
        op
    }

    /// A decreasing azimuth is walked in the increasing direction, because
    /// native swaps the two endpoints first.
    ///
    /// `Operator_Cylinder::FindPath` (`FDTD/operator_cylinder.cpp:331-348`):
    ///
    /// ```text
    ///     while (fabs(l_stop[1]-l_start[1])>PI) { ... }   // pi fold
    ///     if (l_start[1]>l_stop[1]) { for (n) swap(start[n], stop[n]); }
    /// ```
    ///
    /// so 90 deg -> 0 deg is handed to the base walk as 0 deg -> 90 deg and
    /// yields exactly the edges of the increasing case. On this mesh (alpha
    /// lines 0/30/60/90 deg, unit rho lines, z = 3 mm) the increasing path was
    /// derived from `operator.cpp:391-497` in
    /// `find_path_on_a_cylindrical_mesh_uses_the_cartesian_metric`: rho 3->4 at
    /// alpha 0, then three alpha edges at rho 3, then rho 3->4 at alpha 90.
    ///
    /// Dropping the swap would walk the same chord backwards (the probes only
    /// accept `foot > currFoot`), and the first accepted neighbour from
    /// (4, 90 deg) would be the rho-lower node (0, 3) at foot 0.125, distance
    /// 0.707, so that path would start with `(0, [3, 3, 3])` and end with
    /// `(0, [3, 0, 3])` — the mirrored order asserted against below.
    #[test]
    fn cylindrical_find_path_swaps_a_decreasing_azimuth() {
        let op = quarter_turn_mesh();
        let expected = vec![
            (0, [3, 0, 3]),
            (1, [3, 0, 3]),
            (1, [3, 1, 3]),
            (1, [3, 2, 3]),
            (0, [3, 3, 3]),
        ];
        assert_eq!(
            path_edges(&op.find_path([4.0, 0.0, 3.0], [4.0, ALPHA_90, 3.0])),
            expected,
            "the increasing azimuth must walk the documented path"
        );
        assert_eq!(
            path_edges(&op.find_path([4.0, ALPHA_90, 3.0], [4.0, 0.0, 3.0])),
            expected,
            "the decreasing azimuth must be swapped onto the same path"
        );
        // The fold itself is not involved here: the delta is exactly pi/2.
        assert_eq!(op.fit_to_alpha_range(ALPHA_90, false), ALPHA_90);
        assert_eq!(op.fit_to_alpha_range(0.0, false), 0.0);
    }

    /// An azimuth span above pi is folded into +/-pi, so a curve that runs
    /// 90 deg -> 300 deg is walked the short way round — backwards, out of the
    /// open mesh, where it clamps to alpha line 0.
    ///
    /// Native (`operator_cylinder.cpp:331-348`) folds first and swaps after:
    ///
    /// ```text
    ///     |300-90| > PI, stop > start  =>  l_stop[1] = 300 - 360 = -60 deg
    ///     l_start[1] (90) > l_stop[1] (-60)  =>  swap  =>
    ///         l_start = (4, -60 deg, 2), l_stop = (4, 90 deg, 2)
    /// ```
    ///
    /// `FitToAlphaRange` (`:226-250`) then runs with `fullMesh=false`, so its
    /// upper bound is `GetDiscLine(1, GetNumberOfLines(1,false)-1)` = 270 deg:
    /// -60 deg is below the minimum, `+2*pi` lands on 300 deg which is above
    /// the bound, so native returns `300 - 2*pi = -60 deg` unchanged
    /// (`:240-247`), and 90 deg is inside the range. `a_stop >= a_start`, so
    /// the base walk runs (`:353-358`); its `SnapLine2Mesh` folds -60 deg again
    /// and clamps it to line 0 (outside the mesh), while 90 deg snaps to line 1
    /// (the dual line is 0.5*(90+180) = 135 deg).
    ///
    /// Walk (`operator.cpp:391-497`), chord from (4,-60 deg) = (2, -3.4641) to
    /// (4, 90 deg) = (0, 4) in Cartesian. The snapped start node (4, 0 deg, 2)
    /// sits at foot 0.366 and the snapped stop node (4, 90 deg, 2) at foot 1, so
    /// the loop runs: the rho-lower node (3, 0 deg, 2) is accepted first (foot
    /// 0.399, distance 1.86), then replaced by the alpha-upper node (4, 90 deg,
    /// 2) — the snapped stop node itself, distance 0 and foot 1. One alpha
    /// edge, and the closure loop adds nothing because the walk ends on the
    /// snapped stop index.
    #[test]
    fn cylindrical_find_path_folds_an_azimuth_span_above_pi() {
        let op = three_quarter_mesh();
        // 5*pi/3 is not the exact mesh literal; the value a Cartesian curve
        // point would produce is -pi/3, and both fold into the same interval.
        let stop = 5.0 * std::f64::consts::PI / 3.0;
        assert_eq!(
            path_edges(&op.find_path([4.0, ALPHA_90, 2.0], [4.0, stop, 2.0])),
            vec![(1, [4, 0, 2])]
        );
        // The fold itself: 300 deg is not in [0, 270 deg], and neither is
        // -60 deg, so both are returned unchanged; 90 deg stays put.
        assert_eq!(op.fit_to_alpha_range(ALPHA_300, false), ALPHA_300);
        // `:232-239` adds 2*pi, finds the result above the 270 deg bound and
        // subtracts it again (`:235-236`), so the value is the round trip.
        let folded_back = -std::f64::consts::FRAC_PI_3 + 2.0 * std::f64::consts::PI;
        assert_eq!(
            op.fit_to_alpha_range(-std::f64::consts::FRAC_PI_3, false),
            folded_back - 2.0 * std::f64::consts::PI
        );
    }

    /// A curve that crosses the azimuth seam of a closed mesh is split at the
    /// alpha = 360 deg plane, and the second half's radial edges move onto the
    /// duplicated seam plane.
    ///
    /// Native `Operator_Cylinder::FindPath` (`operator_cylinder.cpp:320-475`),
    /// step by step for the curve (rho 2, alpha 300 deg) -> (rho 5, alpha
    /// 10 deg) at z = 2 mm:
    ///
    /// 1. fold (`:331-337`): `|10-300| = 290 > pi` and `stop > start`, so
    ///    `l_stop[1] = 370 deg`; no swap (300 < 370).
    /// 2. fit (`:350-351`, `:226-250`): `a_start = Fit(300) = 300 deg`
    ///    (`GetDiscLine(1, GetNumberOfLines(1,false)-1)` = line 5 = 300 deg is
    ///    the upper bound), `a_stop = Fit(370) = 370 - 360 = 10 deg`. Since
    ///    `a_stop < a_start`, the seam branch runs.
    /// 3. seam plane (`:381-396`): `p0 = (0, 360 deg, 0)`, `p1 = (6, 360 deg,
    ///    0)`, `p2 = (0, 360 deg, 4)` in mesh coordinates, i.e. the half plane
    ///    y = 0 in Cartesian. The segment (1, -1.7321, 2) -> (4.9239, 0.8682, 2)
    ///    crosses it at t = 1.7321/2.6003 = 0.6661 and x = 3.6131, so
    ///    `intersect = (3.6131, 360 deg, 2)` and `LinePlaneIntersection`
    ///    returns 0 (u = 0.60, v = 0.5, both inside the parallelogram).
    /// 4. `path1 = Operator::FindPath((2, 300 deg, 2), (3.6131, 360 deg, 2))`
    ///    (`:415`), i.e. the base walk: rho 2 and alpha 300 deg snap to
    ///    (2, 5), the intersection snaps to rho 4 / alpha 0, which the
    ///    cylindrical `SnapLine2Mesh` remaps to alpha 6 (`:311-314`) because
    ///    the raw stop alpha is greater. Chord (1,-1.7321) -> (3.6131, 0):
    ///
    ///    | node after the step | edge taken | foot |
    ///    | --- | --- | --- |
    ///    | (2,6,2) | alpha 300->360 at rho 2, dist 0.891 beats rho-lower 0.998 | 0.571 |
    ///    | (3,6,2) | rho 2->3 at alpha 360, dist 0.339 beats alpha 420 at 3.33 | 0.837 |
    ///    | (4,6,2) | rho 3->4 at alpha 360, dist 0.214, foot = stopFoot | 1.103 |
    ///
    /// 5. closed alpha: the lower-bound block (`:420-440`) is skipped and
    ///    `ret`/`intersect` still hold the upper-bound values, so
    ///    (`:442-449`) `path2 = Operator::FindPath((3.6131, 0, 2), (5, 10 deg,
    ///    2))`: rho 4 -> rho 5, one radial edge `(0, [4, 0, 2])`.
    /// 6. the closed-alpha remap (`:467-472`) moves that edge to
    ///    `numLines[1] - 2 = 6`.
    ///
    /// Result: alpha 300->360 at rho 2, then rho 2->3, 3->4 and 4->5 along the
    /// seam plane. Without the split the base walk would snap 10 deg onto the
    /// start node and walk a different set.
    #[test]
    fn cylindrical_find_path_splits_a_seam_crossing_segment() {
        let op = closed_six_sector_mesh();
        let stop_alpha = 10.0 * std::f64::consts::PI / 180.0;
        let expected = vec![
            (1, [2, 5, 2]),
            (0, [2, 6, 2]),
            (0, [3, 6, 2]),
            (0, [4, 6, 2]),
        ];
        assert_eq!(
            path_edges(&op.find_path([2.0, ALPHA_300, 2.0], [5.0, stop_alpha, 2.0])),
            expected
        );
        // The same curve as a Cartesian point would present it: atan2 returns
        // 300 deg as -60 deg, and the fold normalises both to the same edges.
        assert_eq!(
            path_edges(&op.find_path(
                [2.0, -std::f64::consts::FRAC_PI_3, 2.0],
                [5.0, stop_alpha, 2.0]
            )),
            expected
        );
        // `FitToAlphaRange` on the closed mesh folds into [0, 300 deg]: 370 deg
        // is reduced by one full turn (`:240-247`), 300 deg is on the bound, and
        // the full-mesh range reaches the pad line at 420 deg.
        assert_eq!(
            op.fit_to_alpha_range(370.0_f64.to_radians(), false),
            370.0_f64.to_radians() - 2.0 * std::f64::consts::PI
        );
        assert_eq!(op.fit_to_alpha_range(ALPHA_300, true), ALPHA_300);
        assert_eq!(op.fit_to_alpha_range(ALPHA_300, false), ALPHA_300);
    }

    /// End to end: an `<Excitation>` whose primitive is a cylindrical `<Curve>`
    /// that crosses the azimuth seam excites exactly the edges native
    /// `Operator_Ext_Excitation` would excite.
    ///
    /// `FDTD/extensions/operator_ext_excitation.cpp:235-288` walks every
    /// consecutive point pair with `m_Op->FindPath` — i.e. the virtual
    /// cylindrical override — and turns every path edge into a voltage
    /// excitation with `GetWeightedExcitation(n, volt_coord) * GetEdgeLength(n,
    /// pos)`. The curve points are resolved into `m_Op->m_MeshType`, so on this
    /// mesh the vertices are `(rho, alpha, z)` and the path is the one derived
    /// in `cylindrical_find_path_splits_a_seam_crossing_segment`:
    /// `[(1,[2,5,2]), (0,[2,6,2]), (0,[3,6,2]), (0,[4,6,2])]`.
    ///
    /// The alpha edge runs from 300 deg to 360 deg at rho line 2, so its native
    /// length is `dalpha_m * rho_line_2` = `(pi/3)*1e-3 * 2` (the drawing-unit
    /// rho factor of `Operator_Cylinder::GetEdgeLength`,
    /// `operator_cylinder.cpp:205-211`).
    #[test]
    fn cylindrical_curve_excitation_crosses_the_azimuth_seam() {
        let xml = r#"<ContinuousStructure CoordSystem="1">
            <RectilinearGrid DeltaUnit="0.001">
                <XLines>0,1,2,3,4,5,6</XLines>
                <YLines>0,1.0471975511965976,2.0943951023931953,3.141592653589793,4.1887902047863905,5.235987755982989,6.283185307179586</YLines>
                <ZLines>0,1,2,3,4</ZLines>
            </RectilinearGrid>
            <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
            <Properties>
                <Excitation Name="seam" Type="0" Excite="1,1,1">
                    <Primitives>
                        <Curve Priority="0">
                            <Vertex X="2" Y="5.235987755982989" Z="2"/>
                            <Vertex X="5" Y="0.17453292519943295" Z="2"/>
                        </Curve>
                    </Primitives>
                </Excitation>
            </Properties>
        </ContinuousStructure>"#;
        let mut op = Operator::new();
        op.load_csx(xml)
            .expect("the cylindrical curve CSX must load");
        assert!(op.closed_alpha());
        op.base.m_bc = [crate::BC_PEC; 6];
        let dt = op.calc_timestep();
        op.calc_ec_operator();
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        let mut signal = crate::excitation::Excitation::gauss(1.0e9, 1.0e9);
        signal
            .build(dt, 100)
            .expect("the gaussian signal must build");
        let extension = op.build_excitation_extension(&signal);
        let edges: Vec<(usize, [usize; 3])> = extension
            .voltage_excitations
            .iter()
            .map(|excitation| (excitation.component, excitation.pos))
            .collect();
        assert_eq!(
            edges,
            vec![
                (1, [2, 5, 2]),
                (0, [2, 6, 2]),
                (0, [3, 6, 2]),
                (0, [4, 6, 2]),
            ],
            "the curve must excite the seam-split path edges in path order"
        );

        // Native amplitude of the alpha edge: `weight * excitation * length`
        // with a unit weight and a unit excitation, and the length of an alpha
        // edge is `dalpha * rho` in drawing units times the grid delta.
        let alpha_edge = extension
            .voltage_excitations
            .iter()
            .find(|excitation| excitation.component == 1)
            .expect("the alpha edge must be excited");
        let expected = ((ALPHA_360 - ALPHA_300) * 1.0e-3 * 2.0) as f32;
        assert!(
            (alpha_edge.amp - expected).abs() < 1.0e-9,
            "alpha edge amplitude {} is not the native {}",
            alpha_edge.amp,
            expected
        );
        assert_eq!(alpha_edge.pos, [2, 5, 2]);

        // The curve is marked used (native `prim->SetPrimitiveUsed(true)`).
        let used = op
            .structure
            .as_ref()
            .and_then(|structure| structure.property_by_index(0))
            .and_then(|property| property.primitive(0))
            .map(|primitive| primitive.is_used());
        assert_eq!(used, Some(true));
    }

    /// The base walk is untouched on a Cartesian mesh: no azimuth fold, no
    /// swap, no seam — `find_path` dispatches on the mesh type exactly like the
    /// native virtual `Operator::FindPath` / `Operator_Cylinder::FindPath`.
    #[test]
    fn cartesian_find_path_ignores_the_cylindrical_override() {
        let op = find_path_mesh(1e-3, 7);
        assert!(!op.is_cylindrical());

        // A decreasing y walks back one line. Native emits the edge index of
        // the *lower* node in the down case (`operator.cpp:458-460`: `minPos`
        // is decremented), so the single edge is the one from y = 0 to y = 1.
        assert_eq!(
            path_edges(&op.find_path([4.0, 1.5, 3.0], [4.0, 0.0, 3.0])),
            vec![(1, [4, 0, 3])]
        );

        // A span above pi is *not* folded on a Cartesian mesh: 0 -> 5 walks five
        // y edges instead of turning round.
        assert_eq!(
            path_edges(&op.find_path([4.0, 0.0, 3.0], [4.0, 5.0, 3.0])),
            vec![
                (1, [4, 0, 3]),
                (1, [4, 1, 3]),
                (1, [4, 2, 3]),
                (1, [4, 3, 3]),
                (1, [4, 4, 3]),
            ]
        );
    }

    /// An `<Excitation>` property whose primitive is a `<Curve>` excites the
    /// edges of the snapped path.
    ///
    /// Mirrors `FDTD/extensions/operator_ext_excitation.cpp:235-288`: the curve
    /// (2,2,2) -> (4,2,2) walks two x edges, each with the native amplitude
    /// `weight * excitation * GetEdgeLength` = `1 * 1 * 1 mm`. The signal starts
    /// at zero (`build_gauss` fills from index 1), so after two time steps the
    /// only non-zero electric field is the source on those two edges.
    #[test]
    fn curve_excitation_maps_onto_the_wire_path() {
        let values: Vec<String> = (0..7).map(|index| index.to_string()).collect();
        let joined = values.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Excitation Name="wire" Type="0" Excite="1,1,1">
                        <Primitives>
                            <Curve Priority="0">
                                <Vertex X="2" Y="2" Z="2"/>
                                <Vertex X="4" Y="2" Z="2"/>
                            </Curve>
                        </Primitives>
                    </Excitation>
                </Properties>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("curve excitation CSX should load");
        op.base.m_bc = [crate::BC_PEC; 6];
        let dt = op.calc_timestep();
        op.calc_ec_operator();
        op.apply_electric_bc();
        op.apply_magnetic_bc();

        let mut signal = crate::excitation::Excitation::gauss(1.0e9, 1.0e9);
        signal
            .build(dt, 100)
            .expect("the gaussian signal must build");
        let extension = op.build_excitation_extension(&signal);
        assert_eq!(
            extension.voltage_excitations.len(),
            2,
            "the two mesh edges of the wire must be excited"
        );
        let mut edges: Vec<(usize, [usize; 3], f32)> = extension
            .voltage_excitations
            .iter()
            .map(|excitation| (excitation.component, excitation.pos, excitation.amp))
            .collect();
        edges.sort_by_key(|edge| edge.1);
        assert_eq!(edges[0].0, 0);
        assert_eq!(edges[0].1, [2, 2, 2]);
        assert_eq!(edges[1].0, 0);
        assert_eq!(edges[1].1, [3, 2, 2]);
        for edge in &edges {
            assert!(
                (edge.2 - 1.0e-3f32).abs() < 1.0e-9,
                "edge {:?} amplitude {} is not the 1 mm edge length",
                edge.1,
                edge.2
            );
        }

        // The curve primitive is marked as used (native `SetPrimitiveUsed`).
        let used = op
            .structure
            .as_ref()
            .and_then(|structure| structure.property_by_index(0))
            .and_then(|property| property.primitive(0))
            .map(|primitive| primitive.is_used());
        assert_eq!(used, Some(true));

        let amp = edges[0].2;
        let expected = amp * signal.voltage(1);
        assert!(expected != 0.0, "the second signal sample must be non-zero");
        let mut engine = crate::engine::Engine::new(Box::new(op));
        engine.add_extension(Box::new(extension));
        engine.iterate_ts(2);
        for position in [[2, 2, 2], [3, 2, 2]] {
            assert_eq!(
                *engine
                    .volt_ptr
                    .get(0, position[0], position[1], position[2]),
                expected,
                "edge {position:?} must carry the wire source"
            );
        }
        // Everything off the wire is still untouched after two steps.
        for position in [[4, 2, 2], [2, 3, 2], [2, 2, 3], [0, 0, 0], [6, 6, 6]] {
            assert_eq!(
                *engine
                    .volt_ptr
                    .get(0, position[0], position[1], position[2]),
                0.0,
                "edge {position:?} must not carry the wire source"
            );
        }
    }

    /// Only a `Curve` is walked as a path. Native `CSPrimitives::ToCurve()`
    /// (`CSXCAD/src/CSPrimitives.h:163`) tests `Type == CURVE`, and a `Wire`
    /// sets `Type == WIRE` (`CSPrimitives.cpp:30`), so a
    /// `<Wire WireRadius="…">` excitation is *not* excited along its axis: it
    /// only reaches the field through the containment test, which a thin,
    /// off-grid wire never satisfies. The native solver reports
    /// `Voltage excitations: 0` and an identically zero probe for this case.
    #[test]
    fn wire_excitation_is_not_walked_as_a_curve() {
        fn excitation_xml(primitive: &str) -> String {
            let values: Vec<String> = (0..7).map(|index| index.to_string()).collect();
            let joined = values.join(",");
            format!(
                r#"<ContinuousStructure CoordSystem="0">
                    <RectilinearGrid DeltaUnit="0.001">
                        <XLines>{joined}</XLines>
                        <YLines>{joined}</YLines>
                        <ZLines>{joined}</ZLines>
                    </RectilinearGrid>
                    <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                    <Properties>
                        <Excitation Name="ex" Type="0" Excite="1,1,1">
                            <Primitives>
                                {primitive}
                            </Primitives>
                        </Excitation>
                    </Properties>
                </ContinuousStructure>"#
            )
        }
        let vertices = r#"<Vertex X="2.1" Y="2.1" Z="2.1"/><Vertex X="5.1" Y="5.1" Z="5.1"/>"#;
        let signal = crate::excitation::Excitation::gauss(4.5e9, 4.5e9);

        // The same two vertices as a `Wire`: no edge is excited.
        let mut op = Operator::new();
        op.load_csx(&excitation_xml(&format!(
            r#"<Wire Priority="0" WireRadius="0.0001">{vertices}</Wire>"#
        )))
        .expect("the wire excitation CSX should load");
        op.base.d_t = 1.0e-12;
        let extension = op.build_excitation_extension(&signal);
        assert!(
            extension.voltage_excitations.is_empty(),
            "a wire must not be walked like a curve: {:?}",
            extension
                .voltage_excitations
                .iter()
                .map(|excitation| (excitation.component, excitation.pos))
                .collect::<Vec<_>>()
        );

        // As a `Curve` the very same segment is a path and is excited.
        let mut op = Operator::new();
        op.load_csx(&excitation_xml(&format!(
            r#"<Curve Priority="0">{vertices}</Curve>"#
        )))
        .expect("the curve excitation CSX should load");
        op.base.d_t = 1.0e-12;
        let extension = op.build_excitation_extension(&signal);
        assert!(!extension.voltage_excitations.is_empty());
    }

    /// Native `Operator::CalcPEC_Curves` (`FDTD/operator.cpp:2064-2097`): a
    /// metal `Curve` is walked with `FindPath` and every edge on the path gets
    /// `VV = VI = 0`; a `Wire` is not a curve and never marks anything.
    #[test]
    fn metal_curve_marks_the_path_as_pec() {
        let values: Vec<String> = (0..7).map(|index| index.to_string()).collect();
        let joined = values.join(",");
        let xml = format!(
            r#"<ContinuousStructure CoordSystem="0">
                <RectilinearGrid DeltaUnit="0.001">
                    <XLines>{joined}</XLines>
                    <YLines>{joined}</YLines>
                    <ZLines>{joined}</ZLines>
                </RectilinearGrid>
                <BackgroundMaterial Epsilon="1" Mue="1" Sigma="0" Kappa="0"/>
                <Properties>
                    <Metal Name="strip">
                        <Primitives>
                            <Curve Priority="0">
                                <Vertex X="2" Y="2" Z="2"/>
                                <Vertex X="4" Y="2" Z="2"/>
                            </Curve>
                        </Primitives>
                    </Metal>
                </Properties>
            </ContinuousStructure>"#
        );
        let mut op = Operator::new();
        op.load_csx(&xml).expect("metal curve CSX should load");
        op.base.m_bc = [crate::BC_PEC; 6];
        op.calc_timestep();
        op.calc_ec_operator();

        let vv = op.vv_ptr.as_ref().expect("vv allocated");
        let vi = op.vi_ptr.as_ref().expect("vi allocated");
        for pos in [[2, 2, 2], [3, 2, 2]] {
            assert_eq!(*vv.get(0, pos[0], pos[1], pos[2]), 0.0, "vv at {pos:?}");
            assert_eq!(*vi.get(0, pos[0], pos[1], pos[2]), 0.0, "vi at {pos:?}");
        }
        // The neighbouring edge on the same line is unaffected.
        assert_ne!(*vv.get(0, 1, 2, 2), 0.0);
        assert!(*vi.get(0, 1, 2, 2) > 0.0);

        let used = op
            .structure
            .as_ref()
            .and_then(|structure| structure.property_by_index(0))
            .and_then(|property| property.primitive(0))
            .map(|primitive| primitive.is_used());
        assert_eq!(used, Some(true), "the metal curve must be marked used");
    }
}
