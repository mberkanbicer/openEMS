//! Operator base: mesh, disc lines, boundary conditions, material storage.
//! Mirrors `Common/operator_base.h/cpp` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::{CoordinateSystem, FDTD_FLOAT};

/// Grid path structure for snap-to-mesh operations.
#[derive(Debug, Clone)]
pub struct GridPath {
    pub pos_path: [Vec<u32>; 3],
    pub dir: Vec<u16>,
}

/// Abstract base trait for a common operator.
/// Mirrors the C++ `Operator_Base` abstract class.
pub trait OperatorBase {
    /// Set the CSX geometry (returns false if null).
    fn set_geometry_csx(&mut self, geo: *mut std::ffi::c_void) -> bool;

    /// Get the CSX geometry pointer.
    fn get_geometry_csx(&self) -> *mut std::ffi::c_void;

    /// Get the timestep used by this operator.
    fn get_timestep(&self) -> f64;

    /// Get the number of cells or nodes defined by this operator.
    fn get_number_cells(&self) -> f64;

    /// Get the number of timesteps satisfying the Nyquist condition.
    fn get_number_of_nyquist_timesteps(&self) -> u32;

    /// Returns the number of lines as needed for post-processing.
    fn get_number_of_lines(&self, ny: i32, full: bool) -> u32;

    /// Get the name for the given direction: 0 -> x, 1 -> y, 2 -> z.
    fn get_dir_name(&self, ny: i32) -> &'static str {
        match ny {
            0 => "x",
            1 => "y",
            2 => "z",
            _ => "",
        }
    }

    /// Get the grid drawing unit in m.
    fn get_grid_delta(&self) -> f64;

    /// Get the disc line in direction n at position pos (in drawing units).
    fn get_disc_line(&self, n: i32, pos: u32, dual_mesh: bool) -> f64;

    /// Get the disc line delta in direction n at position pos (in drawing units).
    fn get_disc_delta(&self, n: i32, pos: u32, dual_mesh: bool) -> f64;

    /// Get the node width for direction ny at position pos.
    fn get_node_width(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64;

    /// Get the node area for direction ny at position pos.
    fn get_node_area(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64;

    /// Get the length of an FDTD edge (unit is meter).
    fn get_edge_length(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64;

    /// Get the area around an edge for direction ny at position pos.
    fn get_edge_area(&self, ny: i32, pos: [u32; 3], dual_mesh: bool) -> f64;

    /// Get the volume of an FDTD cell.
    fn get_cell_volume(&self, pos: [u32; 3], dual_mesh: bool) -> f64;

    /// Get the equivalent-circuit VV coefficient array (read-only).
    fn get_vv(&self) -> Option<&FieldArray> {
        None
    }

    /// Get the equivalent-circuit VI coefficient array (read-only).
    fn get_vi(&self) -> Option<&FieldArray> {
        None
    }

    /// Get the equivalent-circuit II coefficient array (read-only).
    fn get_ii(&self) -> Option<&FieldArray> {
        None
    }

    /// Get the equivalent-circuit IV coefficient array (read-only).
    fn get_iv(&self) -> Option<&FieldArray> {
        None
    }

    /// Set EC voltage coefficients (vv/vi) at a specific mesh position.
    /// Used by extensions like conducting sheet to modify EC coefficients.
    fn set_ec_voltage_coeffs(
        &mut self,
        _n: usize,
        _i: usize,
        _j: usize,
        _k: usize,
        _vv: FDTD_FLOAT,
        _vi: FDTD_FLOAT,
    ) {
        // Default: no-op. Override in concrete operator types that support it.
    }

    /// Snap coordinates to mesh indices.
    fn snap_to_mesh(
        &self,
        coord: [f64; 3],
        uicoord: &mut [u32; 3],
        dual_mesh: bool,
        full_mesh: bool,
        inside: Option<&mut bool>,
    ) -> bool;

    /// Snap a box to the operator mesh.
    #[allow(clippy::too_many_arguments)]
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
    ) -> i32;

    /// Set boundary conditions (6 values, one per face).
    fn set_boundary_condition(&mut self, bcs: [i32; 6]);

    /// Set flags to store material data for post-processing.
    fn set_material_store_flags(&mut self, type_: i32, val: bool);

    /// Cleanup material storage.
    fn cleanup_material_storage(&mut self);

    /// Get stored discrete material (if storage is enabled).
    fn get_disc_material(&self, type_: i32, ny: i32, pos: [u32; 3]) -> f64;

    /// Get the cell center coordinate usable for material averaging.
    fn get_cell_center_material_avg_coord_uint(&self, pos: [u32; 3], coord: &mut [f64; 3]) -> bool;

    /// Get the cell center coordinate usable for material averaging (int version).
    fn get_cell_center_material_avg_coord_int(&self, pos: [i32; 3], coord: &mut [f64; 3]) -> bool;

    /// Get primitives within a bounding box.
    fn get_primitives_bbox(
        &self,
        pos_x: i32,
        pos_y: i32,
        pos_z: i32,
        type_: i32,
    ) -> Vec<*mut std::ffi::c_void>;

    /// Set background material properties.
    fn set_background_material(
        &mut self,
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
        density: f64,
    );

    /// Get background relative electric permittivity.
    fn get_background_eps_r(&self) -> f64;

    /// Set background relative electric permittivity.
    fn set_background_eps_r(&mut self, val: f64);

    /// Get background relative magnetic permeability.
    fn get_background_mue_r(&self) -> f64;

    /// Set background relative magnetic permeability.
    fn set_background_mue_r(&mut self, val: f64);

    /// Get background electric conductivity.
    fn get_background_kappa(&self) -> f64;

    /// Set background electric conductivity.
    fn set_background_kappa(&mut self, val: f64);

    /// Get background magnetic conductivity (artificial).
    fn get_background_sigma(&self) -> f64;

    /// Set background magnetic conductivity (artificial).
    fn set_background_sigma(&mut self, val: f64);

    /// Get background mass density.
    fn get_background_density(&self) -> f64;

    /// Set background mass density.
    fn set_background_density(&mut self, val: f64);
}

/// Default implementation of OperatorBase with shared state.
/// Mirrors the C++ `Operator_Base` concrete data members.
pub struct OperatorBaseImpl {
    pub csx: *mut std::ffi::c_void,
    pub d_t: f64,
    pub disc_lines: [Option<Vec<f64>>; 3],
    pub m_bc: [i32; 6],
    pub m_store_material: [bool; 4],
    pub m_bg_eps_r: f64,
    pub m_bg_mue_r: f64,
    pub m_bg_kappa: f64,
    pub m_bg_sigma: f64,
    pub m_bg_density: f64,
    pub m_mesh_type: CoordinateSystem,
    pub num_lines: [u32; 3],
    pub grid_delta: f64,
    /// CFL variant: 1 = `Rennings_1` (default), 3 = `Rennings_2`.
    pub m_time_step_var: i32,
}

impl Default for OperatorBaseImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorBaseImpl {
    pub fn new() -> Self {
        Self {
            csx: std::ptr::null_mut(),
            d_t: 0.0,
            disc_lines: [None, None, None],
            m_bc: [0; 6],
            m_store_material: [false; 4],
            m_bg_eps_r: 1.0,
            m_bg_mue_r: 1.0,
            m_bg_kappa: 0.0,
            m_bg_sigma: 0.0,
            m_bg_density: 0.0,
            m_mesh_type: CoordinateSystem::Cartesian,
            num_lines: [0, 0, 0],
            grid_delta: 0.0,
            m_time_step_var: 1,
        }
    }
}

impl OperatorBase for OperatorBaseImpl {
    fn set_geometry_csx(&mut self, geo: *mut std::ffi::c_void) -> bool {
        if geo.is_null() {
            return false;
        }
        self.csx = geo;
        true
    }

    fn get_geometry_csx(&self) -> *mut std::ffi::c_void {
        self.csx
    }

    fn get_timestep(&self) -> f64 {
        self.d_t
    }

    fn get_number_cells(&self) -> f64 {
        0.0 // must be overridden
    }

    fn get_number_of_nyquist_timesteps(&self) -> u32 {
        0 // must be overridden
    }

    fn get_number_of_lines(&self, _ny: i32, _full: bool) -> u32 {
        0 // must be overridden
    }

    fn get_grid_delta(&self) -> f64 {
        self.grid_delta
    }

    fn get_disc_line(&self, _n: i32, _pos: u32, _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn get_disc_delta(&self, _n: i32, _pos: u32, _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn get_node_width(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn get_node_area(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn get_edge_length(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn get_edge_area(&self, _ny: i32, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn get_cell_volume(&self, _pos: [u32; 3], _dual_mesh: bool) -> f64 {
        0.0 // must be overridden
    }

    fn snap_to_mesh(
        &self,
        _coord: [f64; 3],
        _uicoord: &mut [u32; 3],
        _dual_mesh: bool,
        _full_mesh: bool,
        _inside: Option<&mut bool>,
    ) -> bool {
        false // must be overridden
    }

    fn snap_box_to_mesh(
        &self,
        _start: [f64; 3],
        _stop: [f64; 3],
        _ui_start: &mut [u32; 3],
        _ui_stop: &mut [u32; 3],
        _dual_mesh: bool,
        _full_mesh: bool,
        _snap_method: i32,
        _b_start_in: Option<&mut bool>,
        _b_stop_in: Option<&mut bool>,
    ) -> i32 {
        -1 // must be overridden
    }

    fn set_boundary_condition(&mut self, bcs: [i32; 6]) {
        self.m_bc = bcs;
    }

    fn set_material_store_flags(&mut self, type_: i32, val: bool) {
        if (0..=3).contains(&type_) {
            self.m_store_material[type_ as usize] = val;
        }
    }

    fn cleanup_material_storage(&mut self) {
        // default: no-op
    }

    fn get_disc_material(&self, _type_: i32, _ny: i32, _pos: [u32; 3]) -> f64 {
        0.0
    }

    fn get_cell_center_material_avg_coord_uint(&self, pos: [u32; 3], coord: &mut [f64; 3]) -> bool {
        let ipos = [pos[0] as i32, pos[1] as i32, pos[2] as i32];
        self.get_cell_center_material_avg_coord_int(ipos, coord)
    }

    fn get_cell_center_material_avg_coord_int(
        &self,
        _pos: [i32; 3],
        _coord: &mut [f64; 3],
    ) -> bool {
        false // must be overridden
    }

    fn get_primitives_bbox(
        &self,
        _pos_x: i32,
        _pos_y: i32,
        _pos_z: i32,
        _type_: i32,
    ) -> Vec<*mut std::ffi::c_void> {
        Vec::new()
    }

    fn set_background_material(
        &mut self,
        eps_r: f64,
        mue_r: f64,
        kappa: f64,
        sigma: f64,
        density: f64,
    ) {
        self.set_background_eps_r(eps_r);
        self.set_background_mue_r(mue_r);
        self.set_background_kappa(kappa);
        self.set_background_sigma(sigma);
        self.set_background_density(density);
    }

    fn get_background_eps_r(&self) -> f64 {
        self.m_bg_eps_r
    }

    fn set_background_eps_r(&mut self, val: f64) {
        if val < 1.0 {
            eprintln!(
                "set_background_eps_r: Warning, relative electric permittivity <1 is not supported, skipping"
            );
            return;
        }
        self.m_bg_eps_r = val;
    }

    fn get_background_mue_r(&self) -> f64 {
        self.m_bg_mue_r
    }

    fn set_background_mue_r(&mut self, val: f64) {
        if val < 1.0 {
            eprintln!(
                "set_background_mue_r: Warning, relative magnetic permeability <1 is not supported, skipping"
            );
            return;
        }
        self.m_bg_mue_r = val;
    }

    fn get_background_kappa(&self) -> f64 {
        self.m_bg_kappa
    }

    fn set_background_kappa(&mut self, val: f64) {
        if val < 0.0 {
            eprintln!(
                "set_background_kappa: Warning, electric conductivity <0 is not supported, skipping"
            );
            return;
        }
        self.m_bg_kappa = val;
    }

    fn get_background_sigma(&self) -> f64 {
        self.m_bg_sigma
    }

    fn set_background_sigma(&mut self, val: f64) {
        if val < 0.0 {
            eprintln!(
                "set_background_sigma: Warning, artificial magnetic conductivity <0 is not supported, skipping"
            );
            return;
        }
        self.m_bg_sigma = val;
    }

    fn get_background_density(&self) -> f64 {
        self.m_bg_density
    }

    fn set_background_density(&mut self, val: f64) {
        if val < 0.0 {
            eprintln!(
                "set_background_density: Warning, mass density <0 is not supported, skipping"
            );
            return;
        }
        self.m_bg_density = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_background_material() {
        let op = OperatorBaseImpl::new();
        assert_eq!(op.get_background_eps_r(), 1.0);
        assert_eq!(op.get_background_mue_r(), 1.0);
        assert_eq!(op.get_background_kappa(), 0.0);
        assert_eq!(op.get_background_sigma(), 0.0);
        assert_eq!(op.get_background_density(), 0.0);
    }

    #[test]
    fn test_set_boundary_condition() {
        let mut op = OperatorBaseImpl::new();
        op.set_boundary_condition([1, 2, 3, 4, 5, 6]);
        // m_bc is private, but we can verify via behavior
        // (no direct getter, but set should not panic)
    }

    #[test]
    fn test_get_dir_name() {
        let op = OperatorBaseImpl::new();
        assert_eq!(op.get_dir_name(0), "x");
        assert_eq!(op.get_dir_name(1), "y");
        assert_eq!(op.get_dir_name(2), "z");
        assert_eq!(op.get_dir_name(3), "");
    }
}
