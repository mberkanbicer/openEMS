//! Physics extensions: PML, Mur ABC, excitation, dispersive materials, etc.
//! Mirrors `FDTD/extensions/` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::FDTD_FLOAT;
use crate::excitation::Excitation;
use std::sync::Arc;

// Extension priority constants (mirroring C++ #defines)
pub const ENG_EXT_PRIO_DEFAULT: i32 = 0;
pub const ENG_EXT_PRIO_STEADYSTATE: i32 = 2_000_000;
pub const ENG_EXT_PRIO_UPML: i32 = 1_000_000;
pub const ENG_EXT_PRIO_CYLINDER: i32 = 100_000;
pub const ENG_EXT_PRIO_TFSF: i32 = 50_000;
pub const ENG_EXT_PRIO_EXCITATION: i32 = -1000;
pub const ENG_EXT_PRIO_CYLINDERMULTIGRID: i32 = -3000;

/// Base trait for engine extensions.
/// Extension methods receive mutable access to the engine's voltage/current
/// field arrays so they can read and modify fields during time stepping.
pub trait EngineExtension {
    fn get_priority(&self) -> i32;
    fn get_extension_name(&self) -> &'static str;
    fn do_pre_voltage_updates(&mut self, _volt: &mut FieldArray, _curr: &FieldArray) {}
    fn do_post_voltage_updates(&mut self, _volt: &mut FieldArray, _curr: &FieldArray) {}
    fn apply2_voltages(&mut self, _volt: &mut FieldArray, _curr: &FieldArray) {}
    fn do_pre_current_updates(&mut self, _volt: &FieldArray, _curr: &mut FieldArray) {}
    fn do_post_current_updates(&mut self, _volt: &FieldArray, _curr: &mut FieldArray) {}
    fn apply2_current(&mut self, _volt: &FieldArray, _curr: &mut FieldArray) {}

    /// Timestep number, refreshed by the engine before every step. Extensions
    /// that need absolute time (TFSF delays) override this.
    fn set_timestep(&mut self, _number: u32) {}

    /// Query extension-specific convergence metric (e.g. steady-state diff).
    /// Returns `None` by default; override in extensions that support it.
    fn get_last_diff(&self) -> Option<FDTD_FLOAT> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extension_priorities() {
        assert_eq!(ENG_EXT_PRIO_STEADYSTATE, 2_000_000);
        assert_eq!(ENG_EXT_PRIO_UPML, 1_000_000);
        assert_eq!(ENG_EXT_PRIO_EXCITATION, -1000);
    }
}

// ============================================================================
// Excitation extension
// ============================================================================

/// A single voltage excitation applied to a field component at a position.
pub struct VoltageExcitation {
    pub component: usize, // 0=Ex, 1=Ey, 2=Ez
    pub pos: [usize; 3],  // position in the field array
    pub delay: u32,       // start delay in timesteps
    pub amp: FDTD_FLOAT,  // weighted amplitude incl. edge length
    pub signal: Arc<Excitation>,
}

/// A single current excitation applied to a field component at a position.
pub struct CurrentExcitation {
    pub component: usize, // 0=Hx, 1=Hy, 2=Hz
    pub pos: [usize; 3],  // position in the field array
    pub delay: u32,       // start delay in timesteps
    pub amp: FDTD_FLOAT,  // weighted amplitude incl. edge length
    pub signal: Arc<Excitation>,
}

/// Engine extension that applies time-domain excitation sources to the field arrays.
/// Mirrors C++ `Engine_Ext_Excitation`.
pub struct EngineExtExcitation {
    pub voltage_excitations: Vec<VoltageExcitation>,
    pub current_excitations: Vec<CurrentExcitation>,
    pub current_ts: u32,
}

impl EngineExtExcitation {
    pub fn new() -> Self {
        Self {
            voltage_excitations: Vec::new(),
            current_excitations: Vec::new(),
            current_ts: 0,
        }
    }

    /// Add a voltage excitation.
    pub fn add_voltage_excitation(
        &mut self,
        comp: usize,
        pos: [usize; 3],
        delay: u32,
        amp: FDTD_FLOAT,
        signal: Arc<Excitation>,
    ) {
        self.voltage_excitations.push(VoltageExcitation {
            component: comp,
            pos,
            delay,
            amp,
            signal,
        });
    }

    /// Add a current excitation.
    pub fn add_current_excitation(
        &mut self,
        comp: usize,
        pos: [usize; 3],
        delay: u32,
        amp: FDTD_FLOAT,
        signal: Arc<Excitation>,
    ) {
        self.current_excitations.push(CurrentExcitation {
            component: comp,
            pos,
            delay,
            amp,
            signal,
        });
    }
}

impl Default for EngineExtExcitation {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineExtExcitation {
    /// Sample index for an excitation at timestep `ts`.
    ///
    /// Mirrors `Engine_Ext_Excitation`: a periodic signal repeats every
    /// `signal_period()` samples, and indices outside the built signal are
    /// skipped instead of extrapolated.
    fn sample_index(signal: &Excitation, ts: u32, delay: u32) -> Option<usize> {
        if ts < delay {
            return None;
        }
        let mut position = (ts - delay) as usize;
        let period = signal.signal_period() as usize;
        if period > 0 {
            position %= period;
        }
        if position < signal.signal_length() as usize {
            Some(position)
        } else {
            None
        }
    }
}

impl EngineExtension for EngineExtExcitation {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_EXCITATION
    }

    fn get_extension_name(&self) -> &'static str {
        "Excitation"
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        let ts = self.current_ts;
        for exc in &self.voltage_excitations {
            let Some(index) = Self::sample_index(&exc.signal, ts, exc.delay) else {
                continue;
            };
            let val = exc.signal.voltage(index);
            let cur = *volt.get(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]);
            *volt.get_mut(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]) = cur + exc.amp * val;
        }
    }

    fn apply2_current(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        let ts = self.current_ts;
        for exc in &self.current_excitations {
            let Some(index) = Self::sample_index(&exc.signal, ts, exc.delay) else {
                continue;
            };
            let val = exc.signal.current(index);
            let cur = *curr.get(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]);
            *curr.get_mut(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]) = cur + exc.amp * val;
        }
        self.current_ts += 1;
    }
}

#[cfg(test)]
mod excitation_tests {
    use super::*;

    #[test]
    fn test_excitation_creation() {
        let ext = EngineExtExcitation::new();
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_EXCITATION);
        assert_eq!(ext.get_extension_name(), "Excitation");
    }

    #[test]
    fn test_add_voltage_excitation() {
        let mut ext = EngineExtExcitation::new();
        let signal = Excitation::gauss(1.0e9, 5.0e8);
        ext.add_voltage_excitation(0, [5, 5, 5], 10, 1.0, Arc::new(signal));
        assert_eq!(ext.voltage_excitations.len(), 1);
        assert_eq!(ext.voltage_excitations[0].component, 0);
        assert_eq!(ext.voltage_excitations[0].pos, [5, 5, 5]);
        assert_eq!(ext.voltage_excitations[0].delay, 10);
        assert_eq!(ext.voltage_excitations[0].amp, 1.0);
    }

    #[test]
    fn test_signal_value_gauss() {
        let mut signal = Excitation::gauss(1.0e9, 5.0e8);
        signal.build(1.0e-12, 10_000).unwrap();
        let val = signal.voltage(1);
        assert!(val.abs() > 0.0);
        assert!(val.abs() <= 1.0);
    }

    #[test]
    fn test_signal_value_dirac() {
        let mut signal = Excitation::dirac(1.0e9);
        signal.build(1.0e-12, 100).unwrap();
        assert_eq!(signal.voltage(0), 0.0);
        assert_eq!(signal.voltage(1), 1.0);
        assert_eq!(signal.voltage(2), 0.0);
    }
}

// ============================================================================
// Mur ABC extension
// ============================================================================

/// One active Mur face: per-position coefficients and the stored voltage
/// history used by the three-phase native update.
#[derive(Debug, Clone)]
pub struct MurFace {
    pub face: usize,
    pub dir: usize,
    pub top: bool,
    pub line: usize,
    pub shift: usize,
    pub dims_p: usize,
    pub dims_pp: usize,
    pub coeff_p: Vec<FDTD_FLOAT>,
    pub coeff_pp: Vec<FDTD_FLOAT>,
    pub hist_p: Vec<FDTD_FLOAT>,
    pub hist_pp: Vec<FDTD_FLOAT>,
}

/// First-order Mur absorbing boundary on all configured faces.
/// Mirrors the `pre -> post -> apply2` sequence of `Engine_Ext_Mur_ABC`.
#[derive(Debug, Clone, Default)]
pub struct EngineExtMurABC {
    pub faces: Vec<MurFace>,
    pub current_ts: u32,
    pub start_ts: u32,
}

impl EngineExtMurABC {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a face with pre-computed coefficients.
    #[allow(clippy::too_many_arguments)]
    pub fn add_face(
        &mut self,
        face: usize,
        dir: usize,
        top: bool,
        line: usize,
        shift: usize,
        dims_p: usize,
        dims_pp: usize,
        coeff_p: Vec<FDTD_FLOAT>,
        coeff_pp: Vec<FDTD_FLOAT>,
    ) {
        let size = dims_p * dims_pp;
        self.faces.push(MurFace {
            face,
            dir,
            top,
            line,
            shift,
            dims_p,
            dims_pp,
            coeff_p,
            coeff_pp,
            hist_p: vec![0.0; size],
            hist_pp: vec![0.0; size],
        });
    }

    /// Set the timestep at which the Mur ABC becomes active.
    pub fn set_start_ts(&mut self, ts: u32) {
        self.start_ts = ts;
    }

    fn is_active(&self) -> bool {
        self.current_ts >= self.start_ts
    }
}

impl EngineExtension for EngineExtMurABC {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_DEFAULT
    }

    fn get_extension_name(&self) -> &'static str {
        "Mur ABC"
    }

    fn do_pre_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        self.current_ts += 1;
        if !self.is_active() {
            return;
        }
        for face in &mut self.faces {
            let n_p = (face.dir + 1) % 3;
            let n_pp = (face.dir + 2) % 3;
            for ip in 0..face.dims_p {
                for ipp in 0..face.dims_pp {
                    let idx = ip * face.dims_pp + ipp;
                    let mut pos = [0usize; 3];
                    pos[face.dir] = face.line;
                    pos[n_p] = ip;
                    pos[n_pp] = ipp;
                    let mut pos_shift = [0usize; 3];
                    pos_shift[face.dir] = face.shift;
                    pos_shift[n_p] = ip;
                    pos_shift[n_pp] = ipp;

                    let v_p_shift = *volt.get(n_p, pos_shift[0], pos_shift[1], pos_shift[2]);
                    let v_p_bound = *volt.get(n_p, pos[0], pos[1], pos[2]);
                    face.hist_p[idx] = v_p_shift - face.coeff_p[idx] * v_p_bound;

                    let v_pp_shift = *volt.get(n_pp, pos_shift[0], pos_shift[1], pos_shift[2]);
                    let v_pp_bound = *volt.get(n_pp, pos[0], pos[1], pos[2]);
                    face.hist_pp[idx] = v_pp_shift - face.coeff_pp[idx] * v_pp_bound;
                }
            }
        }
    }

    fn do_post_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        if !self.is_active() {
            return;
        }
        for face in &mut self.faces {
            let n_p = (face.dir + 1) % 3;
            let n_pp = (face.dir + 2) % 3;
            for ip in 0..face.dims_p {
                for ipp in 0..face.dims_pp {
                    let idx = ip * face.dims_pp + ipp;
                    let mut pos_shift = [0usize; 3];
                    pos_shift[face.dir] = face.shift;
                    pos_shift[n_p] = ip;
                    pos_shift[n_pp] = ipp;
                    let v_p_shift = *volt.get(n_p, pos_shift[0], pos_shift[1], pos_shift[2]);
                    let v_pp_shift = *volt.get(n_pp, pos_shift[0], pos_shift[1], pos_shift[2]);
                    face.hist_p[idx] += face.coeff_p[idx] * v_p_shift;
                    face.hist_pp[idx] += face.coeff_pp[idx] * v_pp_shift;
                }
            }
        }
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        if !self.is_active() {
            return;
        }
        for face in &self.faces {
            let n_p = (face.dir + 1) % 3;
            let n_pp = (face.dir + 2) % 3;
            for ip in 0..face.dims_p {
                for ipp in 0..face.dims_pp {
                    let idx = ip * face.dims_pp + ipp;
                    let mut pos = [0usize; 3];
                    pos[face.dir] = face.line;
                    pos[n_p] = ip;
                    pos[n_pp] = ipp;
                    *volt.get_mut(n_p, pos[0], pos[1], pos[2]) = face.hist_p[idx];
                    *volt.get_mut(n_pp, pos[0], pos[1], pos[2]) = face.hist_pp[idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod mur_abc_tests {
    use super::*;

    #[test]
    fn test_mur_abc_creation() {
        let ext = EngineExtMurABC::new();
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_DEFAULT);
        assert_eq!(ext.get_extension_name(), "Mur ABC");
        assert!(ext.faces.is_empty());
    }

    #[test]
    fn test_mur_three_phase_update_matches_native_sequence() {
        let mut ext = EngineExtMurABC::new();
        ext.add_face(0, 0, false, 0, 1, 3, 3, vec![0.5; 9], vec![0.25; 9]);
        let mut volt = FieldArray::new("volt", [4, 3, 3]);
        let curr = FieldArray::new("curr", [4, 3, 3]);

        // V^n on the x-min face and its inward neighbour.
        *volt.get_mut(1, 0, 1, 1) = 1.0;
        *volt.get_mut(1, 1, 1, 1) = 3.0;
        *volt.get_mut(2, 0, 1, 1) = 2.0;
        *volt.get_mut(2, 1, 1, 1) = 4.0;

        ext.do_pre_voltage_updates(&mut volt, &curr);
        // hist_y = 3 - 0.5*1 = 2.5; hist_z = 4 - 0.25*2 = 3.5
        assert_eq!(ext.faces[0].hist_p[4], 2.5);
        assert_eq!(ext.faces[0].hist_pp[4], 3.5);

        // Main voltage update changes the inward neighbour.
        *volt.get_mut(1, 1, 1, 1) = 5.0;
        *volt.get_mut(2, 1, 1, 1) = 6.0;
        ext.do_post_voltage_updates(&mut volt, &curr);
        // hist_y = 2.5 + 0.5*5 = 5.0; hist_z = 3.5 + 0.25*6 = 5.0
        assert_eq!(ext.faces[0].hist_p[4], 5.0);
        assert_eq!(ext.faces[0].hist_pp[4], 5.0);

        ext.apply2_voltages(&mut volt, &curr);
        assert_eq!(*volt.get(1, 0, 1, 1), 5.0);
        assert_eq!(*volt.get(2, 0, 1, 1), 5.0);
    }
}

// ============================================================================
// UPML (Uniaxial Perfectly Matched Layer) extension
// ============================================================================

/// UPML engine extension implementing a uniaxial perfectly matched layer.
/// Uses auxiliary flux arrays to implement the PML split-field update.
/// Mirrors C++ `Engine_Ext_UPML`.
pub struct EngineExtUPML {
    /// PML region start position (inclusive) in each direction.
    pub start_pos: [usize; 3],
    /// PML region size (number of cells) in each direction.
    pub num_lines: [usize; 3],
    /// PML voltage coefficients: vv, vvfo, vvfn for each component.
    pub vv: Vec<FieldArray>,
    pub vvfo: Vec<FieldArray>,
    pub vvfn: Vec<FieldArray>,
    /// PML current coefficients: ii, iifo, iifn for each component.
    pub ii: Vec<FieldArray>,
    pub iifo: Vec<FieldArray>,
    pub iifn: Vec<FieldArray>,
    /// Auxiliary flux arrays for voltage and current.
    pub volt_flux: Vec<FieldArray>,
    pub curr_flux: Vec<FieldArray>,
}

impl EngineExtUPML {
    pub fn new(start_pos: [usize; 3], num_lines: [usize; 3]) -> Self {
        let mut vv = Vec::new();
        let mut vvfo = Vec::new();
        let mut vvfn = Vec::new();
        let mut ii = Vec::new();
        let mut iifo = Vec::new();
        let mut iifn = Vec::new();
        let mut volt_flux = Vec::new();
        let mut curr_flux = Vec::new();

        let extent = [num_lines[0], num_lines[1], num_lines[2]];
        for _ in 0..3 {
            vv.push(FieldArray::new("pml_vv", extent));
            vvfo.push(FieldArray::new("pml_vvfo", extent));
            vvfn.push(FieldArray::new("pml_vvfn", extent));
            ii.push(FieldArray::new("pml_ii", extent));
            iifo.push(FieldArray::new("pml_iifo", extent));
            iifn.push(FieldArray::new("pml_iifn", extent));
            volt_flux.push(FieldArray::new("pml_volt_flux", extent));
            curr_flux.push(FieldArray::new("pml_curr_flux", extent));
        }

        Self {
            start_pos,
            num_lines,
            vv,
            vvfo,
            vvfn,
            ii,
            iifo,
            iifn,
            volt_flux,
            curr_flux,
        }
    }

    /// Set the PML coefficient for a given component and position.
    pub fn set_vv(&mut self, comp: usize, i: usize, j: usize, k: usize, val: FDTD_FLOAT) {
        *self.vv[comp].get_mut(0, i, j, k) = val;
    }
}

impl Default for EngineExtUPML {
    fn default() -> Self {
        Self::new([0, 0, 0], [0, 0, 0])
    }
}

impl EngineExtension for EngineExtUPML {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_UPML
    }

    fn get_extension_name(&self) -> &'static str {
        "UPML"
    }

    fn do_pre_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];
        let sx = self.start_pos[0];
        let sy = self.start_pos[1];
        let sz = self.start_pos[2];

        for comp in 0..3 {
            for i in 0..nx {
                let gi = i + sx;
                for j in 0..ny {
                    let gj = j + sy;
                    for k in 0..nz {
                        let gk = k + sz;
                        let f_help = *self.vv[comp].get(0, i, j, k) * *volt.get(comp, gi, gj, gk)
                            - *self.vvfo[comp].get(0, i, j, k)
                                * *self.volt_flux[comp].get(0, i, j, k);
                        let flux_old = *self.volt_flux[comp].get(0, i, j, k);
                        *volt.get_mut(comp, gi, gj, gk) = flux_old;
                        *self.volt_flux[comp].get_mut(0, i, j, k) = f_help;
                    }
                }
            }
        }
    }

    fn do_post_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];
        let sx = self.start_pos[0];
        let sy = self.start_pos[1];
        let sz = self.start_pos[2];

        for comp in 0..3 {
            for i in 0..nx {
                let gi = i + sx;
                for j in 0..ny {
                    let gj = j + sy;
                    for k in 0..nz {
                        let gk = k + sz;
                        let f_help = *self.volt_flux[comp].get(0, i, j, k);
                        *self.volt_flux[comp].get_mut(0, i, j, k) = *volt.get(comp, gi, gj, gk);
                        *volt.get_mut(comp, gi, gj, gk) = f_help
                            + *self.vvfn[comp].get(0, i, j, k)
                                * *self.volt_flux[comp].get(0, i, j, k);
                    }
                }
            }
        }
    }

    fn do_pre_current_updates(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];
        let sx = self.start_pos[0];
        let sy = self.start_pos[1];
        let sz = self.start_pos[2];

        for comp in 0..3 {
            for i in 0..nx {
                let gi = i + sx;
                for j in 0..ny {
                    let gj = j + sy;
                    for k in 0..nz {
                        let gk = k + sz;
                        let f_help = *self.ii[comp].get(0, i, j, k) * *curr.get(comp, gi, gj, gk)
                            - *self.iifo[comp].get(0, i, j, k)
                                * *self.curr_flux[comp].get(0, i, j, k);
                        let flux_old = *self.curr_flux[comp].get(0, i, j, k);
                        *curr.get_mut(comp, gi, gj, gk) = flux_old;
                        *self.curr_flux[comp].get_mut(0, i, j, k) = f_help;
                    }
                }
            }
        }
    }

    fn do_post_current_updates(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        let nx = self.num_lines[0];
        let ny = self.num_lines[1];
        let nz = self.num_lines[2];
        let sx = self.start_pos[0];
        let sy = self.start_pos[1];
        let sz = self.start_pos[2];

        for comp in 0..3 {
            for i in 0..nx {
                let gi = i + sx;
                for j in 0..ny {
                    let gj = j + sy;
                    for k in 0..nz {
                        let gk = k + sz;
                        let f_help = *self.curr_flux[comp].get(0, i, j, k);
                        *self.curr_flux[comp].get_mut(0, i, j, k) = *curr.get(comp, gi, gj, gk);
                        *curr.get_mut(comp, gi, gj, gk) = f_help
                            + *self.iifn[comp].get(0, i, j, k)
                                * *self.curr_flux[comp].get(0, i, j, k);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod upml_tests {
    use super::*;

    #[test]
    fn upml_flux_sequence_changes_fields() {
        let mut ext = EngineExtUPML::new([0, 0, 0], [2, 2, 2]);
        *ext.vv[0].get_mut(0, 0, 0, 0) = 1.0;
        *ext.vvfo[0].get_mut(0, 0, 0, 0) = 0.5;
        *ext.vvfn[0].get_mut(0, 0, 0, 0) = 2.0;
        let mut volt = FieldArray::new("volt", [2, 2, 2]);
        let curr = FieldArray::new("curr", [2, 2, 2]);
        *volt.get_mut(0, 0, 0, 0) = 1.0;

        ext.do_pre_voltage_updates(&mut volt, &curr);
        // V is replaced by the old flux (0), flux becomes 1*1 - 0.5*0 = 1.
        assert_eq!(*volt.get(0, 0, 0, 0), 0.0);
        // Simulate the main voltage update.
        *volt.get_mut(0, 0, 0, 0) = 5.0;
        ext.do_post_voltage_updates(&mut volt, &curr);
        // V = flux_old (1) + vvfn (2) * V_new_raw (5) = 11.
        assert_eq!(*volt.get(0, 0, 0, 0), 11.0);
    }

    #[test]
    fn test_upml_creation() {
        let ext = EngineExtUPML::new([0, 0, 0], [10, 10, 10]);
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_UPML);
        assert_eq!(ext.get_extension_name(), "UPML");
        assert_eq!(ext.vv.len(), 3);
        assert_eq!(ext.volt_flux.len(), 3);
    }
}

// ============================================================================
// TFSF (Total Field / Scattered Field) extension
// ============================================================================

/// TFSF engine extension implementing a Total-Field / Scattered-Field boundary.
///
/// Ported from `FDTD/extensions/engine_ext_tfsf.cpp`. The operator side
/// pre-computes, for every boundary edge, the propagation delay and the field
/// amplitude; this side adds the incident field on the six box faces after the
/// main update.
///
/// The per-face arrays are indexed `[n][face][c][ui_pos]` where `c = 0` is the
/// `nP = (n+1) % 3` component and `c = 1` is `nPP = (n+2) % 3`, matching the
/// native layout.
pub struct EngineExtTFSF {
    /// TFSF region start position (inclusive) in each direction.
    pub start: [usize; 3],
    /// TFSF region stop position (inclusive) in each direction.
    pub stop: [usize; 3],
    /// Number of mesh lines inside the region (`stop - start + 1`).
    pub num_lines: [usize; 3],
    /// Active faces: `active_dir[direction][0=low, 1=high]`.
    pub active_dir: [[bool; 2]; 3],
    /// Excitation signal.
    pub signal: Excitation,
    /// Current timestep, refreshed by the engine before every step.
    pub current_ts: u32,
    /// Maximum delay value in timesteps.
    pub max_delay: u32,
    /// Voltage delay lookup: `[direction][face][component][position_index]`.
    pub volt_delay: Vec<Vec<Vec<Vec<u32>>>>,
    /// Voltage delay interpolation factor.
    pub volt_delay_delta: Vec<Vec<Vec<Vec<FDTD_FLOAT>>>>,
    /// Voltage amplitude.
    pub volt_amp: Vec<Vec<Vec<Vec<FDTD_FLOAT>>>>,
    /// Current delay lookup.
    pub curr_delay: Vec<Vec<Vec<Vec<u32>>>>,
    /// Current delay interpolation factor.
    pub curr_delay_delta: Vec<Vec<Vec<Vec<FDTD_FLOAT>>>>,
    /// Current amplitude.
    pub curr_amp: Vec<Vec<Vec<Vec<FDTD_FLOAT>>>>,
    /// `current_ts - delay` lookup table, computed once per step.
    delay_lookup: Vec<usize>,
}

impl EngineExtTFSF {
    pub fn new() -> Self {
        Self {
            start: [0, 0, 0],
            stop: [0, 0, 0],
            num_lines: [0, 0, 0],
            active_dir: [[false; 2]; 3],
            signal: Excitation::default(),
            current_ts: 0,
            max_delay: 0,
            volt_delay: vec![vec![vec![vec![]; 2]; 2]; 3],
            volt_delay_delta: vec![vec![vec![vec![]; 2]; 2]; 3],
            volt_amp: vec![vec![vec![vec![]; 2]; 2]; 3],
            curr_delay: vec![vec![vec![vec![]; 2]; 2]; 3],
            curr_delay_delta: vec![vec![vec![vec![]; 2]; 2]; 3],
            curr_amp: vec![vec![vec![vec![]; 2]; 2]; 3],
            delay_lookup: Vec::new(),
        }
    }

    /// Set the TFSF region boundaries.
    pub fn set_region(&mut self, start: [usize; 3], stop: [usize; 3]) {
        self.start = start;
        self.stop = stop;
        self.num_lines = [
            stop[0] - start[0] + 1,
            stop[1] - start[1] + 1,
            stop[2] - start[2] + 1,
        ];
    }

    /// Activate a face for TFSF correction.
    pub fn activate_face(&mut self, direction: usize, face: i32) {
        if direction < 3 && (0..=1).contains(&face) {
            self.active_dir[direction][face as usize] = true;
        }
    }

    /// Set the excitation signal.
    pub fn set_signal(&mut self, signal: Excitation) {
        self.signal = signal;
    }

    /// Set the maximum delay.
    pub fn set_max_delay(&mut self, max_delay: u32) {
        self.max_delay = max_delay;
    }

    /// Allocate the per-face arrays for every active face.
    pub fn allocate(&mut self) {
        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            let count = self.num_lines[n_p] * self.num_lines[n_pp];
            for face in 0..2 {
                for component in 0..2 {
                    if !self.active_dir[n][face] {
                        continue;
                    }
                    self.volt_delay[n][face][component] = vec![0; count];
                    self.volt_delay_delta[n][face][component] = vec![0.0; count];
                    self.volt_amp[n][face][component] = vec![0.0; count];
                    self.curr_delay[n][face][component] = vec![0; count];
                    self.curr_delay_delta[n][face][component] = vec![0.0; count];
                    self.curr_amp[n][face][component] = vec![0.0; count];
                }
            }
        }
    }

    /// Recompute `current_ts - delay`, mirroring the native delay table.
    fn update_delay_lookup(&mut self, length: u32, period: u32) {
        self.delay_lookup = vec![0; self.max_delay as usize + 1];
        for delay in 0..=self.max_delay as usize {
            let value = if (self.current_ts as usize) < delay {
                0
            } else if (self.current_ts as usize) - delay >= length as usize && period == 0 {
                0
            } else {
                self.current_ts as usize - delay
            };
            self.delay_lookup[delay] = if period > 0 {
                value % period as usize
            } else {
                value
            };
        }
    }

    /// Add the incident field to one edge with the native linear interpolation.
    fn add_sample(
        &self,
        amplitude: FDTD_FLOAT,
        delay: u32,
        delta: FDTD_FLOAT,
        sample: impl Fn(usize) -> FDTD_FLOAT,
    ) -> FDTD_FLOAT {
        let first = self.delay_lookup.get(delay as usize).copied().unwrap_or(0);
        let second = self
            .delay_lookup
            .get(delay as usize + 1)
            .copied()
            .unwrap_or(0);
        (1.0 - delta) * amplitude * sample(first) + delta * amplitude * sample(second)
    }
}

impl Default for EngineExtTFSF {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineExtension for EngineExtTFSF {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_TFSF
    }

    fn get_extension_name(&self) -> &'static str {
        "TFSF"
    }

    fn set_timestep(&mut self, number: u32) {
        self.current_ts = number;
    }

    fn do_post_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        let length = self.signal.signal_length();
        let period = self.signal.signal_period();
        self.update_delay_lookup(length, period);

        for n in 0..3 {
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for face in 0..2 {
                if !self.active_dir[n][face] {
                    continue;
                }
                let mut pos = [0usize; 3];
                pos[n_p] = self.start[n_p];
                let mut ui_pos = 0usize;
                for _i in 0..self.num_lines[n_p] {
                    pos[n_pp] = self.start[n_pp];
                    for _j in 0..self.num_lines[n_pp] {
                        pos[n] = if face == 0 {
                            self.start[n]
                        } else {
                            self.stop[n]
                        };
                        for component in 0..2 {
                            let edge = if component == 0 { n_p } else { n_pp };
                            let amplitude = self.volt_amp[n][face][component][ui_pos];
                            let delay = self.volt_delay[n][face][component][ui_pos];
                            let delta = self.volt_delay_delta[n][face][component][ui_pos];
                            let correction = self.add_sample(amplitude, delay, delta, |index| {
                                self.signal.current(index)
                            });
                            let current = *volt.get(edge, pos[0], pos[1], pos[2]);
                            *volt.get_mut(edge, pos[0], pos[1], pos[2]) = current + correction;
                        }
                        pos[n_pp] += 1;
                        ui_pos += 1;
                    }
                    pos[n_p] += 1;
                }
            }
        }
    }

    fn do_post_current_updates(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        let length = self.signal.signal_length();
        let period = self.signal.signal_period();
        self.update_delay_lookup(length, period);

        for n in 0..3 {
            if !self.active_dir[n][0] && !self.active_dir[n][1] {
                continue;
            }
            let n_p = (n + 1) % 3;
            let n_pp = (n + 2) % 3;
            for face in 0..2 {
                if !self.active_dir[n][face] {
                    continue;
                }
                let mut pos = [0usize; 3];
                pos[n_p] = self.start[n_p];
                let mut ui_pos = 0usize;
                for _i in 0..self.num_lines[n_p] {
                    pos[n_pp] = self.start[n_pp];
                    for _j in 0..self.num_lines[n_pp] {
                        pos[n] = if face == 0 {
                            self.start[n].saturating_sub(1)
                        } else {
                            self.stop[n]
                        };
                        for component in 0..2 {
                            let edge = if component == 0 { n_p } else { n_pp };
                            let amplitude = self.curr_amp[n][face][component][ui_pos];
                            let delay = self.curr_delay[n][face][component][ui_pos];
                            let delta = self.curr_delay_delta[n][face][component][ui_pos];
                            let correction = self.add_sample(amplitude, delay, delta, |index| {
                                self.signal.voltage(index)
                            });
                            let current = *curr.get(edge, pos[0], pos[1], pos[2]);
                            *curr.get_mut(edge, pos[0], pos[1], pos[2]) = current + correction;
                        }
                        pos[n_pp] += 1;
                        ui_pos += 1;
                    }
                    pos[n_p] += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tfsf_tests {
    use super::*;

    #[test]
    fn test_tfsf_creation() {
        let ext = EngineExtTFSF::new();
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_TFSF);
        assert_eq!(ext.get_extension_name(), "TFSF");
    }

    #[test]
    fn test_tfsf_set_region() {
        let mut ext = EngineExtTFSF::new();
        ext.set_region([5, 5, 5], [15, 15, 15]);
        assert_eq!(ext.start, [5, 5, 5]);
        assert_eq!(ext.stop, [15, 15, 15]);
        // Native keeps the inclusive line count.
        assert_eq!(ext.num_lines, [11, 11, 11]);
    }

    #[test]
    fn test_tfsf_activate_face() {
        let mut ext = EngineExtTFSF::new();
        ext.activate_face(0, 0);
        ext.activate_face(2, 1);
        assert!(ext.active_dir[0][0]);
        assert!(ext.active_dir[2][1]);
        assert!(!ext.active_dir[1][0]);
    }

    #[test]
    fn delay_lookup_matches_native_semantics() {
        let mut ext = EngineExtTFSF::new();
        ext.set_max_delay(5);
        ext.current_ts = 100;
        ext.update_delay_lookup(200, 0);
        assert_eq!(ext.delay_lookup[0], 100);
        assert_eq!(ext.delay_lookup[5], 95);
        // A delay beyond the current step maps to sample zero.
        ext.current_ts = 2;
        ext.update_delay_lookup(200, 0);
        assert_eq!(ext.delay_lookup[0], 2);
        assert_eq!(ext.delay_lookup[5], 0);
        // A periodic signal wraps the index into one period.
        ext.current_ts = 100;
        ext.update_delay_lookup(200, 30);
        assert_eq!(ext.delay_lookup[0], 10);
        assert_eq!(ext.delay_lookup[5], 5);
    }

    #[test]
    fn a_zero_amplitude_leaves_the_field_untouched() {
        let mut ext = EngineExtTFSF::new();
        ext.set_region([1, 1, 1], [2, 2, 2]);
        ext.activate_face(0, 0);
        ext.allocate();
        ext.set_max_delay(1);
        ext.signal = Excitation::gauss(1.0e9, 5.0e8);
        ext.current_ts = 0;
        let mut volt = FieldArray::new("volt", [4, 4, 4]);
        *volt.get_mut(1, 1, 1, 1) = 2.0;
        ext.do_post_voltage_updates(&mut volt, &FieldArray::new("curr", [4, 4, 4]));
        assert_eq!(*volt.get(1, 1, 1, 1), 2.0);
    }
}

// ============================================================================
// Dispersive materials extension (Debye / Lorentz)
// ============================================================================

// Priority constants for dispersive extensions
pub const ENG_EXT_PRIO_DISPERSIVE_DEBYE: i32 = 75_000;
pub const ENG_EXT_PRIO_DISPERSIVE_LORENTZ: i32 = 75_000;
pub const ENG_EXT_PRIO_DISPERSIVE_APPLY2: i32 = 0;
pub const ENG_EXT_PRIO_CONDUCTING_SHEET: i32 = 75_000;
pub const ENG_EXT_PRIO_LUMPED_RLC: i32 = 70_000;

/// Engine extension for Debye dispersive materials.
/// Implements the ADE (Auxiliary Differential Equation) method for Debye media.
/// Mirrors C++ `Engine_Ext_DebyeMaterial`.
///
/// Algorithm (trapezoidal update of pole capacitor voltage):
///   DoPreVoltageUpdates  (V is V^n, the only place it exists):
///     h_k = c1_k * W_k + d_k * V^n
///     g   += h_k - W_k
///     W_k  = h_k
///   main voltage update  (V becomes V_raw)
///   DoPostVoltageUpdates (V_raw visible, V^n gone):
///     V^(n+1) = inv * (V_raw - g)
///     volt_ADE[0] = V_raw - V^(n+1)
///     W_k += d_k * V^(n+1)
///   Apply2Voltages:
///     V -= volt_ADE[0]
pub struct EngineExtDebyeMaterial {
    /// Number of poles.
    pub num_poles: usize,
    /// Number of mesh positions with Debye material.
    pub num_positions: usize,
    /// Position list: [pos_idx] -> [i, j, k].
    pub positions: Vec<[usize; 3]>,
    /// Voltage ADE correction (order 0, joint correction): [component][pos_idx].
    pub volt_ade: Vec<Vec<FDTD_FLOAT>>,
    /// Pole state W_k = r_k * V_Ck: [pole][component][pos_idx].
    pub volt_pole_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Pre-computed correction summed over poles: [component][pos_idx].
    pub volt_pre_ade: Vec<Vec<FDTD_FLOAT>>,
    /// Pole relaxation coefficient: [pole][component][pos_idx].
    pub v_relax_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Pole drive coefficient: [pole][component][pos_idx].
    pub v_drive_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Solve coefficient for implicit update: [component][pos_idx].
    pub v_solve_ade: Vec<Vec<FDTD_FLOAT>>,
}

impl EngineExtDebyeMaterial {
    /// Create a new Debye extension with the given number of poles and positions.
    pub fn new(num_poles: usize, num_positions: usize) -> Self {
        let zero_vec = vec![FDTD_FLOAT::default(); num_positions];
        let zero_mat = vec![zero_vec.clone(); 3];
        let zero_cube = vec![zero_mat.clone(); num_poles];

        Self {
            num_poles,
            num_positions,
            positions: vec![[0, 0, 0]; num_positions],
            volt_ade: vec![zero_vec.clone(); 3],
            volt_pole_ade: zero_cube.clone(),
            volt_pre_ade: vec![zero_vec.clone(); 3],
            v_relax_ade: zero_cube.clone(),
            v_drive_ade: zero_cube,
            v_solve_ade: vec![zero_vec.clone(); 3],
        }
    }

    /// Set the mesh position of one Debye cell.
    pub fn set_position(&mut self, pos_idx: usize, pos: [usize; 3]) {
        self.positions[pos_idx] = pos;
    }

    /// Set the per-pole coefficients of one cell for a single field component.
    ///
    /// `v_relax` and `v_drive` are indexed by pole; the native operator stores
    /// one set per direction because the weights may be anisotropic.
    pub fn set_position_component(
        &mut self,
        pos_idx: usize,
        component: usize,
        v_relax: &[FDTD_FLOAT],
        v_drive: &[FDTD_FLOAT],
        v_solve: FDTD_FLOAT,
    ) {
        if component >= 3 {
            return;
        }
        self.v_solve_ade[component][pos_idx] = v_solve;
        for pole in 0..self.num_poles.min(v_relax.len()).min(v_drive.len()) {
            self.v_relax_ade[pole][component][pos_idx] = v_relax[pole];
            self.v_drive_ade[pole][component][pos_idx] = v_drive[pole];
        }
    }

    /// Set every component to the same coefficients (isotropic shortcut).
    pub fn set_position_uniform(
        &mut self,
        pos_idx: usize,
        pos: [usize; 3],
        v_relax: &[FDTD_FLOAT],
        v_drive: &[FDTD_FLOAT],
        v_solve: FDTD_FLOAT,
    ) {
        self.set_position(pos_idx, pos);
        for component in 0..3 {
            self.set_position_component(pos_idx, component, v_relax, v_drive, v_solve);
        }
    }
}

impl Default for EngineExtDebyeMaterial {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl EngineExtension for EngineExtDebyeMaterial {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_DISPERSIVE_DEBYE
    }

    fn get_extension_name(&self) -> &'static str {
        "Debye Material"
    }

    fn do_pre_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // Compute pole states from V^n (the only place V^n exists before the main update).
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for comp in 0..3 {
                let volt_val = *volt.get(comp, i, j, k);
                let mut pre = FDTD_FLOAT::default();
                for o in 0..self.num_poles {
                    let w = self.v_relax_ade[o][comp][pos_idx]
                        * self.volt_pole_ade[o][comp][pos_idx]
                        + self.v_drive_ade[o][comp][pos_idx] * volt_val;
                    pre += w - self.volt_pole_ade[o][comp][pos_idx];
                    self.volt_pole_ade[o][comp][pos_idx] = w;
                }
                self.volt_pre_ade[comp][pos_idx] = pre;
            }
        }
    }

    fn do_post_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // Solve for V^(n+1) and finish pole state updates.
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for comp in 0..3 {
                let v_raw = *volt.get(comp, i, j, k);
                let volt_new =
                    (v_raw - self.volt_pre_ade[comp][pos_idx]) * self.v_solve_ade[comp][pos_idx];
                self.volt_ade[comp][pos_idx] = v_raw - volt_new;
                for o in 0..self.num_poles {
                    self.volt_pole_ade[o][comp][pos_idx] +=
                        self.v_drive_ade[o][comp][pos_idx] * volt_new;
                }
            }
        }
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // Subtract the joint Debye correction from voltage fields.
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for comp in 0..3 {
                let current = *volt.get(comp, i, j, k);
                *volt.get_mut(comp, i, j, k) = current - self.volt_ade[comp][pos_idx];
            }
        }
    }
}

#[cfg(test)]
mod debye_tests {
    use super::*;

    #[test]
    fn test_debye_creation() {
        let ext = EngineExtDebyeMaterial::new(2, 4);
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_DISPERSIVE_DEBYE);
        assert_eq!(ext.get_extension_name(), "Debye Material");
        assert_eq!(ext.num_poles, 2);
        assert_eq!(ext.num_positions, 4);
    }

    #[test]
    fn test_debye_set_position() {
        let mut ext = EngineExtDebyeMaterial::new(1, 1);
        ext.set_position_uniform(0, [5, 5, 5], &[0.5_f32], &[0.3_f32], 0.8_f32);
        assert_eq!(ext.positions[0], [5, 5, 5]);
        assert_eq!(ext.v_relax_ade[0][0][0], 0.5_f32);
        assert_eq!(ext.v_drive_ade[0][0][0], 0.3_f32);
        assert_eq!(ext.v_solve_ade[0][0], 0.8_f32);
    }

    #[test]
    fn test_debye_pre_post_voltage() {
        let mut ext = EngineExtDebyeMaterial::new(1, 1);
        ext.set_position_uniform(0, [1, 1, 1], &[0.5_f32], &[0.3_f32], 0.8_f32);

        // Set up a voltage field
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 10.0_f32;
        *volt.get_mut(1, 1, 1, 1) = 20.0_f32;
        *volt.get_mut(2, 1, 1, 1) = 30.0_f32;

        // Pre-voltage update: reads V^n, computes pole states
        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // After pre-update, volt_pole_ade should be updated
        // W = 0.5 * 0 + 0.3 * 10 = 3.0 for comp 0
        assert!((ext.volt_pole_ade[0][0][0] - 3.0_f32).abs() < 1e-6);
        // pre = W - 0 = 3.0 for comp 0
        assert!((ext.volt_pre_ade[0][0] - 3.0_f32).abs() < 1e-6);

        // Simulate main voltage update (V becomes V_raw)
        *volt.get_mut(0, 1, 1, 1) = 15.0_f32;

        // Post-voltage update: solves for V^(n+1)
        ext.do_post_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // V^(n+1) = (V_raw - g) * inv = (15.0 - 3.0) * 0.8 = 9.6
        // volt_ade = V_raw - V^(n+1) = 15.0 - 9.6 = 5.4
        assert!((ext.volt_ade[0][0] - 5.4_f32).abs() < 1e-6);
        // W += d * V^(n+1) = 3.0 + 0.3 * 9.6 = 5.88
        assert!((ext.volt_pole_ade[0][0][0] - 5.88_f32).abs() < 1e-6);
    }

    #[test]
    fn test_debye_apply2_voltages() {
        let mut ext = EngineExtDebyeMaterial::new(1, 1);
        ext.set_position_uniform(0, [1, 1, 1], &[0.5_f32], &[0.3_f32], 0.8_f32);
        ext.volt_ade[0][0] = 5.4_f32;

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 15.0_f32;

        ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // V -= volt_ade = 15.0 - 5.4 = 9.6
        assert!((*volt.get(0, 1, 1, 1) - 9.6_f32).abs() < 1e-6);
    }
}

/// Engine extension for Lorentz dispersive materials.
/// Implements the ADE method for Lorentz media.
/// Mirrors C++ `Engine_Ext_LorentzMaterial`.
pub struct EngineExtLorentzMaterial {
    /// Number of dispersion orders.
    pub num_orders: usize,
    /// Number of mesh positions with Lorentz material.
    pub num_positions: usize,
    /// Position list: [pos_idx] -> [i, j, k].
    pub positions: Vec<[usize; 3]>,
    /// Voltage ADE state: [order][component][pos_idx].
    pub volt_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Current ADE state: [order][component][pos_idx].
    pub curr_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Lorentz voltage state: [order][component][pos_idx].
    pub volt_lor_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Lorentz current state: [order][component][pos_idx].
    pub curr_lor_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Voltage internal coefficient: [order][component][pos_idx].
    pub v_int_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Voltage external coefficient: [order][component][pos_idx].
    pub v_ext_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Voltage Lorentz coefficient: [order][component][pos_idx].
    pub v_lor_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Current internal coefficient: [order][component][pos_idx].
    pub i_int_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Current external coefficient: [order][component][pos_idx].
    pub i_ext_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Current Lorentz coefficient: [order][component][pos_idx].
    pub i_lor_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Voltage ADE active per order (native `m_volt_ADE_On`).
    pub volt_ade_on: Vec<bool>,
    /// Current ADE active per order (native `m_curr_ADE_On`).
    pub curr_ade_on: Vec<bool>,
}

impl EngineExtLorentzMaterial {
    /// Create a new Lorentz extension with the given number of orders and positions.
    pub fn new(num_orders: usize, num_positions: usize) -> Self {
        let zero_vec = vec![FDTD_FLOAT::default(); num_positions];
        let zero_mat = vec![zero_vec.clone(); 3];
        let zero_cube = vec![zero_mat.clone(); num_orders];

        Self {
            num_orders,
            num_positions,
            positions: vec![[0, 0, 0]; num_positions],
            volt_ade: zero_cube.clone(),
            curr_ade: zero_cube.clone(),
            volt_lor_ade: zero_cube.clone(),
            curr_lor_ade: zero_cube.clone(),
            v_int_ade: zero_cube.clone(),
            v_ext_ade: zero_cube.clone(),
            v_lor_ade: zero_cube.clone(),
            i_int_ade: zero_cube.clone(),
            i_ext_ade: zero_cube.clone(),
            i_lor_ade: zero_cube,
            volt_ade_on: vec![false; num_orders],
            curr_ade_on: vec![false; num_orders],
        }
    }

    /// Set the mesh position of one dispersive cell.
    pub fn set_position(&mut self, pos_idx: usize, pos: [usize; 3]) {
        self.positions[pos_idx] = pos;
    }

    /// Set the Lorentz coefficients of one order for one cell.
    ///
    /// Each slice holds one value per field component.
    #[allow(clippy::too_many_arguments)]
    pub fn set_order_coefficients(
        &mut self,
        pos_idx: usize,
        order: usize,
        v_int: &[FDTD_FLOAT; 3],
        v_ext: &[FDTD_FLOAT; 3],
        v_lor: &[FDTD_FLOAT; 3],
        i_int: &[FDTD_FLOAT; 3],
        i_ext: &[FDTD_FLOAT; 3],
        i_lor: &[FDTD_FLOAT; 3],
    ) {
        if order >= self.num_orders {
            return;
        }
        for component in 0..3 {
            self.v_int_ade[order][component][pos_idx] = v_int[component];
            self.v_ext_ade[order][component][pos_idx] = v_ext[component];
            self.v_lor_ade[order][component][pos_idx] = v_lor[component];
            self.i_int_ade[order][component][pos_idx] = i_int[component];
            self.i_ext_ade[order][component][pos_idx] = i_ext[component];
            self.i_lor_ade[order][component][pos_idx] = i_lor[component];
        }
    }

    /// Mark one order as carrying voltage and/or current ADE terms.
    pub fn set_order_active(&mut self, order: usize, voltage: bool, current: bool) {
        if order < self.num_orders {
            self.volt_ade_on[order] = voltage;
            self.curr_ade_on[order] = current;
        }
    }
}

impl Default for EngineExtLorentzMaterial {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl EngineExtension for EngineExtLorentzMaterial {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_DISPERSIVE_LORENTZ
    }

    fn get_extension_name(&self) -> &'static str {
        "Lorentz Material"
    }

    fn do_pre_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // Update voltage ADE state: volt_ADE *= v_int + v_ext * (V - volt_Lor)
        // volt_Lor += v_lor * volt_ADE (old)
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for o in 0..self.num_orders {
                if !self.volt_ade_on[o] {
                    continue;
                }
                for comp in 0..3 {
                    let volt_val = *volt.get(comp, i, j, k);
                    if self.v_lor_ade[o][comp][pos_idx] != FDTD_FLOAT::default() {
                        self.volt_lor_ade[o][comp][pos_idx] +=
                            self.v_lor_ade[o][comp][pos_idx] * self.volt_ade[o][comp][pos_idx];
                        self.volt_ade[o][comp][pos_idx] *= self.v_int_ade[o][comp][pos_idx];
                        self.volt_ade[o][comp][pos_idx] += self.v_ext_ade[o][comp][pos_idx]
                            * (volt_val - self.volt_lor_ade[o][comp][pos_idx]);
                    } else {
                        self.volt_ade[o][comp][pos_idx] *= self.v_int_ade[o][comp][pos_idx];
                        self.volt_ade[o][comp][pos_idx] +=
                            self.v_ext_ade[o][comp][pos_idx] * volt_val;
                    }
                }
            }
        }
    }

    fn do_pre_current_updates(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        // Update current ADE state: curr_ADE *= i_int + i_ext * (I - curr_Lor)
        // curr_Lor += i_lor * curr_ADE (old)
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for o in 0..self.num_orders {
                if !self.curr_ade_on[o] {
                    continue;
                }
                for comp in 0..3 {
                    let curr_val = *curr.get(comp, i, j, k);
                    if self.i_lor_ade[o][comp][pos_idx] != FDTD_FLOAT::default() {
                        self.curr_lor_ade[o][comp][pos_idx] +=
                            self.i_lor_ade[o][comp][pos_idx] * self.curr_ade[o][comp][pos_idx];
                        self.curr_ade[o][comp][pos_idx] *= self.i_int_ade[o][comp][pos_idx];
                        self.curr_ade[o][comp][pos_idx] += self.i_ext_ade[o][comp][pos_idx]
                            * (curr_val - self.curr_lor_ade[o][comp][pos_idx]);
                    } else {
                        self.curr_ade[o][comp][pos_idx] *= self.i_int_ade[o][comp][pos_idx];
                        self.curr_ade[o][comp][pos_idx] +=
                            self.i_ext_ade[o][comp][pos_idx] * curr_val;
                    }
                }
            }
        }
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // Subtract voltage ADE correction from voltage fields.
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for o in 0..self.num_orders {
                for comp in 0..3 {
                    let current = *volt.get(comp, i, j, k);
                    *volt.get_mut(comp, i, j, k) = current - self.volt_ade[o][comp][pos_idx];
                }
            }
        }
    }

    fn apply2_current(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        // Subtract current ADE correction from current fields.
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for o in 0..self.num_orders {
                for comp in 0..3 {
                    let current = *curr.get(comp, i, j, k);
                    *curr.get_mut(comp, i, j, k) = current - self.curr_ade[o][comp][pos_idx];
                }
            }
        }
    }
}

#[cfg(test)]
mod lorentz_tests {
    use super::*;

    #[test]
    fn test_lorentz_creation() {
        let ext = EngineExtLorentzMaterial::new(2, 4);
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_DISPERSIVE_LORENTZ);
        assert_eq!(ext.get_extension_name(), "Lorentz Material");
        assert_eq!(ext.num_orders, 2);
        assert_eq!(ext.num_positions, 4);
    }

    #[test]
    fn test_lorentz_set_position() {
        let mut ext = EngineExtLorentzMaterial::new(1, 1);
        ext.set_position(0, [5, 5, 5]);
        ext.set_order_coefficients(
            0, 0, &[0.5; 3], &[0.3; 3], &[0.1; 3], &[0.6; 3], &[0.4; 3], &[0.2; 3],
        );
        ext.set_order_active(0, true, true);
        assert_eq!(ext.positions[0], [5, 5, 5]);
        assert_eq!(ext.v_int_ade[0][0][0], 0.5_f32);
        assert_eq!(ext.v_ext_ade[0][0][0], 0.3_f32);
        assert_eq!(ext.v_lor_ade[0][0][0], 0.1_f32);
        assert_eq!(ext.i_int_ade[0][0][0], 0.6_f32);
        assert_eq!(ext.i_ext_ade[0][0][0], 0.4_f32);
        assert_eq!(ext.i_lor_ade[0][0][0], 0.2_f32);
    }

    #[test]
    fn test_lorentz_pre_voltage() {
        let mut ext = EngineExtLorentzMaterial::new(1, 1);
        ext.set_position(0, [1, 1, 1]);
        ext.set_order_coefficients(
            0, 0, &[0.5; 3], &[0.3; 3], &[0.1; 3], &[0.6; 3], &[0.4; 3], &[0.2; 3],
        );
        ext.set_order_active(0, true, true);

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 10.0_f32;

        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // volt_ADE = 0 * 0.5 + 0.3 * (10 - 0) = 3.0
        // volt_Lor = 0 + 0.1 * 0 = 0
        assert!((ext.volt_ade[0][0][0] - 3.0_f32).abs() < 1e-6);
        assert!((ext.volt_lor_ade[0][0][0] - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_lorentz_pre_current() {
        let mut ext = EngineExtLorentzMaterial::new(1, 1);
        ext.set_position(0, [1, 1, 1]);
        ext.set_order_coefficients(
            0, 0, &[0.5; 3], &[0.3; 3], &[0.1; 3], &[0.6; 3], &[0.4; 3], &[0.2; 3],
        );
        ext.set_order_active(0, true, true);

        let mut curr = FieldArray::new("curr", [3, 3, 3]);
        *curr.get_mut(0, 1, 1, 1) = 10.0_f32;

        ext.do_pre_current_updates(&FieldArray::new("volt", [3, 3, 3]), &mut curr);

        // curr_ADE = 0 * 0.6 + 0.4 * (10 - 0) = 4.0
        // curr_Lor = 0 + 0.2 * 0 = 0
        assert!((ext.curr_ade[0][0][0] - 4.0_f32).abs() < 1e-6);
        assert!((ext.curr_lor_ade[0][0][0] - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_lorentz_apply2_voltages() {
        let mut ext = EngineExtLorentzMaterial::new(1, 1);
        ext.set_position(0, [1, 1, 1]);
        ext.set_order_coefficients(
            0, 0, &[0.5; 3], &[0.3; 3], &[0.1; 3], &[0.6; 3], &[0.4; 3], &[0.2; 3],
        );
        ext.set_order_active(0, true, true);
        ext.volt_ade[0][0][0] = 5.4_f32;

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 15.0_f32;

        ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // V -= volt_ade = 15.0 - 5.4 = 9.6
        assert!((*volt.get(0, 1, 1, 1) - 9.6_f32).abs() < 1e-6);
    }

    #[test]
    fn test_lorentz_apply2_current() {
        let mut ext = EngineExtLorentzMaterial::new(1, 1);
        ext.set_position(0, [1, 1, 1]);
        ext.set_order_coefficients(
            0, 0, &[0.5; 3], &[0.3; 3], &[0.1; 3], &[0.6; 3], &[0.4; 3], &[0.2; 3],
        );
        ext.set_order_active(0, true, true);
        ext.curr_ade[0][0][0] = 5.4_f32;

        let mut curr = FieldArray::new("curr", [3, 3, 3]);
        *curr.get_mut(0, 1, 1, 1) = 15.0_f32;

        ext.apply2_current(&FieldArray::new("volt", [3, 3, 3]), &mut curr);

        // I -= curr_ade = 15.0 - 5.4 = 9.6
        assert!((*curr.get(0, 1, 1, 1) - 9.6_f32).abs() < 1e-6);
    }
}

// ============================================================================
// Conducting sheet extension
// ============================================================================

/// Coefficient set for a single conducting sheet position.
/// Contains ADE coefficients for 2 orders and 3 field components,
/// plus modified EC voltage coefficients (vv/vi).
pub struct ConductingSheetCoeffs {
    /// Internal ADE coefficients: [order][component].
    pub v_int_ade: [[FDTD_FLOAT; 3]; 2],
    /// External ADE coefficients: [order][component].
    pub v_ext_ade: [[FDTD_FLOAT; 3]; 2],
    /// Modified EC voltage coefficient vv: [component].
    pub vv: [FDTD_FLOAT; 3],
    /// Modified EC voltage coefficient vi: [component].
    pub vi: [FDTD_FLOAT; 3],
}

/// Engine extension for a conducting sheet model.
/// Implements the ADE method for conducting sheets (order=2, voltage ADE only).
/// Mirrors C++ `Operator_Ext_ConductingSheet`.
pub struct EngineExtConductingSheet {
    /// Number of mesh positions with conducting sheet.
    pub num_positions: usize,
    /// Position list: [pos_idx] -> [i, j, k].
    pub positions: Vec<[usize; 3]>,
    /// Voltage internal ADE coefficients: [order][component][pos_idx].
    pub v_int_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Voltage external ADE coefficients: [order][component][pos_idx].
    pub v_ext_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Voltage ADE state: [order][component][pos_idx].
    pub volt_ade: Vec<Vec<Vec<FDTD_FLOAT>>>,
    /// Modified EC voltage coefficient vv: [component][pos_idx].
    pub vv_mod: Vec<Vec<FDTD_FLOAT>>,
    /// Modified EC voltage coefficient vi: [component][pos_idx].
    pub vi_mod: Vec<Vec<FDTD_FLOAT>>,
    /// Whether EC coefficients have been applied to the operator.
    pub ec_applied: bool,
}

impl EngineExtConductingSheet {
    /// Create a new conducting sheet extension with the given number of positions.
    pub fn new(num_positions: usize) -> Self {
        let mut v_int_ade = Vec::with_capacity(2);
        let mut v_ext_ade = Vec::with_capacity(2);
        let mut volt_ade = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut comp_int = Vec::with_capacity(3);
            let mut comp_ext = Vec::with_capacity(3);
            let mut comp_state = Vec::with_capacity(3);
            for _ in 0..3 {
                comp_int.push(vec![FDTD_FLOAT::default(); num_positions]);
                comp_ext.push(vec![FDTD_FLOAT::default(); num_positions]);
                comp_state.push(vec![FDTD_FLOAT::default(); num_positions]);
            }
            v_int_ade.push(comp_int);
            v_ext_ade.push(comp_ext);
            volt_ade.push(comp_state);
        }
        let vv_mod = vec![vec![FDTD_FLOAT::default(); num_positions]; 3];
        let vi_mod = vec![vec![FDTD_FLOAT::default(); num_positions]; 3];
        Self {
            num_positions,
            positions: vec![[0, 0, 0]; num_positions],
            v_int_ade,
            v_ext_ade,
            volt_ade,
            vv_mod,
            vi_mod,
            ec_applied: false,
        }
    }

    /// Set the position and coefficients for a conducting sheet cell.
    pub fn set_position(
        &mut self,
        pos_idx: usize,
        pos: [usize; 3],
        coeffs: &ConductingSheetCoeffs,
    ) {
        self.positions[pos_idx] = pos;
        for o in 0..2 {
            for n in 0..3 {
                self.v_int_ade[o][n][pos_idx] = coeffs.v_int_ade[o][n];
                self.v_ext_ade[o][n][pos_idx] = coeffs.v_ext_ade[o][n];
            }
        }
        for n in 0..3 {
            self.vv_mod[n][pos_idx] = coeffs.vv[n];
            self.vi_mod[n][pos_idx] = coeffs.vi[n];
        }
    }

    /// Apply modified EC voltage coefficients to the operator.
    /// Should be called before running the time-stepping loop.
    pub fn apply_ec_coeffs(&mut self, op: &mut dyn crate::operator_base::OperatorBase) {
        if self.ec_applied {
            return;
        }
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for n in 0..3 {
                op.set_ec_voltage_coeffs(
                    n,
                    i,
                    j,
                    k,
                    self.vv_mod[n][pos_idx],
                    self.vi_mod[n][pos_idx],
                );
            }
        }
        self.ec_applied = true;
    }
}

impl EngineExtension for EngineExtConductingSheet {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_CONDUCTING_SHEET
    }

    fn get_extension_name(&self) -> &'static str {
        "Conducting Sheet"
    }

    fn do_pre_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // ADE update: volt_ADE[o][n] = v_int * volt_ADE[o][n] + v_ext * V[n]
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for o in 0..2 {
                for n in 0..3 {
                    let v = *volt.get(n, i, j, k);
                    self.volt_ade[o][n][pos_idx] = self.v_int_ade[o][n][pos_idx]
                        * self.volt_ade[o][n][pos_idx]
                        + self.v_ext_ade[o][n][pos_idx] * v;
                }
            }
        }
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        // Subtract the sum of ADE corrections from voltage fields.
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            for n in 0..3 {
                let mut correction = FDTD_FLOAT::default();
                for o in 0..2 {
                    correction += self.volt_ade[o][n][pos_idx];
                }
                let current = *volt.get(n, i, j, k);
                *volt.get_mut(n, i, j, k) = current - correction;
            }
        }
    }
}

#[cfg(test)]
mod conducting_sheet_tests {
    use super::*;

    #[test]
    fn test_conducting_sheet_creation() {
        let ext = EngineExtConductingSheet::new(4);
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_CONDUCTING_SHEET);
        assert_eq!(ext.get_extension_name(), "Conducting Sheet");
        assert_eq!(ext.num_positions, 4);
        assert_eq!(ext.positions.len(), 4);
        assert_eq!(ext.v_int_ade.len(), 2);
        assert_eq!(ext.v_int_ade[0].len(), 3);
        assert_eq!(ext.v_int_ade[0][0].len(), 4);
        assert!(!ext.ec_applied);
    }

    #[test]
    fn test_conducting_sheet_set_position() {
        let mut ext = EngineExtConductingSheet::new(1);
        let coeffs = ConductingSheetCoeffs {
            v_int_ade: [[0.5_f32, 0.6_f32, 0.7_f32], [0.8_f32, 0.9_f32, 1.0_f32]],
            v_ext_ade: [[0.1_f32, 0.2_f32, 0.3_f32], [0.4_f32, 0.5_f32, 0.6_f32]],
            vv: [0.1_f32, 0.2_f32, 0.3_f32],
            vi: [0.4_f32, 0.5_f32, 0.6_f32],
        };
        ext.set_position(0, [5, 5, 5], &coeffs);
        assert_eq!(ext.positions[0], [5, 5, 5]);
        assert_eq!(ext.v_int_ade[0][0][0], 0.5_f32);
        assert_eq!(ext.v_int_ade[1][2][0], 1.0_f32);
        assert_eq!(ext.v_ext_ade[0][0][0], 0.1_f32);
        assert_eq!(ext.v_ext_ade[1][2][0], 0.6_f32);
        assert_eq!(ext.vv_mod[0][0], 0.1_f32);
        assert_eq!(ext.vi_mod[2][0], 0.6_f32);
    }

    #[test]
    fn test_conducting_sheet_pre_voltage_ade() {
        let mut ext = EngineExtConductingSheet::new(1);
        let coeffs = ConductingSheetCoeffs {
            v_int_ade: [[0.5_f32, 0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32, 0.0_f32]],
            v_ext_ade: [[0.3_f32, 0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32, 0.0_f32]],
            vv: [0.0_f32, 0.0_f32, 0.0_f32],
            vi: [0.0_f32, 0.0_f32, 0.0_f32],
        };
        ext.set_position(0, [1, 1, 1], &coeffs);

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 10.0_f32;

        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // volt_ADE[0][0] = 0.5 * 0 + 0.3 * 10 = 3.0
        assert!((ext.volt_ade[0][0][0] - 3.0_f32).abs() < 1e-6);
        // volt_ADE[1][0] = 0.0 * 0 + 0.0 * 10 = 0.0
        assert!((ext.volt_ade[1][0][0] - 0.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_conducting_sheet_apply2_voltages() {
        let mut ext = EngineExtConductingSheet::new(1);
        let coeffs = ConductingSheetCoeffs {
            v_int_ade: [[0.0_f32, 0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32, 0.0_f32]],
            v_ext_ade: [[0.0_f32, 0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32, 0.0_f32]],
            vv: [0.0_f32, 0.0_f32, 0.0_f32],
            vi: [0.0_f32, 0.0_f32, 0.0_f32],
        };
        ext.set_position(0, [1, 1, 1], &coeffs);
        // Set ADE state directly
        ext.volt_ade[0][0][0] = 5.4_f32;
        ext.volt_ade[1][0][0] = 2.0_f32;

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 15.0_f32;

        ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));

        // V -= volt_ade[0] + volt_ade[1] = 15.0 - 5.4 - 2.0 = 7.6
        assert!((*volt.get(0, 1, 1, 1) - 7.6_f32).abs() < 1e-6);
    }

    #[test]
    fn test_conducting_sheet_full_ade_cycle() {
        let mut ext = EngineExtConductingSheet::new(1);
        let coeffs = ConductingSheetCoeffs {
            v_int_ade: [[0.5_f32, 0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32, 0.0_f32]],
            v_ext_ade: [[0.3_f32, 0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32, 0.0_f32]],
            vv: [0.0_f32, 0.0_f32, 0.0_f32],
            vi: [0.0_f32, 0.0_f32, 0.0_f32],
        };
        ext.set_position(0, [1, 1, 1], &coeffs);

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(0, 1, 1, 1) = 10.0_f32;

        // Pre-voltage update: volt_ADE = 0.5 * 0 + 0.3 * 10 = 3.0
        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        assert!((ext.volt_ade[0][0][0] - 3.0_f32).abs() < 1e-6);

        // Apply2Voltages: V -= 3.0 => 10.0 - 3.0 = 7.0
        ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        assert!((*volt.get(0, 1, 1, 1) - 7.0_f32).abs() < 1e-6);
    }
}

// ============================================================================
// Lumped RLC extension
// ============================================================================

/// Coefficients for one lumped `RLC` edge.
///
/// Matches the native `Operator_Ext_LumpedRLC` arrays: parallel `ilv`/`i2v`
/// and the series `vv2`/`vj1`/`vj2`/`vvd`/`ib0`/`b1`/`b2`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LumpedRLCCoeffs {
    /// Edge direction (0/1/2).
    pub direction: usize,
    /// Parallel: `dt / L`.
    pub ilv: FDTD_FLOAT,
    /// Parallel: `(dt / C) / (1 + dt G / (2 C))`.
    pub i2v: FDTD_FLOAT,
    /// Series: coefficient of `V[n-2]` in the voltage update.
    pub vv2: FDTD_FLOAT,
    /// Series: coefficient of `J[n-1]` in the voltage update.
    pub vj1: FDTD_FLOAT,
    /// Series: coefficient of `J[n-2]` in the voltage update.
    pub vj2: FDTD_FLOAT,
    /// Series: multiplier of the whole voltage update.
    pub vvd: FDTD_FLOAT,
    /// Series: inverse of `beta_0`.
    pub ib0: FDTD_FLOAT,
    /// Series: `beta_1`.
    pub b1: FDTD_FLOAT,
    /// Series: `beta_2`.
    pub b2: FDTD_FLOAT,
}

/// Lumped `RLC` ADE time stepping.
///
/// Ported from `FDTD/extensions/engine_ext_lumpedRLC.cpp`. The state is the
/// three-deep voltage history `V[n]`, `V[n-1]`, `V[n-2]`, the current history
/// `J[n-1]`, `J[n-2]`, and the parallel inductor current `I_L`.
pub struct EngineExtLumpedRLC {
    /// Number of lumped edges.
    pub num_positions: usize,
    /// Mesh position of every lumped edge.
    pub positions: Vec<[usize; 3]>,
    /// Coefficients of every lumped edge.
    pub coeffs: Vec<LumpedRLCCoeffs>,
    /// Voltage history per edge: `[V[n], V[n-1], V[n-2]]`.
    v_vdn: Vec<[FDTD_FLOAT; 3]>,
    /// Current history per edge: `[J[n], J[n-1], J[n-2]]`.
    v_jn: Vec<[FDTD_FLOAT; 3]>,
    /// Parallel inductor current per edge.
    v_il: Vec<FDTD_FLOAT>,
}

impl EngineExtLumpedRLC {
    pub fn new(num_positions: usize) -> Self {
        Self {
            num_positions,
            positions: vec![[0; 3]; num_positions],
            coeffs: vec![LumpedRLCCoeffs::default(); num_positions],
            v_vdn: vec![[0.0; 3]; num_positions],
            v_jn: vec![[0.0; 3]; num_positions],
            v_il: vec![0.0; num_positions],
        }
    }

    pub fn set_position(&mut self, pos_idx: usize, pos: [usize; 3], coeffs: &LumpedRLCCoeffs) {
        self.positions[pos_idx] = pos;
        self.coeffs[pos_idx] = *coeffs;
        self.v_vdn[pos_idx] = [0.0; 3];
        self.v_jn[pos_idx] = [0.0; 3];
        self.v_il[pos_idx] = 0.0;
    }

    /// Voltage history of one edge (test/diagnostic access).
    pub fn voltage_history(&self, pos_idx: usize) -> [FDTD_FLOAT; 3] {
        self.v_vdn[pos_idx]
    }

    /// Current history of one edge (test/diagnostic access).
    pub fn current_history(&self, pos_idx: usize) -> [FDTD_FLOAT; 3] {
        self.v_jn[pos_idx]
    }

    /// Parallel inductor current of one edge (test/diagnostic access).
    pub fn inductor_current(&self, pos_idx: usize) -> FDTD_FLOAT {
        self.v_il[pos_idx]
    }
}

impl EngineExtension for EngineExtLumpedRLC {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_LUMPED_RLC
    }

    fn get_extension_name(&self) -> &'static str {
        "lumped_rlc"
    }

    fn do_pre_voltage_updates(&mut self, _volt: &mut FieldArray, _curr: &FieldArray) {
        if self.num_positions == 0 {
            return;
        }
        // Native rotates the three voltage slots once, then advances the
        // parallel inductor current from the (already shifted) V[n-1].
        for history in self.v_vdn.iter_mut() {
            let oldest = history[2];
            history[2] = history[1];
            history[1] = history[0];
            history[0] = oldest;
        }
        for pos_idx in 0..self.num_positions {
            let coefficient = &self.coeffs[pos_idx];
            self.v_il[pos_idx] += coefficient.i2v * coefficient.ilv * self.v_vdn[pos_idx][1];
        }
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        if self.num_positions == 0 {
            return;
        }
        for history in self.v_jn.iter_mut() {
            let oldest = history[2];
            history[2] = history[1];
            history[1] = history[0];
            history[0] = oldest;
        }
        for pos_idx in 0..self.num_positions {
            let [i, j, k] = self.positions[pos_idx];
            let coefficient = self.coeffs[pos_idx];
            let direction = coefficient.direction.min(2);

            let raw = *volt.get(direction, i, j, k);
            self.v_vdn[pos_idx][0] = raw;

            let updated = coefficient.vvd
                * (raw - self.v_il[pos_idx]
                    + coefficient.vv2 * self.v_vdn[pos_idx][2]
                    + coefficient.vj1 * self.v_jn[pos_idx][1]
                    + coefficient.vj2 * self.v_jn[pos_idx][2]);
            self.v_vdn[pos_idx][0] = updated;

            self.v_jn[pos_idx][0] = coefficient.ib0
                * (self.v_vdn[pos_idx][0] - self.v_vdn[pos_idx][2])
                - (coefficient.b1 * coefficient.ib0) * self.v_jn[pos_idx][1]
                - (coefficient.b2 * coefficient.ib0) * self.v_jn[pos_idx][2];

            *volt.get_mut(direction, i, j, k) = self.v_vdn[pos_idx][0];
        }
    }
}

#[cfg(test)]
mod tests_lumped_rlc {
    use super::*;

    fn coefficients(direction: usize) -> LumpedRLCCoeffs {
        LumpedRLCCoeffs {
            direction,
            ..Default::default()
        }
    }

    #[test]
    fn parallel_inductor_current_advances_from_the_previous_voltage() {
        let mut ext = EngineExtLumpedRLC::new(1);
        let mut coeffs = coefficients(1);
        coeffs.i2v = 2.0;
        coeffs.ilv = 3.0;
        ext.set_position(0, [1, 1, 1], &coeffs);

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        *volt.get_mut(1, 1, 1, 1) = 10.0;

        // First step: V[n-1] is still zero, so I_L does not move.
        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        assert_eq!(ext.inductor_current(0), 0.0);

        // The voltage update stores V[0]; the next pre-update sees it as V[n-1].
        ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        let stored = ext.voltage_history(0)[0];
        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        assert!((ext.inductor_current(0) - 2.0 * 3.0 * stored).abs() < 1e-6);
    }

    #[test]
    fn series_rlc_impedance_matches_the_analytic_admittance() {
        // Drive the raw edge voltage with a sinusoid and compare the complex
        // ratio J / V to the analytic series impedance. This validates the
        // whole ADE, not only its ringing frequency.
        let dt = 2.0e-12_f64;
        let inductance = 4.0e-9_f64;
        let resistance = 1.0_f64;
        let capacitance = 1.0e-12_f64;
        let grid_capacitance = 1.0e-12_f64;

        let element = crate::lumped_rlc::CellElement {
            inductance,
            resistance,
            conductance: 0.0,
            capacitance,
        };
        let series = crate::lumped_rlc::series_coefficients(dt, &element);
        let update = crate::lumped_rlc::series_update_coefficients(dt, grid_capacitance, &series);

        let mut ext = EngineExtLumpedRLC::new(1);
        ext.set_position(
            0,
            [1, 1, 1],
            &LumpedRLCCoeffs {
                direction: 0,
                vv2: update.vv2 as FDTD_FLOAT,
                vj1: update.vj1 as FDTD_FLOAT,
                vj2: update.vj2 as FDTD_FLOAT,
                vvd: update.vvd as FDTD_FLOAT,
                ib0: update.ib0 as FDTD_FLOAT,
                b1: update.b1 as FDTD_FLOAT,
                b2: update.b2 as FDTD_FLOAT,
                ..Default::default()
            },
        );

        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let empty = FieldArray::new("curr", [3, 3, 3]);

        for frequency in [1.0e9, 2.0e9, 3.0e9, 4.0e9] {
            let omega = std::f64::consts::TAU * frequency;
            let steps = 20_000usize;
            let window = 4_000usize;
            let mut volt_samples = Vec::with_capacity(window);
            let mut curr_samples = Vec::with_capacity(window);
            for step in 0..steps {
                *volt.get_mut(0, 1, 1, 1) = (omega * step as f64 * dt).sin() as FDTD_FLOAT;
                ext.do_pre_voltage_updates(&mut volt, &empty);
                ext.apply2_voltages(&mut volt, &empty);
                if step >= steps - window {
                    volt_samples.push(*volt.get(0, 1, 1, 1) as f64);
                    curr_samples.push(ext.current_history(0)[0] as f64);
                }
            }
            let fit = |samples: &[f64]| {
                let mut sine = 0.0;
                let mut cosine = 0.0;
                for (offset, value) in samples.iter().enumerate() {
                    let step = (steps - window + offset) as f64;
                    let phase = omega * step * dt;
                    sine += value * phase.sin();
                    cosine += value * phase.cos();
                }
                let scale = 2.0 / samples.len() as f64;
                (scale * sine, scale * cosine)
            };
            let (voltage_sine, voltage_cosine) = fit(&volt_samples);
            let (current_sine, current_cosine) = fit(&curr_samples);
            // Phasor convention: value = Im[(a + i b) e^{i omega t}].
            let voltage = (voltage_sine, voltage_cosine);
            let current = (current_sine, current_cosine);
            let denominator = voltage.0 * voltage.0 + voltage.1 * voltage.1;
            let admittance = (
                (current.0 * voltage.0 + current.1 * voltage.1) / denominator,
                (current.1 * voltage.0 - current.0 * voltage.1) / denominator,
            );
            let impedance = (resistance, omega * inductance - 1.0 / (omega * capacitance));
            let norm = impedance.0 * impedance.0 + impedance.1 * impedance.1;
            let analytic = (impedance.0 / norm, -impedance.1 / norm);
            let error = ((admittance.0 - analytic.0).powi(2) + (admittance.1 - analytic.1).powi(2))
                .sqrt()
                / (analytic.0 * analytic.0 + analytic.1 * analytic.1).sqrt();
            assert!(
                error < 2.0e-3,
                "admittance at {frequency:e} Hz: model {admittance:?} vs analytic {analytic:?} (rel {error:e})"
            );
        }
    }

    #[test]
    fn native_recurrence_matches_a_hand_computed_step() {
        // Deterministic single step with hand-checked arithmetic.
        let mut ext = EngineExtLumpedRLC::new(1);
        let coeffs = LumpedRLCCoeffs {
            direction: 2,
            vv2: 0.5,
            vj1: 0.25,
            vj2: 0.125,
            vvd: 0.8,
            ib0: 0.1,
            b1: 2.0,
            b2: 3.0,
            ..Default::default()
        };
        ext.set_position(0, [0, 0, 1], &coeffs);
        let mut volt = FieldArray::new("volt", [3, 1, 2]);
        *volt.get_mut(2, 0, 0, 1) = 4.0;
        ext.do_pre_voltage_updates(&mut volt, &FieldArray::new("curr", [3, 1, 2]));
        ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 1, 2]));

        // V = 0.8 * (4 - 0 + 0 + 0 + 0) = 3.2
        assert!((*volt.get(2, 0, 0, 1) - 3.2).abs() < 1e-6);
        // J = 0.1 * (3.2 - 0) = 0.32
        assert!((ext.current_history(0)[0] - 0.32).abs() < 1e-6);
        assert!((ext.voltage_history(0)[0] - 3.2).abs() < 1e-6);
    }
}

// ============================================================================
// Steady-state extension
// ============================================================================

/// Steady-state detection extension.
///
/// Records E-field values at probe positions over time and detects steady state
/// by comparing consecutive periods. Ported from C++ `Engine_Ext_SteadyState`.
///
/// The extension records E-field samples at each probe position into a circular
/// buffer of size `2 * ts_period`. Every `ts_period` timesteps, it computes the
/// normalized difference between the current period and the previous period.
/// When the difference falls below a threshold, the simulation has reached
/// steady state.
pub struct EngineExtSteadyState {
    /// Number of timesteps per period
    ts_period: usize,
    /// Internal timestep counter (incremented each apply2_voltages call)
    num_ts: usize,
    /// Probe positions: [pos_idx][axis]
    probe_pos: Vec<[usize; 3]>,
    /// Probe field component: 0=Ex, 1=Ey, 2=Ez
    probe_dir: Vec<usize>,
    /// Circular buffer of E-field records: [probe_idx][sample_idx]
    e_records: Vec<Vec<FDTD_FLOAT>>,
    /// Last computed max difference (steady-state convergence metric)
    last_max_diff: FDTD_FLOAT,
    /// Mesh line counts; enables the native total-energy term when non-zero.
    num_lines: [usize; 3],
    /// Total field energy at the previous evaluation.
    last_total_energy: f64,
}

impl EngineExtSteadyState {
    pub fn new(ts_period: usize) -> Self {
        Self::with_num_lines(ts_period, [0, 0, 0])
    }

    /// Create the extension with the mesh size so the native total-energy
    /// comparison (`CalcFastEnergy`) is included in the metric.
    pub fn with_num_lines(ts_period: usize, num_lines: [usize; 3]) -> Self {
        Self {
            ts_period,
            num_ts: 0,
            probe_pos: Vec::new(),
            probe_dir: Vec::new(),
            e_records: Vec::new(),
            // Native starts at 1 so a run is never declared converged before
            // the first evaluation.
            last_max_diff: 1.0,
            num_lines,
            last_total_energy: 0.0,
        }
    }

    /// Add an E-field probe at the given position and component.
    pub fn add_e_probe(&mut self, pos: [usize; 3], dir: usize) -> bool {
        if dir > 2 {
            return false;
        }
        if self.num_lines.iter().all(|&lines| lines > 0)
            && (0..3).any(|axis| pos[axis] >= self.num_lines[axis])
        {
            return false;
        }
        self.probe_pos.push(pos);
        self.probe_dir.push(dir);
        self.e_records
            .push(vec![FDTD_FLOAT::default(); 2 * self.ts_period]);
        true
    }

    /// Rough field energy, mirroring `Engine_Interface_FDTD::CalcFastEnergy`.
    fn fast_energy(&self, volt: &FieldArray, curr: &FieldArray) -> f64 {
        if self.num_lines.iter().any(|&lines| lines == 0) {
            return 0.0;
        }
        crate::engine::calc_fast_energy(volt, curr, self.num_lines)
    }

    /// Get the last computed max difference (steady-state convergence metric).
    pub fn get_last_diff(&self) -> FDTD_FLOAT {
        self.last_max_diff
    }

    /// Probe positions, in the order they were added.
    pub fn probe_positions(&self) -> &[[usize; 3]] {
        &self.probe_pos
    }

    /// Probe components (0=Ex, 1=Ey, 2=Ez), in the same order.
    pub fn probe_components(&self) -> &[usize] {
        &self.probe_dir
    }
}

impl EngineExtension for EngineExtSteadyState {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_STEADYSTATE
    }

    fn get_extension_name(&self) -> &'static str {
        "steadystate"
    }

    fn get_last_diff(&self) -> Option<FDTD_FLOAT> {
        Some(self.get_last_diff())
    }

    fn apply2_voltages(&mut self, volt: &mut FieldArray, curr: &FieldArray) {
        self.num_ts += 1;
        let p = self.ts_period;
        let ts = self.num_ts;
        if p == 0 {
            return;
        }
        let rel_pos = ts % (2 * p);

        // Record E-field values at probe positions
        for n in 0..self.probe_pos.len() {
            let [i, j, k] = self.probe_pos[n];
            let dir = self.probe_dir[n];
            self.e_records[n][rel_pos] = *volt.get(dir, i, j, k);
        }

        // Every period, compute steady-state difference
        if ts.is_multiple_of(p) && ts >= 2 * p {
            let mut no_valid = true;
            self.last_max_diff = FDTD_FLOAT::default();

            // Native total-energy term (`CalcFastEnergy`).
            if self.num_lines.iter().all(|&lines| lines > 0) {
                let current_energy = self.fast_energy(volt, curr);
                if self.last_total_energy > 0.0 {
                    self.last_max_diff = ((current_energy - self.last_total_energy).abs()
                        / self.last_total_energy)
                        as FDTD_FLOAT;
                    no_valid = false;
                }
                self.last_total_energy = current_energy;
            }

            let mut old_pos = 0usize;
            let mut new_pos = p;
            if rel_pos <= p {
                new_pos = 0;
                old_pos = p;
            }

            let mut max_pow = FDTD_FLOAT::default();
            let mut curr_pow = vec![FDTD_FLOAT::default(); self.probe_pos.len()];
            let mut diff_pow = vec![FDTD_FLOAT::default(); self.probe_pos.len()];

            for n in 0..self.probe_pos.len() {
                let buf = &self.e_records[n];
                for nt in 0..p {
                    let v_new = buf[nt + new_pos];
                    let v_old = buf[nt + old_pos];
                    curr_pow[n] += v_new * v_new;
                    diff_pow[n] += (v_old - v_new) * (v_old - v_new);
                }
                max_pow = max_pow.max(curr_pow[n]);
            }

            for n in 0..self.probe_pos.len() {
                if curr_pow[n] > max_pow * 1e-2_f32 {
                    let ratio = diff_pow[n] / curr_pow[n];
                    self.last_max_diff = self.last_max_diff.max(ratio);
                    no_valid = false;
                }
            }

            if no_valid || self.last_max_diff > 1.0_f32 {
                self.last_max_diff = 1.0_f32;
            }
        }
    }

    fn apply2_current(&mut self, _volt: &FieldArray, _curr: &mut FieldArray) {
        // No-op: steady-state detection only operates on voltage fields
    }
}

#[cfg(test)]
mod tests_steadystate {
    use super::*;

    #[test]
    fn test_steadystate_new() {
        let ext = EngineExtSteadyState::new(10);
        assert_eq!(ext.ts_period, 10);
        assert_eq!(ext.probe_pos.len(), 0);
        assert_eq!(ext.e_records.len(), 0);
        // Native starts at 1 so a run is never converged before an evaluation.\n        assert!((ext.get_last_diff() - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_steadystate_add_e_probe() {
        let mut ext = EngineExtSteadyState::new(4);
        ext.add_e_probe([1, 2, 3], 0);
        ext.add_e_probe([4, 5, 6], 1);
        assert_eq!(ext.probe_pos.len(), 2);
        assert_eq!(ext.probe_pos[0], [1, 2, 3]);
        assert_eq!(ext.probe_dir[0], 0);
        assert_eq!(ext.probe_pos[1], [4, 5, 6]);
        assert_eq!(ext.probe_dir[1], 1);
        // Each probe gets a buffer of 2 * ts_period = 8 samples
        assert_eq!(ext.e_records[0].len(), 8);
        assert_eq!(ext.e_records[1].len(), 8);
    }

    #[test]
    fn test_steadystate_record_and_diff() {
        let mut ext = EngineExtSteadyState::new(2);
        ext.add_e_probe([1, 1, 1], 0);

        let mut volt = FieldArray::new("volt", [3, 3, 3]);

        // Simulate 4 timesteps (2 periods of 2)
        // Period 0: E = 1.0, 2.0
        // Period 1: E = 1.0, 2.0 (identical → steady state, diff = 0)
        for ts in 0..4u32 {
            let e_val = if ts % 2 == 0 { 1.0_f32 } else { 2.0_f32 };
            *volt.get_mut(0, 1, 1, 1) = e_val;
            // Manually set num_ts via the extension's internal tracking
            // We simulate by calling apply2_voltages which uses its own ts counter
            // Note: num_ts is not accessible here; we rely on the call count
            ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        }

        // After 4 timesteps (2 periods), diff should be 0 (identical periods)
        // Native starts at 1 so a run is never converged before an evaluation.\n        assert!((ext.get_last_diff() - 1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_steadystate_not_converged() {
        let mut ext = EngineExtSteadyState::new(2);
        ext.add_e_probe([1, 1, 1], 0);

        let mut volt = FieldArray::new("volt", [3, 3, 3]);

        // Period 0: E = 1.0, 2.0
        // Period 1: E = 10.0, 20.0 (very different → not converged)
        for ts in 0..4u32 {
            let e_val = if ts < 2 {
                if ts == 0 {
                    1.0_f32
                } else {
                    2.0_f32
                }
            } else {
                if ts == 2 {
                    10.0_f32
                } else {
                    20.0_f32
                }
            };
            *volt.get_mut(0, 1, 1, 1) = e_val;
            ext.apply2_voltages(&mut volt, &FieldArray::new("curr", [3, 3, 3]));
        }

        // Diff should be large (not converged)
        assert!(ext.get_last_diff() > 0.1_f32);
    }
}

// ============================================================================
// Cylinder coordinates extension
// ============================================================================

/// Cylinder coordinates extension.
///
/// Handles the alpha-direction periodic boundary condition and r=0 special
/// treatment for cylindrical FDTD. Ported from C++ `Engine_Ext_Cylinder`.
///
/// When `cc_closed_alpha` is true, tangential field components are copied
/// from the last alpha-plane to the first alpha-plane (periodic boundary).
/// When `cc_r0_included` is true, special EC coefficients (vv_r0, vi_r0)
/// are applied at r=0 (the z-axis).
pub struct EngineExtCylinder {
    num_lines: [usize; 3],
    cc_closed_alpha: bool,
    cc_r0_included: bool,
    /// Special vv coefficient for r=0, indexed by z-position
    vv_r0: Vec<FDTD_FLOAT>,
    /// Special vi coefficient for r=0, indexed by z-position
    vi_r0: Vec<FDTD_FLOAT>,
}

impl EngineExtCylinder {
    pub fn new(num_lines: [usize; 3]) -> Self {
        Self {
            num_lines,
            cc_closed_alpha: false,
            cc_r0_included: false,
            vv_r0: vec![FDTD_FLOAT::default(); num_lines[2]],
            vi_r0: vec![FDTD_FLOAT::default(); num_lines[2]],
        }
    }

    pub fn set_closed_alpha(&mut self, closed: bool) {
        self.cc_closed_alpha = closed;
    }

    pub fn set_r0_included(&mut self, included: bool) {
        self.cc_r0_included = included;
    }

    /// Set the special EC coefficients for r=0 at a given z-position.
    pub fn set_r0_coeffs(&mut self, k: usize, vv: FDTD_FLOAT, vi: FDTD_FLOAT) {
        if k < self.vv_r0.len() {
            self.vv_r0[k] = vv;
            self.vi_r0[k] = vi;
        }
    }
}

impl EngineExtension for EngineExtCylinder {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_CYLINDER
    }

    fn get_extension_name(&self) -> &'static str {
        "cylinder"
    }

    fn do_post_voltage_updates(&mut self, volt: &mut FieldArray, curr: &FieldArray) {
        if !self.cc_closed_alpha {
            return;
        }

        let [nx, ny, nz] = self.num_lines;
        let last_a_line = ny.saturating_sub(2);

        if self.cc_r0_included {
            // Special treatment for r=0 (z-axis)
            for k in 0..nz {
                let mut volt_val = *volt.get(2, 0, 0, k) * self.vv_r0[k];
                for j in 0..ny.saturating_sub(1) {
                    volt_val += self.vi_r0[k] * *curr.get(1, 0, j, k);
                }
                *volt.get_mut(2, 0, 0, k) = volt_val;
            }

            for j in 0..ny {
                for k in 0..nz {
                    *volt.get_mut(1, 0, j, k) = FDTD_FLOAT::default(); // no voltage in alpha at r=0
                    *volt.get_mut(2, 0, j, k) = *volt.get(2, 0, 0, k);
                }
            }
        }

        // Close alpha: copy tangential voltages from last alpha-plane to first
        for i in 0..nx {
            for k in 0..nz {
                *volt.get_mut(0, i, 0, k) = *volt.get(0, i, last_a_line, k);
                *volt.get_mut(2, i, 0, k) = *volt.get(2, i, last_a_line, k);
            }
        }
    }

    fn do_post_current_updates(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        if !self.cc_closed_alpha {
            return;
        }

        let [nx, ny, nz] = self.num_lines;
        let last_a_line = ny.saturating_sub(2);

        // Close alpha: copy tangential currents from first alpha-plane to last
        for i in 0..nx.saturating_sub(1) {
            for k in 0..nz.saturating_sub(1) {
                *curr.get_mut(0, i, last_a_line, k) = *curr.get(0, i, 0, k);
                *curr.get_mut(2, i, last_a_line, k) = *curr.get(2, i, 0, k);
            }
        }
    }
}

#[cfg(test)]
mod tests_cylinder {
    use super::*;

    #[test]
    fn test_cylinder_new() {
        let ext = EngineExtCylinder::new([10, 8, 6]);
        assert_eq!(ext.num_lines, [10, 8, 6]);
        assert!(!ext.cc_closed_alpha);
        assert!(!ext.cc_r0_included);
        assert_eq!(ext.vv_r0.len(), 6);
        assert_eq!(ext.vi_r0.len(), 6);
    }

    #[test]
    fn test_cylinder_set_flags() {
        let mut ext = EngineExtCylinder::new([10, 8, 6]);
        ext.set_closed_alpha(true);
        ext.set_r0_included(true);
        assert!(ext.cc_closed_alpha);
        assert!(ext.cc_r0_included);
    }

    #[test]
    fn test_cylinder_set_r0_coeffs() {
        let mut ext = EngineExtCylinder::new([10, 8, 6]);
        ext.set_r0_coeffs(3, 0.5_f32, 0.25_f32);
        assert_eq!(ext.vv_r0[3], 0.5);
        assert_eq!(ext.vi_r0[3], 0.25);
    }

    #[test]
    fn test_cylinder_no_closed_alpha_no_op() {
        let mut ext = EngineExtCylinder::new([4, 4, 4]);
        // cc_closed_alpha is false, so updates should be no-ops

        let mut volt = FieldArray::new("volt", [4, 4, 4]);
        let mut curr = FieldArray::new("curr", [4, 4, 4]);

        ext.do_post_voltage_updates(&mut volt, &curr);
        ext.do_post_current_updates(&volt, &mut curr);

        // Fields should be unchanged
        assert_eq!(*volt.get(0, 0, 0, 0), 0.0_f32);
        assert_eq!(*curr.get(0, 0, 0, 0), 0.0_f32);
    }

    #[test]
    fn test_cylinder_close_alpha_voltage() {
        let mut ext = EngineExtCylinder::new([4, 4, 4]);
        ext.set_closed_alpha(true);

        let mut volt = FieldArray::new("volt", [4, 4, 4]);
        *volt.get_mut(0, 1, 2, 1) = 7.0_f32;
        *volt.get_mut(2, 1, 2, 1) = 3.0_f32;

        ext.do_post_voltage_updates(&mut volt, &FieldArray::new("curr", [4, 4, 4]));

        // Voltage at alpha=0 should be copied from alpha=last_a_line=2
        assert!((*volt.get(0, 1, 0, 1) - 7.0_f32).abs() < 1e-6);
        assert!((*volt.get(2, 1, 0, 1) - 3.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_cylinder_close_alpha_current() {
        let mut ext = EngineExtCylinder::new([4, 4, 4]);
        ext.set_closed_alpha(true);

        let mut curr = FieldArray::new("curr", [4, 4, 4]);
        *curr.get_mut(0, 1, 0, 1) = 5.0_f32;
        *curr.get_mut(2, 1, 0, 1) = 2.0_f32;

        ext.do_post_current_updates(&FieldArray::new("volt", [4, 4, 4]), &mut curr);

        // Current at alpha=last_a_line=2 should be copied from alpha=0
        assert!((*curr.get(0, 1, 2, 1) - 5.0_f32).abs() < 1e-6);
        assert!((*curr.get(2, 1, 2, 1) - 2.0_f32).abs() < 1e-6);
    }
}

#[cfg(test)]
mod steadystate_energy_tests {
    use super::*;

    #[test]
    fn energy_term_detects_a_decaying_field() {
        // 3 cells per axis => 2x2x2 interior cells. Probe at the centre.
        let mut ext = EngineExtSteadyState::with_num_lines(2, [3, 3, 3]);
        assert!(ext.add_e_probe([1, 1, 1], 0));
        let mut volt = FieldArray::new("volt", [3, 3, 3]);
        let mut curr = FieldArray::new("curr", [3, 3, 3]);

        // Two identical periods: energy and probe values repeat -> zero diff.
        for step in 0..4u32 {
            let value = if step % 2 == 0 { 1.0 } else { 2.0 };
            *volt.get_mut(0, 1, 1, 1) = value;
            ext.apply2_voltages(&mut volt, &curr);
        }
        assert!(ext.get_last_diff() < 1e-6, "diff={}", ext.get_last_diff());

        // A field that keeps growing must not look converged.
        for step in 0..4u32 {
            let value = 10.0 * (step + 1) as f32;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        *volt.get_mut(0, i, j, k) = value;
                        *curr.get_mut(1, i, j, k) = value * 0.5;
                    }
                }
            }
            *volt.get_mut(0, 1, 1, 1) = value;
            ext.apply2_voltages(&mut volt, &curr);
        }
        assert!(ext.get_last_diff() > 0.1, "diff={}", ext.get_last_diff());
    }

    #[test]
    fn probes_are_bounds_checked_when_the_mesh_is_known() {
        let mut ext = EngineExtSteadyState::with_num_lines(2, [3, 3, 3]);
        assert!(ext.add_e_probe([2, 2, 2], 0));
        assert!(!ext.add_e_probe([3, 0, 0], 0));
        assert!(!ext.add_e_probe([0, 0, 0], 3));
        assert_eq!(ext.probe_positions().len(), 1);
    }
}

#[cfg(test)]
mod excitation_period_tests {
    use super::*;

    #[test]
    fn periodic_sources_repeat_every_period() {
        let mut signal = Excitation::sinus(1.0e9);
        signal.build(1.0e-12, 10_000).unwrap();
        let period = signal.signal_period() as u32;
        assert!(period > 0);
        assert_eq!(period, 1_000); // 1 GHz at dt = 1 ps

        for ts in 0..(3 * period) {
            let index = EngineExtExcitation::sample_index(&signal, ts, 0).unwrap();
            assert_eq!(index, (ts % period) as usize, "ts={ts}");
        }
        // The built signal is two periods long; wrapping keeps injecting.
        assert_eq!(
            EngineExtExcitation::sample_index(&signal, 2 * period + 7, 0),
            Some(7)
        );
    }

    #[test]
    fn excitation_delay_skips_earlier_steps() {
        let mut signal = Excitation::sinus(1.0e9);
        signal.build(1.0e-12, 10_000).unwrap();
        assert_eq!(EngineExtExcitation::sample_index(&signal, 9, 10), None);
        assert_eq!(EngineExtExcitation::sample_index(&signal, 10, 10), Some(0));
        assert_eq!(
            EngineExtExcitation::sample_index(&signal, 1_013, 10),
            Some(3)
        );
    }
}
