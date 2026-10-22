//! Physics extensions: PML, Mur ABC, excitation, dispersive materials, etc.
//! Mirrors `FDTD/extensions/` in the C++ codebase.

use crate::array_nijk::FieldArray;
use crate::constants::FDTD_FLOAT;
use crate::excitation::{Excitation, ExcitationType};

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
    pub component: usize,   // 0=Ex, 1=Ey, 2=Ez
    pub pos: [usize; 3],    // position in the field array
    pub delay: u32,         // start delay in timesteps
    pub signal: Excitation, // the time-domain signal
}

/// A single current excitation applied to a field component at a position.
pub struct CurrentExcitation {
    pub component: usize,   // 0=Hx, 1=Hy, 2=Hz
    pub pos: [usize; 3],    // position in the field array
    pub delay: u32,         // start delay in timesteps
    pub signal: Excitation, // the time-domain signal
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
        signal: Excitation,
    ) {
        self.voltage_excitations.push(VoltageExcitation {
            component: comp,
            pos,
            delay,
            signal,
        });
    }

    /// Add a current excitation.
    pub fn add_current_excitation(
        &mut self,
        comp: usize,
        pos: [usize; 3],
        delay: u32,
        signal: Excitation,
    ) {
        self.current_excitations.push(CurrentExcitation {
            component: comp,
            pos,
            delay,
            signal,
        });
    }

    /// Compute the signal value at the given timestep.
    fn signal_value(signal: &Excitation, ts: u32) -> f64 {
        let t = ts as f64;
        match signal.exc_type {
            ExcitationType::Gauss => {
                // Gaussian pulse: exp(-((t - t0)/sigma)^2)
                let fc = signal.fc;
                let sigma = 1.0 / (2.0 * std::f64::consts::PI * fc);
                let t0 = 3.0 * sigma;
                let dt = t - t0;
                (-((dt / sigma).powi(2))).exp()
            }
            ExcitationType::Sinus => (2.0 * std::f64::consts::PI * signal.f0 * t).sin(),
            ExcitationType::Dirac => {
                if ts == 0 {
                    1.0
                } else {
                    0.0
                }
            }
            ExcitationType::Step => {
                if t >= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ExcitationType::Custom => 0.0,
        }
    }
}

impl Default for EngineExtExcitation {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineExtension for EngineExtExcitation {
    fn get_priority(&self) -> i32 {
        ENG_EXT_PRIO_EXCITATION
    }

    fn get_extension_name(&self) -> &'static str {
        "Excitation"
    }

    fn do_pre_voltage_updates(&mut self, volt: &mut FieldArray, _curr: &FieldArray) {
        self.current_ts += 1;
        let ts = self.current_ts;
        for exc in &self.voltage_excitations {
            if ts < exc.delay {
                continue;
            }
            let rel_ts = ts - exc.delay;
            let val = Self::signal_value(&exc.signal, rel_ts) as FDTD_FLOAT;
            let cur = *volt.get(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]);
            *volt.get_mut(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]) = cur + val;
        }
    }

    fn do_pre_current_updates(&mut self, _volt: &FieldArray, curr: &mut FieldArray) {
        let ts = self.current_ts;
        for exc in &self.current_excitations {
            if ts < exc.delay {
                continue;
            }
            let rel_ts = ts - exc.delay;
            let val = Self::signal_value(&exc.signal, rel_ts) as FDTD_FLOAT;
            let cur = *curr.get(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]);
            *curr.get_mut(exc.component, exc.pos[0], exc.pos[1], exc.pos[2]) = cur + val;
        }
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
        ext.add_voltage_excitation(0, [5, 5, 5], 10, signal);
        assert_eq!(ext.voltage_excitations.len(), 1);
        assert_eq!(ext.voltage_excitations[0].component, 0);
        assert_eq!(ext.voltage_excitations[0].pos, [5, 5, 5]);
        assert_eq!(ext.voltage_excitations[0].delay, 10);
    }

    #[test]
    fn test_signal_value_gauss() {
        let signal = Excitation::gauss(1.0e9, 5.0e8);
        let val = EngineExtExcitation::signal_value(&signal, 0);
        assert!(val > 0.0);
        assert!(val <= 1.0);
    }

    #[test]
    fn test_signal_value_dirac() {
        let signal = Excitation::dirac(1.0e9);
        assert_eq!(EngineExtExcitation::signal_value(&signal, 0), 1.0);
        assert_eq!(EngineExtExcitation::signal_value(&signal, 1), 0.0);
    }
}

// ============================================================================
// Mur ABC extension
// ============================================================================

/// Mur Absorbing Boundary Condition engine extension.
/// Implements a first-order Mur ABC on the specified boundary faces.
/// Mirrors C++ `Engine_Ext_Mur_ABC`.
pub struct EngineExtMurABC {
    /// Boundary face indices (0=x-min, 1=x-max, 2=y-min, 3=y-max, 4=z-min, 5=z-max) that are active.
    pub active_faces: [bool; 6],
    /// Field history for each face: [face][position_index] -> previous voltage values.
    /// Used for the Mur ABC time-delayed update.
    pub volt_history: Vec<Vec<FDTD_FLOAT>>,
    pub current_ts: u32,
    pub start_ts: u32,
}

impl EngineExtMurABC {
    pub fn new() -> Self {
        Self {
            active_faces: [false; 6],
            volt_history: Vec::new(),
            current_ts: 0,
            start_ts: 0,
        }
    }

    /// Activate a boundary face for Mur ABC.
    pub fn activate_face(&mut self, face: i32) {
        if (0..=5).contains(&face) {
            self.active_faces[face as usize] = true;
        }
    }

    /// Set the timestep at which the Mur ABC becomes active.
    pub fn set_start_ts(&mut self, ts: u32) {
        self.start_ts = ts;
    }

    fn is_active(&self) -> bool {
        self.current_ts >= self.start_ts
    }
}

impl Default for EngineExtMurABC {
    fn default() -> Self {
        Self::new()
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
        let nx = volt.ni();
        let ny = volt.nj();
        let nz = volt.nk();

        // First-order Mur ABC: E(n+1) at boundary = E(n) at boundary + (c*dt/dx - 1)/(c*dt/dx + 1) * (E(n+1) at neighbor - E(n) at neighbor)
        // Simplified: copy the neighbor value inward (first-order approximation)
        for comp in 0..3 {
            // x-min face (face 0)
            if self.active_faces[0] {
                for j in 0..ny {
                    for k in 0..nz {
                        let val = *volt.get(comp, 1, j, k);
                        *volt.get_mut(comp, 0, j, k) = val;
                    }
                }
            }
            // x-max face (face 1)
            if self.active_faces[1] {
                for j in 0..ny {
                    for k in 0..nz {
                        let val = *volt.get(comp, nx - 2, j, k);
                        *volt.get_mut(comp, nx - 1, j, k) = val;
                    }
                }
            }
            // y-min face (face 2)
            if self.active_faces[2] {
                for i in 0..nx {
                    for k in 0..nz {
                        let val = *volt.get(comp, i, 1, k);
                        *volt.get_mut(comp, i, 0, k) = val;
                    }
                }
            }
            // y-max face (face 3)
            if self.active_faces[3] {
                for i in 0..nx {
                    for k in 0..nz {
                        let val = *volt.get(comp, i, ny - 2, k);
                        *volt.get_mut(comp, i, ny - 1, k) = val;
                    }
                }
            }
            // z-min face (face 4)
            if self.active_faces[4] {
                for i in 0..nx {
                    for j in 0..ny {
                        let val = *volt.get(comp, i, j, 1);
                        *volt.get_mut(comp, i, j, 0) = val;
                    }
                }
            }
            // z-max face (face 5)
            if self.active_faces[5] {
                for i in 0..nx {
                    for j in 0..ny {
                        let val = *volt.get(comp, i, j, nz - 2);
                        *volt.get_mut(comp, i, j, nz - 1) = val;
                    }
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
    }

    #[test]
    fn test_activate_face() {
        let mut ext = EngineExtMurABC::new();
        ext.activate_face(0);
        ext.activate_face(5);
        assert!(ext.active_faces[0]);
        assert!(ext.active_faces[5]);
        assert!(!ext.active_faces[1]);
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
                        *self.volt_flux[comp].get_mut(0, i, j, k) = *volt.get(comp, gi, gj, gk);
                        *volt.get_mut(comp, gi, gj, gk) = f_help;
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
                        *self.curr_flux[comp].get_mut(0, i, j, k) = *curr.get(comp, gi, gj, gk);
                        *curr.get_mut(comp, gi, gj, gk) = f_help;
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
    fn test_upml_creation() {
        let ext = EngineExtUPML::new([0, 0, 0], [10, 10, 10]);
        assert_eq!(ext.get_priority(), ENG_EXT_PRIO_UPML);
        assert_eq!(ext.get_extension_name(), "UPML");
        assert_eq!(ext.vv.len(), 3);
        assert_eq!(ext.volt_flux.len(), 3);
    }
}
