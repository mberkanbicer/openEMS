//! Physics extensions: PML, Mur ABC, excitation, dispersive materials, etc.
//! Mirrors `FDTD/extensions/` in the C++ codebase.

// Extension priority constants (mirroring C++ #defines)
pub const ENG_EXT_PRIO_DEFAULT: i32 = 0;
pub const ENG_EXT_PRIO_STEADYSTATE: i32 = 2_000_000;
pub const ENG_EXT_PRIO_UPML: i32 = 1_000_000;
pub const ENG_EXT_PRIO_CYLINDER: i32 = 100_000;
pub const ENG_EXT_PRIO_TFSF: i32 = 50_000;
pub const ENG_EXT_PRIO_EXCITATION: i32 = -1000;
pub const ENG_EXT_PRIO_CYLINDERMULTIGRID: i32 = -3000;

/// Base trait for engine extensions.
pub trait EngineExtension {
    fn get_priority(&self) -> i32;
    fn get_extension_name(&self) -> &'static str;
    fn do_pre_voltage_updates(&mut self) {}
    fn do_post_voltage_updates(&mut self) {}
    fn apply2_voltages(&mut self) {}
    fn do_pre_current_updates(&mut self) {}
    fn do_post_current_updates(&mut self) {}
    fn apply2_current(&mut self) {}
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
