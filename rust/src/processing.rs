//! Processing: post-processing (probes, dumps, SAR, NF2FF interface).
//! Mirrors `Common/processing.h/cpp` and subclasses in the C++ codebase.

/// Base trait for processing objects that sample the running simulation.
pub trait Processing {
    fn process(&mut self);
    fn get_name(&self) -> String;
}

/// A field probe that samples E/H fields at a point during the simulation.
pub struct FieldProbe {
    pub name: String,
    pub position: [f64; 3],
}

impl FieldProbe {
    pub fn new(name: &str, position: [f64; 3]) -> Self {
        Self {
            name: name.to_string(),
            position,
        }
    }
}

impl Processing for FieldProbe {
    fn process(&mut self) {
        // Placeholder: sample fields at position
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_probe() {
        let mut probe = FieldProbe::new("probe1", [0.0, 0.0, 0.0]);
        assert_eq!(probe.get_name(), "probe1");
        probe.process(); // should not panic
    }
}
