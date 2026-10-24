//! Operator-side extension registry.
//!
//! Mirrors `Operator_Extension` and `Operator::AddExtension` +
//! `BuildExtension` in the native code: every extension declares a priority,
//! the registry sorts once, and each active extension builds its engine-side
//! counterpart from the finished operator.
//!
//! Physics that needs to pre-compute coefficients or rewrite the operator
//! belongs here, not in `Operator` itself.

use crate::excitation::Excitation;
use crate::extensions::{
    EngineExtension, ENG_EXT_PRIO_DEFAULT, ENG_EXT_PRIO_EXCITATION,
    ENG_EXT_PRIO_STEADYSTATE, ENG_EXT_PRIO_UPML,
};
use crate::operator::Operator;
use std::sync::Arc;

/// One operator-side extension.
pub trait OperatorExtension {
    /// Stable name used in logs and tests.
    fn name(&self) -> &'static str;

    /// Execution priority; lower values build first.
    fn priority(&self) -> i32;

    /// Extensions that have nothing to do can disable themselves.
    fn is_active(&self) -> bool {
        true
    }

    /// Build the engine extension(s) from the finished operator.
    ///
    /// Returning an empty vector is allowed (for example a Mur extension with
    /// no active faces). Errors abort the setup instead of silently dropping
    /// physics.
    fn build(
        &mut self,
        operator: &mut Operator,
    ) -> Result<Vec<Box<dyn EngineExtension>>, String>;
}

/// Ordered collection of operator extensions.
#[derive(Default)]
pub struct OperatorExtensionRegistry {
    extensions: Vec<Box<dyn OperatorExtension>>,
}

impl OperatorExtensionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, extension: Box<dyn OperatorExtension>) {
        self.extensions.push(extension);
    }

    pub fn len(&self) -> usize {
        self.extensions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }

    /// Names in registration order.
    pub fn names(&self) -> Vec<&'static str> {
        self.extensions
            .iter()
            .map(|extension| extension.name())
            .collect()
    }

    /// Stable sort by ascending priority; equal priorities keep registration
    /// order.
    pub fn sort(&mut self) {
        self.extensions.sort_by_key(|extension| extension.priority());
    }

    /// Sort, then build every active extension in order.
    pub fn build(
        &mut self,
        operator: &mut Operator,
    ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
        self.sort();
        let mut engine_extensions = Vec::new();
        for extension in self.extensions.iter_mut() {
            if !extension.is_active() {
                continue;
            }
            engine_extensions.extend(extension.build(operator)?);
        }
        Ok(engine_extensions)
    }
}

impl std::fmt::Debug for OperatorExtensionRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OperatorExtensionRegistry")
            .field("names", &self.names())
            .finish()
    }
}

/// First-order Mur absorbing boundaries (`Operator_Ext_Mur_ABC`).
pub struct MurOperatorExtension {
    pub phase_velocity: [f64; 6],
}

impl OperatorExtension for MurOperatorExtension {
    fn name(&self) -> &'static str {
        "Mur"
    }

    fn priority(&self) -> i32 {
        ENG_EXT_PRIO_DEFAULT
    }

    fn build(
        &mut self,
        operator: &mut Operator,
    ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
        let extension = operator.build_mur_extension(self.phase_velocity);
        if extension.faces.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![Box::new(extension)])
        }
    }
}

/// UPML layers (`Operator_Ext_UPML`, one extension per region).
pub struct UpmlOperatorExtension {
    pub pml_size: [u32; 6],
}

impl OperatorExtension for UpmlOperatorExtension {
    fn name(&self) -> &'static str {
        "UPML"
    }

    fn priority(&self) -> i32 {
        ENG_EXT_PRIO_UPML
    }

    fn is_active(&self) -> bool {
        self.pml_size.iter().any(|&size| size > 0)
    }

    fn build(
        &mut self,
        operator: &mut Operator,
    ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
        let regions = operator.build_upml_extensions(self.pml_size)?;
        Ok(regions
            .into_iter()
            .map(|region| Box::new(region) as Box<dyn EngineExtension>)
            .collect())
    }
}

/// Steady-state detection for periodic excitations
/// (`Operator_Ext_SteadyState`).
pub struct SteadyStateOperatorExtension {
    pub period_ts: u32,
}

impl OperatorExtension for SteadyStateOperatorExtension {
    fn name(&self) -> &'static str {
        "SteadyState"
    }

    fn priority(&self) -> i32 {
        ENG_EXT_PRIO_STEADYSTATE
    }

    fn is_active(&self) -> bool {
        self.period_ts > 0
    }

    fn build(
        &mut self,
        operator: &mut Operator,
    ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
        Ok(operator
            .build_steady_state_extension(self.period_ts)
            .into_iter()
            .map(|extension| Box::new(extension) as Box<dyn EngineExtension>)
            .collect())
    }
}

/// Time-domain sources (`Operator_Ext_Excitation`).
pub struct ExcitationOperatorExtension {
    pub signal: Arc<Excitation>,
}

impl OperatorExtension for ExcitationOperatorExtension {
    fn name(&self) -> &'static str {
        "Excitation"
    }

    fn priority(&self) -> i32 {
        ENG_EXT_PRIO_EXCITATION
    }

    fn build(
        &mut self,
        operator: &mut Operator,
    ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
        let extension = operator.build_excitation_extension(self.signal.as_ref());
        if extension.voltage_excitations.is_empty() && extension.current_excitations.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![Box::new(extension)])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_nijk::FieldArray;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockExtension {
        name: &'static str,
        priority: i32,
        active: bool,
        calls: Arc<AtomicUsize>,
        result: Result<(), String>,
    }

    impl OperatorExtension for MockExtension {
        fn name(&self) -> &'static str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn is_active(&self) -> bool {
            self.active
        }

        fn build(
            &mut self,
            _operator: &mut Operator,
        ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.result.clone()?;
            Ok(Vec::new())
        }
    }

    struct NoopEngineExtension;

    impl EngineExtension for NoopEngineExtension {
        fn get_priority(&self) -> i32 {
            0
        }

        fn get_extension_name(&self) -> &'static str {
            "noop"
        }

        fn do_pre_voltage_updates(&mut self, _volt: &mut FieldArray, _curr: &FieldArray) {}
        fn do_post_voltage_updates(&mut self, _volt: &mut FieldArray, _curr: &FieldArray) {}
        fn do_pre_current_updates(&mut self, _volt: &FieldArray, _curr: &mut FieldArray) {}
        fn do_post_current_updates(&mut self, _volt: &FieldArray, _curr: &mut FieldArray) {}
    }

    #[test]
    fn extensions_build_in_priority_order() {
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        struct Ordered {
            name: &'static str,
            priority: i32,
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl OperatorExtension for Ordered {
            fn name(&self) -> &'static str {
                self.name
            }
            fn priority(&self) -> i32 {
                self.priority
            }
            fn build(
                &mut self,
                _operator: &mut Operator,
            ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
                self.order.lock().unwrap().push(self.name);
                Ok(Vec::new())
            }
        }

        let mut registry = OperatorExtensionRegistry::new();
        registry.add(Box::new(Ordered {
            name: "late",
            priority: 100,
            order: order.clone(),
        }));
        registry.add(Box::new(Ordered {
            name: "early",
            priority: -5,
            order: order.clone(),
        }));
        registry.add(Box::new(Ordered {
            name: "middle",
            priority: 10,
            order: order.clone(),
        }));
        let mut operator = Operator::new();
        registry.build(&mut operator).expect("build");

        assert_eq!(*order.lock().unwrap(), vec!["early", "middle", "late"]);
        // `build` sorts the registry in place.
        assert_eq!(registry.names(), vec!["early", "middle", "late"]);
    }

    #[test]
    fn inactive_extensions_are_skipped_and_errors_propagate() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = OperatorExtensionRegistry::new();
        registry.add(Box::new(MockExtension {
            name: "off",
            priority: 1,
            active: false,
            calls: calls.clone(),
            result: Ok(()),
        }));
        registry.add(Box::new(MockExtension {
            name: "ok",
            priority: 2,
            active: true,
            calls: calls.clone(),
            result: Ok(()),
        }));
        let mut operator = Operator::new();
        assert!(registry.build(&mut operator).unwrap().is_empty());
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let mut failing = OperatorExtensionRegistry::new();
        failing.add(Box::new(MockExtension {
            name: "bad",
            priority: 0,
            active: true,
            calls: calls.clone(),
            result: Err("boom".to_string()),
        }));
        let error = failing
            .build(&mut operator)
            .err()
            .expect("the extension must fail");
        assert_eq!(error, "boom");
    }

    #[test]
    fn engine_extensions_are_forwarded_in_order() {
        struct Producing {
            priority: i32,
        }
        impl OperatorExtension for Producing {
            fn name(&self) -> &'static str {
                "producing"
            }
            fn priority(&self) -> i32 {
                self.priority
            }
            fn build(
                &mut self,
                _operator: &mut Operator,
            ) -> Result<Vec<Box<dyn EngineExtension>>, String> {
                Ok(vec![Box::new(NoopEngineExtension)])
            }
        }
        let mut registry = OperatorExtensionRegistry::new();
        registry.add(Box::new(Producing { priority: 20 }));
        registry.add(Box::new(Producing { priority: 10 }));
        let mut operator = Operator::new();
        let built = registry.build(&mut operator).unwrap();
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].get_extension_name(), "noop");
    }

    #[test]
    fn upml_is_inactive_without_sizes() {
        let extension = UpmlOperatorExtension { pml_size: [0; 6] };
        assert!(!extension.is_active());
        let extension = UpmlOperatorExtension { pml_size: [4; 6] };
        assert!(extension.is_active());
        assert_eq!(extension.priority(), ENG_EXT_PRIO_UPML);
    }
}
