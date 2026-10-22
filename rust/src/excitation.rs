//! Excitation: time-domain source signals.
//! Mirrors `FDTD/excitation.h/cpp` in the C++ codebase.

/// Types of excitation signals supported by the solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcitationType {
    Gauss,
    Sinus,
    Dirac,
    Step,
    Custom,
}

/// An excitation time signal.
pub struct Excitation {
    pub exc_type: ExcitationType,
    pub f0: f64,
    pub fc: f64,
    pub f_max: f64,
}

impl Excitation {
    /// Create a Gaussian excitation.
    pub fn gauss(f0: f64, fc: f64) -> Self {
        Self {
            exc_type: ExcitationType::Gauss,
            f0,
            fc,
            f_max: 0.0,
        }
    }

    /// Create a sinusoidal excitation.
    pub fn sinus(f0: f64) -> Self {
        Self {
            exc_type: ExcitationType::Sinus,
            f0,
            fc: 0.0,
            f_max: 0.0,
        }
    }

    /// Create a Dirac excitation.
    pub fn dirac(f_max: f64) -> Self {
        Self {
            exc_type: ExcitationType::Dirac,
            f0: 0.0,
            fc: 0.0,
            f_max,
        }
    }

    /// Create a step excitation.
    pub fn step(f_max: f64) -> Self {
        Self {
            exc_type: ExcitationType::Step,
            f0: 0.0,
            fc: 0.0,
            f_max,
        }
    }
}

impl Default for Excitation {
    fn default() -> Self {
        Self {
            exc_type: ExcitationType::Gauss,
            f0: 1.0e9,
            fc: 5.0e8,
            f_max: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_excitation() {
        let exc = Excitation::gauss(1.0e9, 5.0e8);
        assert_eq!(exc.exc_type, ExcitationType::Gauss);
        assert_eq!(exc.f0, 1.0e9);
        assert_eq!(exc.fc, 5.0e8);
    }

    #[test]
    fn test_sinus_excitation() {
        let exc = Excitation::sinus(2.4e9);
        assert_eq!(exc.exc_type, ExcitationType::Sinus);
        assert_eq!(exc.f0, 2.4e9);
    }
}
