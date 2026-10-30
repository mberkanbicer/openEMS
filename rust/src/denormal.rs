//! Flush denormals to zero during a solver run (native `Denormal::FlushToZero`).
//!
//! Every native engine does this: subnormal values are flushed to zero while
//! the simulation runs, which is fast and gives the same results for all
//! engines. The caller's setting is restored when the run ends, so Python
//! post-processing keeps its own behaviour.

/// RAII guard that enables flush-to-zero and denormals-are-zero on x86 and
/// restores the previous MXCSR when dropped.
pub struct FlushToZero {
    saved: Option<u32>,
}

#[cfg(target_arch = "x86_64")]
fn enable() -> Option<u32> {
    unsafe {
        let saved = std::arch::x86_64::_mm_getcsr();
        // bit 15: flush to zero, bit 6: denormals are zero
        std::arch::x86_64::_mm_setcsr(saved | (1 << 15) | (1 << 6));
        Some(saved)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn enable() -> Option<u32> {
    None
}

#[cfg(target_arch = "x86_64")]
fn restore(saved: u32) {
    unsafe {
        std::arch::x86_64::_mm_setcsr(saved);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn restore(_saved: u32) {}

#[cfg(all(test, target_arch = "x86_64"))]
fn current() -> u32 {
    unsafe { std::arch::x86_64::_mm_getcsr() }
}

impl FlushToZero {
    pub fn new() -> Self {
        Self { saved: enable() }
    }
}

impl Default for FlushToZero {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FlushToZero {
    fn drop(&mut self) {
        if let Some(saved) = self.saved {
            restore(saved);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_denormal_flush_is_set_and_restored() {
        let before = current();
        {
            let _guard = FlushToZero::new();
            let inside = current();
            assert_eq!(inside & (1 << 15), 1 << 15, "flush-to-zero set");
            assert_eq!(inside & (1 << 6), 1 << 6, "denormals-are-zero set");
        }
        assert_eq!(current() & (1 << 15), before & (1 << 15));
        assert_eq!(current() & (1 << 6), before & (1 << 6));
    }
}
