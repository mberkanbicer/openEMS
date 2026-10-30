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
fn read_mxcsr() -> u32 {
    let mut value: u32 = 0;
    // SAFETY: `stmxcsr` stores the 32-bit control/status word into `value`.
    unsafe {
        std::arch::asm!("stmxcsr [{}]", in(reg) &mut value, options(nostack, preserves_flags));
    }
    value
}

#[cfg(target_arch = "x86_64")]
fn write_mxcsr(value: u32) {
    // SAFETY: `ldmxcsr` loads a valid MXCSR value; the caller passes one read
    // from this CPU or that value with the FTZ/DAZ control bits set.
    unsafe {
        std::arch::asm!("ldmxcsr [{}]", in(reg) &value, options(nostack, preserves_flags));
    }
}

#[cfg(target_arch = "x86_64")]
fn enable() -> Option<u32> {
    let saved = read_mxcsr();
    // bit 15: flush to zero, bit 6: denormals are zero
    write_mxcsr(saved | (1 << 15) | (1 << 6));
    Some(saved)
}

#[cfg(not(target_arch = "x86_64"))]
fn enable() -> Option<u32> {
    None
}

#[cfg(target_arch = "x86_64")]
fn restore(saved: u32) {
    write_mxcsr(saved);
}

#[cfg(not(target_arch = "x86_64"))]
fn restore(_saved: u32) {}

#[cfg(all(test, target_arch = "x86_64"))]
fn current() -> u32 {
    read_mxcsr()
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
