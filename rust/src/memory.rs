//! Peak resident memory of the process (native `PeakResidentBytes`).
//!
//! The array accounting in `Operator::memory_bytes` counts what the solver
//! asked for; this reports what the OS counted, so the two can be compared.

/// Peak resident set size in bytes, or `None` when the platform cannot report
/// it.
pub fn peak_resident_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kilobytes = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                return Some(kilobytes * 1024);
            }
        }
        None
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }

        #[link(name = "psapi")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn GetProcessMemoryInfo(
                process: isize,
                counters: *mut ProcessMemoryCounters,
                size: u32,
            ) -> i32;
        }

        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        let ok = unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut counters,
                std::mem::size_of::<ProcessMemoryCounters>() as u32,
            )
        };
        (ok != 0).then_some(counters.peak_working_set_size as u64)
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_peak_resident_memory_is_plausible() {
        if let Some(bytes) = peak_resident_bytes() {
            // Any process running these tests uses at least one page.
            assert!(bytes >= 4096, "peak RSS {bytes} bytes");
        }
    }
}
