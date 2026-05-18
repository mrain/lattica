//! Process-level utilities available on `std` builds.

use alloc::string::String;

#[cfg(any(target_os = "linux", target_os = "android"))]
use std::fs;

/// Best-effort process memory snapshot for reporting and examples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessMemorySnapshot {
    /// Current resident set size in KiB, if available.
    pub current_rss_kib: Option<u64>,
    /// Peak resident set size in KiB, if available.
    pub peak_rss_kib: Option<u64>,
}

impl ProcessMemorySnapshot {
    /// Capture current and peak process RSS.
    ///
    /// Supported platforms:
    /// - Linux / Android: `/proc/self/statm` + `getrusage(RUSAGE_SELF)`
    /// - Apple targets: Mach `task_info(MACH_TASK_BASIC_INFO)`
    /// - Windows: `GetProcessMemoryInfo`
    ///
    /// Other targets currently return an empty snapshot.
    pub fn capture() -> Self {
        capture_process_memory_snapshot()
    }

    /// Return a compact description of this snapshot.
    pub fn describe(&self) -> String {
        format!(
            "rss={} peak={}",
            format_opt_kib(self.current_rss_kib),
            format_opt_kib(self.peak_rss_kib)
        )
    }

    /// Return a compact description of this snapshot relative to an earlier one.
    pub fn describe_since(&self, earlier: &Self) -> String {
        let current_delta = match (self.current_rss_kib, earlier.current_rss_kib) {
            (Some(now), Some(prev)) => Some(now as i64 - prev as i64),
            _ => None,
        };
        let peak_delta = match (self.peak_rss_kib, earlier.peak_rss_kib) {
            (Some(now), Some(prev)) => Some(now as i64 - prev as i64),
            _ => None,
        };
        format!(
            "{} delta_rss={} delta_peak={}",
            self.describe(),
            format_opt_delta_kib(current_delta),
            format_opt_delta_kib(peak_delta)
        )
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn capture_process_memory_snapshot() -> ProcessMemorySnapshot {
    ProcessMemorySnapshot {
        current_rss_kib: current_rss_kib_procfs(),
        peak_rss_kib: peak_rss_kib_rusage_linux_like(),
    }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "watchos",
    target_os = "tvos",
    target_os = "visionos"
))]
fn capture_process_memory_snapshot() -> ProcessMemorySnapshot {
    apple_task_basic_info()
}

#[cfg(windows)]
fn capture_process_memory_snapshot() -> ProcessMemorySnapshot {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut counters = core::mem::MaybeUninit::<PROCESS_MEMORY_COUNTERS>::zeroed();
    let size = core::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    unsafe {
        let counters_ptr = counters.as_mut_ptr();
        (*counters_ptr).cb = size;
        if K32GetProcessMemoryInfo(GetCurrentProcess(), counters_ptr, size) == 0 {
            return ProcessMemorySnapshot::default();
        }
        let counters = counters.assume_init();
        ProcessMemorySnapshot {
            current_rss_kib: bytes_to_kib_u64(counters.WorkingSetSize as u64),
            peak_rss_kib: bytes_to_kib_u64(counters.PeakWorkingSetSize as u64),
        }
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "watchos",
    target_os = "tvos",
    target_os = "visionos",
    windows
)))]
fn capture_process_memory_snapshot() -> ProcessMemorySnapshot {
    ProcessMemorySnapshot::default()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn current_rss_kib_procfs() -> Option<u64> {
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages = parse_statm_resident_pages(&statm)?;
    let page_size = page_size_bytes()?;
    resident_pages
        .checked_mul(page_size)
        .and_then(bytes_to_kib_u64)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn peak_rss_kib_rusage_linux_like() -> Option<u64> {
    let mut usage = core::mem::MaybeUninit::<libc::rusage>::uninit();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    let usage = unsafe { usage.assume_init() };
    u64::try_from(usage.ru_maxrss).ok()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn page_size_bytes() -> Option<u64> {
    let bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if bytes <= 0 {
        None
    } else {
        u64::try_from(bytes).ok()
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn parse_statm_resident_pages(statm: &str) -> Option<u64> {
    let mut fields = statm.split_whitespace();
    let _program_size_pages = fields.next()?;
    fields.next()?.parse::<u64>().ok()
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "watchos",
    target_os = "tvos",
    target_os = "visionos"
))]
fn apple_task_basic_info() -> ProcessMemorySnapshot {
    let mut info = core::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    let kr = unsafe {
        libc::task_info(
            mach2::traps::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<libc::integer_t>(),
            &mut count,
        )
    };
    if kr != libc::KERN_SUCCESS {
        return ProcessMemorySnapshot::default();
    }
    let info = unsafe { info.assume_init() };
    let current_bytes = unsafe { core::ptr::addr_of!(info.resident_size).read_unaligned() };
    let peak_bytes = unsafe { core::ptr::addr_of!(info.resident_size_max).read_unaligned() };
    ProcessMemorySnapshot {
        current_rss_kib: bytes_to_kib_u64(current_bytes),
        peak_rss_kib: bytes_to_kib_u64(peak_bytes),
    }
}

fn bytes_to_kib_u64(bytes: u64) -> Option<u64> {
    Some(bytes / 1024)
}

fn format_opt_kib(value: Option<u64>) -> String {
    match value {
        Some(value) => format!("{value} KiB"),
        None => "n/a".to_owned(),
    }
}

fn format_opt_delta_kib(value: Option<i64>) -> String {
    match value {
        Some(value) => format!("{value:+} KiB"),
        None => "n/a".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_parse_statm_resident_pages_reads_second_field() {
        assert_eq!(parse_statm_resident_pages("100 25 10 0 0 0 0\n"), Some(25));
        assert_eq!(parse_statm_resident_pages("100\n"), None);
    }

    #[test]
    fn test_bytes_to_kib_rounds_down() {
        assert_eq!(bytes_to_kib_u64(0), Some(0));
        assert_eq!(bytes_to_kib_u64(1023), Some(0));
        assert_eq!(bytes_to_kib_u64(1024), Some(1));
        assert_eq!(bytes_to_kib_u64(4097), Some(4));
    }

    #[test]
    fn test_describe_since_formats_delta() {
        let earlier = ProcessMemorySnapshot {
            current_rss_kib: Some(100),
            peak_rss_kib: Some(150),
        };
        let later = ProcessMemorySnapshot {
            current_rss_kib: Some(140),
            peak_rss_kib: Some(210),
        };
        let description = later.describe_since(&earlier);
        assert!(description.contains("rss=140 KiB"));
        assert!(description.contains("peak=210 KiB"));
        assert!(description.contains("delta_rss=+40 KiB"));
        assert!(description.contains("delta_peak=+60 KiB"));
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn test_capture_returns_some_memory_data() {
        let snapshot = ProcessMemorySnapshot::capture();
        assert!(snapshot.current_rss_kib.is_some());
        assert!(snapshot.peak_rss_kib.is_some());
    }
}
