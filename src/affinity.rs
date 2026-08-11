//! Self-pinned CPU affinity for the queens solver.
//!
//! Reuses the approach proven in the `sessions` TUI daemon
//! (`~/src/sessions/src/daemon/affinity.rs`): detect performance cores by peak cpufreq,
//! pin the search rayon pool across all allowed cores **perf-first** (deterministic 1:1
//! placement — no scheduler migration, which this heterogeneous box is hypersensitive to),
//! and confine the freeze-build pool + the watcher thread to the **efficiency** cores so
//! housekeeping never preempts the latency-critical search on a high-clock core.
//!
//! On the dev box (AMD HX 370): perf logicals `{0-3,12-15}` (Zen5, ~5.16 GHz) vs efficiency
//! `{4-11,16-23}` (Zen5c, ~3.29 GHz). `QUEENS_AFFINITY=off` disables; `=on` forces even on a
//! homogeneous machine; default `auto` engages only when a distinct efficiency tier exists.
//! No-op on non-Linux. Respects an inherited `taskset`/cgroup mask (we partition *within* it).

use std::sync::OnceLock;

/// Resolved placement, computed once from the host topology (and the inherited affinity mask).
pub struct CorePlan {
    pub engaged: bool,
    /// Search rayon-pool worker `i` pins to `search_cpus[i]` (perf cores first).
    pub search_cpus: Vec<u32>,
    /// CPUs the freeze-build pool + watcher + freeze-orchestrator threads float across
    /// (efficiency cores; falls back to the full allowed set when there is no distinct tier).
    pub aux_cpus: Vec<u32>,
}

impl CorePlan {
    fn disabled() -> Self {
        CorePlan {
            engaged: false,
            search_cpus: Vec::new(),
            aux_cpus: Vec::new(),
        }
    }
}

static PLAN: OnceLock<CorePlan> = OnceLock::new();

/// Board sizes below this don't engage under `auto`: small boards never freeze (so the
/// build/watcher confinement — the actual win — is inert) and the 1:1 search pin *regressed*
/// them ~5% in the n=14 A/B (the spiky search wants the scheduler's freedom to keep active
/// threads on the perf cores). `QUEENS_AFFINITY=on` overrides; `=off` always disables.
const AFFINITY_MIN_N: u32 = 16;

/// Resolve the plan for an `n`x`n` solve and, when engaged, pin the global rayon pool's
/// search workers (perf-first, 1:1). Call once at the top of `solve`, before any `par_iter`
/// or store build. No-op (rayon's default pool) unless engaged — under `auto`, only n >=
/// [`AFFINITY_MIN_N`] on heterogeneous hardware.
pub fn configure(n: u32) {
    let plan = PLAN.get_or_init(|| build_plan(n));
    if !plan.engaged {
        return;
    }
    let search: &'static [u32] = &plan.search_cpus;
    let res = rayon::ThreadPoolBuilder::new()
        .num_threads(search.len())
        .thread_name(|i| format!("queens-search-{i}"))
        .start_handler(move |idx| {
            if let Some(&cpu) = search.get(idx) {
                pin_to_cpus(&[cpu], "search");
            }
        })
        .build_global();
    match res {
        Ok(()) => eprintln!(
            "\x1b[90m(affinity: search pinned 1:1 to {} core(s) [perf-first {:?}…]; build/watcher on {} efficiency core(s) {:?})\x1b[0m",
            search.len(),
            &search[..search.len().min(8)],
            plan.aux_cpus.len(),
            plan.aux_cpus,
        ),
        Err(e) => eprintln!("\x1b[90m(affinity: build_global failed: {e}; default pool)\x1b[0m"),
    }
}

/// Pin the calling thread to the auxiliary (efficiency) mask. Used by the freeze-build pool's
/// `start_handler`, the watcher thread, and the per-freeze orchestrator thread. No-op when the
/// plan isn't engaged.
pub fn pin_aux(label: &str) {
    if let Some(plan) = PLAN.get() {
        if plan.engaged && !plan.aux_cpus.is_empty() {
            pin_to_cpus(&plan.aux_cpus, label);
        }
    }
}

fn build_plan(n: u32) -> CorePlan {
    let policy = std::env::var("QUEENS_AFFINITY").ok();
    let policy = policy.as_deref();
    if policy == Some("off") {
        return CorePlan::disabled();
    }
    let forced = policy == Some("on");
    // Default (`auto`): only engage on the large boards that freeze; `on` overrides the gate.
    if !forced && n < AFFINITY_MIN_N {
        return CorePlan::disabled();
    }

    let allowed = allowed_cpus(); // respects an inherited taskset/cgroup mask
    if allowed.len() < 2 {
        return CorePlan::disabled();
    }
    // Peak frequency per allowed CPU; bail (leave the scheduler alone) if any is unreadable.
    let mut freqs: Vec<(u32, u64)> = Vec::with_capacity(allowed.len());
    for &c in &allowed {
        match max_freq(c) {
            Some(f) => freqs.push((c, f)),
            None => return CorePlan::disabled(),
        }
    }
    let maxf = freqs.iter().map(|(_, f)| *f).max().unwrap_or(0);
    let mut perf: Vec<u32> = freqs
        .iter()
        .filter(|(_, f)| *f == maxf)
        .map(|(c, _)| *c)
        .collect();
    let mut eff: Vec<u32> = freqs
        .iter()
        .filter(|(_, f)| *f != maxf)
        .map(|(c, _)| *c)
        .collect();
    perf.sort_unstable();
    eff.sort_unstable();

    // `auto` engages only on a heterogeneous part (a distinct efficiency tier exists); a
    // homogeneous box is left to the scheduler unless `QUEENS_AFFINITY=on` forces pinning.
    if eff.is_empty() && !forced {
        return CorePlan::disabled();
    }
    let aux = if eff.is_empty() {
        perf.clone()
    } else {
        eff.clone()
    };
    let mut search = perf;
    search.extend(eff); // perf cores first, then efficiency
    CorePlan {
        engaged: true,
        search_cpus: search,
        aux_cpus: aux,
    }
}

#[cfg(target_os = "linux")]
fn max_freq(cpu: u32) -> Option<u64> {
    std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
    ))
    .ok()
    .and_then(|s| s.trim().parse().ok())
}

#[cfg(not(target_os = "linux"))]
fn max_freq(_cpu: u32) -> Option<u64> {
    None
}

/// The CPUs this process is allowed to run on (inherited affinity mask). Empty on non-Linux
/// or if the query fails ⇒ a disabled plan.
#[cfg(target_os = "linux")]
fn allowed_cpus() -> Vec<u32> {
    use std::mem::{size_of, zeroed};
    // SAFETY: `sched_getaffinity` fills a zeroed `cpu_set_t` of the size we pass; on success
    // we only read it via the `CPU_ISSET` macro. No aliasing or lifetime concerns.
    unsafe {
        let mut set: libc::cpu_set_t = zeroed();
        if libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&c| libc::CPU_ISSET(c, &set))
            .map(|c| c as u32)
            .collect()
    }
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpus() -> Vec<u32> {
    Vec::new()
}

/// Pin the calling thread to `cpus` (best-effort; a failure leaves the mask untouched).
#[cfg(target_os = "linux")]
fn pin_to_cpus(cpus: &[u32], _label: &str) {
    use std::mem::{size_of, zeroed};
    // SAFETY: build a `cpu_set_t` with the standard `CPU_ZERO`/`CPU_SET` macros (bounds-checked
    // against `CPU_SETSIZE`) and hand it to `sched_setaffinity` for the current thread (pid 0).
    // All are async-signal-safe POSIX calls that touch only the local set and this thread's mask.
    unsafe {
        let mut set: libc::cpu_set_t = zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            if (c as usize) < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c as usize, &mut set);
            }
        }
        let _ = libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_cpus(_cpus: &[u32], _label: &str) {}
