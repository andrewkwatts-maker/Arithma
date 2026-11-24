//! A counting global allocator for tests.
//!
//! The one sanctioned copy of Themelios `alloc_probe::Counting`: Arithma is a shared
//! repo and cannot path-depend on Themelios. Install with
//! `#[global_allocator] static A: Counting = Counting::new();` and read
//! [`Counting::thread_count`] before and after the window under test.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    static THREAD_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn bump_thread() {
    // `try_with` skips the count if the thread-local is gone (thread teardown).
    let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get() + 1));
}

/// Forwards to the system allocator and counts every allocation and reallocation.
pub struct Counting {
    allocs: AtomicUsize,
}

impl Counting {
    /// A counter at zero.
    pub const fn new() -> Self {
        Self {
            allocs: AtomicUsize::new(0),
        }
    }

    /// Allocations and reallocations so far, across all threads.
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.allocs.load(Ordering::SeqCst)
    }

    /// Allocations and reallocations made so far by the calling thread only.
    pub fn thread_count() -> u64 {
        THREAD_ALLOCS.try_with(Cell::get).unwrap_or(0)
    }
}

impl Default for Counting {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every method forwards its arguments unchanged to `System`, which upholds the
// `GlobalAlloc` contract; the counter is an atomic and touches no allocation state.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        self.allocs.fetch_add(1, Ordering::SeqCst);
        bump_thread();
        // SAFETY: caller's layout contract is passed straight through.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` and `l` came from this allocator, which forwards to `System`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        self.allocs.fetch_add(1, Ordering::SeqCst);
        bump_thread();
        // SAFETY: `p` and `l` came from this allocator, which forwards to `System`.
        unsafe { System.realloc(p, l, n) }
    }
}
