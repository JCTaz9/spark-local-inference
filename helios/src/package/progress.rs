//! Per-acquisition byte and elapsed-time accounting.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One shared handle for an acquisition. Clones observe the same counters.
#[derive(Clone, Debug)]
pub struct AcquisitionProgress {
    start: Instant,
    counters: Arc<Mutex<Counters>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    network_bytes: u64,
    local_source_bytes: u64,
    materialized_copy_bytes: u64,
    materialized_fill_bytes: u64,
    verification_bytes: u64,
}

/// Immutable acquisition counters for phase and final reporting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AcquisitionSnapshot {
    pub network_bytes: u64,
    /// Bytes successfully read from local source payloads.
    pub local_source_bytes: u64,
    /// Bytes successfully written by derivation copy operations.
    pub materialized_copy_bytes: u64,
    /// Bytes successfully written by derivation fill operations.
    pub materialized_fill_bytes: u64,
    /// Bytes successfully read for byte comparison or fed to the final target hash.
    pub verification_bytes: u64,
    pub elapsed: Duration,
}

impl AcquisitionSnapshot {
    pub fn materialized_bytes(&self) -> u64 {
        self.materialized_copy_bytes
            .saturating_add(self.materialized_fill_bytes)
    }
}

impl Default for AcquisitionProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl AcquisitionProgress {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            counters: Arc::new(Mutex::new(Counters::default())),
        }
    }

    pub fn record_network_bytes(&self, bytes: u64) {
        self.update(|counters| counters.network_bytes = counters.network_bytes.saturating_add(bytes));
    }

    pub fn record_local_source_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.local_source_bytes = counters.local_source_bytes.saturating_add(bytes)
        });
    }

    pub fn record_materialized_copy_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.materialized_copy_bytes =
                counters.materialized_copy_bytes.saturating_add(bytes)
        });
    }

    pub fn record_materialized_fill_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.materialized_fill_bytes =
                counters.materialized_fill_bytes.saturating_add(bytes)
        });
    }

    pub fn record_verification_bytes(&self, bytes: u64) {
        self.update(|counters| {
            counters.verification_bytes = counters.verification_bytes.saturating_add(bytes)
        });
    }

    pub fn snapshot(&self) -> AcquisitionSnapshot {
        let counters = *self
            .counters
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        AcquisitionSnapshot {
            network_bytes: counters.network_bytes,
            local_source_bytes: counters.local_source_bytes,
            materialized_copy_bytes: counters.materialized_copy_bytes,
            materialized_fill_bytes: counters.materialized_fill_bytes,
            verification_bytes: counters.verification_bytes,
            elapsed: self.start.elapsed(),
        }
    }

    fn update(&self, update: impl FnOnce(&mut Counters)) {
        let mut counters = self
            .counters
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        update(&mut counters);
    }
}
