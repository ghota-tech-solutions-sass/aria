//! # Activity Tracking - Sparse Update System
//!
//! This module implements the "sleep/wake" mechanism that allows ARIA
//! to scale to millions of cells. Inactive cells "sleep" and consume
//! zero CPU/GPU cycles.
//!
//! ## Philosophy
//!
//! Like biological neurons, cells should only fire when stimulated.
//! A sleeping cell is not dead - it's conserving energy and can wake
//! instantly when a signal reaches it.
//!
//! ## Benefits
//!
//! - **90% CPU reduction**: Only active cells are computed
//! - **Better focus**: Attention concentrates on stimulated areas
//! - **Memory consolidation**: Sleeping regions stabilize their DNA
//! - **Specialization**: Cells that sleep together evolve together

use serde::{Deserialize, Serialize};

/// Activity state for sparse updates
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ActivityState {
    /// Is this cell currently sleeping?
    pub sleeping: bool,

    /// How many consecutive ticks without significant change
    pub idle_ticks: u32,

    /// Last energy delta (for detecting stasis)
    pub last_energy_delta: f32,

    /// Last state magnitude (for detecting stasis)
    pub last_state_magnitude: f32,

    /// Ticks since last wake-up
    pub ticks_since_wake: u32,
}

impl ActivityState {
    /// Create a new activity state (awake by default)
    pub fn new() -> Self {
        Self {
            sleeping: false,
            idle_ticks: 0,
            last_energy_delta: 0.0,
            last_state_magnitude: 0.0,
            ticks_since_wake: 0,
        }
    }

    /// Check if cell should go to sleep
    ///
    /// A cell sleeps when:
    /// - Energy change is tiny (|delta| < threshold)
    /// - State magnitude is stable
    /// - Has been idle for enough ticks
    pub fn should_sleep(&self, config: &SleepConfig) -> bool {
        !self.sleeping
            && self.idle_ticks >= config.idle_ticks_to_sleep
            && self.last_energy_delta.abs() < config.energy_delta_threshold
    }

    /// Check if cell should wake up given a stimulus
    ///
    /// A cell wakes when:
    /// - Receives a signal above wake threshold
    /// - Has slept long enough to avoid oscillation
    pub fn should_wake(&self, stimulus: f32, config: &SleepConfig) -> bool {
        self.sleeping
            && stimulus.abs() >= config.wake_threshold
            && self.ticks_since_wake >= config.min_sleep_ticks
    }

    /// Update activity tracking after a tick
    pub fn update(&mut self, energy_delta: f32, state_magnitude: f32, config: &SleepConfig) {
        self.last_energy_delta = energy_delta;
        self.last_state_magnitude = state_magnitude;

        if energy_delta.abs() < config.energy_delta_threshold {
            self.idle_ticks = self.idle_ticks.saturating_add(1);
        } else {
            self.idle_ticks = 0;
        }

        if self.sleeping {
            self.ticks_since_wake = self.ticks_since_wake.saturating_add(1);
        }
    }

    /// Put cell to sleep
    pub fn sleep(&mut self) {
        self.sleeping = true;
        self.ticks_since_wake = 0;
    }

    /// Wake cell up
    pub fn wake(&mut self) {
        self.sleeping = false;
        self.idle_ticks = 0;
        self.ticks_since_wake = 0;
    }
}

/// Configuration for sleep/wake behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SleepConfig {
    /// Energy change below this = idle
    pub energy_delta_threshold: f32,

    /// Ticks of idleness before sleeping
    pub idle_ticks_to_sleep: u32,

    /// Stimulus needed to wake
    pub wake_threshold: f32,

    /// Minimum ticks to stay asleep (prevents oscillation)
    pub min_sleep_ticks: u32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            energy_delta_threshold: 0.001,
            idle_ticks_to_sleep: 100,
            wake_threshold: 0.1,
            min_sleep_ticks: 50,
        }
    }
}

/// Tracks global activity statistics
#[derive(Clone, Debug, Default)]
pub struct ActivityTracker {
    /// Number of cells currently awake
    pub awake_count: u64,

    /// Number of cells sleeping
    pub sleeping_count: u64,

    /// Cells woken this tick
    pub woken_this_tick: u64,

    /// Cells put to sleep this tick
    pub slept_this_tick: u64,

    /// Rolling average of awake ratio
    pub awake_ratio: f32,
}

impl ActivityTracker {
    /// Create new tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Update statistics
    pub fn update(&mut self, awake: u64, sleeping: u64, woken: u64, slept: u64) {
        self.awake_count = awake;
        self.sleeping_count = sleeping;
        self.woken_this_tick = woken;
        self.slept_this_tick = slept;

        let total = awake + sleeping;
        if total > 0 {
            let current_ratio = awake as f32 / total as f32;
            // Exponential moving average
            self.awake_ratio = self.awake_ratio * 0.95 + current_ratio * 0.05;
        }
    }

    /// Get percentage of cells that are awake
    pub fn awake_percentage(&self) -> f32 {
        self.awake_ratio * 100.0
    }

    /// Estimate CPU savings from sparse updates
    pub fn cpu_savings_percentage(&self) -> f32 {
        (1.0 - self.awake_ratio) * 100.0
    }
}

/// Region-based activity for spatial locality
///
/// Cells in the same region tend to sleep/wake together,
/// enabling efficient batch processing on GPU.
#[derive(Clone, Debug)]
pub struct RegionActivity {
    /// Region identifier (spatial hash)
    pub region_id: u64,

    /// Number of awake cells in region
    pub awake_count: u32,

    /// Average activity level
    pub avg_activity: f32,

    /// Is this region "hot" (worth computing)?
    pub is_hot: bool,
}

impl RegionActivity {
    /// Create a new region
    pub fn new(region_id: u64) -> Self {
        Self {
            region_id,
            awake_count: 0,
            avg_activity: 0.0,
            is_hot: false,
        }
    }

    /// Update region activity
    pub fn update(&mut self, awake: u32, total: u32, hot_threshold: f32) {
        self.awake_count = awake;
        if total > 0 {
            self.avg_activity = awake as f32 / total as f32;
            self.is_hot = self.avg_activity >= hot_threshold;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_sleep_cycle() {
        let mut activity = ActivityState::new();
        let config = SleepConfig::default();

        assert!(!activity.sleeping);

        // Simulate idle ticks
        for _ in 0..150 {
            activity.update(0.0001, 0.1, &config);
        }

        assert!(activity.should_sleep(&config));
        activity.sleep();
        assert!(activity.sleeping);

        // Simulate stimulus
        for _ in 0..60 {
            activity.update(0.0, 0.0, &config);
        }

        assert!(activity.should_wake(0.5, &config));
        activity.wake();
        assert!(!activity.sleeping);
    }

    #[test]
    fn test_activity_tracker() {
        let mut tracker = ActivityTracker::new();
        tracker.update(1000, 9000, 50, 30);

        assert_eq!(tracker.awake_count, 1000);
        assert!(tracker.cpu_savings_percentage() > 80.0);
    }
}
