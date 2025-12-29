//! Exploration Memory - ARIA remembers what she tried
//!
//! Tracks word combinations ARIA has explored and their outcomes.

use serde::{Deserialize, Serialize};

/// Result of an exploration attempt (combination of words)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExplorationResult {
    /// Number of times this combination was tried
    pub attempts: u64,
    /// Number of positive feedbacks received
    pub positive_feedback: u64,
    /// Number of negative feedbacks received
    pub negative_feedback: u64,
    /// Average intensity when this was expressed
    pub avg_intensity: f32,
    /// Last tick when this was tried
    pub last_attempt: u64,
}

impl ExplorationResult {
    pub fn new(tick: u64, intensity: f32) -> Self {
        Self {
            attempts: 1,
            positive_feedback: 0,
            negative_feedback: 0,
            avg_intensity: intensity,
            last_attempt: tick,
        }
    }

    /// Calculate exploration score - novelty + success rate
    pub fn exploration_score(&self) -> f32 {
        let success_rate = if self.attempts > 0 {
            self.positive_feedback as f32 / self.attempts as f32
        } else {
            0.5 // Unknown = neutral
        };

        // Prefer: high success rate, but also some novelty (not tried too many times)
        let novelty = 1.0 / (1.0 + self.attempts as f32 * 0.1);

        success_rate * 0.7 + novelty * 0.3
    }

    /// Is this combination worth trying again?
    pub fn should_retry(&self) -> bool {
        // Retry if: few attempts, or good success rate
        self.attempts < 3 || (self.positive_feedback > self.negative_feedback)
    }
}
