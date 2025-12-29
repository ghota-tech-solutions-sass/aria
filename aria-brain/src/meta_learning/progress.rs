//! Progress Tracker - Awareness of learning progress
//!
//! ARIA tracks her own competence level and learning trend.

use serde::{Deserialize, Serialize};
use super::reward::InternalReward;

/// Tracks ARIA's awareness of her own progress
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProgressTracker {
    /// Rolling average of internal rewards (overall learning quality)
    pub learning_quality: f32,
    /// How fast is learning improving?
    pub learning_velocity: f32,
    /// How many successful explorations recently?
    pub recent_successes: u32,
    /// How many failed explorations recently?
    pub recent_failures: u32,
    /// Current competence level (0.0 = novice, 1.0 = expert)
    pub competence_level: f32,
    /// History of competence for trend detection
    pub competence_history: Vec<f32>,
    /// Is ARIA improving, stable, or declining?
    pub trend: LearningTrend,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum LearningTrend {
    Improving,
    #[default]
    Stable,
    Declining,
}

impl ProgressTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an exploration result
    pub fn record_exploration(&mut self, reward: &InternalReward) {
        // Update recent counts
        if reward.is_positive() {
            self.recent_successes += 1;
        } else {
            self.recent_failures += 1;
        }

        // Decay old counts
        if self.recent_successes + self.recent_failures > 20 {
            self.recent_successes = (self.recent_successes as f32 * 0.9) as u32;
            self.recent_failures = (self.recent_failures as f32 * 0.9) as u32;
        }

        // Update learning quality (exponential moving average)
        self.learning_quality = self.learning_quality * 0.9 + reward.total_score * 0.1;

        // Calculate success rate
        let total = (self.recent_successes + self.recent_failures) as f32;
        let success_rate = if total > 0.0 {
            self.recent_successes as f32 / total
        } else {
            0.5
        };

        // Update competence level
        let new_competence = (self.learning_quality * 0.5 + success_rate * 0.5).min(1.0);
        let old_competence = self.competence_level;
        self.competence_level = self.competence_level * 0.95 + new_competence * 0.05;

        // Track history for trend detection
        self.competence_history.push(self.competence_level);
        if self.competence_history.len() > 50 {
            self.competence_history.remove(0);
        }

        // Calculate velocity and trend
        self.learning_velocity = self.competence_level - old_competence;
        self.update_trend();
    }

    fn update_trend(&mut self) {
        if self.competence_history.len() < 10 {
            self.trend = LearningTrend::Stable;
            return;
        }

        // Compare recent average to older average
        let recent: f32 = self.competence_history.iter().rev().take(5).sum::<f32>() / 5.0;
        let older: f32 = self.competence_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;

        let diff = recent - older;
        self.trend = if diff > 0.02 {
            LearningTrend::Improving
        } else if diff < -0.02 {
            LearningTrend::Declining
        } else {
            LearningTrend::Stable
        };
    }

    /// Get a description of current state
    pub fn status_description(&self) -> String {
        let trend_str = match self.trend {
            LearningTrend::Improving => "s'améliore",
            LearningTrend::Stable => "stable",
            LearningTrend::Declining => "en difficulté",
        };

        let level_str = if self.competence_level > 0.7 {
            "experte"
        } else if self.competence_level > 0.4 {
            "compétente"
        } else {
            "débutante"
        };

        format!("ARIA est {} ({:.0}%), apprentissage {}",
            level_str,
            self.competence_level * 100.0,
            trend_str
        )
    }
}
