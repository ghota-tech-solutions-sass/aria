//! Internal Goals - ARIA sets her own objectives
//!
//! Instead of waiting for external tasks, ARIA creates goals for herself.

use serde::{Deserialize, Serialize};

/// A goal ARIA sets for herself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalGoal {
    /// Unique goal ID
    pub id: u64,
    /// Human-readable description
    pub description: String,
    /// Goal type
    pub goal_type: GoalType,
    /// Progress toward goal (0.0 to 1.0)
    pub progress: f32,
    /// Target value to achieve
    pub target: f32,
    /// Current value
    pub current: f32,
    /// When this goal was set
    pub created_at: u64,
    /// Is the goal complete?
    pub completed: bool,
    /// Reward when completed
    pub completion_reward: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GoalType {
    /// Learn N new words
    LearnWords,
    /// Achieve N successful explorations
    SuccessfulExplorations,
    /// Improve a strategy's performance
    ImproveStrategy,
    /// Reduce boredom by exploring
    CureBoredom,
    /// Find connections in a topic
    ExploreCluster,
    /// Maintain emotional stability
    EmotionalBalance,
}

impl InternalGoal {
    pub fn new(
        id: u64,
        description: String,
        goal_type: GoalType,
        target: f32,
        tick: u64,
    ) -> Self {
        Self {
            id,
            description,
            goal_type,
            progress: 0.0,
            target,
            current: 0.0,
            created_at: tick,
            completed: false,
            completion_reward: 0.5, // Default reward
        }
    }

    /// Update progress toward goal
    pub fn update(&mut self, new_value: f32) -> bool {
        self.current = new_value;
        self.progress = (self.current / self.target).min(1.0);

        if self.progress >= 1.0 && !self.completed {
            self.completed = true;
            tracing::info!("🎯 GOAL COMPLETED: {}", self.description);
            return true;
        }
        false
    }

    /// Increment progress by a delta
    pub fn increment(&mut self, delta: f32) -> bool {
        self.update(self.current + delta)
    }
}
