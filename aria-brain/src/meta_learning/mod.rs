//! Meta-Learning System for ARIA
//!
//! This module implements ARIA's ability to learn HOW to learn.
//! Instead of relying on external feedback, ARIA evaluates herself
//! and develops her own learning strategies.
//!
//! ## Module Structure (refactored)
//!
//! - `reward` - InternalReward: ARIA evaluates her own explorations
//! - `strategy` - StrategyType, ExplorationStrategy: Learned patterns
//! - `goals` - InternalGoal, GoalType: Self-directed objectives
//! - `progress` - ProgressTracker, LearningTrend: Self-awareness
//! - `self_modifier` - SelfModification, SelfModifier: ARIA changes herself

pub mod reward;
pub mod strategy;
pub mod goals;
pub mod progress;
pub mod self_modifier;

// Re-export all types for convenience
pub use reward::InternalReward;
pub use strategy::{StrategyType, ExplorationStrategy};
pub use goals::{InternalGoal, GoalType};
pub use progress::{ProgressTracker, LearningTrend};
pub use self_modifier::{
    ModifiableParam, CurrentParams, SelfModifier,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

// ============================================================================
// META LEARNER - The core of self-directed learning
// ============================================================================

/// The MetaLearner coordinates all meta-learning activities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearner {
    /// Exploration strategies and their performance
    pub strategies: HashMap<StrategyType, ExplorationStrategy>,
    /// Current internal goals
    pub goals: Vec<InternalGoal>,
    /// Next goal ID
    pub next_goal_id: u64,
    /// Progress tracking
    pub progress: ProgressTracker,
    /// Currently active strategy
    pub current_strategy: StrategyType,
    /// Last exploration's internal reward
    pub last_reward: Option<InternalReward>,
    /// Total explorations evaluated
    pub total_evaluations: u64,
    /// Configuration
    pub config: MetaLearnerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnerConfig {
    /// How much to explore vs exploit (0 = pure exploit, 1 = pure explore)
    pub exploration_rate: f32,
    /// Minimum explorations before trusting strategy scores
    pub min_explorations_for_trust: u64,
    /// How often to create new goals (in ticks)
    pub goal_creation_interval: u64,
    /// Maximum concurrent goals
    pub max_goals: usize,
}

impl Default for MetaLearnerConfig {
    fn default() -> Self {
        Self {
            exploration_rate: 0.3, // 30% random exploration
            min_explorations_for_trust: 10,
            goal_creation_interval: 5000,
            max_goals: 3,
        }
    }
}

#[allow(dead_code)]
impl MetaLearner {
    pub fn new() -> Self {
        // Initialize all strategies
        let mut strategies = HashMap::new();
        for strategy_type in StrategyType::all() {
            strategies.insert(strategy_type, ExplorationStrategy::new(strategy_type));
        }

        Self {
            strategies,
            goals: Vec::new(),
            next_goal_id: 0,
            progress: ProgressTracker::new(),
            current_strategy: StrategyType::Random, // Start random
            last_reward: None,
            total_evaluations: 0,
            config: MetaLearnerConfig::default(),
        }
    }

    /// Select the next exploration strategy
    pub fn select_strategy(&mut self, current_tick: u64) -> StrategyType {
        let mut rng = rand::thread_rng();

        // Exploration vs exploitation
        if rng.gen::<f32>() < self.config.exploration_rate {
            // Random exploration
            let strategies: Vec<StrategyType> = self.strategies.keys().copied().collect();
            let idx = rng.gen_range(0..strategies.len());
            self.current_strategy = strategies[idx];
            tracing::debug!("🎲 RANDOM strategy: {:?}", self.current_strategy);
        } else {
            // Select best performing strategy
            let best = self.strategies.iter()
                .max_by(|a, b| {
                    let score_a = a.1.selection_score(current_tick);
                    let score_b = b.1.selection_score(current_tick);
                    score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(t, _)| *t)
                .unwrap_or(StrategyType::Random);

            self.current_strategy = best;
            tracing::debug!("🎯 BEST strategy: {:?} (score: {:.2})",
                self.current_strategy,
                self.strategies.get(&best).map(|s| s.selection_score(current_tick)).unwrap_or(0.0)
            );
        }

        self.current_strategy
    }

    /// Evaluate an exploration and update everything
    pub fn evaluate_exploration(
        &mut self,
        coherence: f32,
        novelty: f32,
        intensity: f32,
        emotional_before: f32,
        emotional_after: f32,
        expected_intensity: f32,
        current_tick: u64,
    ) -> InternalReward {
        // Compute internal reward
        let reward = InternalReward::compute(
            coherence,
            novelty,
            intensity,
            emotional_before,
            emotional_after,
            expected_intensity,
        );

        // Update current strategy
        if let Some(strategy) = self.strategies.get_mut(&self.current_strategy) {
            strategy.record_usage(reward.total_score, current_tick);

            tracing::info!(
                "📊 INTERNAL REWARD: {:.2} (strategy: {}, coherence: {:.2}, surprise: {:.2})",
                reward.total_score,
                self.current_strategy.name(),
                reward.coherence,
                reward.surprise
            );
        }

        // Update progress tracker
        self.progress.record_exploration(&reward);

        // Update goals
        self.update_goals(&reward, current_tick);

        // Save last reward
        self.last_reward = Some(reward.clone());
        self.total_evaluations += 1;

        // Adapt exploration rate based on trend
        self.adapt_exploration_rate();

        reward
    }

    /// Update goals based on exploration result
    fn update_goals(&mut self, reward: &InternalReward, _current_tick: u64) {
        for goal in &mut self.goals {
            if goal.completed {
                continue;
            }

            match goal.goal_type {
                GoalType::SuccessfulExplorations => {
                    if reward.is_positive() {
                        goal.increment(1.0);
                    }
                }
                GoalType::CureBoredom => {
                    // Exploration itself reduces boredom
                    goal.increment(0.2);
                }
                GoalType::ImproveStrategy => {
                    if reward.is_excellent() {
                        goal.increment(0.3);
                    }
                }
                GoalType::EmotionalBalance => {
                    if reward.emotional_delta > 0.0 {
                        goal.increment(reward.emotional_delta);
                    }
                }
                _ => {}
            }
        }

        // Remove completed goals
        self.goals.retain(|g| !g.completed);
    }

    /// Adapt exploration rate based on learning trend
    fn adapt_exploration_rate(&mut self) {
        match self.progress.trend {
            LearningTrend::Improving => {
                // Things are going well, exploit more
                self.config.exploration_rate = (self.config.exploration_rate - 0.01).max(0.1);
            }
            LearningTrend::Declining => {
                // Need to try new things
                self.config.exploration_rate = (self.config.exploration_rate + 0.02).min(0.5);
            }
            LearningTrend::Stable => {
                // Drift toward default
                let target = 0.3;
                self.config.exploration_rate += (target - self.config.exploration_rate) * 0.1;
            }
        }
    }

    /// Create a new goal if conditions are met
    pub fn maybe_create_goal(&mut self, current_tick: u64, boredom: f32, success_rate: f32) {
        // Don't create too many goals
        if self.goals.len() >= self.config.max_goals {
            return;
        }

        let mut rng = rand::thread_rng();

        // Choose goal type based on current state
        let goal_type = if boredom > 0.6 {
            GoalType::CureBoredom
        } else if success_rate < 0.3 {
            GoalType::ImproveStrategy
        } else if self.progress.trend == LearningTrend::Declining {
            GoalType::EmotionalBalance
        } else {
            // Random goal
            match rng.gen_range(0..3) {
                0 => GoalType::SuccessfulExplorations,
                1 => GoalType::LearnWords,
                _ => GoalType::ExploreCluster,
            }
        };

        let (description, target) = match goal_type {
            GoalType::CureBoredom => ("Réduire l'ennui par l'exploration".to_string(), 1.0),
            GoalType::SuccessfulExplorations => ("Réussir 5 explorations".to_string(), 5.0),
            GoalType::ImproveStrategy => ("Améliorer mes stratégies".to_string(), 1.0),
            GoalType::LearnWords => ("Apprendre 3 nouveaux mots".to_string(), 3.0),
            GoalType::ExploreCluster => ("Explorer un groupe de mots".to_string(), 3.0),
            GoalType::EmotionalBalance => ("Retrouver mon équilibre".to_string(), 0.5),
        };

        let goal = InternalGoal::new(
            self.next_goal_id,
            description.clone(),
            goal_type,
            target,
            current_tick,
        );

        tracing::info!("🎯 NEW GOAL: {}", description);
        self.goals.push(goal);
        self.next_goal_id += 1;
    }

    /// Get strategy recommendation for a word pair
    pub fn classify_exploration_strategy(
        &self,
        _word1: &str,
        _word2: &str,
        word1_category: Option<&str>,
        word2_category: Option<&str>,
        word1_valence: f32,
        word2_valence: f32,
        are_in_same_cluster: bool,
    ) -> StrategyType {
        // Determine what kind of exploration this is

        // Cross-category if categories differ
        if let (Some(cat1), Some(cat2)) = (word1_category, word2_category) {
            if cat1 != cat2 {
                return StrategyType::CrossCategory;
            }
        }

        // Emotional if both have strong valence
        if word1_valence.abs() > 0.3 && word2_valence.abs() > 0.3 {
            return StrategyType::Emotional;
        }

        // Semantic if in same cluster
        if are_in_same_cluster {
            return StrategyType::Semantic;
        }

        // Default to random
        StrategyType::Random
    }

    /// Get a summary of strategy performance
    pub fn strategy_summary(&self) -> String {
        let mut lines = vec!["📈 Strategy Performance:".to_string()];

        let mut sorted: Vec<_> = self.strategies.iter().collect();
        sorted.sort_by(|a, b| {
            b.1.avg_reward.partial_cmp(&a.1.avg_reward).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (strategy_type, strategy) in sorted {
            if strategy.usage_count > 0 {
                lines.push(format!(
                    "  {:15} avg:{:.2} best:{:.2} uses:{}",
                    strategy_type.name(),
                    strategy.avg_reward,
                    strategy.best_reward,
                    strategy.usage_count
                ));
            }
        }

        lines.join("\n")
    }

    /// Get current goals as string
    pub fn goals_summary(&self) -> String {
        if self.goals.is_empty() {
            return "No active goals".to_string();
        }

        let mut lines = vec!["🎯 Current Goals:".to_string()];
        for goal in &self.goals {
            let progress_bar = "█".repeat((goal.progress * 10.0) as usize);
            let empty_bar = "░".repeat(10 - (goal.progress * 10.0) as usize);
            lines.push(format!(
                "  [{}{}] {:.0}% {}",
                progress_bar, empty_bar,
                goal.progress * 100.0,
                goal.description
            ));
        }

        lines.join("\n")
    }
}

impl Default for MetaLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_internal_reward_computation() {
        let reward = InternalReward::compute(
            0.7,   // coherence
            0.8,   // novelty
            0.6,   // intensity achieved
            0.0,   // emotional before
            0.2,   // emotional after (improved)
            0.3,   // expected intensity
        );

        assert!(reward.total_score > 0.4, "Good exploration should have positive reward");
        assert!(reward.surprise > 0.0, "Unexpected intensity should cause surprise");
        assert!(reward.emotional_delta > 0.0, "Emotional improvement should be positive");
    }

    #[test]
    fn test_strategy_selection() {
        let mut learner = MetaLearner::new();

        // Record some good results for Semantic strategy
        for _ in 0..5 {
            learner.strategies.get_mut(&StrategyType::Semantic)
                .unwrap()
                .record_usage(0.8, 100);
        }

        // After enough good results, Semantic should be favored
        learner.config.exploration_rate = 0.0; // No random exploration
        let selected = learner.select_strategy(200);

        // Should select Semantic (best performing)
        assert_eq!(selected, StrategyType::Semantic);
    }

    #[test]
    fn test_goal_progress() {
        let mut goal = InternalGoal::new(
            1,
            "Test goal".to_string(),
            GoalType::SuccessfulExplorations,
            5.0,
            0,
        );

        assert!(!goal.completed);

        goal.increment(3.0);
        assert_eq!(goal.progress, 0.6);

        goal.increment(2.0);
        assert!(goal.completed);
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new();

        // Record some positive explorations
        for _ in 0..5 {
            let reward = InternalReward::compute(0.7, 0.5, 0.5, 0.0, 0.1, 0.3);
            tracker.record_exploration(&reward);
        }

        assert!(tracker.competence_level > 0.0);
        assert!(tracker.recent_successes > 0);
    }
}
