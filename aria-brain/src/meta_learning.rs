//! Meta-Learning System for ARIA
//!
//! This module implements ARIA's ability to learn HOW to learn.
//! Instead of relying on external feedback, ARIA evaluates herself
//! and develops her own learning strategies.
//!
//! Key concepts:
//! - InternalReward: ARIA evaluates her own explorations
//! - ExplorationStrategy: Learned patterns of exploration
//! - MetaLearner: Tracks and improves learning strategies
//! - SelfAssessment: Awareness of progress and competence

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rand::Rng;

// ============================================================================
// INTERNAL REWARD - ARIA evaluates herself
// ============================================================================

/// Internal reward computed by ARIA herself, without external feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalReward {
    /// How coherent was the exploration? (cells agreed)
    pub coherence: f32,
    /// How surprising/novel was it?
    pub surprise: f32,
    /// Did it satisfy curiosity?
    pub curiosity_satisfaction: f32,
    /// Did emotional state improve?
    pub emotional_delta: f32,
    /// Combined score (computed)
    pub total_score: f32,
}

impl InternalReward {
    /// Compute internal reward from exploration results
    pub fn compute(
        coherence: f32,           // 0-1: how coherent the cell response was
        novelty: f32,             // 0-1: how novel the combination was
        intensity_achieved: f32,  // 0-1: how strong the response was
        emotional_before: f32,    // -1 to 1: emotional state before
        emotional_after: f32,     // -1 to 1: emotional state after
        expected_intensity: f32,  // what we expected to happen
    ) -> Self {
        // Coherence reward: cells working together = good
        let coherence_reward = coherence;

        // Surprise reward: unexpected strong response = very interesting!
        let surprise = if intensity_achieved > expected_intensity {
            (intensity_achieved - expected_intensity).min(1.0)
        } else {
            0.0
        };

        // Curiosity satisfaction: novelty + coherence = satisfied curiosity
        let curiosity_satisfaction = novelty * 0.5 + coherence * 0.5;

        // Emotional delta: did we feel better after?
        let emotional_delta = (emotional_after - emotional_before).max(-0.5).min(0.5);

        // Total score combines all factors
        // Weights reflect what matters for learning:
        // - Coherence (30%): Internal consistency is fundamental
        // - Surprise (25%): Learning happens when expectations are violated
        // - Curiosity satisfaction (25%): Following curiosity leads to growth
        // - Emotional improvement (20%): Good feelings = good direction
        let total_score = coherence_reward * 0.30
            + surprise * 0.25
            + curiosity_satisfaction * 0.25
            + emotional_delta * 0.20;

        Self {
            coherence: coherence_reward,
            surprise,
            curiosity_satisfaction,
            emotional_delta,
            total_score: total_score.max(0.0).min(1.0),
        }
    }

    /// Is this reward good enough to reinforce the behavior?
    pub fn is_positive(&self) -> bool {
        self.total_score > 0.4
    }

    /// Is this reward very good (should strongly reinforce)?
    pub fn is_excellent(&self) -> bool {
        self.total_score > 0.7
    }
}

// ============================================================================
// EXPLORATION STRATEGY - How ARIA explores
// ============================================================================

/// Type of exploration strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyType {
    /// Combine semantically related words
    Semantic,
    /// Combine phonetically similar words
    Phonetic,
    /// Combine emotionally charged words
    Emotional,
    /// Combine words from different categories (noun + verb, etc.)
    CrossCategory,
    /// Random exploration (discovery mode)
    Random,
    /// Repeat successful patterns
    Exploitation,
}

impl StrategyType {
    pub fn all() -> Vec<StrategyType> {
        vec![
            StrategyType::Semantic,
            StrategyType::Phonetic,
            StrategyType::Emotional,
            StrategyType::CrossCategory,
            StrategyType::Random,
            StrategyType::Exploitation,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            StrategyType::Semantic => "semantic",
            StrategyType::Phonetic => "phonetic",
            StrategyType::Emotional => "emotional",
            StrategyType::CrossCategory => "cross-category",
            StrategyType::Random => "random",
            StrategyType::Exploitation => "exploitation",
        }
    }
}

/// A learned exploration strategy with performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationStrategy {
    /// Type of strategy
    pub strategy_type: StrategyType,
    /// Number of times this strategy was used
    pub usage_count: u64,
    /// Total internal reward accumulated
    pub total_reward: f32,
    /// Average internal reward (computed)
    pub avg_reward: f32,
    /// Best reward ever achieved with this strategy
    pub best_reward: f32,
    /// Recent performance (last 10 uses)
    pub recent_rewards: Vec<f32>,
    /// When was this strategy last used
    pub last_used: u64,
}

impl ExplorationStrategy {
    pub fn new(strategy_type: StrategyType) -> Self {
        Self {
            strategy_type,
            usage_count: 0,
            total_reward: 0.0,
            avg_reward: 0.5, // Start neutral
            best_reward: 0.0,
            recent_rewards: Vec::new(),
            last_used: 0,
        }
    }

    /// Record a usage of this strategy
    pub fn record_usage(&mut self, reward: f32, tick: u64) {
        self.usage_count += 1;
        self.total_reward += reward;
        self.avg_reward = self.total_reward / self.usage_count as f32;
        self.best_reward = self.best_reward.max(reward);
        self.last_used = tick;

        // Track recent performance
        self.recent_rewards.push(reward);
        if self.recent_rewards.len() > 10 {
            self.recent_rewards.remove(0);
        }
    }

    /// Get recent average (more responsive to changes)
    pub fn recent_avg(&self) -> f32 {
        if self.recent_rewards.is_empty() {
            self.avg_reward
        } else {
            self.recent_rewards.iter().sum::<f32>() / self.recent_rewards.len() as f32
        }
    }

    /// Calculate selection score (for choosing this strategy)
    pub fn selection_score(&self, current_tick: u64) -> f32 {
        // Balance exploitation (good avg) with exploration (less used)
        let exploitation = self.recent_avg() * 0.7 + self.avg_reward * 0.3;

        // Novelty bonus: strategies not used recently get a boost
        let recency = (current_tick.saturating_sub(self.last_used)) as f32 / 1000.0;
        let novelty_bonus = (recency * 0.1).min(0.3);

        // Uncertainty bonus: less used strategies might be underexplored
        let uncertainty_bonus = if self.usage_count < 5 {
            0.2
        } else if self.usage_count < 20 {
            0.1
        } else {
            0.0
        };

        exploitation + novelty_bonus + uncertainty_bonus
    }
}

// ============================================================================
// INTERNAL GOAL - ARIA sets her own objectives
// ============================================================================

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

// ============================================================================
// PROGRESS TRACKER - Awareness of learning progress
// ============================================================================

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
        word1: &str,
        word2: &str,
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

        // Phonetic if similar sound (simple heuristic)
        if word1.len() >= 2 && word2.len() >= 2 {
            let prefix1 = &word1[..2];
            let prefix2 = &word2[..2];
            if prefix1 == prefix2 {
                return StrategyType::Phonetic;
            }
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
