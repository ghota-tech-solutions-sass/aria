//! Exploration Strategies - How ARIA explores
//!
//! ARIA learns which exploration strategies work best and adapts.

use serde::{Deserialize, Serialize};

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
