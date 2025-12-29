//! Self-Modification - ARIA changes herself (Session 16)
//!
//! This is a major step toward AGI: ARIA consciously analyzes her
//! performance and decides which parameters to modify.

use serde::{Deserialize, Serialize};
use super::progress::{ProgressTracker, LearningTrend};

/// A parameter that ARIA can modify
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModifiableParam {
    /// Threshold for emitting responses (0.05-0.5)
    EmissionThreshold,
    /// Probability of responding to input (0.3-1.0)
    ResponseProbability,
    /// Base learning rate (0.1-0.8)
    LearningRate,
    /// Spontaneous speech probability (0.01-0.3)
    Spontaneity,
    /// Exploration rate in meta-learning (0.1-0.5)
    ExplorationRate,
}

impl ModifiableParam {
    pub fn all() -> &'static [ModifiableParam] {
        &[
            ModifiableParam::EmissionThreshold,
            ModifiableParam::ResponseProbability,
            ModifiableParam::LearningRate,
            ModifiableParam::Spontaneity,
            ModifiableParam::ExplorationRate,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            ModifiableParam::EmissionThreshold => "emission_threshold",
            ModifiableParam::ResponseProbability => "response_probability",
            ModifiableParam::LearningRate => "learning_rate",
            ModifiableParam::Spontaneity => "spontaneity",
            ModifiableParam::ExplorationRate => "exploration_rate",
        }
    }

    pub fn range(&self) -> (f32, f32) {
        match self {
            ModifiableParam::EmissionThreshold => (0.05, 0.5),
            ModifiableParam::ResponseProbability => (0.3, 1.0),
            ModifiableParam::LearningRate => (0.1, 0.8),
            ModifiableParam::Spontaneity => (0.01, 0.3),
            ModifiableParam::ExplorationRate => (0.1, 0.5),
        }
    }
}

/// Snapshot of metrics at a given time (for measuring modification success)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsSnapshot {
    pub competence_level: f32,
    pub learning_quality: f32,
    pub recent_successes: u32,
    pub recent_failures: u32,
    pub tick: u64,
}

impl MetricsSnapshot {
    pub fn from_progress(progress: &ProgressTracker, tick: u64) -> Self {
        Self {
            competence_level: progress.competence_level,
            learning_quality: progress.learning_quality,
            recent_successes: progress.recent_successes,
            recent_failures: progress.recent_failures,
            tick,
        }
    }

    /// Calculate overall score (higher = better)
    pub fn score(&self) -> f32 {
        let success_rate = if self.recent_successes + self.recent_failures > 0 {
            self.recent_successes as f32 / (self.recent_successes + self.recent_failures) as f32
        } else {
            0.5
        };
        self.competence_level * 0.4 + self.learning_quality * 0.3 + success_rate * 0.3
    }
}

/// A proposed self-modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModification {
    /// Which parameter to modify
    pub param: ModifiableParam,
    /// Current value
    pub current_value: f32,
    /// Proposed new value
    pub new_value: f32,
    /// Change direction (-1, 0, +1)
    pub direction: i8,
    /// Reasoning for this change
    pub reasoning: String,
    /// Confidence in this change (0-1)
    pub confidence: f32,
    /// When this was proposed
    pub proposed_at: u64,
    /// Was this applied?
    pub applied: bool,
    /// Baseline metrics when this was applied (for measuring success)
    #[serde(default)]
    pub baseline: Option<MetricsSnapshot>,
    /// Was this modification evaluated?
    #[serde(default)]
    pub evaluated: bool,
    /// Was this modification successful? (metrics improved)
    #[serde(default)]
    pub was_successful: Option<bool>,
}

impl SelfModification {
    pub fn new(
        param: ModifiableParam,
        current: f32,
        new: f32,
        reasoning: String,
        confidence: f32,
        tick: u64,
    ) -> Self {
        let direction = if new > current { 1 } else if new < current { -1 } else { 0 };
        Self {
            param,
            current_value: current,
            new_value: new,
            direction,
            reasoning,
            confidence,
            proposed_at: tick,
            applied: false,
            baseline: None,
            evaluated: false,
            was_successful: None,
        }
    }

    /// Set baseline metrics (called when modification is applied)
    pub fn set_baseline(&mut self, progress: &ProgressTracker, tick: u64) {
        self.baseline = Some(MetricsSnapshot::from_progress(progress, tick));
    }

    pub fn clamp_to_range(&mut self) {
        let (min, max) = self.param.range();
        self.new_value = self.new_value.max(min).min(max);
    }
}

/// Current parameter values (passed to analyzer)
#[derive(Debug, Clone)]
pub struct CurrentParams {
    pub emission_threshold: f32,
    pub response_probability: f32,
    pub learning_rate: f32,
    pub spontaneity: f32,
    pub exploration_rate: f32,
}

/// Self-modification engine - ARIA decides how to change herself
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SelfModifier {
    /// History of modifications (for learning from past changes)
    pub modification_history: Vec<SelfModification>,
    /// How often to consider self-modification (in ticks)
    pub modification_interval: u64,
    /// Last tick when modification was considered
    pub last_modification_tick: u64,
    /// Minimum confidence required to apply a change
    pub min_confidence: f32,
    /// Total modifications applied
    pub total_modifications: u64,
    /// Successful modifications (improved metrics)
    pub successful_modifications: u64,
}

impl SelfModifier {
    pub fn new() -> Self {
        Self {
            modification_history: Vec::new(),
            modification_interval: 2_000, // Every 2k ticks (faster for testing)
            last_modification_tick: 0,
            min_confidence: 0.6,
            total_modifications: 0,
            successful_modifications: 0,
        }
    }

    /// Analyze current state and propose modifications
    pub fn analyze_and_propose(
        &self,
        progress: &ProgressTracker,
        current_params: &CurrentParams,
        current_tick: u64,
    ) -> Vec<SelfModification> {
        let mut proposals = Vec::new();

        // === ANALYSIS ===

        // 1. If learning quality is declining, we need to change something
        if progress.trend == LearningTrend::Declining {
            // Try increasing learning rate
            if current_params.learning_rate < 0.6 {
                proposals.push(SelfModification::new(
                    ModifiableParam::LearningRate,
                    current_params.learning_rate,
                    current_params.learning_rate + 0.1,
                    "Apprentissage en déclin → augmenter learning_rate".to_string(),
                    0.7,
                    current_tick,
                ));
            }

            // Or try more exploration
            if current_params.exploration_rate < 0.4 {
                proposals.push(SelfModification::new(
                    ModifiableParam::ExplorationRate,
                    current_params.exploration_rate,
                    current_params.exploration_rate + 0.1,
                    "Apprentissage en déclin → explorer plus".to_string(),
                    0.6,
                    current_tick,
                ));
            }
        }

        // 2. If too many failures, lower emission threshold (be more selective)
        let failure_rate = if progress.recent_successes + progress.recent_failures > 0 {
            progress.recent_failures as f32 /
                (progress.recent_successes + progress.recent_failures) as f32
        } else {
            0.0
        };

        if failure_rate > 0.6 && current_params.emission_threshold < 0.4 {
            proposals.push(SelfModification::new(
                ModifiableParam::EmissionThreshold,
                current_params.emission_threshold,
                current_params.emission_threshold + 0.05,
                format!("Taux d'échec élevé ({:.0}%) → être plus sélectif", failure_rate * 100.0),
                0.65,
                current_tick,
            ));
        }

        // 3. If very few responses, increase response probability
        let total_responses = progress.recent_successes + progress.recent_failures;
        if total_responses < 5 && current_params.response_probability < 0.9 {
            proposals.push(SelfModification::new(
                ModifiableParam::ResponseProbability,
                current_params.response_probability,
                current_params.response_probability + 0.1,
                "Peu de réponses → augmenter probabilité de réponse".to_string(),
                0.7,
                current_tick,
            ));
        }

        // 4. If learning is improving and stable, we can be more spontaneous
        if progress.trend == LearningTrend::Improving && progress.competence_level > 0.5 {
            if current_params.spontaneity < 0.2 {
                proposals.push(SelfModification::new(
                    ModifiableParam::Spontaneity,
                    current_params.spontaneity,
                    current_params.spontaneity + 0.03,
                    format!("Compétence élevée ({:.0}%) → plus de spontanéité",
                        progress.competence_level * 100.0),
                    0.6,
                    current_tick,
                ));
            }
        }

        // 5. If competence is high, reduce exploration (exploit more)
        if progress.competence_level > 0.7 && current_params.exploration_rate > 0.2 {
            proposals.push(SelfModification::new(
                ModifiableParam::ExplorationRate,
                current_params.exploration_rate,
                current_params.exploration_rate - 0.05,
                format!("Compétence très élevée ({:.0}%) → exploiter davantage",
                    progress.competence_level * 100.0),
                0.65,
                current_tick,
            ));
        }

        // Clamp all proposals to valid ranges
        for p in &mut proposals {
            p.clamp_to_range();
        }

        proposals
    }

    /// Decide which modification to apply (if any)
    pub fn decide(&mut self, proposals: Vec<SelfModification>, progress: &ProgressTracker, current_tick: u64) -> Option<SelfModification> {
        if proposals.is_empty() {
            return None;
        }

        // Find the highest confidence proposal
        let best = proposals.into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap());

        if let Some(mut modification) = best {
            if modification.confidence >= self.min_confidence {
                modification.applied = true;
                modification.set_baseline(progress, current_tick);
                self.modification_history.push(modification.clone());
                self.total_modifications += 1;

                tracing::info!(
                    "🔧 AUTO-MODIFICATION: {} {:.3} → {:.3} (confidence: {:.0}%)",
                    modification.param.name(),
                    modification.current_value,
                    modification.new_value,
                    modification.confidence * 100.0
                );
                tracing::info!("   Raison: {}", modification.reasoning);

                return Some(modification);
            }
        }

        None
    }

    /// Check outcomes of past modifications
    /// Returns number of modifications that were evaluated as successful
    pub fn check_outcomes(&mut self, progress: &ProgressTracker, current_tick: u64, eval_delay: u64) -> u32 {
        let current_snapshot = MetricsSnapshot::from_progress(progress, current_tick);
        let current_score = current_snapshot.score();
        let mut successes = 0;

        for modification in &mut self.modification_history {
            // Skip already evaluated or unapplied modifications
            if modification.evaluated || !modification.applied {
                continue;
            }

            // Check if enough time has passed
            if let Some(ref baseline) = modification.baseline {
                if current_tick >= baseline.tick + eval_delay {
                    let baseline_score = baseline.score();
                    let improved = current_score > baseline_score + 0.01; // Small threshold

                    modification.evaluated = true;
                    modification.was_successful = Some(improved);

                    if improved {
                        self.successful_modifications += 1;
                        successes += 1;
                        tracing::info!(
                            "✅ MODIFICATION SUCCESS: {} {:.3} → {:.3} (score: {:.2} → {:.2})",
                            modification.param.name(),
                            modification.current_value,
                            modification.new_value,
                            baseline_score,
                            current_score
                        );
                    } else {
                        tracing::info!(
                            "❌ MODIFICATION NEUTRAL/FAIL: {} {:.3} → {:.3} (score: {:.2} → {:.2})",
                            modification.param.name(),
                            modification.current_value,
                            modification.new_value,
                            baseline_score,
                            current_score
                        );
                    }
                }
            }
        }

        successes
    }

    /// Record that a modification was successful (metrics improved after)
    pub fn record_outcome(&mut self, success: bool) {
        if success {
            self.successful_modifications += 1;
        }
    }

    /// Get success rate of modifications
    pub fn success_rate(&self) -> f32 {
        if self.total_modifications > 0 {
            self.successful_modifications as f32 / self.total_modifications as f32
        } else {
            0.0
        }
    }

    /// Should we consider self-modification now?
    pub fn should_modify(&self, current_tick: u64) -> bool {
        current_tick >= self.last_modification_tick + self.modification_interval
    }

    /// Update last modification tick
    pub fn mark_modified(&mut self, tick: u64) {
        self.last_modification_tick = tick;
    }

    /// Get recent modifications for display
    pub fn recent_modifications(&self, limit: usize) -> Vec<&SelfModification> {
        self.modification_history.iter().rev().take(limit).collect()
    }
}
