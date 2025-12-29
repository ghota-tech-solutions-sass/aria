//! Self-Modification & Meta-Learning for ARIA's Substrate
//!
//! ARIA consciously analyzes herself and decides how to change.
//! This is a major step toward AGI.

use super::*;
use crate::memory::{CurrentParams, InternalReward};

impl Substrate {
    /// ARIA consciously analyzes herself and decides to change parameters
    pub(super) fn maybe_self_modify(&self, current_tick: u64) {
        // Check if it's time to consider self-modification
        let should_modify = {
            let memory = self.memory.read();
            memory.self_modifier.should_modify(current_tick)
        };

        if !should_modify {
            return;
        }

        // Gather current state
        let (progress, current_params, _exploration_rate) = {
            let memory = self.memory.read();
            let params = self.adaptive_params.read();

            let current = CurrentParams {
                emission_threshold: params.emission_threshold,
                response_probability: params.response_probability,
                learning_rate: params.learning_rate,
                spontaneity: params.spontaneity,
                exploration_rate: memory.meta_learner.config.exploration_rate,
            };

            (memory.meta_learner.progress.clone(), current, memory.meta_learner.config.exploration_rate)
        };

        // First, check outcomes of past modifications (evaluate after 2000 ticks)
        {
            let mut memory = self.memory.write();
            memory.self_modifier.check_outcomes(&progress, current_tick, 2000);
        }

        // Analyze and propose new modifications
        let proposals = {
            let memory = self.memory.read();
            memory.self_modifier.analyze_and_propose(&progress, &current_params, current_tick)
        };

        // Decide and apply
        if !proposals.is_empty() {
            let mut memory = self.memory.write();
            memory.self_modifier.mark_modified(current_tick);

            if let Some(modification) = memory.self_modifier.decide(proposals, &progress, current_tick) {
                // Apply the modification
                match modification.param {
                    crate::memory::ModifiableParam::EmissionThreshold => {
                        let mut params = self.adaptive_params.write();
                        params.emission_threshold = modification.new_value;
                    }
                    crate::memory::ModifiableParam::ResponseProbability => {
                        let mut params = self.adaptive_params.write();
                        params.response_probability = modification.new_value;
                    }
                    crate::memory::ModifiableParam::LearningRate => {
                        let mut params = self.adaptive_params.write();
                        params.learning_rate = modification.new_value;
                    }
                    crate::memory::ModifiableParam::Spontaneity => {
                        let mut params = self.adaptive_params.write();
                        params.spontaneity = modification.new_value;
                    }
                    crate::memory::ModifiableParam::ExplorationRate => {
                        memory.meta_learner.config.exploration_rate = modification.new_value;
                    }
                }

                // Sync to memory
                drop(memory);
                self.sync_adaptive_params_to_memory();
            }
        } else {
            // Just mark that we checked
            let mut memory = self.memory.write();
            memory.self_modifier.mark_modified(current_tick);
        }
    }

    /// Evaluate ANY response (not just explorations)
    pub fn evaluate_response(&self, coherence: f32, intensity: f32, current_tick: u64) {
        // Get emotional state
        let emotional = self.emotional_state.read();
        let emotional_after = emotional.happiness;
        let curiosity = emotional.curiosity;
        drop(emotional);

        // Check if this was an exploration (has last_exploration set)
        let last_expl = self.last_exploration.read().clone();
        let (novelty, is_exploration) = if let Some(ref expl) = last_expl {
            // This was an exploration - get novelty from history
            let memory = self.memory.read();
            let key = expl.to_lowercase();
            let nov = memory.exploration_history.get(&key)
                .map(|r| 1.0 / (1.0 + r.attempts as f32 * 0.2))
                .unwrap_or(1.0);
            (nov, true)
        } else {
            // Regular response - compute novelty from coherence variance
            // High coherence = cells agree = less novel but more reliable
            let nov = 0.3 + (1.0 - coherence) * 0.4; // Moderate novelty for conversations
            (nov, false)
        };

        // Use curiosity as "before" state proxy
        let emotional_before = curiosity * 0.5;

        // Log what we're evaluating
        if is_exploration {
            tracing::debug!("📊 Evaluating exploration: coherence={:.2}", coherence);
        } else {
            tracing::debug!("📊 Evaluating response: coherence={:.2}", coherence);
        }

        // Compute internal reward
        self.compute_internal_reward(
            coherence,
            intensity,
            novelty,
            emotional_before,
            emotional_after,
            current_tick,
        );

        // Clear last_exploration after evaluation (if it was one)
        if is_exploration {
            let mut last_expl_write = self.last_exploration.write();
            *last_expl_write = None;
        }
    }

    /// Compute internal reward for an exploration (ARIA evaluates herself)
    fn compute_internal_reward(
        &self,
        coherence: f32,
        intensity: f32,
        novelty: f32,
        emotional_before: f32,
        emotional_after: f32,
        current_tick: u64,
    ) {
        // Compute the internal reward
        let expected_intensity = 0.3; // Default expectation
        let reward = InternalReward::compute(
            coherence,
            novelty,
            intensity,
            emotional_before,
            emotional_after,
            expected_intensity,
        );

        // Update MetaLearner
        let mut memory = self.memory.write();
        memory.meta_learner.evaluate_exploration(
            coherence,
            novelty,
            intensity,
            emotional_before,
            emotional_after,
            expected_intensity,
            current_tick,
        );

        // Log the internal evaluation
        if reward.is_excellent() {
            tracing::info!("🌟 INTERNAL REWARD: {:.2} (EXCELLENT!) - coherence:{:.2} surprise:{:.2}",
                reward.total_score, reward.coherence, reward.surprise);
        } else if reward.is_positive() {
            tracing::info!("✅ INTERNAL REWARD: {:.2} (good) - coherence:{:.2} surprise:{:.2}",
                reward.total_score, reward.coherence, reward.surprise);
        } else {
            tracing::debug!("📉 INTERNAL REWARD: {:.2} (low) - coherence:{:.2}",
                reward.total_score, reward.coherence);
        }

        // Maybe create a new goal based on current state
        let boredom = self.emotional_state.read().boredom;
        let (total, successful, _failed) = memory.exploration_stats();
        let success_rate = if total > 0 { successful as f32 / total as f32 } else { 0.5 };
        memory.meta_learner.maybe_create_goal(current_tick, boredom, success_rate);
    }
}
