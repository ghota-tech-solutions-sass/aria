//! Internal Reward - ARIA evaluates herself
//!
//! Instead of relying on external feedback ("Bravo!"), ARIA computes
//! her own reward based on coherence, surprise, and emotional state.

use serde::{Deserialize, Serialize};

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
