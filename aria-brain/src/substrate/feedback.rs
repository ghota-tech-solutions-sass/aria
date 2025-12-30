//! Feedback Processing for ARIA's Substrate
//!
//! ## Physical Intelligence (Session 20)
//!
//! Feedback no longer updates "word valences" - there are no words.
//! Feedback directly modifies ARIA's emotional state and adaptive parameters.
//! The cells learn through resonance, not through semantic reinforcement.

use super::*;

impl Substrate {
    /// Process feedback from user (Bravo!, Non, etc.)
    ///
    /// **Physical Intelligence**: Feedback affects emotional state and behavior,
    /// not word valences. The substrate learns through resonance patterns.
    pub(super) fn process_feedback(&self, label: &str, _current_tick: u64) {
        // Feedback should be short, dedicated messages
        if label.len() > 15 {
            return;
        }

        let lower_label = label.to_lowercase().trim().to_string();

        // Feedback detection
        let positive_feedback = [
            "bravo", "bien!", "super", "génial", "parfait", "excellent",
            "good", "great", "yes", "perfect", "awesome", "👏", "👍", "oui"
        ];

        let negative_feedback = [
            "non", "mauvais", "faux", "arrête", "stop",
            "no", "wrong", "bad", "👎"
        ];

        let is_positive = positive_feedback.iter().any(|w|
            lower_label == *w || lower_label.starts_with(&format!("{} ", w)) || lower_label.starts_with(&format!("{}!", w))
        );
        let is_negative = negative_feedback.iter().any(|w|
            lower_label == *w || lower_label.starts_with(&format!("{} ", w)) || lower_label.starts_with(&format!("{}!", w))
        );

        if !is_positive && !is_negative {
            return;
        }

        // NOTE: Word valence updates removed in Session 20 (Physical Intelligence)
        // Cells learn through resonance, not through semantic reinforcement

        // Update emotional state
        {
            let mut emotional = self.emotional_state.write();
            if is_positive {
                emotional.happiness = (emotional.happiness + 0.3).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort + 0.2).clamp(-1.0, 1.0);
                tracing::info!("⚡ FEEDBACK POSITIVE: happiness={:.2}, comfort={:.2}",
                    emotional.happiness, emotional.comfort);
            } else {
                emotional.happiness = (emotional.happiness - 0.2).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort - 0.1).clamp(-1.0, 1.0);
                tracing::info!("⚡ FEEDBACK NEGATIVE: happiness={:.2}, comfort={:.2}",
                    emotional.happiness, emotional.comfort);
            }
        }

        // Update adaptive params (behavior modification)
        {
            let mut params = self.adaptive_params.write();
            if is_positive {
                params.reinforce_positive();
            } else {
                params.reinforce_negative();
            }
        }
    }

    /// Sync current adaptive params to long-term memory
    pub(super) fn sync_adaptive_params_to_memory(&self) {
        let params = self.adaptive_params.read();
        let mut mem = self.memory.write();

        mem.adaptive_emission_threshold = params.emission_threshold;
        mem.adaptive_response_probability = params.response_probability;
        mem.adaptive_learning_rate = params.learning_rate;
        mem.adaptive_spontaneity = params.spontaneity;
        mem.adaptive_feedback_positive = params.positive_count();
        mem.adaptive_feedback_negative = params.negative_count();
    }
}
