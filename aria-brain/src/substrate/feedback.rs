//! Feedback Processing for ARIA's Substrate
//!
//! Handles positive/negative feedback and adaptive parameter synchronization.

use super::*;

impl Substrate {
    /// Process feedback from user (Bravo!, Non, etc.)
    pub(super) fn process_feedback(&self, label: &str, _current_tick: u64) {
        // Feedback should be short, dedicated messages - not part of conversation
        // If the message is too long (> 15 chars), it's probably not feedback
        if label.len() > 15 {
            return;
        }

        let lower_label = label.to_lowercase().trim().to_string();

        // Exact match or starts with feedback words (allow "Bravo!" but not "C'est bien fait")
        let positive_feedback = [
            "bravo", "bien!", "super", "génial", "parfait", "excellent",
            "good", "great", "yes", "perfect", "awesome", "👏", "👍", "oui"
        ];

        let negative_feedback = [
            "non", "mauvais", "faux", "arrête", "stop",
            "no", "wrong", "bad", "👎"
        ];

        // Check if the message IS the feedback (not contains it)
        let is_positive = positive_feedback.iter().any(|w|
            lower_label == *w || lower_label.starts_with(&format!("{} ", w)) || lower_label.starts_with(&format!("{}!", w))
        );
        let is_negative = negative_feedback.iter().any(|w|
            lower_label == *w || lower_label.starts_with(&format!("{} ", w)) || lower_label.starts_with(&format!("{}!", w))
        );

        if !is_positive && !is_negative {
            return;
        }

        // Step 1: Get recent expressions (release lock immediately)
        let recent_expr: Vec<String> = self.recent_expressions.read().clone();

        // Step 2: Update word valences in memory (separate scope)
        {
            let mut memory = self.memory.write();
            for word in &recent_expr {
                if let Some(freq) = memory.word_frequencies.get_mut(word) {
                    let old = freq.emotional_valence;
                    if is_positive {
                        freq.emotional_valence = (freq.emotional_valence + 0.3).clamp(-2.0, 2.0);
                        freq.count += 2;
                        tracing::info!("FEEDBACK POSITIVE! '{}' ({:.2} → {:.2})", word, old, freq.emotional_valence);
                    } else {
                        freq.emotional_valence = (freq.emotional_valence - 0.3).clamp(-2.0, 2.0);
                        tracing::info!("FEEDBACK NEGATIVE! '{}' ({:.2} → {:.2})", word, old, freq.emotional_valence);
                    }
                }
            }
        } // memory lock released here

        // Step 3: Update emotional state (separate scope)
        {
            let mut emotional = self.emotional_state.write();
            if is_positive {
                emotional.happiness = (emotional.happiness + 0.3).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort + 0.2).clamp(-1.0, 1.0);
            } else {
                emotional.happiness = (emotional.happiness - 0.2).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort - 0.1).clamp(-1.0, 1.0);
            }
        } // emotional lock released here

        // Step 4: Update adaptive params (separate scope)
        {
            let mut params = self.adaptive_params.write();
            if is_positive {
                params.reinforce_positive();
            } else {
                params.reinforce_negative();
            }
        } // params lock released here

        // Step 5: Update exploration history (separate scope)
        {
            let last_expl = self.last_exploration.read().clone();
            if let Some(combination) = last_expl {
                let mut memory = self.memory.write();
                memory.feedback_exploration(&combination, is_positive);
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
