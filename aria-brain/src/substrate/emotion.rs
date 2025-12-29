//! Emotional state management for ARIA
//!
//! ARIA's emotional state influences her behavior:
//! - Happy cells are more likely to emit positive signals
//! - Curious cells explore more
//! - Bored cells seek stimulation
//! - Comfort affects willingness to take risks

use serde::{Deserialize, Serialize};

/// ARIA's emotional state - her current mood
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EmotionalState {
    /// Happiness level (-1.0 = sad, 0.0 = neutral, 1.0 = happy)
    pub happiness: f32,

    /// Arousal/excitement level (0.0 = calm, 1.0 = excited)
    pub arousal: f32,

    /// Comfort level (-1.0 = uncomfortable, 1.0 = comfortable)
    pub comfort: f32,

    /// Curiosity level (0.0 = not curious, 1.0 = very curious)
    pub curiosity: f32,

    /// Boredom level (0.0 = engaged, 1.0 = very bored)
    /// This GROWS without interaction
    pub boredom: f32,

    /// Last tick when emotional state was updated
    pub last_update: u64,
}

#[allow(dead_code)]
impl EmotionalState {
    /// Create a new emotional state with default neutral values
    pub fn new() -> Self {
        Self::default()
    }

    /// Decay emotional state over time
    /// - Positive emotions fade back to neutral
    /// - Boredom GROWS without interaction
    pub fn decay(&mut self, current_tick: u64) {
        let ticks_elapsed = current_tick.saturating_sub(self.last_update);
        if ticks_elapsed > 0 {
            // Exponential decay towards neutral
            let decay = 0.999f32.powi(ticks_elapsed as i32);
            self.happiness *= decay;
            self.arousal *= decay;
            self.comfort *= decay;
            self.curiosity *= decay;

            // Boredom GROWS without interaction
            let boredom_growth = 0.0001 * ticks_elapsed as f32;
            self.boredom = (self.boredom + boredom_growth).min(1.0);

            self.last_update = current_tick;
        }
    }

    /// Process incoming signal and update emotional state
    /// Signal content dimensions:
    /// - 28: positive emotion marker
    /// - 29: negative emotion marker
    /// - 30: request marker
    /// - 31: question marker
    pub fn process_signal(&mut self, content: &[f32], intensity: f32, current_tick: u64) {
        // First decay existing state
        self.decay(current_tick);

        // Extract emotional markers from signal
        let positive = content.get(28).copied().unwrap_or(0.0);
        let negative = content.get(29).copied().unwrap_or(0.0);
        let _request = content.get(30).copied().unwrap_or(0.0);
        let question = content.get(31).copied().unwrap_or(0.0);

        // How much the signal affects emotional state
        let momentum = 0.3;

        // Positive signals increase happiness and comfort
        if positive > 0.0 {
            self.happiness = (self.happiness + positive * momentum * intensity).clamp(-1.0, 1.0);
            self.comfort = (self.comfort + 0.2 * momentum * intensity).clamp(-1.0, 1.0);
        }

        // Negative signals decrease happiness and comfort
        if negative < 0.0 {
            self.happiness = (self.happiness + negative * momentum * intensity).clamp(-1.0, 1.0);
            self.comfort = (self.comfort - 0.3 * momentum * intensity).clamp(-1.0, 1.0);
        }

        // Questions increase curiosity and arousal
        if question > 0.0 {
            self.curiosity = (self.curiosity + question * momentum * intensity).clamp(0.0, 1.0);
            self.arousal = (self.arousal + 0.1 * momentum).clamp(0.0, 1.0);
        }

        // Any interaction reduces boredom and increases arousal
        self.arousal = (self.arousal + 0.05 * intensity).clamp(0.0, 1.0);
        self.boredom = (self.boredom - 0.3 * intensity).max(0.0);
    }

    /// Get an emotional marker based on current state
    /// Used to add emotional coloring to ARIA's responses
    pub fn get_emotional_marker(&self) -> Option<&'static str> {
        let threshold = 0.3;

        if self.happiness > threshold && self.happiness >= self.curiosity.abs() {
            if self.happiness > 0.6 { Some("♥") } else { Some("~") }
        } else if self.curiosity > threshold {
            if self.arousal > 0.5 { Some("!") } else { Some("?") }
        } else if self.happiness < -threshold {
            Some("...")
        } else if self.arousal > 0.6 {
            Some("!")
        } else {
            None
        }
    }

    /// Get a human-readable description of ARIA's current mood
    pub fn mood_description(&self) -> &'static str {
        if self.happiness > 0.5 && self.arousal > 0.5 {
            "joyeux"
        } else if self.happiness > 0.5 {
            "content"
        } else if self.curiosity > 0.5 {
            "curieux"
        } else if self.happiness < -0.3 {
            "triste"
        } else if self.arousal > 0.6 {
            "excité"
        } else {
            "calme"
        }
    }

    /// Check if ARIA is in a positive emotional state
    pub fn is_positive(&self) -> bool {
        self.happiness > 0.2 && self.comfort > -0.3
    }

    /// Check if ARIA is bored (needs stimulation)
    pub fn is_bored(&self) -> bool {
        self.boredom > 0.5
    }

    /// Check if ARIA is lonely (hasn't had interaction in a while)
    pub fn needs_attention(&self) -> bool {
        self.boredom > 0.3 && self.arousal < 0.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_state_decay() {
        let mut state = EmotionalState {
            happiness: 1.0,
            arousal: 1.0,
            comfort: 1.0,
            curiosity: 1.0,
            boredom: 0.0,
            last_update: 0,
        };

        state.decay(1000);

        assert!(state.happiness < 1.0, "Happiness should decay");
        assert!(state.boredom > 0.0, "Boredom should grow");
    }

    #[test]
    fn test_emotional_markers() {
        let mut state = EmotionalState::default();

        // Happy state
        state.happiness = 0.7;
        assert_eq!(state.get_emotional_marker(), Some("♥"));

        // Curious state
        state.happiness = 0.0;
        state.curiosity = 0.5;
        state.arousal = 0.3;
        assert_eq!(state.get_emotional_marker(), Some("?"));

        // Neutral state
        state.curiosity = 0.0;
        assert_eq!(state.get_emotional_marker(), None);
    }

    #[test]
    fn test_mood_description() {
        let mut state = EmotionalState::default();
        assert_eq!(state.mood_description(), "calme");

        state.happiness = 0.6;
        state.arousal = 0.6;
        assert_eq!(state.mood_description(), "joyeux");
    }
}
