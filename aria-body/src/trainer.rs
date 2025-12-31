//! Auto-Training Module for ARIA
//!
//! Automatically teaches ARIA vocabulary, associations, and emotional responses.
//! Toggle with 'a' key in visual mode.

use rand::seq::SliceRandom;
use rand::Rng;
use std::time::{Duration, Instant};

/// Training phases - progression from basic to complex
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingPhase {
    /// Basic words: objects, animals, emotions
    Vocabulary,
    /// Word pairs: "chat moka", "joli chat"
    Associations,
    /// Simple interactions: greetings, questions
    Conversation,
    /// Emotional responses: "Bravo!", "Non", varying tones
    Emotional,
}

impl TrainingPhase {
    pub fn name(&self) -> &'static str {
        match self {
            TrainingPhase::Vocabulary => "Vocabulary",
            TrainingPhase::Associations => "Associations",
            TrainingPhase::Conversation => "Conversation",
            TrainingPhase::Emotional => "Emotional",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            TrainingPhase::Vocabulary => TrainingPhase::Associations,
            TrainingPhase::Associations => TrainingPhase::Conversation,
            TrainingPhase::Conversation => TrainingPhase::Emotional,
            TrainingPhase::Emotional => TrainingPhase::Vocabulary,
        }
    }
}

/// Auto-trainer state
pub struct Trainer {
    /// Is auto-training enabled?
    pub enabled: bool,

    /// Current training phase
    pub phase: TrainingPhase,

    /// Last time we sent a stimulus
    last_stimulus: Instant,

    /// Interval between stimuli (adapts based on ARIA's responses)
    pub interval: Duration,

    /// Last stimulus sent (for feedback)
    last_sent: Option<String>,

    /// Count of stimuli sent this phase
    stimuli_count: usize,

    /// Count of responses received this phase
    responses_count: usize,

    /// Stimuli per phase before advancing
    stimuli_per_phase: usize,

    /// Basic vocabulary words
    vocabulary: Vec<&'static str>,

    /// Word pairs for associations
    associations: Vec<(&'static str, &'static str)>,

    /// Conversation starters
    conversations: Vec<&'static str>,

    /// Emotional expressions
    emotional: Vec<&'static str>,

    /// Positive feedback variants
    positive_feedback: Vec<&'static str>,

    /// Negative feedback variants
    negative_feedback: Vec<&'static str>,

    /// Should we send feedback next?
    pending_feedback: Option<bool>,

    /// Stats
    pub total_sent: usize,
    pub total_responses: usize,
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            enabled: false,
            phase: TrainingPhase::Vocabulary,
            last_stimulus: Instant::now(),
            interval: Duration::from_secs(3),
            last_sent: None,
            stimuli_count: 0,
            responses_count: 0,
            stimuli_per_phase: 20,

            // French vocabulary - objects, animals, emotions, actions
            vocabulary: vec![
                // Animals (Mickael's cats!)
                "moka", "chat", "obrigada",
                // Basic words
                "bonjour", "salut", "merci", "oui", "non",
                // Emotions
                "content", "triste", "curieux", "fatigué",
                // Objects
                "eau", "soleil", "lune", "maison",
                // Actions
                "jouer", "dormir", "manger", "parler",
                // Adjectives
                "joli", "grand", "petit", "bon",
            ],

            associations: vec![
                ("chat", "moka"),
                ("joli", "chat"),
                ("bon", "matin"),
                ("bonne", "nuit"),
                ("petit", "chat"),
                ("chat", "dort"),
                ("chat", "joue"),
                ("soleil", "chaud"),
                ("lune", "nuit"),
                ("eau", "fraiche"),
            ],

            conversations: vec![
                "bonjour",
                "comment vas-tu?",
                "tu es là?",
                "salut ARIA",
                "bonne nuit",
                "qu'est-ce que tu fais?",
                "tu me reconnais?",
                "je suis content",
                "à bientôt",
            ],

            emotional: vec![
                "Bravo!",
                "Super!",
                "Bien joué!",
                "Non",
                "Pas ça",
                "Encore",
                "Continue",
                "C'est bien",
                "Oui!",
                "Parfait!",
            ],

            positive_feedback: vec![
                "Bravo!",
                "Oui!",
                "Super!",
                "Bien!",
                "C'est ça!",
                "Parfait!",
            ],

            negative_feedback: vec![
                "Non",
                "Pas ça",
                "Essaie encore",
            ],

            pending_feedback: None,
            total_sent: 0,
            total_responses: 0,
        }
    }

    /// Toggle auto-training on/off
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        if self.enabled {
            self.last_stimulus = Instant::now();
        }
    }

    /// Notify trainer that ARIA responded
    pub fn on_response(&mut self, _response: &str) {
        self.responses_count += 1;
        self.total_responses += 1;

        // If ARIA responds, we'll give positive feedback
        self.pending_feedback = Some(true);

        // Speed up if ARIA is responsive
        if self.responses_count > 3 && self.interval > Duration::from_secs(2) {
            self.interval = Duration::from_secs(2);
        }
    }

    /// Get the next stimulus to send (if any)
    pub fn tick(&mut self) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let now = Instant::now();

        // First, check if we need to send feedback
        if let Some(positive) = self.pending_feedback.take() {
            if now.duration_since(self.last_stimulus) > Duration::from_millis(500) {
                self.last_stimulus = now;
                let feedback = if positive {
                    self.positive_feedback.choose(&mut rand::thread_rng())
                } else {
                    self.negative_feedback.choose(&mut rand::thread_rng())
                };
                return feedback.map(|s: &&str| s.to_string());
            }
        }

        // Check if it's time for a new stimulus
        if now.duration_since(self.last_stimulus) < self.interval {
            return None;
        }

        self.last_stimulus = now;
        self.stimuli_count += 1;
        self.total_sent += 1;

        // Check if we should advance to next phase
        if self.stimuli_count >= self.stimuli_per_phase {
            self.phase = self.phase.next();
            self.stimuli_count = 0;
            self.responses_count = 0;
            // Reset interval for new phase
            self.interval = Duration::from_secs(3);
        }

        // Generate stimulus based on phase
        let stimulus = self.generate_stimulus();
        self.last_sent = Some(stimulus.clone());
        Some(stimulus)
    }

    fn generate_stimulus(&self) -> String {
        let mut rng = rand::thread_rng();

        match self.phase {
            TrainingPhase::Vocabulary => {
                // Single word
                self.vocabulary
                    .choose(&mut rng)
                    .map(|s: &&str| s.to_string())
                    .unwrap_or_else(|| "bonjour".to_string())
            }

            TrainingPhase::Associations => {
                // Word pair
                if let Some((a, b)) = self.associations.choose(&mut rng) {
                    format!("{} {}", a, b)
                } else {
                    "chat moka".to_string()
                }
            }

            TrainingPhase::Conversation => {
                // Full phrase
                self.conversations
                    .choose(&mut rng)
                    .map(|s: &&str| s.to_string())
                    .unwrap_or_else(|| "bonjour".to_string())
            }

            TrainingPhase::Emotional => {
                // Emotional expression with varying intensity
                let base = self.emotional
                    .choose(&mut rng)
                    .map(|s: &&str| s.to_string())
                    .unwrap_or_else(|| "Bravo!".to_string());

                // Sometimes add emphasis
                if rng.gen_bool(0.3) {
                    format!("{}!", base.trim_end_matches('!'))
                } else {
                    base
                }
            }
        }
    }

    /// Get status string for display
    pub fn status(&self) -> String {
        if self.enabled {
            format!(
                "AUTO: {} ({}/{}) | Sent: {} | Responses: {}",
                self.phase.name(),
                self.stimuli_count,
                self.stimuli_per_phase,
                self.total_sent,
                self.total_responses
            )
        } else {
            "AUTO: Off (press 'a' to start)".to_string()
        }
    }

    /// Get response rate
    pub fn response_rate(&self) -> f32 {
        if self.total_sent > 0 {
            self.total_responses as f32 / self.total_sent as f32
        } else {
            0.0
        }
    }
}
