//! Episodic Memory - ARIA remembers specific moments
//!
//! Unlike semantic memory (word associations), episodes are autobiographical:
//! "I remember when you first said you loved me"

use serde::{Deserialize, Serialize};

/// An episodic memory - a specific moment ARIA remembers
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Episode {
    /// Unique episode ID
    pub id: u64,

    /// When this happened (tick)
    pub timestamp: u64,

    /// When this happened (real time, for display)
    #[serde(default)]
    pub real_time: Option<String>,

    /// What was said to ARIA
    pub input: String,

    /// What ARIA responded
    pub response: Option<String>,

    /// Key words in this episode
    pub keywords: Vec<String>,

    /// Emotional state at the time
    pub emotion: EpisodeEmotion,

    /// How important/significant this episode was (0.0 to 1.0)
    pub importance: f32,

    /// How many times this episode has been recalled
    pub recall_count: u64,

    /// Last time this episode was recalled (tick)
    pub last_recalled: u64,

    /// Was this a first time? (first greeting, first "I love you", etc.)
    pub first_of_kind: Option<String>,

    /// Category of episode
    pub category: EpisodeCategory,
}

/// Emotional state captured in an episode
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct EpisodeEmotion {
    pub happiness: f32,
    pub arousal: f32,
    pub comfort: f32,
    pub curiosity: f32,
}

/// Category of episodic memory
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum EpisodeCategory {
    /// First time something happened
    FirstTime,
    /// Emotionally significant moment
    Emotional,
    /// Learning something new
    Learning,
    /// Social interaction (greeting, farewell, etc.)
    Social,
    /// Question and answer
    Question,
    /// Positive feedback received
    Praise,
    /// Negative feedback received
    Correction,
    /// General conversation
    General,
}

impl Default for EpisodeCategory {
    fn default() -> Self {
        EpisodeCategory::General
    }
}

impl Episode {
    /// Create a new episode
    pub fn new(
        id: u64,
        timestamp: u64,
        input: String,
        response: Option<String>,
        keywords: Vec<String>,
        emotion: EpisodeEmotion,
        importance: f32,
        category: EpisodeCategory,
    ) -> Self {
        // Get current real time for display
        let real_time = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();

        Self {
            id,
            timestamp,
            real_time: Some(real_time),
            input,
            response,
            keywords,
            emotion,
            importance,
            recall_count: 0,
            last_recalled: timestamp,
            first_of_kind: None,
            category,
        }
    }

    /// Mark this as a "first time" episode
    pub fn mark_first_of_kind(&mut self, kind: &str) {
        self.first_of_kind = Some(kind.to_string());
        self.importance = (self.importance + 0.3).min(1.0); // First times are more important
    }

    /// Recall this episode (strengthens it)
    pub fn recall(&mut self, current_tick: u64) {
        self.recall_count += 1;
        self.last_recalled = current_tick;
        // Recalling strengthens importance slightly
        self.importance = (self.importance + 0.05).min(1.0);
    }

    /// Check if this episode matches given keywords
    pub fn matches_context(&self, context_words: &[String]) -> f32 {
        if context_words.is_empty() || self.keywords.is_empty() {
            return 0.0;
        }

        let matches = self.keywords.iter()
            .filter(|kw| context_words.iter().any(|cw| cw.to_lowercase() == kw.to_lowercase()))
            .count();

        matches as f32 / self.keywords.len().max(1) as f32
    }

    /// Calculate memory strength (for forgetting curve)
    pub fn memory_strength(&self, current_tick: u64) -> f32 {
        let age = current_tick.saturating_sub(self.timestamp) as f32;
        let recency = current_tick.saturating_sub(self.last_recalled) as f32;

        // Base decay (older memories fade)
        let age_factor = (-age / 1_000_000.0).exp(); // Very slow decay

        // Recency boost (recently recalled = stronger)
        let recency_factor = (-recency / 100_000.0).exp();

        // Recall strengthening (more recalls = stronger)
        let recall_factor = 1.0 + (self.recall_count as f32).ln().max(0.0) * 0.2;

        // Importance matters
        let importance_factor = 0.5 + self.importance * 0.5;

        (age_factor * 0.3 + recency_factor * 0.4 + importance_factor * 0.3) * recall_factor
    }
}

/// Compressed episode for long-term storage
/// Gemini suggestion: compress old episodes to save memory
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CompressedEpisode {
    /// Original episode ID
    pub id: u64,
    /// When this happened (tick)
    pub timestamp: u64,
    /// Summary keywords (reduced from original)
    pub keywords: Vec<String>,
    /// Overall emotional tone
    pub emotional_tone: f32,
    /// Importance score
    pub importance: f32,
    /// Category
    pub category: EpisodeCategory,
    /// Was this a first time?
    pub first_of_kind: Option<String>,
}

impl CompressedEpisode {
    /// Compress a full episode into a compact form
    pub fn from_episode(episode: &Episode) -> Self {
        // Keep only top 3 keywords
        let keywords: Vec<String> = episode.keywords.iter()
            .take(3)
            .cloned()
            .collect();

        // Compute overall emotional tone
        let emotional_tone = (episode.emotion.happiness + episode.emotion.comfort) / 2.0;

        Self {
            id: episode.id,
            timestamp: episode.timestamp,
            keywords,
            emotional_tone,
            importance: episode.importance,
            category: episode.category.clone(),
            first_of_kind: episode.first_of_kind.clone(),
        }
    }
}
