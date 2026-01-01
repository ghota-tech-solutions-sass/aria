//! Vocabulary - Word frequencies and semantic associations
//!
//! Tracks how ARIA learns words and builds semantic connections.

use serde::{Deserialize, Serialize};
use super::types::{WordCategory, UsagePattern};

/// Tracks how often a word is heard and its emotional context
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WordFrequency {
    /// How many times this word was heard
    pub count: u64,
    /// First time heard (tick)
    pub first_heard: u64,
    /// Last time heard (tick)
    pub last_heard: u64,
    /// Average vector representation (learned from context)
    pub learned_vector: [f32; 8],
    /// Emotional associations (positive = 1.0, negative = -1.0)
    pub emotional_valence: f32,
    /// How special this word is (0.0 = common, 1.0 = very special like "Moka")
    pub familiarity_boost: f32,
    /// Grammatical category (learned from context)
    #[serde(default)]
    pub category: WordCategory,
    /// Confidence scores for each category (noun, verb, adjective)
    /// Used for probabilistic classification
    #[serde(default)]
    pub category_scores: [f32; 3], // [noun, verb, adjective]
    /// Usage patterns - when/how this word is typically used
    #[serde(default)]
    pub usage_pattern: UsagePattern,
}

/// Semantic association between two words
/// When words appear together, they become associated
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WordAssociation {
    /// How many times these words appeared together
    pub co_occurrences: u64,
    /// Strength of association (0.0 to 1.0)
    pub strength: f32,
    /// Last time they appeared together
    pub last_seen: u64,
    /// Emotional context of the association
    pub emotional_valence: f32,
}

/// Semantic cluster - group of related words
/// Words in the same cluster are semantically connected
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SemanticCluster {
    /// Unique cluster ID
    pub id: u32,
    /// Human-readable label (optional, can be inferred)
    pub label: Option<String>,
    /// Words in this cluster with their membership strength
    pub words: Vec<(String, f32)>,
    /// Average emotional valence of the cluster
    pub emotional_valence: f32,
    /// Dominant category in this cluster
    pub dominant_category: WordCategory,
}

/// Word meaning - learned from context
#[derive(Serialize, Deserialize, Clone)]
pub struct WordMeaning {
    /// The vector representation
    pub vector: [f32; 8],
    /// How confident we are in this meaning
    pub confidence: f32,
    /// Examples of usage
    pub examples: Vec<String>,
    /// Times encountered
    pub frequency: u64,
}

/// ProtoConcept - Bridge between GPU clusters and symbolic language (Phase 6)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProtoConcept {
    /// ID of the cluster in the GPU backend
    pub cluster_id: u32,
    /// Temporary name (e.g., "Concept-12") until a word is mapped
    pub name: String,
    /// Semantic signature (average vector of member cells)
    pub signature: [f32; 8],
    /// Survival/Stability score (based on hysteresis)
    pub stability: f32,
    /// Words that are most associated with this conceptual signature
    pub related_words: Vec<(String, f32)>,
    /// When this concept first emerged
    pub emerged_at: u64,
}
