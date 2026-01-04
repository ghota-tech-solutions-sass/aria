//! Persistence types for ARIA's long-term memory
//!
//! Core data structures that are serialized to disk via bincode.
//! Used by LongTermMemory for patterns, associations, elite DNA, etc.

use aria_core::DNA;
use serde::{Deserialize, Serialize};

/// Elite DNA - preserved genetic heritage from successful lineages
#[derive(Serialize, Deserialize, Clone)]
pub struct EliteDNA {
    pub dna: DNA,
    pub fitness_score: f32,
    pub generation: u64,
    pub specialization: String,
    pub preserved_at: u64,
}

/// Elite structural code - validated by Shadow Brain
#[derive(Serialize, Deserialize, Clone)]
pub struct EliteStructuralCode {
    pub checksum: u64,
    pub validation_score: f32,
    pub discovered_at: u64,
}

/// Proto-concept - emergent abstraction from GPU clusters (Phase 6)
/// NOTE: related_words field removed in Session 31 (Physical Intelligence)
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ProtoConcept {
    pub cluster_id: u32,
    pub name: String,
    pub signature: [f32; 8],
    pub stability: f32,
    pub emerged_at: u64,
}

/// Learned pattern - sequences that repeat
#[derive(Serialize, Deserialize, Clone)]
pub struct Pattern {
    /// The pattern itself (sequence of vectors)
    pub sequence: Vec<[f32; 8]>,
    /// How many times observed
    pub frequency: u64,
    /// What typically follows this pattern
    pub typical_response: [f32; 8],
    /// Emotional valence
    pub valence: f32,
    /// When first learned
    pub first_seen: u64,
    /// When last seen
    pub last_seen: u64,
}

/// Stimulus-response association
#[derive(Serialize, Deserialize, Clone)]
pub struct Association {
    pub stimulus: [f32; 16],
    pub response: [f32; 16],
    pub strength: f32,
    pub last_reinforced: u64,
    pub times_reinforced: u64,
}

/// Important memory (high emotional moment)
#[derive(Serialize, Deserialize, Clone)]
pub struct Memory {
    pub timestamp: u64,
    pub trigger: String,
    pub internal_state: [f32; 32],
    pub emotional_intensity: f32,
    pub outcome: Outcome,
}

/// Outcome of an interaction
#[derive(Serialize, Deserialize, Clone)]
pub enum Outcome {
    Positive(f32),
    Negative(f32),
    Neutral,
}

/// Global statistics
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct GlobalStats {
    pub total_ticks: u64,
    pub total_interactions: u64,
    pub total_births: u64,
    pub total_deaths: u64,
    pub peak_population: usize,
    pub longest_lineage: u64,
    pub total_patterns_learned: u64,
    pub total_memories: u64,
}

// Default values for adaptive params (used in serde)
pub fn default_emission_threshold() -> f32 {
    0.15
}
pub fn default_response_probability() -> f32 {
    0.8
}
pub fn default_learning_rate() -> f32 {
    0.3
}
pub fn default_spontaneity() -> f32 {
    0.05
}
