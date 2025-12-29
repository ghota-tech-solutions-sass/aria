//! Visual Memory - ARIA remembers what she sees
//!
//! Images are converted to 32D feature vectors and stored with labels.

use serde::{Deserialize, Serialize};

/// A visual memory - ARIA remembers seeing this
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VisualMemory {
    /// Unique ID
    pub id: u64,
    /// The visual signature (32D vector)
    pub signature: [f32; 32],
    /// Human-readable description generated at creation
    pub description: String,
    /// Labels associated with this image (learned from context)
    pub labels: Vec<String>,
    /// When this was first seen
    pub first_seen: u64,
    /// How many times seen
    pub times_seen: u64,
    /// Last time seen
    pub last_seen: u64,
    /// Emotional valence when first seen (-1 to 1)
    pub emotional_context: f32,
    /// Source/origin (e.g., "webcam", "file:moka.jpg")
    pub source: String,
}

impl VisualMemory {
    pub fn new(id: u64, signature: [f32; 32], description: String, source: String, tick: u64, emotional_context: f32) -> Self {
        Self {
            id,
            signature,
            description,
            labels: Vec::new(),
            first_seen: tick,
            times_seen: 1,
            last_seen: tick,
            emotional_context,
            source,
        }
    }

    /// Calculate similarity to another visual signature (0 to 1)
    pub fn similarity(&self, other: &[f32; 32]) -> f32 {
        let dot: f32 = self.signature.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
        let mag_a: f32 = self.signature.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a > 0.0 && mag_b > 0.0 {
            (dot / (mag_a * mag_b)).max(0.0)
        } else {
            0.0
        }
    }

    /// Is this likely the same thing? (high similarity threshold)
    pub fn is_same(&self, other: &[f32; 32]) -> bool {
        self.similarity(other) > 0.85
    }

    /// Is this similar enough to be related?
    pub fn is_related(&self, other: &[f32; 32]) -> bool {
        self.similarity(other) > 0.6
    }
}

/// Visual-word link - connects visual features to words
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VisualWordLink {
    /// The word being linked
    pub word: String,
    /// Visual prototype (average of all images associated with this word)
    pub visual_prototype: [f32; 32],
    /// Number of associations (how confident is this link)
    pub association_count: u64,
    /// Average similarity of associations
    pub avg_similarity: f32,
    /// Last time this link was reinforced
    pub last_reinforced: u64,
}
