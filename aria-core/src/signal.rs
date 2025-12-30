//! # Signal - Quantum of Information
//!
//! Signals are how cells communicate and how the outside world
//! interacts with ARIA. They travel through the substrate like
//! waves through water.
//!
//! ## Types
//!
//! - **Perception**: External input (from humans)
//! - **Expression**: Emergent output (from ARIA)
//! - **Internal**: Between cells

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::SIGNAL_DIMS;

/// A signal fragment for cell-to-cell communication (GPU-friendly)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct SignalFragment {
    /// Source cell ID (0 = external)
    pub source_id: u64,
    /// Signal content (8D vector)
    pub content: [f32; SIGNAL_DIMS],
    /// Target position in semantic space (8D) - where to inject the signal
    pub position: [f32; SIGNAL_DIMS],
    /// Signal intensity
    pub intensity: f32,
    /// Padding for alignment
    _pad: [f32; 3],
}

impl SignalFragment {
    /// Create a new signal fragment with position
    pub fn new(source_id: u64, content: [f32; SIGNAL_DIMS], position: [f32; SIGNAL_DIMS], intensity: f32) -> Self {
        Self {
            source_id,
            content,
            position,
            intensity,
            _pad: [0.0; 3],
        }
    }

    /// Create an external signal (from outside ARIA)
    /// Position defaults to content (legacy behavior, but should be set explicitly)
    pub fn external(content: [f32; SIGNAL_DIMS], intensity: f32) -> Self {
        Self::new(0, content, content, intensity)
    }

    /// Create an external signal with explicit position
    pub fn external_at(content: [f32; SIGNAL_DIMS], position: [f32; SIGNAL_DIMS], intensity: f32) -> Self {
        Self::new(0, content, position, intensity)
    }

    /// Create a zeroed signal fragment
    pub fn zeroed() -> Self {
        Self {
            source_id: 0,
            content: [0.0; SIGNAL_DIMS],
            position: [0.0; SIGNAL_DIMS],
            intensity: 0.0,
            _pad: [0.0; 3],
        }
    }
}

/// Full signal with metadata (CPU-side, not GPU-transferred)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Signal {
    /// Vector content (variable length for flexibility)
    pub content: Vec<f32>,

    /// Global intensity
    pub intensity: f32,

    /// Human-readable label (for debugging and learning)
    pub label: String,

    /// Type of signal
    pub signal_type: SignalType,

    /// Timestamp (tick when created)
    pub timestamp: u64,

    /// Semantic position in space (where to inject)
    pub position: Option<[f32; 16]>,
}

/// Type of signal
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// External perception (input from outside)
    Perception,
    /// Emergent expression (output from ARIA)
    Expression,
    /// Internal signal between cells
    Internal,
}

impl Signal {
    /// Create a perception signal from text
    pub fn from_text(text: &str) -> Self {
        let mut content = vec![0.0f32; 32];

        // Character-based encoding (first 20 chars)
        for (i, ch) in text.chars().take(20).enumerate() {
            content[i] = (ch as u32 as f32) / 256.0;
        }

        // Statistical features
        let len = text.len() as f32;
        content[20] = (len / 100.0).min(1.0);
        content[21] = text.chars().filter(|c| c.is_uppercase()).count() as f32 / len.max(1.0);
        content[22] = text.chars().filter(|c| c.is_whitespace()).count() as f32 / len.max(1.0);
        content[23] = text.chars().filter(|c| c.is_numeric()).count() as f32 / len.max(1.0);

        // Punctuation
        content[24] = if text.ends_with('?') { 1.0 } else { 0.0 };
        content[25] = if text.ends_with('!') { 1.0 } else { 0.0 };
        content[26] = if text.ends_with('.') { 1.0 } else { 0.0 };
        content[27] = if text.contains(',') { 0.5 } else { 0.0 };

        // Emotional markers
        let lower = text.to_lowercase();
        content[28] = Self::detect_positive_emotion(&lower);
        content[29] = Self::detect_negative_emotion(&lower);
        content[30] = Self::detect_request(&lower);
        content[31] = Self::detect_question(&lower);

        Self {
            content,
            intensity: (len / 50.0).min(1.0).max(0.1),
            label: text.chars().take(50).collect(),
            signal_type: SignalType::Perception,
            timestamp: 0,
            position: None,
        }
    }

    /// Create an expression signal from a vector
    pub fn from_vector(v: [f32; SIGNAL_DIMS], label: String) -> Self {
        Self {
            content: v.to_vec(),
            intensity: v.iter().map(|x| x.abs()).sum::<f32>() / SIGNAL_DIMS as f32,
            label,
            signal_type: SignalType::Expression,
            timestamp: 0,
            position: None,
        }
    }

    /// Convert to fixed-size vector for cell processing
    pub fn to_fragment(&self) -> SignalFragment {
        let mut content = [0.0f32; SIGNAL_DIMS];
        let mut position = [0.0f32; SIGNAL_DIMS];
        for (i, v) in self.content.iter().take(SIGNAL_DIMS).enumerate() {
            content[i] = *v;
            // Scale content to cell space [-10, 10]
            position[i] = *v * 20.0 - 10.0;
        }
        SignalFragment::new(0, content, position, self.intensity)
    }

    /// Get semantic position for spatial injection
    pub fn semantic_position(&self) -> [f32; 16] {
        if let Some(pos) = self.position {
            pos
        } else {
            let mut pos = [0.0f32; 16];
            for (i, v) in self.content.iter().take(16).enumerate() {
                pos[i] = *v;
            }
            pos
        }
    }

    /// Check if this is a question
    pub fn is_question(&self) -> bool {
        self.content.get(24).copied().unwrap_or(0.0) > 0.5
            || self.content.get(31).copied().unwrap_or(0.0) > 0.5
    }

    /// Check if emotionally charged
    pub fn is_emotional(&self) -> bool {
        let positive = self.content.get(28).copied().unwrap_or(0.0);
        let negative = self.content.get(29).copied().unwrap_or(0.0);
        (positive + negative.abs()) > 0.5
    }

    fn detect_positive_emotion(text: &str) -> f32 {
        const POSITIVE: &[&str] = &[
            "love", "happy", "good", "great", "beautiful", "nice", "sweet", "cute",
            "aime", "adore", "content", "heureux", "heureuse", "bien", "super", "génial",
            "joli", "jolie", "beau", "belle", "mignon", "mignonne", "bisou", "calin",
            "merci", "bravo", "cool", "chouette", "sympa", "♥", "❤", ":)", "<3",
        ];
        if POSITIVE.iter().any(|w| text.contains(w)) { 1.0 } else { 0.0 }
    }

    fn detect_negative_emotion(text: &str) -> f32 {
        const NEGATIVE: &[&str] = &[
            "hate", "sad", "bad", "angry", "fear", "scared", "hurt", "pain",
            "triste", "mal", "peur", "colère", "fâché", "fâchée", "méchant", "nul",
            "déteste", "horrible", "moche", "pleure", "pleurer", ":(", ":'(",
        ];
        if NEGATIVE.iter().any(|w| text.contains(w)) { -1.0 } else { 0.0 }
    }

    fn detect_request(text: &str) -> f32 {
        const REQUEST: &[&str] = &[
            "help", "please", "want", "need",
            "aide", "s'il te plaît", "stp", "veux", "voudrais", "besoin", "peux",
        ];
        if REQUEST.iter().any(|w| text.contains(w)) { 0.5 } else { 0.0 }
    }

    fn detect_question(text: &str) -> f32 {
        const QUESTION: &[&str] = &[
            "why", "how", "what", "when", "where", "who",
            "pourquoi", "comment", "quoi", "quand", "où", "qui", "est-ce",
        ];
        if QUESTION.iter().any(|w| text.contains(w)) { 0.8 } else { 0.0 }
    }
}

impl Default for Signal {
    fn default() -> Self {
        Self {
            content: vec![0.0; SIGNAL_DIMS],
            intensity: 0.0,
            label: String::new(),
            signal_type: SignalType::Internal,
            timestamp: 0,
            position: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_fragment_size() {
        // Should be 48 bytes (8 + 32 + 4 + 4)
        // source_id(8) + content(32) + position(32) + intensity(4) + padding(12) = 88 bytes
        assert_eq!(std::mem::size_of::<SignalFragment>(), 88);
    }

    #[test]
    fn test_signal_from_text() {
        let signal = Signal::from_text("Bonjour ARIA!");
        assert!(signal.intensity > 0.0);
        assert_eq!(signal.signal_type, SignalType::Perception);
    }

    #[test]
    fn test_question_detection() {
        let q = Signal::from_text("Comment vas-tu?");
        assert!(q.is_question());

        let s = Signal::from_text("Je vais bien.");
        assert!(!s.is_question());
    }
}
