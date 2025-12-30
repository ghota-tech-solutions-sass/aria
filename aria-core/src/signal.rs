//! # Signal - Quantum of Information
//!
//! Signals are how cells communicate and how the outside world
//! interacts with ARIA. They travel through the substrate like
//! waves through water.
//!
//! ## Physical Intelligence (Session 20)
//!
//! ARIA no longer "understands" language. She FEELS the tension.
//! Text is converted to raw physical tension vectors, not semantic encodings.
//! Cells resonate with tension patterns, not words.
//!
//! ## Types
//!
//! - **Perception**: External input (from humans) - now pure tension
//! - **Expression**: Emergent output (from ARIA)
//! - **Internal**: Between cells

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::tension::{text_to_tension, tension_intensity, tension_to_position};
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
    ///
    /// **Physical Intelligence**: Text is converted to pure TENSION, not semantic encoding.
    /// ARIA doesn't "understand" words - she feels the vibration of the message.
    pub fn from_text(text: &str) -> Self {
        // Convert text to 8D tension vector
        let tension = text_to_tension(text);

        // Extend to 32D for backward compatibility, but only 8D matters
        let mut content = vec![0.0f32; 32];
        for (i, t) in tension.iter().enumerate() {
            content[i] = *t;
        }

        // Calculate position in cell space [-10, 10]
        let position_8d = tension_to_position(&tension);
        let mut position_16d = [0.0f32; 16];
        for (i, p) in position_8d.iter().enumerate() {
            position_16d[i] = *p;
        }

        Self {
            content,
            intensity: tension_intensity(&tension),
            label: text.chars().take(50).collect(),
            signal_type: SignalType::Perception,
            timestamp: 0,
            position: Some(position_16d),
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
    ///
    /// Physical Intelligence: Questions have high urgency (dimension 2)
    /// from question marks and short message length.
    pub fn is_question(&self) -> bool {
        // In Physical Intelligence, urgency (dim 2) encodes question-like tension
        // question_marks * 0.2 + short_msg bonus means questions often have urgency > 0.4
        self.content.get(2).copied().unwrap_or(0.0) > 0.4
    }

    /// Check if emotionally charged (based on tension valence)
    pub fn is_emotional(&self) -> bool {
        // In Physical Intelligence, check tension dimensions
        // Dimension 0 = arousal, Dimension 1 = valence
        let arousal = self.content.get(0).copied().unwrap_or(0.0);
        let valence = self.content.get(1).copied().unwrap_or(0.0).abs();
        arousal > 0.5 || valence > 0.3
    }

    // NOTE: detect_positive_emotion, detect_negative_emotion, detect_request, detect_question
    // removed in Session 20 (Physical Intelligence)
    // Emotional detection now happens through tension vectors, not word matching
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
