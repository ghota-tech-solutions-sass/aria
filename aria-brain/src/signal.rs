//! Signal - Quantum of information that travels through ARIA
//!
//! Signals are how cells communicate and how the outside world
//! interacts with ARIA.

use serde::{Deserialize, Serialize};

/// A signal is a quantum of information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Signal {
    /// Vector content (the actual information)
    pub content: Vec<f32>,

    /// Global intensity
    pub intensity: f32,

    /// Human-readable label (for debugging)
    pub label: String,

    /// Type of signal
    pub signal_type: SignalType,

    /// Timestamp (tick when created)
    #[serde(default)]
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    /// External perception (input from outside)
    Perception,
    /// Emergent expression (output from ARIA)
    Expression,
    /// Internal signal between cells
    Internal,
}

impl Signal {
    /// Create a signal from text input
    ///
    /// This encodes the text into a vector representation
    /// that cells can process.
    #[allow(dead_code)]
    pub fn from_text(text: &str) -> Self {
        let mut content = vec![0.0f32; 32];

        // Character-based encoding
        for (i, ch) in text.chars().take(20).enumerate() {
            content[i] = (ch as u32 as f32) / 256.0;
        }

        // Statistical features
        let len = text.len() as f32;
        content[20] = (len / 100.0).min(1.0); // Length
        content[21] = text.chars().filter(|c| c.is_uppercase()).count() as f32 / len.max(1.0); // Uppercase ratio
        content[22] = text.chars().filter(|c| c.is_whitespace()).count() as f32 / len.max(1.0); // Whitespace ratio
        content[23] = text.chars().filter(|c| c.is_numeric()).count() as f32 / len.max(1.0); // Numeric ratio

        // Punctuation signals
        content[24] = if text.ends_with('?') { 1.0 } else { 0.0 }; // Question
        content[25] = if text.ends_with('!') { 1.0 } else { 0.0 }; // Exclamation
        content[26] = if text.ends_with('.') { 1.0 } else { 0.0 }; // Statement
        content[27] = if text.contains(',') { 0.5 } else { 0.0 }; // Complexity

        // Emotional markers (simple heuristics)
        let lower = text.to_lowercase();
        content[28] = if lower.contains("love") || lower.contains("happy") || lower.contains("good") { 1.0 } else { 0.0 };
        content[29] = if lower.contains("hate") || lower.contains("sad") || lower.contains("bad") { -1.0 } else { 0.0 };
        content[30] = if lower.contains("help") || lower.contains("please") { 0.5 } else { 0.0 };
        content[31] = if lower.contains("why") || lower.contains("how") || lower.contains("what") { 0.8 } else { 0.0 };

        Self {
            content,
            intensity: (len / 50.0).min(1.0).max(0.1),
            label: text.chars().take(30).collect(),
            signal_type: SignalType::Perception,
            timestamp: 0,
        }
    }

    /// Create a signal from a raw vector (emergent from cells)
    pub fn from_vector(v: [f32; 8], label: String) -> Self {
        Self {
            content: v.to_vec(),
            intensity: v.iter().map(|x| x.abs()).sum::<f32>() / 8.0,
            label,
            signal_type: SignalType::Expression,
            timestamp: 0,
        }
    }

    /// Convert to a fixed-size vector for cell processing
    pub fn to_vector(&self) -> [f32; 8] {
        let mut result = [0.0f32; 8];
        for (i, v) in self.content.iter().take(8).enumerate() {
            result[i] = *v;
        }
        result
    }

    /// Get the semantic position of this signal
    pub fn semantic_position(&self) -> [f32; 16] {
        let mut pos = [0.0f32; 16];
        for (i, v) in self.content.iter().take(16).enumerate() {
            pos[i] = *v;
        }
        pos
    }

    /// Convert an emergent signal to a human-readable expression
    ///
    /// This is where ARIA's "language" emerges progressively.
    /// Initially, it's just primitive sounds/symbols.
    #[allow(dead_code)]
    pub fn to_expression(&self) -> String {
        // Find the dominant value and index
        let (dominant_index, dominant_value) = self.content
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap_or((0, 0.0));

        // Primitive vocabulary - will evolve over time
        let primitives = [
            // Pure sounds
            "...", "?", "!", "~", "*", "•", "○", "◊",
            // Movement/direction
            "→", "←", "↑", "↓", "↔", "↕", "⟳", "⟲",
            // Proto-words (phonemes)
            "a", "o", "e", "i", "u", "m", "n", "r",
            // Proto-concepts
            "da", "ma", "pa", "ba", "ta", "ka", "na", "la",
        ];

        // More evolved vocabulary (unlocked with higher coherence)
        let evolved_primitives = [
            "want", "see", "feel", "think",
            "move", "stay", "go", "be",
            "good", "bad", "more", "less",
            "you", "me", "this", "that",
        ];

        // Calculate coherence (how organized the signal is)
        let coherence = self.calculate_coherence();

        let base_index = ((dominant_index as f32 + dominant_value.abs() * 10.0) as usize) % primitives.len();

        let mut expression = if coherence > 0.7 && self.intensity > 0.5 {
            // High coherence: use evolved vocabulary
            let evolved_index = (base_index + (coherence * 10.0) as usize) % evolved_primitives.len();
            evolved_primitives[evolved_index].to_string()
        } else {
            primitives[base_index].to_string()
        };

        // Add intensity markers
        if self.intensity > 0.8 {
            expression = expression.to_uppercase();
        }
        if self.intensity > 0.6 {
            expression.push('!');
        }
        if dominant_value < 0.0 {
            expression = format!("~{}", expression);
        }

        // Add repetition for very high intensity
        if self.intensity > 0.9 {
            expression = format!("{} {}", expression, expression);
        }

        expression
    }

    /// Calculate the coherence of the signal (0.0 = noise, 1.0 = very organized)
    fn calculate_coherence(&self) -> f32 {
        if self.content.is_empty() {
            return 0.0;
        }

        let mean: f32 = self.content.iter().sum::<f32>() / self.content.len() as f32;
        let variance: f32 = self.content.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.content.len() as f32;

        // Low variance = high coherence
        (1.0 - variance.sqrt().min(1.0)).max(0.0)
    }

    /// Check if this signal is a question
    #[allow(dead_code)]
    pub fn is_question(&self) -> bool {
        self.content.get(24).copied().unwrap_or(0.0) > 0.5 ||
        self.content.get(31).copied().unwrap_or(0.0) > 0.5
    }

    /// Check if this signal is emotionally charged
    #[allow(dead_code)]
    pub fn is_emotional(&self) -> bool {
        let positive = self.content.get(28).copied().unwrap_or(0.0);
        let negative = self.content.get(29).copied().unwrap_or(0.0);
        (positive + negative.abs()) > 0.5
    }
}

impl Default for Signal {
    fn default() -> Self {
        Self {
            content: vec![0.0; 8],
            intensity: 0.0,
            label: String::new(),
            signal_type: SignalType::Internal,
            timestamp: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_text() {
        let signal = Signal::from_text("Hello world!");
        assert!(!signal.content.is_empty());
        assert!(signal.intensity > 0.0);
        assert_eq!(signal.signal_type, SignalType::Perception);
    }

    #[test]
    fn test_question_detection() {
        let signal = Signal::from_text("How are you?");
        assert!(signal.is_question());

        let statement = Signal::from_text("I am fine.");
        assert!(!statement.is_question());
    }

    #[test]
    fn test_signal_expression() {
        let signal = Signal::from_vector([0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], "test".into());
        let expression = signal.to_expression();
        assert!(!expression.is_empty());
    }
}
