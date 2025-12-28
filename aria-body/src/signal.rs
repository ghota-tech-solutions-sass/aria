//! Signal types for aria-body
//!
//! Mirrors the signal types from aria-brain for communication.

use serde::{Deserialize, Serialize};

/// A signal is a quantum of information
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Signal {
    /// Vector content
    pub content: Vec<f32>,

    /// Global intensity
    pub intensity: f32,

    /// Human-readable label
    pub label: String,

    /// Signal type
    pub signal_type: SignalType,

    /// Timestamp
    #[serde(default)]
    pub timestamp: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SignalType {
    Perception,
    Expression,
    Internal,
}

impl Signal {
    /// Create a signal from text
    pub fn from_text(text: &str) -> Self {
        let mut content = vec![0.0f32; 32];

        // Character encoding
        for (i, ch) in text.chars().take(20).enumerate() {
            content[i] = (ch as u32 as f32) / 256.0;
        }

        // Features
        let len = text.len() as f32;
        content[20] = (len / 100.0).min(1.0);
        content[21] = text.chars().filter(|c| c.is_uppercase()).count() as f32 / len.max(1.0);
        content[22] = text.chars().filter(|c| c.is_whitespace()).count() as f32 / len.max(1.0);
        content[23] = text.chars().filter(|c| c.is_numeric()).count() as f32 / len.max(1.0);

        content[24] = if text.ends_with('?') { 1.0 } else { 0.0 };
        content[25] = if text.ends_with('!') { 1.0 } else { 0.0 };
        content[26] = if text.ends_with('.') { 1.0 } else { 0.0 };
        content[27] = if text.contains(',') { 0.5 } else { 0.0 };

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

    /// Convert to expression
    pub fn to_expression(&self) -> String {
        let (dominant_index, dominant_value) = self.content
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, v)| (i, *v))
            .unwrap_or((0, 0.0));

        let primitives = [
            "...", "?", "!", "~", "*", "•", "○", "◊",
            "→", "←", "↑", "↓", "↔", "↕", "⟳", "⟲",
            "a", "o", "e", "i", "u", "m", "n", "r",
            "da", "ma", "pa", "ba", "ta", "ka", "na", "la",
        ];

        let evolved = [
            "want", "see", "feel", "think",
            "move", "stay", "go", "be",
            "good", "bad", "more", "less",
            "you", "me", "this", "that",
        ];

        let coherence = self.coherence();
        let base_index = ((dominant_index as f32 + dominant_value.abs() * 10.0) as usize) % primitives.len();

        let mut expression = if coherence > 0.7 && self.intensity > 0.5 {
            let evolved_index = (base_index + (coherence * 10.0) as usize) % evolved.len();
            evolved[evolved_index].to_string()
        } else {
            primitives[base_index].to_string()
        };

        if self.intensity > 0.8 {
            expression = expression.to_uppercase();
        }
        if self.intensity > 0.6 {
            expression.push('!');
        }
        if dominant_value < 0.0 {
            expression = format!("~{}", expression);
        }

        expression
    }

    fn coherence(&self) -> f32 {
        if self.content.is_empty() {
            return 0.0;
        }
        let mean: f32 = self.content.iter().sum::<f32>() / self.content.len() as f32;
        let variance: f32 = self.content.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.content.len() as f32;
        (1.0 - variance.sqrt().min(1.0)).max(0.0)
    }
}
