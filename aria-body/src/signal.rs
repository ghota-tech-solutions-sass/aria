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

    /// Convert to expression - ARIA's baby babbling
    pub fn to_expression(&self) -> String {
        // If the brain matched a known word, use it!
        if self.label.starts_with("word:") {
            let word = self.label.strip_prefix("word:").unwrap_or(&self.label);
            // Add some baby-like variations
            if self.intensity > 0.5 {
                return format!("{}!", word.to_uppercase());
            } else if self.intensity > 0.3 {
                return word.to_string();
            } else {
                // Whisper/uncertain
                return format!("{}...", word);
            }
        }

        // Vowels - the first sounds a baby makes
        let vowels = ["a", "e", "i", "o", "u", "é", "è", "ô"];

        // Consonants - gradually added
        let consonants = ["m", "n", "p", "b", "t", "d", "k", "g", "l", "r", "s", "f"];

        // Simple syllables - baby babbling
        let syllables = [
            "ma", "pa", "ba", "da", "ta", "na", "la", "ka",
            "me", "pe", "be", "de", "te", "ne", "le", "ke",
            "mi", "pi", "bi", "di", "ti", "ni", "li", "ki",
            "mo", "po", "bo", "do", "to", "no", "lo", "ko",
            "mu", "pu", "bu", "du", "tu", "nu", "lu", "ku",
        ];

        // Emotional sounds
        let emotions = ["...", "?", "!", "~", "♪", "♥", "☆", "○"];

        // Proto-words (emerge with higher coherence)
        let proto_words = [
            "moi", "toi", "oui", "non", "quoi", "ça",
            "veux", "aime", "vois", "sais", "peux",
            "bien", "mal", "plus", "encore",
            "chat", "moka", "ami", "papa", "mama",
        ];

        // Calculate characteristics from the signal
        let coherence = self.coherence();
        let energy: f32 = self.content.iter().map(|x| x.abs()).sum::<f32>();

        // Use multiple values from content for variety
        let v0 = self.content.get(0).copied().unwrap_or(0.0);
        let v1 = self.content.get(1).copied().unwrap_or(0.0);
        let v2 = self.content.get(2).copied().unwrap_or(0.0);
        let v3 = self.content.get(3).copied().unwrap_or(0.0);

        // Create indices from different parts of the signal
        let idx1 = ((v0.abs() * 100.0) as usize) % 8;
        let idx2 = ((v1.abs() * 100.0) as usize) % 12;
        let idx3 = ((v2.abs() * 100.0) as usize) % syllables.len();
        let idx4 = ((v3.abs() * 100.0) as usize) % emotions.len();

        // Build expression based on coherence level
        let mut expression = if coherence > 0.8 && self.intensity > 0.4 {
            // High coherence: proto-words!
            let word_idx = (((v0 + v1 + v2) * 100.0).abs() as usize) % proto_words.len();
            proto_words[word_idx].to_string()
        } else if coherence > 0.6 && self.intensity > 0.3 {
            // Medium coherence: syllables
            syllables[idx3].to_string()
        } else if coherence > 0.4 {
            // Low-medium coherence: consonant + vowel
            format!("{}{}", consonants[idx2], vowels[idx1])
        } else if energy > 5.0 {
            // High energy but low coherence: emotional sounds
            emotions[idx4].to_string()
        } else {
            // Low coherence: simple vowels (baby sounds)
            vowels[idx1].to_string()
        };

        // Sometimes add babbling repetition (like "mamama" or "bababa")
        if coherence > 0.5 && self.intensity > 0.2 && v2 > 0.3 {
            let syllable = &syllables[idx3];
            expression = format!("{}{}", syllable, syllable);
        }

        // Add emotional markers based on intensity
        if self.intensity > 0.5 {
            expression.push('!');
        } else if v3 < 0.0 {
            expression = format!("{}~", expression);
        }

        // Capitalize for very high intensity (like shouting)
        if self.intensity > 0.7 {
            expression = expression.to_uppercase();
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
