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

        // Positive emotions (love, joy, affection) - FR + EN
        let positive_words = [
            "love", "happy", "good", "great", "beautiful", "nice", "sweet", "cute",
            "aime", "adore", "content", "heureux", "heureuse", "bien", "super", "génial",
            "joli", "jolie", "beau", "belle", "mignon", "mignonne", "bisou", "calin",
            "merci", "bravo", "cool", "chouette", "sympa", "♥", "❤", ":)", "<3"
        ];
        content[28] = if positive_words.iter().any(|w| lower.contains(w)) { 1.0 } else { 0.0 };

        // Negative emotions (sadness, anger, fear) - FR + EN
        let negative_words = [
            "hate", "sad", "bad", "angry", "fear", "scared", "hurt", "pain",
            "triste", "mal", "peur", "colère", "fâché", "fâchée", "méchant", "nul",
            "déteste", "horrible", "moche", "pleure", "pleurer", ":(", ":'("
        ];
        content[29] = if negative_words.iter().any(|w| lower.contains(w)) { -1.0 } else { 0.0 };

        // Requests and needs - FR + EN
        let request_words = [
            "help", "please", "want", "need",
            "aide", "s'il te plaît", "stp", "veux", "voudrais", "besoin", "peux"
        ];
        content[30] = if request_words.iter().any(|w| lower.contains(w)) { 0.5 } else { 0.0 };

        // Questions and curiosity - FR + EN
        let question_words = [
            "why", "how", "what", "when", "where", "who",
            "pourquoi", "comment", "quoi", "quand", "où", "qui", "est-ce"
        ];
        content[31] = if question_words.iter().any(|w| lower.contains(w)) { 0.8 } else { 0.0 };

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
        // Parse label for word and emotional marker
        // Format: "word:moka|emotion:♥" or just "word:moka" or "emergence@123"
        let (main_label, emotional_marker) = if let Some(pipe_pos) = self.label.find('|') {
            let main = &self.label[..pipe_pos];
            let emotion_part = &self.label[pipe_pos + 1..];
            let marker = emotion_part.strip_prefix("emotion:").unwrap_or("");
            (main.to_string(), if marker.is_empty() { None } else { Some(marker.to_string()) })
        } else {
            (self.label.clone(), None)
        };

        // If the brain is babbling (learning to communicate)
        if main_label.starts_with("babble:") {
            let syllable = main_label.strip_prefix("babble:").unwrap_or("hm");
            let base_expression = if self.intensity > 0.5 {
                format!("{}!", syllable)
            } else {
                syllable.to_string()
            };

            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
        }

        // If the brain is responding to a social context (greeting, farewell, etc.)
        if main_label.starts_with("social:") {
            // Format: social:greeting:bonjour or social:farewell:bye
            let parts: Vec<&str> = main_label.split(':').collect();
            let (context_type, word) = if parts.len() >= 3 {
                (parts[1], parts[2])
            } else if parts.len() >= 2 {
                (parts[1], "")
            } else {
                ("general", "")
            };

            let base_expression = match context_type {
                "greeting" => {
                    // Greeting response - friendly!
                    if word.is_empty() {
                        "bonjour!".to_string()
                    } else if self.intensity > 0.5 {
                        format!("{}!", word.to_uppercase())
                    } else {
                        format!("{}~", word)
                    }
                }
                "farewell" => {
                    // Saying goodbye
                    if word.is_empty() {
                        "bye...".to_string()
                    } else if self.intensity > 0.5 {
                        format!("{}!", word)
                    } else {
                        format!("{}~", word)
                    }
                }
                "thanks" => {
                    // Responding to thanks
                    if word == "derien" {
                        "de rien~".to_string()
                    } else if word.is_empty() {
                        "~".to_string()
                    } else {
                        format!("{}~", word)
                    }
                }
                "affection" => {
                    // Expressing love back
                    if word.is_empty() {
                        "♥".to_string()
                    } else if self.intensity > 0.5 {
                        format!("{} ♥♥", word.to_uppercase())
                    } else {
                        format!("{} ♥", word)
                    }
                }
                _ => word.to_string(),
            };

            // Add emotional marker if present
            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
        }

        // If the brain is speaking spontaneously (without being asked)
        if main_label.starts_with("spontaneous:") {
            let content = main_label.strip_prefix("spontaneous:").unwrap_or(&main_label);

            // Different spontaneous expressions
            let base_expression = match content {
                "attention" => {
                    // Seeking attention
                    if self.intensity > 0.3 {
                        "...hé ?".to_string()
                    } else {
                        "...".to_string()
                    }
                }
                "joy" => {
                    // General joy
                    "♪~".to_string()
                }
                "excited" => {
                    // Excited babbling
                    "ah!".to_string()
                }
                "curious" => {
                    // General curiosity
                    "hm?".to_string()
                }
                "babble" => {
                    // Soft babbling
                    "mmm~".to_string()
                }
                "bored" => {
                    // Restless, wants stimulation
                    "...hm".to_string()
                }
                word => {
                    // Thinking about a specific word she loves
                    if self.intensity > 0.4 {
                        format!("{}!", word)
                    } else if self.intensity > 0.2 {
                        word.to_string()
                    } else {
                        format!("{}...", word)
                    }
                }
            };

            // Add emotional marker if present
            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
        }

        // If the brain is answering a question (oui/non)
        if main_label.starts_with("answer:") {
            let answer = main_label.strip_prefix("answer:").unwrap_or(&main_label);
            let parts: Vec<&str> = answer.split('+').collect();

            let (response, word) = if parts.len() >= 2 {
                (parts[0], parts[1])
            } else {
                (answer, "")
            };

            // Format based on yes/no response
            let base_expression = if response == "oui" {
                if self.intensity > 0.5 {
                    if word.is_empty() {
                        "OUI!".to_string()
                    } else {
                        format!("OUI {} ♥", word.to_uppercase())
                    }
                } else {
                    if word.is_empty() {
                        "oui!".to_string()
                    } else {
                        format!("oui {} ♥", word)
                    }
                }
            } else if response == "non" {
                if self.intensity > 0.5 {
                    if word.is_empty() {
                        "NON!".to_string()
                    } else {
                        format!("NON {}...", word.to_uppercase())
                    }
                } else {
                    if word.is_empty() {
                        "non...".to_string()
                    } else {
                        format!("non {}...", word)
                    }
                }
            } else {
                // Unknown answer type
                format!("{}?", word)
            };

            // Add emotional marker if present
            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
        }

        // If the brain is recalling a memory
        if main_label.starts_with("memory:") {
            let memory_content = main_label.strip_prefix("memory:").unwrap_or(&main_label);
            let parts: Vec<&str> = memory_content.split('|').collect();

            let base_expression = if parts.len() >= 2 {
                match parts[0] {
                    "first" => {
                        // First time memory - "Je me souviens... [keyword]!"
                        let keyword = parts.get(2).unwrap_or(&"ça");
                        if self.intensity > 0.5 {
                            format!("je me souviens... {}! ✨", keyword)
                        } else {
                            format!("première fois... {} 💭", keyword)
                        }
                    }
                    "emotion" => {
                        // Emotional memory
                        let keyword = parts.get(1).unwrap_or(&"moment");
                        format!("{}... 💭", keyword)
                    }
                    "recall" => {
                        // General memory recall
                        let keyword = parts.get(1).unwrap_or(&"souviens");
                        format!("{}... 💭", keyword)
                    }
                    _ => {
                        // Unknown memory type
                        format!("souviens... 💭")
                    }
                }
            } else {
                format!("souviens... 💭")
            };

            // Add emotional marker if present
            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
        }

        // If the brain matched a known word, use it!
        if main_label.starts_with("word:") {
            let word = main_label.strip_prefix("word:").unwrap_or(&main_label);
            // Add some baby-like variations
            let base_expression = if self.intensity > 0.5 {
                format!("{}!", word.to_uppercase())
            } else if self.intensity > 0.3 {
                word.to_string()
            } else {
                // Whisper/uncertain
                format!("{}...", word)
            };

            // Add emotional marker if present
            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
        }

        // If the brain created a phrase (word association), display the words!
        if main_label.starts_with("phrase:") {
            let phrase = main_label.strip_prefix("phrase:").unwrap_or(&main_label);
            // Parse "word1+word2" or "word1+word2+word3" format
            let words: Vec<&str> = phrase.split('+').collect();

            let base_expression = match words.len() {
                3 => {
                    // 3-word phrase! Like a baby making simple sentences
                    if self.intensity > 0.5 {
                        format!("{} {} {}!", words[0].to_uppercase(), words[1].to_uppercase(), words[2].to_uppercase())
                    } else if self.intensity > 0.3 {
                        format!("{} {} {}", words[0], words[1], words[2])
                    } else {
                        format!("{}... {} {}", words[0], words[1], words[2])
                    }
                }
                2 => {
                    // 2-word phrase
                    if self.intensity > 0.5 {
                        format!("{} {}!", words[0].to_uppercase(), words[1].to_uppercase())
                    } else if self.intensity > 0.3 {
                        format!("{} {}", words[0], words[1])
                    } else {
                        format!("{}... {}", words[0], words[1])
                    }
                }
                _ => phrase.to_string()
            };

            // Add emotional marker if present
            return if let Some(marker) = emotional_marker {
                format!("{} {}", base_expression, marker)
            } else {
                base_expression
            };
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

        // Add emotional marker if present (for babbling too)
        if let Some(marker) = emotional_marker {
            expression = format!("{} {}", expression, marker);
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
