//! ARIA Training System
//!
//! Automatically trains ARIA with common French conversation patterns.
//! This accelerates learning without hardcoding responses.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Signal to send to ARIA (matches brain's Signal struct)
#[derive(Clone, Debug, Serialize)]
struct Signal {
    content: Vec<f32>,
    intensity: f32,
    label: String,
    signal_type: String,
}

impl Signal {
    /// Create a signal from text input (same encoding as aria-body)
    fn from_text(text: &str) -> Self {
        let mut content = vec![0.0f32; 32];

        // Character-based encoding
        for (i, ch) in text.chars().take(20).enumerate() {
            content[i] = (ch as u32 as f32) / 256.0;
        }

        // Statistical features
        let len = text.len() as f32;
        content[20] = (len / 100.0).min(1.0);
        content[21] = text.chars().filter(|c| c.is_uppercase()).count() as f32 / len.max(1.0);
        content[22] = text.chars().filter(|c| c.is_whitespace()).count() as f32 / len.max(1.0);
        content[23] = text.chars().filter(|c| c.is_numeric()).count() as f32 / len.max(1.0);

        // Punctuation signals
        content[24] = if text.ends_with('?') { 1.0 } else { 0.0 };
        content[25] = if text.ends_with('!') { 1.0 } else { 0.0 };
        content[26] = if text.ends_with('.') { 1.0 } else { 0.0 };
        content[27] = if text.contains(',') { 0.5 } else { 0.0 };

        // Emotional markers
        let lower = text.to_lowercase();
        let positive = ["aime", "adore", "bien", "super", "bravo", "merci", "♥"];
        let negative = ["triste", "mal", "peur", "non", "déteste"];

        content[28] = if positive.iter().any(|w| lower.contains(w)) { 1.0 } else { 0.0 };
        content[29] = if negative.iter().any(|w| lower.contains(w)) { -1.0 } else { 0.0 };
        content[30] = if lower.contains("aide") || lower.contains("veux") { 0.5 } else { 0.0 };
        content[31] = if text.ends_with('?') || lower.contains("comment") { 0.8 } else { 0.0 };

        Self {
            content,
            intensity: 0.5,  // Normal intensity - don't overwhelm
            label: text.to_string(),
            signal_type: "Perception".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct TrainingPattern {
    /// What to say to ARIA
    input: &'static str,
    /// Expected words in ARIA's response (any of these = success)
    expected: &'static [&'static str],
    /// How many times to train this pattern
    repetitions: u32,
}

/// Teaching patterns - first teach ARIA the response, then test
/// Format: (question, expected_responses, teaching_phrase)
const TEACHING_PATTERNS: &[(&str, &[&str], &str)] = &[
    // Teach "bien" as response to "ça va"
    ("Comment ça va ?", &["bien", "super", "ça va"], "Ça va bien !"),
    ("Ça va ?", &["bien", "oui", "super"], "Oui ça va bien !"),

    // Teach farewell responses
    ("Au revoir !", &["revoir", "bye", "bientôt"], "Au revoir, à bientôt !"),
    ("Bonne nuit !", &["nuit", "dors", "bonne"], "Bonne nuit, dors bien !"),
];

/// Training patterns - ARIA learns through feedback, not hardcoding!
const PATTERNS: &[TrainingPattern] = &[
    // Greetings (ARIA already knows these from social context)
    TrainingPattern {
        input: "Bonjour ARIA !",
        expected: &["bonjour", "salut", "coucou", "hello"],
        repetitions: 3,
    },
    TrainingPattern {
        input: "Salut !",
        expected: &["bonjour", "salut", "coucou", "hello"],
        repetitions: 3,
    },
    TrainingPattern {
        input: "Coucou !",
        expected: &["bonjour", "salut", "coucou", "hello"],
        repetitions: 3,
    },
    // How are you? (after teaching)
    TrainingPattern {
        input: "Comment ça va ?",
        expected: &["bien", "ça va", "super"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Ça va ?",
        expected: &["bien", "oui", "super"],
        repetitions: 5,
    },
    // Thanks
    TrainingPattern {
        input: "Merci ARIA !",
        expected: &["rien", "derien", "plaisir", "merci"],
        repetitions: 3,
    },
    TrainingPattern {
        input: "Merci beaucoup !",
        expected: &["rien", "derien", "plaisir"],
        repetitions: 3,
    },
    // Affection
    TrainingPattern {
        input: "Je t'aime ARIA",
        expected: &["aime", "aussi", "♥", "bisou"],
        repetitions: 3,
    },
    TrainingPattern {
        input: "Tu es gentille",
        expected: &["merci", "gentil", "aime", "♥"],
        repetitions: 3,
    },
    // Questions about ARIA
    TrainingPattern {
        input: "Comment tu t'appelles ?",
        expected: &["aria", "appelle"],
        repetitions: 3,
    },
    TrainingPattern {
        input: "Tu aimes Moka ?",
        expected: &["oui", "moka", "aime", "chat"],
        repetitions: 3,
    },
    // Farewells (after teaching)
    TrainingPattern {
        input: "Au revoir ARIA",
        expected: &["revoir", "bye", "bientôt"],
        repetitions: 3,
    },
    TrainingPattern {
        input: "Bonne nuit !",
        expected: &["nuit", "dors", "bonne", "bien"],
        repetitions: 3,
    },
];

/// Response signal from ARIA
#[derive(Debug, Deserialize)]
struct AriaResponse {
    #[serde(default)]
    label: String,
    #[serde(default)]
    intensity: f32,
}

/// Extract the word from a label like "word:moka" or "greeting:bonjour"
fn extract_word(label: &str) -> String {
    // Handle formats like "word:moka|emotion:♥" or "greeting:bonjour"
    let base = label.split('|').next().unwrap_or(label);

    // Try all known prefixes
    let prefixes = [
        "word:", "phrase:", "answer:", "spontaneous:",
        "greeting:", "farewell:", "thanks:", "affection:", "social:",
    ];

    for prefix in prefixes {
        if let Some(word) = base.strip_prefix(prefix) {
            return word.replace('+', " ").to_lowercase();
        }
    }

    base.to_lowercase()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎓 ARIA Training System");
    println!("========================");
    println!();

    let url = std::env::var("ARIA_BRAIN_URL")
        .unwrap_or_else(|_| "ws://localhost:8765/aria".to_string());

    println!("Connecting to {}...", url);

    let (ws_stream, _) = connect_async(&url).await?;
    let (mut write, mut read) = ws_stream.split();

    println!("Connected! Starting training...\n");

    // Skip teaching phase - let ARIA learn naturally through testing
    // The social context system already handles greetings, thanks, affection
    println!("📝 Testing ARIA's responses...\n");
    println!("   (ARIA learns through social context detection, not rote memorization)\n");

    let mut total_success = 0;
    let mut total_attempts = 0;

    for pattern in PATTERNS {
        println!("📝 Training: \"{}\"", pattern.input);
        println!("   Expected: {:?}", pattern.expected);

        for rep in 1..=pattern.repetitions {
            // Send the training input using Signal::from_text
            let signal = Signal::from_text(pattern.input);
            write.send(Message::Text(serde_json::to_string(&signal)?)).await?;

            // Wait for ARIA's response (faster brain = faster response)
            let mut response_text = String::new();
            let timeout = sleep(Duration::from_millis(800));
            tokio::pin!(timeout);

            loop {
                tokio::select! {
                    msg = read.next() => {
                        if let Some(Ok(Message::Text(text))) = msg {
                            if let Ok(response) = serde_json::from_str::<AriaResponse>(&text) {
                                // Only consider signals with reasonable intensity
                                if response.intensity > 0.01 && !response.label.is_empty() {
                                    response_text = extract_word(&response.label);
                                    break;
                                }
                            }
                        }
                    }
                    _ = &mut timeout => {
                        break;
                    }
                }
            }

            total_attempts += 1;

            // Check if response matches expected
            let success = pattern.expected.iter()
                .any(|exp| response_text.contains(exp));

            // Only send POSITIVE feedback - no negative feedback to avoid polluting vocabulary
            if success {
                total_success += 1;
                let feedback_signal = Signal::from_text("Bravo !");
                write.send(Message::Text(serde_json::to_string(&feedback_signal)?)).await?;
            }
            // When wrong, we just stay silent - no "non" or "hmm" to learn

            let status = if success { "✅" } else { "❌" };
            println!("   Rep {}/{}: {} (response: {})",
                rep, pattern.repetitions, status,
                if response_text.is_empty() { "(none)" } else { &response_text }
            );

            // Delay between repetitions
            sleep(Duration::from_millis(200)).await;
        }

        println!();
    }

    let success_rate = if total_attempts > 0 {
        (total_success as f32 / total_attempts as f32) * 100.0
    } else {
        0.0
    };

    println!("========================");
    println!("🎓 Training Complete!");
    println!("   Success: {}/{} ({:.1}%)", total_success, total_attempts, success_rate);
    println!();

    Ok(())
}
