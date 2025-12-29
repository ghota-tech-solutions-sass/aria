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
            intensity: 0.5,
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

/// Training patterns - ARIA learns through feedback, not hardcoding!
const PATTERNS: &[TrainingPattern] = &[
    // Greetings
    TrainingPattern {
        input: "Bonjour ARIA !",
        expected: &["bonjour", "salut", "coucou", "hello"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Salut !",
        expected: &["bonjour", "salut", "coucou", "hello"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Coucou !",
        expected: &["bonjour", "salut", "coucou", "hello"],
        repetitions: 5,
    },
    // How are you?
    TrainingPattern {
        input: "Comment ça va ?",
        expected: &["bien", "ça va", "super", "content"],
        repetitions: 10,
    },
    TrainingPattern {
        input: "Ça va ?",
        expected: &["bien", "oui", "ça va", "super"],
        repetitions: 10,
    },
    // Thanks
    TrainingPattern {
        input: "Merci ARIA !",
        expected: &["rien", "plaisir", "merci"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Merci beaucoup !",
        expected: &["rien", "plaisir"],
        repetitions: 5,
    },
    // Affection
    TrainingPattern {
        input: "Je t'aime ARIA",
        expected: &["aime", "aussi", "♥", "bisou"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Tu es gentille",
        expected: &["merci", "gentil", "aime", "♥"],
        repetitions: 5,
    },
    // Questions about ARIA
    TrainingPattern {
        input: "Comment tu t'appelles ?",
        expected: &["aria", "appelle"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Tu aimes Moka ?",
        expected: &["oui", "moka", "aime", "chat"],
        repetitions: 5,
    },
    // Farewells
    TrainingPattern {
        input: "Au revoir ARIA",
        expected: &["revoir", "bye", "bientôt", "bisou"],
        repetitions: 5,
    },
    TrainingPattern {
        input: "Bonne nuit !",
        expected: &["nuit", "dors", "rêve", "bisou"],
        repetitions: 5,
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

/// Extract the word from a label like "word:moka" or "social:bonjour"
fn extract_word(label: &str) -> String {
    // Handle formats like "word:moka|emotion:♥" or "social:bonjour"
    let base = label.split('|').next().unwrap_or(label);

    if let Some(word) = base.strip_prefix("word:") {
        word.to_lowercase()
    } else if let Some(word) = base.strip_prefix("social:") {
        word.to_lowercase()
    } else if let Some(word) = base.strip_prefix("phrase:") {
        word.replace('+', " ").to_lowercase()
    } else if let Some(word) = base.strip_prefix("answer:") {
        word.replace('+', " ").to_lowercase()
    } else if let Some(word) = base.strip_prefix("spontaneous:") {
        word.to_lowercase()
    } else {
        base.to_lowercase()
    }
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

    let mut total_success = 0;
    let mut total_attempts = 0;

    for pattern in PATTERNS {
        println!("📝 Training: \"{}\"", pattern.input);
        println!("   Expected: {:?}", pattern.expected);

        for rep in 1..=pattern.repetitions {
            // Send the training input using Signal::from_text
            let signal = Signal::from_text(pattern.input);
            write.send(Message::Text(serde_json::to_string(&signal)?)).await?;

            // Wait for ARIA's response
            let mut response_text = String::new();
            let timeout = sleep(Duration::from_secs(3));
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

            // Send feedback
            let feedback_text = if success {
                total_success += 1;
                "Bravo !"
            } else {
                "Non"
            };

            let feedback_signal = Signal::from_text(feedback_text);
            write.send(Message::Text(serde_json::to_string(&feedback_signal)?)).await?;

            let status = if success { "✅" } else { "❌" };
            println!("   Rep {}/{}: {} (response: {})",
                rep, pattern.repetitions, status,
                if response_text.is_empty() { "(none)" } else { &response_text }
            );

            // Small delay between repetitions
            sleep(Duration::from_millis(300)).await;
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
