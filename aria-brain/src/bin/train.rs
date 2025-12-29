//! ARIA Training System
//!
//! Automatically trains ARIA with common French conversation patterns.
//! This accelerates learning without hardcoding responses.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};

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

#[derive(Debug, Serialize, Deserialize)]
struct AriaMessage {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    intensity: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
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
            // Send the training input
            let msg = AriaMessage {
                msg_type: "text".to_string(),
                text: Some(pattern.input.to_string()),
                content: None,
                intensity: None,
                label: None,
            };

            write.send(Message::Text(serde_json::to_string(&msg)?)).await?;

            // Wait for ARIA's response
            let mut response_text = String::new();
            let timeout = sleep(Duration::from_secs(5));
            tokio::pin!(timeout);

            loop {
                tokio::select! {
                    msg = read.next() => {
                        if let Some(Ok(Message::Text(text))) = msg {
                            if let Ok(aria_msg) = serde_json::from_str::<AriaMessage>(&text) {
                                if aria_msg.msg_type == "expression" {
                                    if let Some(label) = aria_msg.label {
                                        response_text = label.to_lowercase();
                                        break;
                                    }
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
            let feedback = if success {
                total_success += 1;
                "Bravo !"
            } else {
                "Non"
            };

            let feedback_msg = AriaMessage {
                msg_type: "text".to_string(),
                text: Some(feedback.to_string()),
                content: None,
                intensity: None,
                label: None,
            };

            write.send(Message::Text(serde_json::to_string(&feedback_msg)?)).await?;

            let status = if success { "✅" } else { "❌" };
            println!("   Rep {}/{}: {} (response: {})",
                rep, pattern.repetitions, status,
                if response_text.is_empty() { "(none)" } else { &response_text }
            );

            // Small delay between repetitions
            sleep(Duration::from_millis(500)).await;
        }

        println!();
    }

    let success_rate = (total_success as f32 / total_attempts as f32) * 100.0;
    println!("========================");
    println!("🎓 Training Complete!");
    println!("   Success: {}/{} ({:.1}%)", total_success, total_attempts, success_rate);
    println!();

    Ok(())
}
