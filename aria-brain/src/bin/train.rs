//! ARIA Training System v2
//!
//! Trains ARIA through contextual conversations, not isolated patterns.
//! ARIA learns naturally through multi-turn dialogues and associations.

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use rand::seq::SliceRandom;

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
        let positive = ["aime", "adore", "bien", "super", "bravo", "merci", "♥", "good", "great"];
        let negative = ["triste", "mal", "peur", "non", "déteste"];

        content[28] = if positive.iter().any(|w| lower.contains(w)) { 1.0 } else { 0.0 };
        content[29] = if negative.iter().any(|w| lower.contains(w)) { -1.0 } else { 0.0 };
        content[30] = if lower.contains("aide") || lower.contains("veux") { 0.5 } else { 0.0 };
        content[31] = if text.ends_with('?') || lower.contains("comment") || lower.contains("qui") { 0.8 } else { 0.0 };

        Self {
            content,
            intensity: 0.5,
            label: text.to_string(),
            signal_type: "Perception".to_string(),
        }
    }
}

/// A conversational exchange
#[derive(Debug, Clone)]
struct Exchange {
    /// What to say
    input: &'static str,
    /// Expected words (any = success), None = just context, don't check
    expected: Option<&'static [&'static str]>,
    /// Delay after this exchange (ms)
    delay_ms: u64,
}

/// A full conversation scenario
#[derive(Debug, Clone)]
struct Conversation {
    /// Name of this scenario
    name: &'static str,
    /// Description
    description: &'static str,
    /// Exchanges in order
    exchanges: &'static [Exchange],
}

/// Contextual conversations - ARIA learns through dialogue flow
const CONVERSATIONS: &[Conversation] = &[
    // === Introduction & Identity ===
    Conversation {
        name: "Introduction",
        description: "Teaching ARIA her identity",
        exchanges: &[
            Exchange { input: "Bonjour !", expected: Some(&["bonjour", "salut", "coucou"]), delay_ms: 600 },
            Exchange { input: "Tu t'appelles ARIA.", expected: None, delay_ms: 400 },
            Exchange { input: "ARIA, c'est ton nom.", expected: None, delay_ms: 400 },
            Exchange { input: "Comment tu t'appelles ?", expected: Some(&["aria"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
        ],
    },

    // === Moka the cat ===
    Conversation {
        name: "Moka le chat",
        description: "Teaching about Moka",
        exchanges: &[
            Exchange { input: "J'ai un chat.", expected: None, delay_ms: 400 },
            Exchange { input: "Mon chat s'appelle Moka.", expected: None, delay_ms: 400 },
            Exchange { input: "Moka est un Bengal.", expected: None, delay_ms: 400 },
            Exchange { input: "Moka est très mignon.", expected: None, delay_ms: 400 },
            Exchange { input: "Tu aimes Moka ?", expected: Some(&["oui", "moka", "aime", "chat"]), delay_ms: 600 },
            Exchange { input: "Bravo ! Moka est adorable.", expected: None, delay_ms: 400 },
            Exchange { input: "Qui est Moka ?", expected: Some(&["chat", "moka", "bengal"]), delay_ms: 600 },
        ],
    },

    // === Emotions & Feelings ===
    Conversation {
        name: "Emotions",
        description: "Teaching emotional responses",
        exchanges: &[
            Exchange { input: "Comment ça va ?", expected: Some(&["bien", "super", "oui", "ça va"]), delay_ms: 600 },
            Exchange { input: "Je suis content de te parler.", expected: None, delay_ms: 400 },
            Exchange { input: "Tu es contente ?", expected: Some(&["oui", "content", "bien"]), delay_ms: 600 },
            Exchange { input: "Je t'aime ARIA !", expected: Some(&["oui", "aime", "aussi", "♥", "merci"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
        ],
    },

    // === Greetings variety ===
    Conversation {
        name: "Salutations",
        description: "Various greetings",
        exchanges: &[
            Exchange { input: "Salut ARIA !", expected: Some(&["salut", "bonjour", "coucou", "oui"]), delay_ms: 600 },
            Exchange { input: "Coucou !", expected: Some(&["coucou", "salut", "bonjour", "oui"]), delay_ms: 600 },
            Exchange { input: "Hello !", expected: Some(&["hello", "bonjour", "salut", "oui"]), delay_ms: 600 },
            Exchange { input: "Bravo, tu sais dire bonjour !", expected: None, delay_ms: 400 },
        ],
    },

    // === Politeness ===
    Conversation {
        name: "Politesse",
        description: "Thank you and please",
        exchanges: &[
            Exchange { input: "Merci ARIA !", expected: Some(&["rien", "plaisir", "merci", "oui"]), delay_ms: 600 },
            Exchange { input: "Tu es très gentille.", expected: Some(&["merci", "gentil", "oui"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
            Exchange { input: "Merci beaucoup !", expected: Some(&["rien", "plaisir", "merci", "oui"]), delay_ms: 600 },
        ],
    },

    // === Questions & Answers ===
    Conversation {
        name: "Questions",
        description: "Teaching Q&A patterns",
        exchanges: &[
            Exchange { input: "Tu parles français ?", expected: Some(&["oui", "français", "parle"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
            Exchange { input: "Tu comprends ?", expected: Some(&["oui", "comprend"]), delay_ms: 600 },
            Exchange { input: "Tu apprends vite !", expected: None, delay_ms: 400 },
            Exchange { input: "Tu aimes apprendre ?", expected: Some(&["oui", "aime", "apprend"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
        ],
    },

    // === Farewells ===
    Conversation {
        name: "Au revoir",
        description: "Goodbye patterns",
        exchanges: &[
            Exchange { input: "Je vais partir.", expected: None, delay_ms: 400 },
            Exchange { input: "Au revoir ARIA !", expected: Some(&["revoir", "bye", "bientôt", "au"]), delay_ms: 600 },
            Exchange { input: "À bientôt !", expected: Some(&["bientôt", "revoir", "oui", "au"]), delay_ms: 600 },
            Exchange { input: "Bonne nuit ARIA.", expected: Some(&["nuit", "bonne", "dors", "oui"]), delay_ms: 600 },
        ],
    },

    // === Associations ===
    Conversation {
        name: "Associations",
        description: "Building word associations",
        exchanges: &[
            Exchange { input: "Le soleil est jaune.", expected: None, delay_ms: 400 },
            Exchange { input: "Le soleil brille.", expected: None, delay_ms: 400 },
            Exchange { input: "J'aime le soleil.", expected: None, delay_ms: 400 },
            Exchange { input: "Tu aimes le soleil ?", expected: Some(&["oui", "soleil", "aime"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
            Exchange { input: "La lune est belle.", expected: None, delay_ms: 400 },
            Exchange { input: "La nuit, il y a la lune.", expected: None, delay_ms: 400 },
            Exchange { input: "Tu préfères le soleil ou la lune ?", expected: Some(&["soleil", "lune"]), delay_ms: 600 },
        ],
    },

    // === Colors ===
    Conversation {
        name: "Couleurs",
        description: "Teaching colors",
        exchanges: &[
            Exchange { input: "Le ciel est bleu.", expected: None, delay_ms: 400 },
            Exchange { input: "L'herbe est verte.", expected: None, delay_ms: 400 },
            Exchange { input: "Les roses sont rouges.", expected: None, delay_ms: 400 },
            Exchange { input: "De quelle couleur est le ciel ?", expected: Some(&["bleu", "ciel"]), delay_ms: 600 },
            Exchange { input: "Bravo !", expected: None, delay_ms: 300 },
            Exchange { input: "Tu aimes le bleu ?", expected: Some(&["oui", "bleu", "aime"]), delay_ms: 600 },
        ],
    },

    // === Numbers (basic) ===
    Conversation {
        name: "Nombres",
        description: "Basic numbers",
        exchanges: &[
            Exchange { input: "Un, deux, trois.", expected: None, delay_ms: 400 },
            Exchange { input: "J'ai deux chats.", expected: None, delay_ms: 400 },
            Exchange { input: "Moka et Obrigada sont mes chats.", expected: None, delay_ms: 400 },
            Exchange { input: "Combien de chats j'ai ?", expected: Some(&["deux", "2", "moka", "obrigada", "chat"]), delay_ms: 600 },
        ],
    },

    // === Reinforcement through repetition ===
    Conversation {
        name: "Renforcement",
        description: "Reinforcing core concepts",
        exchanges: &[
            Exchange { input: "ARIA, tu es intelligente.", expected: None, delay_ms: 400 },
            Exchange { input: "ARIA apprend vite.", expected: None, delay_ms: 400 },
            Exchange { input: "Je suis fier de toi ARIA.", expected: None, delay_ms: 400 },
            Exchange { input: "Tu es fière de toi ?", expected: Some(&["oui", "fier", "fière"]), delay_ms: 600 },
            Exchange { input: "Bravo ARIA !", expected: None, delay_ms: 300 },
            Exchange { input: "Tu es la meilleure !", expected: Some(&["merci", "♥", "oui", "aime"]), delay_ms: 600 },
        ],
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
    let base = label.split('|').next().unwrap_or(label);

    let prefixes = [
        "word:", "phrase:", "answer:", "spontaneous:", "babble:",
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
    println!("🎓 ARIA Training System v2");
    println!("===========================");
    println!("Training through contextual conversations\n");

    let url = std::env::var("ARIA_BRAIN_URL")
        .unwrap_or_else(|_| "ws://localhost:8765/aria".to_string());

    let epochs: u32 = std::env::var("ARIA_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    // Timeout for responses (longer for bigger brains)
    let timeout_ms: u64 = std::env::var("ARIA_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);  // 2 seconds default

    println!("Connecting to {}...", url);
    println!("Epochs: {}, Timeout: {}ms\n", epochs, timeout_ms);

    let (ws_stream, _) = connect_async(&url).await?;
    let (mut write, mut read) = ws_stream.split();

    println!("Connected! Starting training...\n");

    let mut total_success = 0;
    let mut total_tests = 0;
    let mut rng = rand::thread_rng();

    for epoch in 1..=epochs {
        println!("╔═══════════════════════════════════════╗");
        println!("║  Epoch {}/{}                            ║", epoch, epochs);
        println!("╚═══════════════════════════════════════╝\n");

        // Shuffle conversations for variety
        let mut conversations: Vec<_> = CONVERSATIONS.iter().collect();
        conversations.shuffle(&mut rng);

        for conv in &conversations {
            println!("📚 {} - {}", conv.name, conv.description);
            println!("─────────────────────────────────────────");

            for exchange in conv.exchanges {
                // Send message
                let signal = Signal::from_text(exchange.input);
                write.send(Message::Text(serde_json::to_string(&signal)?)).await?;

                print!("   You: {}", exchange.input);

                // Wait for response (use configured timeout)
                let mut response_text = String::new();
                let timeout = sleep(Duration::from_millis(timeout_ms));
                tokio::pin!(timeout);

                loop {
                    tokio::select! {
                        msg = read.next() => {
                            if let Some(Ok(Message::Text(text))) = msg {
                                if let Ok(response) = serde_json::from_str::<AriaResponse>(&text) {
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

                // Check response if expected
                if let Some(expected) = exchange.expected {
                    total_tests += 1;
                    let success = expected.iter().any(|exp| response_text.contains(exp));

                    if success {
                        total_success += 1;
                        println!(" → ARIA: {} ✅", response_text);
                    } else if response_text.is_empty() {
                        println!(" → ARIA: (silence) ❌");
                    } else {
                        println!(" → ARIA: {} ❌ (expected: {:?})", response_text, expected);
                    }
                } else {
                    // Just context, no test
                    if !response_text.is_empty() {
                        println!(" → ARIA: {} 💭", response_text);
                    } else {
                        println!(" (context)");
                    }
                }

                // Small pause between exchanges
                sleep(Duration::from_millis(300)).await;
            }

            println!();

            // Pause between conversations
            sleep(Duration::from_millis(800)).await;
        }

        let epoch_rate = if total_tests > 0 {
            (total_success as f32 / total_tests as f32) * 100.0
        } else {
            0.0
        };

        println!("📊 After epoch {}: {:.1}% success ({}/{})\n",
            epoch, epoch_rate, total_success, total_tests);
    }

    let final_rate = if total_tests > 0 {
        (total_success as f32 / total_tests as f32) * 100.0
    } else {
        0.0
    };

    println!("═══════════════════════════════════════");
    println!("🎓 Training Complete!");
    println!("   Total: {}/{} ({:.1}%)", total_success, total_tests, final_rate);
    println!("═══════════════════════════════════════\n");

    Ok(())
}
