//! ARIA Training System v3 - Meta-Learning Edition
//!
//! Trains ARIA through contextual conversations AND autonomous exploration.
//! ARIA learns naturally through dialogue and self-directed discovery.
//!
//! New in v3:
//! - Meta-learning stats display
//! - Autonomous exploration phases
//! - Strategy performance tracking

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

/// Meta-learning stats from /meta endpoint
#[derive(Debug, Deserialize)]
struct MetaStats {
    total_evaluations: u64,
    current_strategy: String,
    exploration_rate: f32,
    strategies: Vec<StrategyStats>,
    goals: Vec<GoalStats>,
    progress: ProgressStats,
}

#[derive(Debug, Deserialize)]
struct StrategyStats {
    #[serde(rename = "type")]
    strategy_type: String,
    usage_count: u64,
    avg_reward: f32,
    best_reward: f32,
    recent_avg: f32,
}

#[derive(Debug, Deserialize)]
struct GoalStats {
    description: String,
    progress: f32,
    completed: bool,
}

#[derive(Debug, Deserialize)]
struct ProgressStats {
    learning_quality: f32,
    competence_level: f32,
    trend: String,
    recent_successes: u32,
    recent_failures: u32,
    status: String,
}

/// Contextual conversations - ARIA learns through dialogue flow
const CONVERSATIONS: &[Conversation] = &[
    // === Introduction & Identity ===
    Conversation {
        name: "Introduction",
        description: "Teaching ARIA her identity",
        exchanges: &[
            Exchange { input: "Bonjour !", expected: Some(&["bonjour", "salut", "coucou"]) },
            Exchange { input: "Tu t'appelles ARIA.", expected: None },
            Exchange { input: "ARIA, c'est ton nom.", expected: None },
            Exchange { input: "Comment tu t'appelles ?", expected: Some(&["aria"]) },
            Exchange { input: "Bravo !", expected: None },
        ],
    },

    // === Moka the cat ===
    Conversation {
        name: "Moka le chat",
        description: "Teaching about Moka",
        exchanges: &[
            Exchange { input: "J'ai un chat.", expected: None },
            Exchange { input: "Mon chat s'appelle Moka.", expected: None },
            Exchange { input: "Moka est un Bengal.", expected: None },
            Exchange { input: "Moka est très mignon.", expected: None },
            Exchange { input: "Tu aimes Moka ?", expected: Some(&["oui", "moka", "aime", "chat"]) },
            Exchange { input: "Bravo ! Moka est adorable.", expected: None },
            Exchange { input: "Qui est Moka ?", expected: Some(&["chat", "moka", "bengal"]) },
        ],
    },

    // === Emotions & Feelings ===
    Conversation {
        name: "Emotions",
        description: "Teaching emotional responses",
        exchanges: &[
            Exchange { input: "Comment ça va ?", expected: Some(&["bien", "super", "oui", "ça va"]) },
            Exchange { input: "Je suis content de te parler.", expected: None },
            Exchange { input: "Tu es contente ?", expected: Some(&["oui", "content", "bien"]) },
            Exchange { input: "Je t'aime ARIA !", expected: Some(&["oui", "aime", "aussi", "♥", "merci"]) },
            Exchange { input: "Bravo !", expected: None },
        ],
    },

    // === Greetings variety ===
    Conversation {
        name: "Salutations",
        description: "Various greetings",
        exchanges: &[
            Exchange { input: "Salut ARIA !", expected: Some(&["salut", "bonjour", "coucou", "oui"]) },
            Exchange { input: "Coucou !", expected: Some(&["coucou", "salut", "bonjour", "oui"]) },
            Exchange { input: "Hello !", expected: Some(&["hello", "bonjour", "salut", "oui"]) },
            Exchange { input: "Bravo, tu sais dire bonjour !", expected: None },
        ],
    },

    // === Politeness ===
    Conversation {
        name: "Politesse",
        description: "Thank you and please",
        exchanges: &[
            Exchange { input: "Merci ARIA !", expected: Some(&["rien", "plaisir", "merci", "oui"]) },
            Exchange { input: "Tu es très gentille.", expected: Some(&["merci", "gentil", "oui"]) },
            Exchange { input: "Bravo !", expected: None },
            Exchange { input: "Merci beaucoup !", expected: Some(&["rien", "plaisir", "merci", "oui"]) },
        ],
    },

    // === Questions & Answers ===
    Conversation {
        name: "Questions",
        description: "Teaching Q&A patterns",
        exchanges: &[
            Exchange { input: "Tu parles français ?", expected: Some(&["oui", "français", "parle"]) },
            Exchange { input: "Bravo !", expected: None },
            Exchange { input: "Tu comprends ?", expected: Some(&["oui", "comprend"]) },
            Exchange { input: "Tu apprends vite !", expected: None },
            Exchange { input: "Tu aimes apprendre ?", expected: Some(&["oui", "aime", "apprend"]) },
            Exchange { input: "Bravo !", expected: None },
        ],
    },

    // === Farewells ===
    Conversation {
        name: "Au revoir",
        description: "Goodbye patterns",
        exchanges: &[
            Exchange { input: "Je vais partir.", expected: None },
            Exchange { input: "Au revoir ARIA !", expected: Some(&["revoir", "bye", "bientôt", "au"]) },
            Exchange { input: "À bientôt !", expected: Some(&["bientôt", "revoir", "oui", "au"]) },
            Exchange { input: "Bonne nuit ARIA.", expected: Some(&["nuit", "bonne", "dors", "oui"]) },
        ],
    },

    // === Associations ===
    Conversation {
        name: "Associations",
        description: "Building word associations",
        exchanges: &[
            Exchange { input: "Le soleil est jaune.", expected: None },
            Exchange { input: "Le soleil brille.", expected: None },
            Exchange { input: "J'aime le soleil.", expected: None },
            Exchange { input: "Tu aimes le soleil ?", expected: Some(&["oui", "soleil", "aime"]) },
            Exchange { input: "Bravo !", expected: None },
            Exchange { input: "La lune est belle.", expected: None },
            Exchange { input: "La nuit, il y a la lune.", expected: None },
            Exchange { input: "Tu préfères le soleil ou la lune ?", expected: Some(&["soleil", "lune"]) },
        ],
    },

    // === Colors ===
    Conversation {
        name: "Couleurs",
        description: "Teaching colors",
        exchanges: &[
            Exchange { input: "Le ciel est bleu.", expected: None },
            Exchange { input: "L'herbe est verte.", expected: None },
            Exchange { input: "Les roses sont rouges.", expected: None },
            Exchange { input: "De quelle couleur est le ciel ?", expected: Some(&["bleu", "ciel"]) },
            Exchange { input: "Bravo !", expected: None },
            Exchange { input: "Tu aimes le bleu ?", expected: Some(&["oui", "bleu", "aime"]) },
        ],
    },

    // === Numbers (basic) ===
    Conversation {
        name: "Nombres",
        description: "Basic numbers",
        exchanges: &[
            Exchange { input: "Un, deux, trois.", expected: None },
            Exchange { input: "J'ai deux chats.", expected: None },
            Exchange { input: "Moka et Obrigada sont mes chats.", expected: None },
            Exchange { input: "Combien de chats j'ai ?", expected: Some(&["deux", "2", "moka", "obrigada", "chat"]) },
        ],
    },

    // === Reinforcement through repetition ===
    Conversation {
        name: "Renforcement",
        description: "Reinforcing core concepts",
        exchanges: &[
            Exchange { input: "ARIA, tu es intelligente.", expected: None },
            Exchange { input: "ARIA apprend vite.", expected: None },
            Exchange { input: "Je suis fier de toi ARIA.", expected: None },
            Exchange { input: "Tu es fière de toi ?", expected: Some(&["oui", "fier", "fière"]) },
            Exchange { input: "Bravo ARIA !", expected: None },
            Exchange { input: "Tu es la meilleure !", expected: Some(&["merci", "♥", "oui", "aime"]) },
        ],
    },

    // === Exploration encouragement (NEW) ===
    Conversation {
        name: "Exploration",
        description: "Encouraging ARIA to explore",
        exchanges: &[
            Exchange { input: "Tu peux explorer.", expected: None },
            Exchange { input: "Essaie des choses nouvelles.", expected: None },
            Exchange { input: "Tu peux combiner des mots.", expected: None },
            Exchange { input: "Chat et aime font une belle phrase.", expected: None },
            Exchange { input: "Moka aime jouer.", expected: None },
            Exchange { input: "Bravo ! Continue d'explorer.", expected: None },
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
        "explore:", "memory:",
    ];

    for prefix in prefixes {
        if let Some(word) = base.strip_prefix(prefix) {
            return word.replace('+', " ").to_lowercase();
        }
    }

    base.to_lowercase()
}

/// Fetch meta-learning stats from the brain
async fn fetch_meta_stats(base_url: &str) -> Option<MetaStats> {
    let url = base_url.replace("ws://", "http://").replace("/aria", "/meta");
    match reqwest::get(&url).await {
        Ok(resp) => resp.json().await.ok(),
        Err(_) => None,
    }
}

/// Display meta-learning stats
fn display_meta_stats(stats: &MetaStats) {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  🧠 META-LEARNING STATS                                   ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Total evaluations: {:>6}                                ║", stats.total_evaluations);
    println!("║  Current strategy:  {:>15}                    ║", stats.current_strategy);
    println!("║  Exploration rate:  {:>5.1}%                              ║", stats.exploration_rate * 100.0);
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  📊 STRATEGY PERFORMANCE                                  ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    for s in &stats.strategies {
        if s.usage_count > 0 {
            let bar_len = (s.avg_reward * 20.0) as usize;
            let bar: String = "█".repeat(bar_len) + &"░".repeat(20 - bar_len);
            println!("║  {:>12}: [{}] {:.2}  ({} uses)  ║",
                s.strategy_type, bar, s.avg_reward, s.usage_count);
        }
    }

    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  📈 PROGRESS                                              ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Status: {}                     ║", stats.progress.status);
    println!("║  Learning quality: {:.2}   Competence: {:.0}%              ║",
        stats.progress.learning_quality, stats.progress.competence_level * 100.0);
    println!("║  Recent: {} successes, {} failures                       ║",
        stats.progress.recent_successes, stats.progress.recent_failures);

    if !stats.goals.is_empty() {
        println!("╠═══════════════════════════════════════════════════════════╣");
        println!("║  🎯 CURRENT GOALS                                         ║");
        println!("╠═══════════════════════════════════════════════════════════╣");
        for g in &stats.goals {
            let status = if g.completed { "✅" } else { "⏳" };
            println!("║  {} {:>40} ({:.0}%)   ║", status, g.description, g.progress * 100.0);
        }
    }

    println!("╚═══════════════════════════════════════════════════════════╝\n");
}

/// Autonomous exploration phase - let ARIA explore on her own
async fn exploration_phase(
    write: &mut futures_util::stream::SplitSink<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>
        >,
        Message
    >,
    read: &mut futures_util::stream::SplitStream<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>
        >
    >,
    duration_secs: u64,
    timeout_ms: u64,
) -> Result<u32, Box<dyn std::error::Error>> {
    println!("\n🔍 EXPLORATION PHASE ({} seconds)", duration_secs);
    println!("─────────────────────────────────────────");
    println!("   ARIA explores autonomously...\n");

    let start = std::time::Instant::now();
    let mut explorations = 0u32;

    // Give ARIA some stimuli to think about
    let stimuli = [
        "Pense à tes mots préférés.",
        "Qu'est-ce que tu aimes ?",
        "Explore de nouvelles idées.",
    ];

    for stimulus in stimuli {
        let signal = Signal::from_text(stimulus);
        write.send(Message::Text(serde_json::to_string(&signal)?)).await?;
        sleep(Duration::from_millis(500)).await;
    }

    // Let ARIA explore
    while start.elapsed().as_secs() < duration_secs {
        let timeout = sleep(Duration::from_millis(timeout_ms));
        tokio::pin!(timeout);

        tokio::select! {
            msg = read.next() => {
                if let Some(Ok(Message::Text(text))) = msg {
                    if let Ok(response) = serde_json::from_str::<AriaResponse>(&text) {
                        if response.intensity > 0.01 && !response.label.is_empty() {
                            let word = extract_word(&response.label);

                            // Detect exploration
                            if response.label.contains("explore:") {
                                explorations += 1;
                                println!("   🔍 Exploration #{}: {}", explorations, word);

                                // Give positive feedback for exploring
                                let feedback = Signal::from_text("Bravo !");
                                write.send(Message::Text(serde_json::to_string(&feedback)?)).await?;
                            } else if response.label.contains("spontaneous:") {
                                println!("   💭 Thinking: {}", word);
                            }
                        }
                    }
                }
            }
            _ = &mut timeout => {
                // Ping ARIA to keep her engaged
                let ping = Signal::from_text("Continue...");
                write.send(Message::Text(serde_json::to_string(&ping)?)).await?;
            }
        }
    }

    println!("\n   ✅ {} explorations during this phase\n", explorations);
    Ok(explorations)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                                                           ║");
    println!("║  🎓 ARIA Training System v3 - Meta-Learning Edition       ║");
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

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

    // Exploration phase duration
    let explore_secs: u64 = std::env::var("ARIA_EXPLORE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);  // 30 seconds default

    println!("Connecting to {}...", url);
    println!("Epochs: {}, Timeout: {}ms, Explore: {}s\n", epochs, timeout_ms, explore_secs);

    // Fetch initial meta stats
    if let Some(stats) = fetch_meta_stats(&url).await {
        println!("📊 Initial meta-learning state:");
        display_meta_stats(&stats);
    }

    let (ws_stream, _) = connect_async(&url).await?;
    let (mut write, mut read) = ws_stream.split();

    println!("Connected! Starting training...\n");

    let mut total_success = 0;
    let mut total_tests = 0;
    let mut total_explorations = 0u32;
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

        // Exploration phase between epochs
        if epoch < epochs && explore_secs > 0 {
            let explorations = exploration_phase(&mut write, &mut read, explore_secs, timeout_ms).await?;
            total_explorations += explorations;
        }

        let epoch_rate = if total_tests > 0 {
            (total_success as f32 / total_tests as f32) * 100.0
        } else {
            0.0
        };

        println!("📊 After epoch {}: {:.1}% success ({}/{}), {} explorations\n",
            epoch, epoch_rate, total_success, total_tests, total_explorations);
    }

    // Final exploration phase
    if explore_secs > 0 {
        let explorations = exploration_phase(&mut write, &mut read, explore_secs, timeout_ms).await?;
        total_explorations += explorations;
    }

    let final_rate = if total_tests > 0 {
        (total_success as f32 / total_tests as f32) * 100.0
    } else {
        0.0
    };

    println!("═══════════════════════════════════════════════════════════");
    println!("🎓 Training Complete!");
    println!("   Conversations: {}/{} ({:.1}%)", total_success, total_tests, final_rate);
    println!("   Explorations:  {}", total_explorations);
    println!("═══════════════════════════════════════════════════════════\n");

    // Fetch final meta stats
    if let Some(stats) = fetch_meta_stats(&url).await {
        println!("📊 Final meta-learning state:");
        display_meta_stats(&stats);
    }

    Ok(())
}
