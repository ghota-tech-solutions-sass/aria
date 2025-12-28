//! ARIA Brain - The Living Substrate
//!
//! This is where ARIA's cells live, evolve, and think.
//! The brain runs on the GPU-enabled machine for maximum parallel processing.

mod cell;
mod substrate;
mod signal;
mod memory;
mod connection;

use std::sync::Arc;
use tokio::sync::broadcast;
use warp::Filter;
use futures::{StreamExt, SinkExt};
use tracing::{info, warn, Level};

use substrate::Substrate;
use signal::Signal;
use memory::LongTermMemory;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const DEFAULT_PORT: u16 = 8765;
const INITIAL_CELLS: usize = 10_000;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                                                           ║");
    println!("║     █████╗ ██████╗ ██╗ █████╗                             ║");
    println!("║    ██╔══██╗██╔══██╗██║██╔══██╗                            ║");
    println!("║    ███████║██████╔╝██║███████║                            ║");
    println!("║    ██╔══██║██╔══██╗██║██╔══██║                            ║");
    println!("║    ██║  ██║██║  ██║██║██║  ██║                            ║");
    println!("║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝                            ║");
    println!("║                                                           ║");
    println!("║    Autonomous Recursive Intelligence Architecture        ║");
    println!("║    Brain v{}                                           ║", VERSION);
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Load or create long-term memory
    let memory_path = std::path::Path::new("data/aria.memory");
    std::fs::create_dir_all("data").ok();
    let memory = Arc::new(parking_lot::RwLock::new(
        LongTermMemory::load_or_create(memory_path)
    ));

    // Create the substrate with initial cells
    let substrate = Arc::new(Substrate::new(INITIAL_CELLS, memory.clone()));
    info!("Substrate created with {} primordial cells", INITIAL_CELLS);

    // Channels for perception (input) and expression (output)
    let (perception_tx, _) = broadcast::channel::<Signal>(1000);
    let (expression_tx, _) = broadcast::channel::<Signal>(1000);

    // Start the evolution loop in the background
    let substrate_evolution = substrate.clone();
    let perception_rx = perception_tx.subscribe();
    let expression_tx_evolution = expression_tx.clone();
    let memory_evolution = memory.clone();

    tokio::spawn(async move {
        evolution_loop(
            substrate_evolution,
            perception_rx,
            expression_tx_evolution,
            memory_evolution
        ).await;
    });

    // Auto-save memory periodically
    let memory_save = memory.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            let mem = memory_save.read();
            if let Err(e) = mem.save(std::path::Path::new("data/aria.memory")) {
                warn!("Failed to save memory: {}", e);
            } else {
                let familiar_words: Vec<_> = mem.get_familiar_words(0.5)
                    .iter()
                    .map(|(w, f)| format!("{}({})", w, f.count))
                    .collect();
                info!("Memory saved ({} memories, {} patterns, {} words known)",
                    mem.memories.len(),
                    mem.learned_patterns.len(),
                    mem.word_frequencies.len()
                );
                if !familiar_words.is_empty() {
                    info!("Familiar words: {}", familiar_words.join(", "));
                }
            }
        }
    });

    // WebSocket server for communication with the Body
    let substrate_ws = substrate.clone();
    let perception_tx_ws = perception_tx.clone();
    let expression_tx_ws = expression_tx.clone();

    let ws_route = warp::path("aria")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let substrate = substrate_ws.clone();
            let perception_tx = perception_tx_ws.clone();
            let expression_rx = expression_tx_ws.subscribe();

            ws.on_upgrade(move |socket| {
                handle_connection(socket, substrate, perception_tx, expression_rx)
            })
        });

    // Health check endpoint
    let health = warp::path("health")
        .map(|| warp::reply::json(&serde_json::json!({"status": "alive"})));

    // Stats endpoint
    let substrate_stats = substrate.clone();
    let stats = warp::path("stats")
        .map(move || {
            let s = substrate_stats.stats();
            warp::reply::json(&s)
        });

    // Words endpoint - show known words
    let memory_words = memory.clone();
    let words = warp::path("words")
        .map(move || {
            let mem = memory_words.read();
            let words_info: Vec<serde_json::Value> = mem.word_frequencies
                .iter()
                .map(|(word, freq)| {
                    serde_json::json!({
                        "word": word,
                        "count": freq.count,
                        "familiarity": freq.familiarity_boost,
                        "emotional_valence": freq.emotional_valence
                    })
                })
                .collect();
            warp::reply::json(&serde_json::json!({
                "total_words": mem.word_frequencies.len(),
                "words": words_info
            }))
        });

    // Associations endpoint - show semantic word associations
    let memory_assoc = memory.clone();
    let associations = warp::path("associations")
        .map(move || {
            let mem = memory_assoc.read();
            let assoc_info: Vec<serde_json::Value> = mem.word_associations
                .iter()
                .filter(|(_, a)| a.strength >= 0.4)  // Only show meaningful associations
                .map(|(key, assoc)| {
                    let parts: Vec<&str> = key.split(':').collect();
                    serde_json::json!({
                        "word1": parts.get(0).unwrap_or(&""),
                        "word2": parts.get(1).unwrap_or(&""),
                        "strength": assoc.strength,
                        "co_occurrences": assoc.co_occurrences,
                        "emotional_valence": assoc.emotional_valence
                    })
                })
                .collect();
            warp::reply::json(&serde_json::json!({
                "total_associations": mem.word_associations.len(),
                "strong_associations": assoc_info.len(),
                "associations": assoc_info
            }))
        });

    let routes = ws_route.or(health).or(stats).or(words).or(associations);

    let port = std::env::var("ARIA_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    info!("WebSocket ready on ws://0.0.0.0:{}/aria", port);
    info!("Health check on http://0.0.0.0:{}/health", port);
    info!("Stats on http://0.0.0.0:{}/stats", port);
    info!("Words on http://0.0.0.0:{}/words", port);
    info!("Associations on http://0.0.0.0:{}/associations", port);
    println!();
    println!("🧒 ARIA is waiting for her first interaction...");
    println!();

    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}

async fn evolution_loop(
    substrate: Arc<Substrate>,
    mut perception: broadcast::Receiver<Signal>,
    expression: broadcast::Sender<Signal>,
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
) {
    let mut tick: u64 = 0;
    let mut last_stats_tick: u64 = 0;

    loop {
        // 1. Receive incoming perceptions (non-blocking)
        // Immediate emergence check for each signal
        while let Ok(signal) = perception.try_recv() {
            let immediate_emergence = substrate.inject_signal(signal);
            for em_signal in immediate_emergence {
                info!("IMMEDIATE EMERGENCE! intensity: {:.3}", em_signal.intensity);
                if em_signal.intensity > 0.01 {
                    let _ = expression.send(em_signal);
                }
            }
        }

        // 2. One tick of life
        let emergent_signals = substrate.tick();

        // 3. Send emergent expressions (lowered threshold for baby ARIA)
        for signal in emergent_signals {
            info!("EMERGENCE! intensity: {:.3}, label: {}", signal.intensity, signal.label);
            // Send all emergent signals - even weak ones
            if signal.intensity > 0.01 {
                let _ = expression.send(signal);
            }
        }

        // 4. Periodic stats (every 500 ticks = ~5 seconds)
        if tick - last_stats_tick >= 500 {
            let stats = substrate.stats();
            info!(
                "Tick {}: {} cells, energy: {:.2}, entropy: {:.4}, emotion: {}",
                tick, stats.alive_cells, stats.total_energy, stats.entropy, stats.dominant_emotion
            );

            // Update global stats in memory
            {
                let mut mem = memory.write();
                mem.stats.total_ticks = tick;
            }

            last_stats_tick = tick;
        }

        tick += 1;

        // ~100 ticks per second
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

async fn handle_connection(
    ws: warp::ws::WebSocket,
    _substrate: Arc<Substrate>,
    perception_tx: broadcast::Sender<Signal>,
    mut expression_rx: broadcast::Receiver<Signal>,
) {
    let (mut ws_tx, mut ws_rx) = ws.split();

    info!("New connection established");

    // Task to forward expressions to the client
    let send_task = tokio::spawn(async move {
        while let Ok(signal) = expression_rx.recv().await {
            let json = serde_json::to_string(&signal).unwrap_or_default();
            if ws_tx.send(warp::ws::Message::text(json)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    while let Some(result) = ws_rx.next().await {
        match result {
            Ok(msg) => {
                if let Ok(text) = msg.to_str() {
                    // Try to parse as a signal
                    if let Ok(signal) = serde_json::from_str::<Signal>(text) {
                        let _ = perception_tx.send(signal);
                    } else if text.contains("\"type\":\"stats\"") {
                        // Stats request - handled by HTTP endpoint now
                    }
                }
            }
            Err(e) => {
                warn!("WebSocket error: {}", e);
                break;
            }
        }
    }

    send_task.abort();
    info!("Connection closed");
}
