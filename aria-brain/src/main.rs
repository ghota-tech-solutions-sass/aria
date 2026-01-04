//! ARIA Brain - The Living Substrate
//!
//! This is where ARIA's cells live, evolve, and think.
//! Supports GPU acceleration with automatic CPU fallback.

mod config;
mod expression;
mod handlers;
mod memory;
mod meta_learning;
mod signal;
mod substrate;
mod vision;
mod web_learner;

use expression::ExpressionGenerator;
use web_learner::WebLearner;

use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, warn, Level};

use config::{print_banner, Config};
use handlers::AppState;
use memory::LongTermMemory;
use signal::Signal;
use substrate::Substrate;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Load configuration with GPU detection
    let config = Config::from_env();

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
    println!(
        "║    Brain v{}                                           ║",
        VERSION
    );
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    print_banner(&config);

    // Load or create long-term memory
    let memory_path = std::path::Path::new("data/aria.memory");
    std::fs::create_dir_all("data").ok();
    let memory = Arc::new(parking_lot::RwLock::new(LongTermMemory::load_or_create(
        memory_path,
    )));

    // Initialize web learner for autonomous learning
    let web_learner = Arc::new(tokio::sync::RwLock::new(WebLearner::new()));
    info!("📚 Web Learner initialized with 2 sources");

    // Initialize expression generator for emergent speech
    let expression_gen = Arc::new(parking_lot::RwLock::new(ExpressionGenerator::new()));
    info!(
        "💬 Expression Generator initialized with {} expressions",
        expression_gen.read().stats().total_expressions
    );

    // Create the substrate with aria-core config
    let aria_config = aria_core::AriaConfig::from_env();
    info!(
        "🧠 Substrate: {} cells ({:?} backend, sparse={})",
        aria_config.population.target_population,
        aria_config.compute.backend,
        aria_config.compute.sparse_updates
    );

    let substrate = Arc::new(parking_lot::RwLock::new(Substrate::new(
        aria_config,
        memory.clone(),
    )));

    // Channels for perception (input) and expression (output)
    let (perception_tx, _) = broadcast::channel::<Signal>(1000);
    let (expression_tx, _) = broadcast::channel::<Signal>(1000);

    // Create shared app state
    let state = AppState {
        substrate: substrate.clone(),
        memory: memory.clone(),
        web_learner: web_learner.clone(),
        expression_gen: expression_gen.clone(),
    };

    // Start the evolution loop in the background
    let substrate_evolution = substrate.clone();
    let perception_rx = perception_tx.subscribe();
    let expression_tx_evolution = expression_tx.clone();
    let memory_evolution = memory.clone();
    let web_learner_evolution = web_learner.clone();
    let expression_gen_evolution = expression_gen.clone();

    tokio::spawn(async move {
        evolution_loop(
            substrate_evolution,
            perception_rx,
            expression_tx_evolution,
            memory_evolution,
            web_learner_evolution,
            expression_gen_evolution,
        )
        .await;
    });

    // Autonomous web learning task
    let web_learner_auto = web_learner.clone();
    let perception_tx_auto = perception_tx.clone();
    tokio::spawn(async move {
        autonomous_learning_loop(web_learner_auto, perception_tx_auto).await;
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
                info!(
                    "Memory saved ({} memories, {} patterns, {} episodes, {} visual)",
                    mem.memories.len(),
                    mem.learned_patterns.len(),
                    mem.episodes.len(),
                    mem.visual_memories.len()
                );
            }
        }
    });

    // Compose all routes using handlers module
    let routes = handlers::routes(state, perception_tx, expression_tx);

    info!("WebSocket ready on ws://0.0.0.0:{}/aria", config.port);
    info!("Health check on http://0.0.0.0:{}/health", config.port);
    info!("Stats on http://0.0.0.0:{}/stats", config.port);
    info!("Episodes on http://0.0.0.0:{}/episodes", config.port);
    info!("Meta-learning on http://0.0.0.0:{}/meta", config.port);
    info!("Vision on POST http://0.0.0.0:{}/vision", config.port);
    info!("Visual memory on http://0.0.0.0:{}/visual", config.port);
    info!("Self-modification on http://0.0.0.0:{}/self", config.port);
    info!("Substrate view on http://0.0.0.0:{}/substrate", config.port);
    info!("Web learning on http://0.0.0.0:{}/learn", config.port);
    info!("Expression stats on http://0.0.0.0:{}/express", config.port);
    println!();
    println!("🧒 ARIA is waiting for her first interaction...");
    println!("📚 Autonomous web learning will start in 30 seconds...");
    println!();

    warp::serve(routes).run(([0, 0, 0, 0], config.port)).await;
}

/// Main evolution loop - ARIA's heartbeat
async fn evolution_loop(
    substrate: Arc<parking_lot::RwLock<Substrate>>,
    mut perception: broadcast::Receiver<Signal>,
    expression: broadcast::Sender<Signal>,
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
    web_learner: Arc<tokio::sync::RwLock<WebLearner>>,
    expression_gen: Arc<parking_lot::RwLock<ExpressionGenerator>>,
) {
    let mut tick: u64 = 0;
    let mut last_stats_tick: u64 = 0;

    loop {
        // 1. Receive incoming perceptions (non-blocking)
        while let Ok(signal) = perception.try_recv() {
            // Learn expressions from user input
            if !signal.label.is_empty() && signal.content.len() >= 8 {
                let tension: [f32; 8] = [
                    signal.content.get(0).copied().unwrap_or(0.0),
                    signal.content.get(1).copied().unwrap_or(0.0),
                    signal.content.get(2).copied().unwrap_or(0.0),
                    signal.content.get(3).copied().unwrap_or(0.0),
                    signal.content.get(4).copied().unwrap_or(0.0),
                    signal.content.get(5).copied().unwrap_or(0.0),
                    signal.content.get(6).copied().unwrap_or(0.0),
                    signal.content.get(7).copied().unwrap_or(0.0),
                ];
                let mut expr_gen = expression_gen.write();
                expr_gen.learn_from_user(&signal.label, tension, tick);
            }

            let immediate_emergence = {
                let mut sub = substrate.write();
                sub.inject_signal(signal)
            };
            for em_signal in immediate_emergence {
                info!("IMMEDIATE EMERGENCE! intensity: {:.3}", em_signal.intensity);
                if em_signal.intensity > 0.01 {
                    let _ = expression.send(em_signal);
                }
            }
        }

        // 1.5. Inject pending web learning content
        {
            let mut wl = web_learner.write().await;
            if let Some(injection) = wl.get_next_injection() {
                let signal = Signal {
                    content: injection.tension.to_vec(),
                    intensity: injection.intensity,
                    label: injection.label,
                    signal_type: signal::SignalType::Perception,
                    timestamp: tick,
                };
                let mut sub = substrate.write();
                let _ = sub.inject_signal(signal);
            }
            wl.tick();
        }

        // 2. One tick of life
        let emergent_signals = {
            let mut sub = substrate.write();
            sub.tick()
        };

        // 3. Send emergent expressions with learned text
        for mut signal in emergent_signals {
            // Try to generate an expression from the tension pattern
            if signal.content.len() >= 8 {
                let tension: [f32; 8] = [
                    signal.content.get(0).copied().unwrap_or(0.0),
                    signal.content.get(1).copied().unwrap_or(0.0),
                    signal.content.get(2).copied().unwrap_or(0.0),
                    signal.content.get(3).copied().unwrap_or(0.0),
                    signal.content.get(4).copied().unwrap_or(0.0),
                    signal.content.get(5).copied().unwrap_or(0.0),
                    signal.content.get(6).copied().unwrap_or(0.0),
                    signal.content.get(7).copied().unwrap_or(0.0),
                ];

                let mut expr_gen = expression_gen.write();
                if let Some(text) = expr_gen.express(&tension, signal.intensity, tick) {
                    // Add expressed text to label
                    signal.label = format!("{}|says:{}", signal.label, text);
                }
            }

            info!(
                "EMERGENCE! intensity: {:.3}, label: {}",
                signal.intensity, signal.label
            );
            if signal.intensity > 0.01 {
                let _ = expression.send(signal);
            }
        }

        // 4. Periodic stats (every 500 ticks)
        if tick - last_stats_tick >= 500 {
            let stats = substrate.read().stats();
            let wl_stats = web_learner.read().await.stats();
            info!(
                "Tick {}: {} cells ({} sleeping, {:.1}% saved), energy: {:.2}, mood: {}, learned: {}",
                tick,
                stats.alive_cells,
                stats.sleeping_cells,
                stats.cpu_savings_percent,
                stats.total_energy,
                stats.mood,
                wl_stats.total_learned
            );

            // Update global stats in memory
            {
                let mut mem = memory.write();
                mem.stats.total_ticks = tick;
            }

            last_stats_tick = tick;
        }

        tick += 1;

        // Rate limit to ~1000 TPS (1ms per tick)
        tokio::time::sleep(tokio::time::Duration::from_micros(1000)).await;
    }
}

/// Autonomous web learning loop - fetches and learns from web content
async fn autonomous_learning_loop(
    web_learner: Arc<tokio::sync::RwLock<WebLearner>>,
    _perception_tx: broadcast::Sender<Signal>,
) {
    // Wait a bit before starting to let the system stabilize
    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
    info!("🌐 Autonomous learning starting...");

    let mut source_idx = 0;
    let mut learn_interval = tokio::time::interval(tokio::time::Duration::from_secs(300));

    loop {
        learn_interval.tick().await;

        let sources_count = web_learner.read().await.sources.len();
        if sources_count == 0 {
            continue;
        }

        // Fetch from current source
        let current_tick = { web_learner.read().await.learning_tick };

        info!("🌐 Fetching knowledge from source {}...", source_idx);

        let injections = {
            let mut wl = web_learner.write().await;
            wl.fetch_and_learn(source_idx, current_tick).await
        };

        // Queue injections for gradual injection into substrate
        if let Some(injs) = injections {
            let mut wl = web_learner.write().await;
            wl.queue_injections(injs);
        }

        // Rotate to next source
        source_idx = (source_idx + 1) % sources_count;
    }
}
