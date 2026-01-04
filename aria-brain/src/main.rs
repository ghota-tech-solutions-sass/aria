//! ARIA Brain - The Living Substrate
//!
//! This is where ARIA's cells live, evolve, and think.
//! Supports GPU acceleration with automatic CPU fallback.

mod substrate;
mod signal;
mod memory;
mod config;
mod meta_learning;
mod vision;
mod web_learner;
mod expression;

use web_learner::WebLearner;
use expression::ExpressionGenerator;

use std::sync::Arc;
use tokio::sync::broadcast;
use warp::Filter;
use futures::{StreamExt, SinkExt};
use tracing::{info, warn, Level};

use substrate::Substrate;
use signal::Signal;
use memory::LongTermMemory;
use config::{Config, print_banner};

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
    println!("║    Brain v{}                                           ║", VERSION);
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    print_banner(&config);

    // Load or create long-term memory
    let memory_path = std::path::Path::new("data/aria.memory");
    std::fs::create_dir_all("data").ok();
    let memory = Arc::new(parking_lot::RwLock::new(
        LongTermMemory::load_or_create(memory_path)
    ));

    // Initialize web learner for autonomous learning
    let web_learner = Arc::new(tokio::sync::RwLock::new(WebLearner::new()));
    info!("📚 Web Learner initialized with 2 sources");

    // Initialize expression generator for emergent speech
    let expression_gen = Arc::new(parking_lot::RwLock::new(ExpressionGenerator::new()));
    info!("💬 Expression Generator initialized with {} expressions", expression_gen.read().stats().total_expressions);

    // Create the substrate with aria-core config
    let aria_config = aria_core::AriaConfig::from_env();
    info!("🧠 Substrate: {} cells ({:?} backend, sparse={})",
        aria_config.population.target_population,
        aria_config.compute.backend,
        aria_config.compute.sparse_updates);

    let substrate = Arc::new(parking_lot::RwLock::new(
        Substrate::new(aria_config, memory.clone())
    ));

    // Channels for perception (input) and expression (output)
    let (perception_tx, _) = broadcast::channel::<Signal>(1000);
    let (expression_tx, _) = broadcast::channel::<Signal>(1000);

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
            expression_gen_evolution
        ).await;
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

            // NOTE: Vocabulary removed in Session 31 (Physical Intelligence)
            let mem = memory_save.read();
            if let Err(e) = mem.save(std::path::Path::new("data/aria.memory")) {
                warn!("Failed to save memory: {}", e);
            } else {
                info!("Memory saved ({} memories, {} patterns, {} episodes, {} visual)",
                    mem.memories.len(),
                    mem.learned_patterns.len(),
                    mem.episodes.len(),
                    mem.visual_memories.len()
                );
            }
        }
    });

    // WebSocket server for communication with the Body
    let perception_tx_ws = perception_tx.clone();
    let expression_tx_ws = expression_tx.clone();

    let ws_route = warp::path("aria")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let perception_tx = perception_tx_ws.clone();
            let expression_rx = expression_tx_ws.subscribe();

            ws.on_upgrade(move |socket| {
                handle_connection(socket, perception_tx, expression_rx)
            })
        });

    // Health check endpoint
    let health = warp::path("health")
        .map(|| warp::reply::json(&serde_json::json!({"status": "alive"})));

    // Stats endpoint
    let substrate_stats = substrate.clone();
    let stats = warp::path("stats")
        .map(move || {
            let s = substrate_stats.read().stats();
            warp::reply::json(&s)
        });

    // Words endpoint - NOTE: Vocabulary removed in Session 31 (Physical Intelligence)
    let words = warp::path("words")
        .map(move || {
            warp::reply::json(&serde_json::json!({
                "message": "Vocabulary removed in Session 31 (Physical Intelligence)",
                "total_words": 0,
                "words": []
            }))
        });

    // Associations endpoint - NOTE: Vocabulary removed in Session 31 (Physical Intelligence)
    let associations = warp::path("associations")
        .map(move || {
            warp::reply::json(&serde_json::json!({
                "message": "Word associations removed in Session 31 (Physical Intelligence)",
                "total_associations": 0,
                "associations": []
            }))
        });

    // Episodes endpoint - show episodic memories
    let memory_episodes = memory.clone();
    let episodes = warp::path("episodes")
        .map(move || {
            let mem = memory_episodes.read();
            let episodes_info: Vec<serde_json::Value> = mem.episodes
                .iter()
                .rev()  // Most recent first
                .take(50)  // Last 50 episodes
                .map(|ep| {
                    serde_json::json!({
                        "id": ep.id,
                        "timestamp": ep.timestamp,
                        "real_time": ep.real_time,
                        "input": ep.input,
                        "response": ep.response,
                        "keywords": ep.keywords,
                        "category": format!("{:?}", ep.category),
                        "importance": ep.importance,
                        "recall_count": ep.recall_count,
                        "first_of_kind": ep.first_of_kind,
                        "emotion": {
                            "happiness": ep.emotion.happiness,
                            "arousal": ep.emotion.arousal,
                            "comfort": ep.emotion.comfort,
                            "curiosity": ep.emotion.curiosity
                        }
                    })
                })
                .collect();

            let first_times: Vec<serde_json::Value> = mem.first_times
                .iter()
                .map(|(kind, id)| {
                    serde_json::json!({
                        "kind": kind,
                        "episode_id": id
                    })
                })
                .collect();

            warp::reply::json(&serde_json::json!({
                "total_episodes": mem.episodes.len(),
                "showing": episodes_info.len(),
                "first_times": first_times,
                "episodes": episodes_info
            }))
        });

    // Clusters endpoint - NOTE: Semantic clusters removed in Session 31 (Physical Intelligence)
    let clusters = warp::path("clusters")
        .map(move || {
            warp::reply::json(&serde_json::json!({
                "message": "Semantic clusters removed in Session 31 (Physical Intelligence)",
                "total_clusters": 0,
                "clusters": []
            }))
        });

    // Meta-learning endpoint - show ARIA's self-learning progress (Session 14)
    let memory_meta = memory.clone();
    let meta = warp::path("meta")
        .map(move || {
            let mem = memory_meta.read();
            let ml = &mem.meta_learner;

            // Strategy performance
            let strategies: Vec<serde_json::Value> = ml.strategies.iter()
                .map(|(st, strategy)| {
                    serde_json::json!({
                        "type": st.name(),
                        "usage_count": strategy.usage_count,
                        "avg_reward": strategy.avg_reward,
                        "best_reward": strategy.best_reward,
                        "recent_avg": strategy.recent_avg()
                    })
                })
                .collect();

            // Goals
            let goals: Vec<serde_json::Value> = ml.goals.iter()
                .map(|goal| {
                    serde_json::json!({
                        "id": goal.id,
                        "description": goal.description,
                        "progress": goal.progress,
                        "completed": goal.completed
                    })
                })
                .collect();

            // Progress tracker
            let progress = &ml.progress;

            warp::reply::json(&serde_json::json!({
                "total_evaluations": ml.total_evaluations,
                "current_strategy": ml.current_strategy.name(),
                "exploration_rate": ml.config.exploration_rate,
                "strategies": strategies,
                "goals": goals,
                "progress": {
                    "learning_quality": progress.learning_quality,
                    "competence_level": progress.competence_level,
                    "trend": format!("{:?}", progress.trend),
                    "recent_successes": progress.recent_successes,
                    "recent_failures": progress.recent_failures,
                    "status": progress.status_description()
                }
            }))
        });

    // Visual perception endpoint - POST /vision with base64 image
    // Accepts: { "image": "<base64>", "source": "name", "labels": ["chat", "moka"] }
    let perception_tx_vision = perception_tx.clone();
    let memory_vision = memory.clone();
    let substrate_vision = substrate.clone();
    let vision_endpoint = warp::path("vision")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| {
            // Extract base64 image from body
            let b64 = body.get("image")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let source = body.get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("upload")
                .to_string();

            // NOTE: Labels removed in Session 31 (Physical Intelligence)
            let _labels: Vec<String> = body.get("labels")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect())
                .unwrap_or_default();

            if b64.is_empty() {
                return warp::reply::json(&serde_json::json!({
                    "error": "Missing 'image' field with base64 data"
                }));
            }

            // Process the image
            let vision_processor = vision::VisualPerception::new();
            match vision_processor.process_base64(b64) {
                Ok(features) => {
                    let visual_signal = vision::VisualSignal::new(features.clone(), source.clone());

                    // Get current tick from substrate
                    let current_tick = substrate_vision.read().current_tick();

                    // Convert 32-element vector to fixed array
                    let mut signature = [0.0f32; 32];
                    for (i, v) in visual_signal.vector.iter().take(32).enumerate() {
                        signature[i] = *v;
                    }

                    // Store in visual memory and check for recognition
                    let (memory_id, is_new, recognition) = {
                        let mut mem = memory_vision.write();
                        mem.see(
                            signature,
                            visual_signal.description.clone(),
                            source,
                            current_tick,
                            features.emotional_valence
                        )
                    };

                    // NOTE: Visual-word linking removed in Session 31 (Physical Intelligence)
                    // Labels are ignored - ARIA learns through physical intelligence, not words

                    // Convert to internal signal and send to substrate
                    let signal = Signal {
                        content: visual_signal.vector.to_vec(),
                        intensity: visual_signal.intensity,
                        label: format!("vision:{}", visual_signal.description),
                        signal_type: signal::SignalType::Visual,
                        timestamp: current_tick,
                    };

                    info!("👁️ VISION: {} (intensity: {:.2}, memory_id: {}, new: {})",
                        visual_signal.description, visual_signal.intensity, memory_id, is_new);

                    if let Err(e) = perception_tx_vision.send(signal) {
                        warn!("Failed to send visual signal: {}", e);
                    }

                    warp::reply::json(&serde_json::json!({
                        "success": true,
                        "description": visual_signal.description,
                        "intensity": visual_signal.intensity,
                        "memory": {
                            "id": memory_id,
                            "is_new": is_new,
                            "recognition": recognition
                        },
                        "features": {
                            "brightness": features.brightness,
                            "warmth": features.warmth,
                            "saturation": features.saturation,
                            "complexity": features.edge_density,
                            "emotional_valence": features.emotional_valence,
                            "nature_score": features.nature_score,
                            "face_likelihood": features.face_likelihood
                        }
                    }))
                }
                Err(e) => {
                    warp::reply::json(&serde_json::json!({
                        "error": format!("Failed to process image: {}", e)
                    }))
                }
            }
        });

    // Visual memory stats endpoint - GET /visual
    let memory_visual_stats = memory.clone();
    let visual_stats_endpoint = warp::path("visual")
        .map(move || {
            let mem = memory_visual_stats.read();
            let total_memories = mem.visual_stats();

            // Get recent memories
            let recent_memories: Vec<serde_json::Value> = mem.visual_memories
                .iter()
                .rev()
                .take(10)
                .map(|m| serde_json::json!({
                    "id": m.id,
                    "description": m.description,
                    "times_seen": m.times_seen,
                    "source": m.source
                }))
                .collect();

            // NOTE: word_links removed in Session 31 (Physical Intelligence)
            warp::reply::json(&serde_json::json!({
                "total_visual_memories": total_memories,
                "recent_memories": recent_memories
            }))
        });

    // Self-modification endpoint - GET /self
    let memory_self = memory.clone();
    let substrate_self = substrate.clone();
    let self_endpoint = warp::path("self")
        .map(move || {
            let mem = memory_self.read();
            let params = {
                let sub = substrate_self.read();
                let p = sub.stats();
                serde_json::json!({
                    "emission_threshold": p.adaptive_emission_threshold,
                    "response_probability": p.adaptive_response_probability,
                    "spontaneity": p.adaptive_spontaneity,
                    "feedback_positive": p.adaptive_feedback_positive,
                    "feedback_negative": p.adaptive_feedback_negative
                })
            };

            let modifier = &mem.self_modifier;
            let recent_mods: Vec<serde_json::Value> = modifier.modification_history
                .iter()
                .rev()
                .take(10)
                .map(|m| serde_json::json!({
                    "param": m.param.name(),
                    "from": m.current_value,
                    "to": m.new_value,
                    "reasoning": m.reasoning,
                    "confidence": m.confidence,
                    "tick": m.proposed_at,
                    "evaluated": m.evaluated,
                    "successful": m.was_successful
                }))
                .collect();

            warp::reply::json(&serde_json::json!({
                "self_modification": {
                    "enabled": true,
                    "total_modifications": modifier.total_modifications,
                    "successful_modifications": modifier.successful_modifications,
                    "success_rate": if modifier.total_modifications > 0 {
                        modifier.successful_modifications as f32 / modifier.total_modifications as f32
                    } else { 0.0 },
                    "last_check_tick": modifier.last_modification_tick,
                    "check_interval": modifier.modification_interval
                },
                "current_params": params,
                "recent_modifications": recent_mods,
                "meta_learning": {
                    "competence_level": mem.meta_learner.progress.competence_level,
                    "learning_quality": mem.meta_learner.progress.learning_quality,
                    "trend": format!("{:?}", mem.meta_learner.progress.trend),
                    "total_evaluations": mem.meta_learner.total_evaluations
                }
            }))
        });

    // Substrate spatial view endpoint - GET /substrate
    // Returns spatial activity data for TUI visualization
    let substrate_view = substrate.clone();
    let substrate_endpoint = warp::path("substrate")
        .map(move || {
            let view = substrate_view.read().spatial_view();
            warp::reply::json(&view)
        });

    // Web learner stats endpoint - GET /learn
    let web_learner_stats = web_learner.clone();
    let learn_endpoint = warp::path("learn")
        .and_then(move || {
            let wl = web_learner_stats.clone();
            async move {
                let stats = wl.read().await.stats();
                Ok::<_, warp::Rejection>(warp::reply::json(&stats))
            }
        });

    // Expression generator stats endpoint - GET /express
    let expression_gen_stats = expression_gen.clone();
    let express_endpoint = warp::path("express")
        .map(move || {
            let stats = expression_gen_stats.read().stats();
            warp::reply::json(&stats)
        });

    let routes = ws_route.or(health).or(stats).or(words).or(associations).or(episodes).or(clusters).or(meta).or(vision_endpoint).or(visual_stats_endpoint).or(self_endpoint).or(substrate_endpoint).or(learn_endpoint).or(express_endpoint);

    info!("WebSocket ready on ws://0.0.0.0:{}/aria", config.port);
    info!("Health check on http://0.0.0.0:{}/health", config.port);
    info!("Stats on http://0.0.0.0:{}/stats", config.port);
    info!("Words on http://0.0.0.0:{}/words", config.port);
    info!("Associations on http://0.0.0.0:{}/associations", config.port);
    info!("Episodes on http://0.0.0.0:{}/episodes", config.port);
    info!("Clusters on http://0.0.0.0:{}/clusters", config.port);
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

            info!("EMERGENCE! intensity: {:.3}, label: {}", signal.intensity, signal.label);
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
                tick, stats.alive_cells, stats.sleeping_cells, stats.cpu_savings_percent,
                stats.total_energy, stats.mood, wl_stats.total_learned
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
        // This prevents cooldowns from being bypassed by high TPS
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
    let mut learn_interval = tokio::time::interval(tokio::time::Duration::from_secs(300)); // Every 5 minutes

    loop {
        learn_interval.tick().await;

        let sources_count = web_learner.read().await.sources.len();
        if sources_count == 0 {
            continue;
        }

        // Fetch from current source
        let current_tick = {
            let wl = web_learner.read().await;
            wl.learning_tick
        };

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

async fn handle_connection(
    ws: warp::ws::WebSocket,
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
