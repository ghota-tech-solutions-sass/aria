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

            // Rebuild semantic clusters before saving
            {
                let mut mem = memory_save.write();
                mem.rebuild_clusters();
            }

            let mem = memory_save.read();
            if let Err(e) = mem.save(std::path::Path::new("data/aria.memory")) {
                warn!("Failed to save memory: {}", e);
            } else {
                let familiar_words: Vec<_> = mem.get_familiar_words(0.5)
                    .iter()
                    .map(|(w, f)| format!("{}({})", w, f.count))
                    .collect();
                info!("Memory saved ({} memories, {} patterns, {} words known, {} clusters)",
                    mem.memories.len(),
                    mem.learned_patterns.len(),
                    mem.word_frequencies.len(),
                    mem.semantic_clusters.len()
                );
                if !familiar_words.is_empty() {
                    info!("Familiar words: {}", familiar_words.join(", "));
                }
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

    // Clusters endpoint - show semantic word clusters
    let memory_clusters = memory.clone();
    let clusters = warp::path("clusters")
        .map(move || {
            let mem = memory_clusters.read();
            let clusters_info: Vec<serde_json::Value> = mem.semantic_clusters
                .iter()
                .map(|cluster| {
                    let words: Vec<serde_json::Value> = cluster.words.iter()
                        .map(|(word, strength)| {
                            serde_json::json!({
                                "word": word,
                                "strength": strength
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "id": cluster.id,
                        "label": cluster.label,
                        "words": words,
                        "word_count": cluster.words.len(),
                        "emotional_valence": cluster.emotional_valence,
                        "dominant_category": format!("{:?}", cluster.dominant_category)
                    })
                })
                .collect();
            warp::reply::json(&serde_json::json!({
                "total_clusters": mem.semantic_clusters.len(),
                "clusters": clusters_info
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

            // Optional labels to associate with this image (for teaching)
            let labels: Vec<String> = body.get("labels")
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

                    // If labels provided, link them to this visual
                    if !labels.is_empty() {
                        let mut mem = memory_vision.write();
                        for label in &labels {
                            mem.link_vision_to_word(&signature, label, current_tick);
                        }
                    }

                    // Ask memory what words this image evokes
                    let suggested_words: Vec<serde_json::Value> = {
                        let mem = memory_vision.read();
                        mem.visual_to_words(&signature)
                            .into_iter()
                            .map(|(word, score)| serde_json::json!({
                                "word": word,
                                "confidence": score
                            }))
                            .collect()
                    };

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
                            "recognition": recognition,
                            "labels_learned": labels
                        },
                        "suggested_words": suggested_words,
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
            let (total_memories, total_links, top_words) = mem.visual_stats();

            // Get recent memories
            let recent_memories: Vec<serde_json::Value> = mem.visual_memories
                .iter()
                .rev()
                .take(10)
                .map(|m| serde_json::json!({
                    "id": m.id,
                    "description": m.description,
                    "labels": m.labels,
                    "times_seen": m.times_seen,
                    "source": m.source
                }))
                .collect();

            let top_links: Vec<serde_json::Value> = top_words.iter()
                .map(|(word, count)| serde_json::json!({
                    "word": word,
                    "associations": count
                }))
                .collect();

            warp::reply::json(&serde_json::json!({
                "total_visual_memories": total_memories,
                "total_word_links": total_links,
                "recent_memories": recent_memories,
                "top_word_links": top_links
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

    let routes = ws_route.or(health).or(stats).or(words).or(associations).or(episodes).or(clusters).or(meta).or(vision_endpoint).or(visual_stats_endpoint).or(self_endpoint).or(substrate_endpoint);

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
    println!();
    println!("🧒 ARIA is waiting for her first interaction...");
    println!();

    warp::serve(routes).run(([0, 0, 0, 0], config.port)).await;
}

/// Main evolution loop - ARIA's heartbeat
async fn evolution_loop(
    substrate: Arc<parking_lot::RwLock<Substrate>>,
    mut perception: broadcast::Receiver<Signal>,
    expression: broadcast::Sender<Signal>,
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
) {
    let mut tick: u64 = 0;
    let mut last_stats_tick: u64 = 0;

    loop {
        // 1. Receive incoming perceptions (non-blocking)
        while let Ok(signal) = perception.try_recv() {
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

        // 2. One tick of life
        let emergent_signals = {
            let mut sub = substrate.write();
            sub.tick()
        };

        // 3. Send emergent expressions
        for signal in emergent_signals {
            info!("EMERGENCE! intensity: {:.3}, label: {}", signal.intensity, signal.label);
            if signal.intensity > 0.01 {
                let _ = expression.send(signal);
            }
        }

        // 4. Periodic stats (every 500 ticks)
        if tick - last_stats_tick >= 500 {
            let stats = substrate.read().stats();
            info!(
                "Tick {}: {} cells ({} sleeping, {:.1}% saved), energy: {:.2}, mood: {}",
                tick, stats.alive_cells, stats.sleeping_cells, stats.cpu_savings_percent,
                stats.total_energy, stats.mood
            );

            // Update global stats in memory
            {
                let mut mem = memory.write();
                mem.stats.total_ticks = tick;
            }

            last_stats_tick = tick;
        }

        tick += 1;

        // Fast mode: yield to allow other tasks
        tokio::task::yield_now().await;
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
