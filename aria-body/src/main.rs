//! ARIA Body - The Interface
//!
//! This is how humans interact with ARIA.
//! It connects to the Brain and provides a terminal UI.

mod signal;
mod visualizer;

use std::io::{self, Write};
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{StreamExt, SinkExt};
use crossterm::{
    execute,
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    cursor::{Show, Hide},
    event::{self, Event, KeyCode, KeyEventKind},
};
use ratatui::{backend::CrosstermBackend, Terminal};

use signal::Signal;
use visualizer::AriaVisualizer;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Extract context information from signal labels for conversation display
fn extract_context(label: &str) -> Option<String> {
    // Parse label format: "type:detail" or just "type"
    if label.is_empty() {
        return None;
    }

    // Known context types
    let context = match label {
        // Memory-related
        l if l.starts_with("memory:") => {
            let detail = l.strip_prefix("memory:").unwrap();
            format!("memory: {}", detail)
        }
        l if l.starts_with("episodic:") => {
            let detail = l.strip_prefix("episodic:").unwrap();
            format!("memory: {}", detail)
        }

        // Visual recognition
        l if l.starts_with("vision:") => {
            let detail = l.strip_prefix("vision:").unwrap();
            format!("sees: {}", detail)
        }
        l if l.starts_with("visual:") => {
            let detail = l.strip_prefix("visual:").unwrap();
            format!("sees: {}", detail)
        }

        // Emotional states
        l if l.starts_with("emotion:") => {
            let detail = l.strip_prefix("emotion:").unwrap();
            format!("feels: {}", detail)
        }
        l if l.starts_with("mood:") => {
            let detail = l.strip_prefix("mood:").unwrap();
            format!("mood: {}", detail)
        }

        // Learning/exploration
        l if l.starts_with("explore:") => {
            let detail = l.strip_prefix("explore:").unwrap();
            format!("exploring: {}", detail)
        }
        l if l.starts_with("learn:") => {
            let detail = l.strip_prefix("learn:").unwrap();
            format!("learned: {}", detail)
        }

        // Spontaneous speech
        "spontaneous" => "spontaneous".to_string(),
        "dream" => "dreaming".to_string(),
        "bored" => "bored".to_string(),
        "curious" => "curious".to_string(),

        // Default: use label as-is if it's short
        l if l.len() <= 20 => l.to_string(),

        // Truncate long labels
        l => format!("{}...", &l[..17]),
    };

    Some(context)
}

#[derive(Clone)]
#[allow(dead_code)]
enum UiMode {
    Simple,      // Just text chat
    Visual,      // Full TUI with stats
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let visual_mode = args.iter().any(|a| a == "--visual" || a == "-v");

    if visual_mode {
        run_visual_mode().await
    } else {
        run_simple_mode().await
    }
}

async fn run_simple_mode() -> Result<(), Box<dyn std::error::Error>> {
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
    println!("║    Body v{} - Simple Mode                              ║", VERSION);
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Get brain URL
    let brain_url = std::env::var("ARIA_BRAIN_URL")
        .unwrap_or_else(|_| {
            println!("Enter the Brain URL (default: ws://localhost:8765/aria):");
            print!("> ");
            io::stdout().flush().ok();
            let mut input = String::new();
            io::stdin().read_line(&mut input).ok();
            let input = input.trim();
            if input.is_empty() {
                "ws://localhost:8765/aria".to_string()
            } else {
                input.to_string()
            }
        });

    println!();
    println!("Connecting to Brain at {}...", brain_url);

    let (ws_stream, _) = match connect_async(&brain_url).await {
        Ok(stream) => stream,
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            eprintln!();
            eprintln!("Make sure aria-brain is running on the target machine.");
            eprintln!("Set ARIA_BRAIN_URL environment variable to the correct address.");
            return Err(e.into());
        }
    };

    let (mut write, mut read) = ws_stream.split();

    println!("✓ Connected to ARIA's Brain!");
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Talk to ARIA. She's young, be patient.");
    println!();
    println!("  \x1B[33mShortcuts:\x1B[0m");
    println!("    \x1B[32my\x1B[0m = Good! (positive feedback)");
    println!("    \x1B[31mn\x1B[0m = No! (negative feedback)");
    println!("    \x1B[90mESC\x1B[0m = Quit");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Channel for expressions from ARIA
    let (expr_tx, mut expr_rx) = mpsc::channel::<String>(100);

    // Task to receive expressions
    tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    if let Ok(signal) = serde_json::from_str::<Signal>(&text) {
                        let expression = signal.to_expression();
                        if expr_tx.send(expression).await.is_err() {
                            break;
                        }
                    }
                }
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    eprintln!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    });

    // Enable raw mode for non-blocking input
    enable_raw_mode()?;

    let mut input_buffer = String::new();
    let mut last_aria_time = std::time::Instant::now();
    let throttle_duration = std::time::Duration::from_millis(500);

    // Print initial prompt
    print!("  \x1B[32mYou:\x1B[0m ");
    io::stdout().flush()?;

    loop {
        // Check for ARIA messages (non-blocking)
        while let Ok(expression) = expr_rx.try_recv() {
            let now = std::time::Instant::now();
            if now.duration_since(last_aria_time) >= throttle_duration {
                // Clear current line, print ARIA message, re-print user input
                print!("\r\x1B[K");  // Clear line
                println!("  \x1B[36mARIA:\x1B[0m {}", expression);
                print!("  \x1B[32mYou:\x1B[0m {}", input_buffer);
                io::stdout().flush()?;
                last_aria_time = now;
            }
        }

        // Handle keyboard input (with timeout to check messages)
        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match key.code {
                    KeyCode::Esc => break,
                    KeyCode::Enter => {
                        println!();  // Move to next line

                        let input = input_buffer.trim().to_string();
                        input_buffer.clear();

                        if input == "/quit" || input == "/exit" {
                            break;
                        }

                        if !input.is_empty() {
                            // Convert text to signal and send
                            let signal = Signal::from_text(&input);
                            if let Ok(json) = serde_json::to_string(&signal) {
                                let _ = write.send(Message::Text(json)).await;
                            }
                        }

                        // Print new prompt
                        print!("  \x1B[32mYou:\x1B[0m ");
                        io::stdout().flush()?;
                    }
                    KeyCode::Backspace => {
                        if input_buffer.pop().is_some() {
                            // Erase character on screen
                            print!("\x08 \x08");
                            io::stdout().flush()?;
                        }
                    }
                    KeyCode::Char(c) => {
                        // Quick feedback shortcuts (only when buffer is empty)
                        if input_buffer.is_empty() {
                            if c == 'y' || c == 'Y' {
                                // Positive feedback
                                print!("\r\x1B[K");
                                println!("  \x1B[32m✓ Good!\x1B[0m");
                                print!("  \x1B[32mYou:\x1B[0m ");
                                io::stdout().flush()?;
                                let signal = Signal::from_text("Bravo!");
                                if let Ok(json) = serde_json::to_string(&signal) {
                                    let _ = write.send(Message::Text(json)).await;
                                }
                                continue;
                            } else if c == 'n' || c == 'N' {
                                // Negative feedback
                                print!("\r\x1B[K");
                                println!("  \x1B[31m✗ No!\x1B[0m");
                                print!("  \x1B[32mYou:\x1B[0m ");
                                io::stdout().flush()?;
                                let signal = Signal::from_text("Non");
                                if let Ok(json) = serde_json::to_string(&signal) {
                                    let _ = write.send(Message::Text(json)).await;
                                }
                                continue;
                            }
                        }
                        input_buffer.push(c);
                        print!("{}", c);
                        io::stdout().flush()?;
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;

    println!();
    println!();
    println!("Goodbye. ARIA continues to live...");
    println!();

    Ok(())
}

async fn run_visual_mode() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Hide)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Get brain URL
    let brain_url = std::env::var("ARIA_BRAIN_URL")
        .unwrap_or_else(|_| "ws://localhost:8765/aria".to_string());

    // Derive HTTP URL from WebSocket URL
    let http_url = brain_url
        .replace("ws://", "http://")
        .replace("wss://", "https://")
        .replace("/aria", "");

    // Try to connect
    let connection = match connect_async(&brain_url).await {
        Ok((stream, _)) => Some(stream),
        Err(e) => {
            // Show error and exit
            disable_raw_mode()?;
            execute!(io::stdout(), LeaveAlternateScreen, Show)?;
            eprintln!("Failed to connect to {}: {}", brain_url, e);
            return Err(e.into());
        }
    };

    let (mut write, mut read) = connection.unwrap().split();

    let mut visualizer = AriaVisualizer::new();
    let mut input_buffer = String::new();

    // Channel for incoming messages
    let (msg_tx, mut msg_rx) = mpsc::channel::<Signal>(100);

    // Channel for stats
    let (stats_tx, mut stats_rx) = mpsc::channel::<visualizer::BrainStats>(10);

    // Channel for substrate view
    let (substrate_tx, mut substrate_rx) = mpsc::channel::<visualizer::SubstrateView>(10);

    // Channel for learning stats
    let (learning_tx, mut learning_rx) = mpsc::channel::<visualizer::LearningStats>(10);

    // Receive task for WebSocket messages
    tokio::spawn(async move {
        while let Some(msg) = read.next().await {
            if let Ok(Message::Text(text)) = msg {
                if let Ok(signal) = serde_json::from_str::<Signal>(&text) {
                    if msg_tx.send(signal).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Fetch stats periodically via HTTP
    let stats_url = format!("{}/stats", http_url);
    let substrate_url = format!("{}/substrate", http_url);

    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            if let Ok(resp) = client.get(&stats_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let stats = visualizer::BrainStats {
                        tick: json.get("tick").and_then(|v| v.as_u64()).unwrap_or(0),
                        alive_cells: json.get("alive_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        sleeping_cells: json.get("sleeping_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        total_energy: json.get("total_energy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        entropy: json.get("entropy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        dominant_emotion: json.get("dominant_emotion").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        mood: json.get("mood").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        happiness: json.get("happiness").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        arousal: json.get("arousal").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        curiosity: json.get("curiosity").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    };
                    let _ = stats_tx.send(stats).await;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });

    // Fetch substrate view periodically via HTTP (for heatmap)
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            if let Ok(resp) = client.get(&substrate_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let view = visualizer::SubstrateView {
                        grid_size: json.get("grid_size").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                        activity_grid: json.get("activity_grid")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                            .unwrap_or_default(),
                        energy_grid: json.get("energy_grid")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                            .unwrap_or_default(),
                        tension_grid: json.get("tension_grid")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                            .unwrap_or_default(),
                        cell_count_grid: json.get("cell_count_grid")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                            .unwrap_or_default(),
                        total_cells: json.get("total_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        alive_cells: json.get("alive_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        sleeping_cells: json.get("sleeping_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        dead_cells: json.get("dead_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        awake_cells: json.get("awake_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        energy_histogram: json.get("energy_histogram")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                            .unwrap_or_default(),
                        // Health and entropy
                        activity_entropy: json.get("activity_entropy").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
                        system_health: json.get("system_health").and_then(|v| v.as_f64()).unwrap_or(0.7) as f32,
                        // Advanced metrics (lineage, sparse dispatch, tension)
                        max_generation: json.get("max_generation").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                        avg_generation: json.get("avg_generation").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        elite_count: json.get("elite_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        sparse_savings_percent: json.get("sparse_savings_percent").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        avg_energy: json.get("avg_energy").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32,
                        avg_tension: json.get("avg_tension").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        total_tension: json.get("total_tension").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        tps: json.get("tps").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                    };
                    let _ = substrate_tx.send(view).await;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await; // Faster update for heatmap
        }
    });

    // Fetch learning stats periodically
    let words_url = format!("{}/words", http_url);
    let assoc_url = format!("{}/associations", http_url);
    let episodes_url = format!("{}/episodes", http_url);
    let meta_url = format!("{}/meta", http_url);

    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            let mut stats = visualizer::LearningStats::default();

            // Get word count
            if let Ok(resp) = client.get(&words_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(arr) = json.as_array() {
                        stats.word_count = arr.len();
                        // Get recent words (first 5)
                        stats.recent_words = arr.iter()
                            .take(5)
                            .filter_map(|w| w.get("word").and_then(|v| v.as_str()).map(|s| s.to_string()))
                            .collect();
                    }
                }
            }

            // Get association count
            if let Ok(resp) = client.get(&assoc_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(arr) = json.as_array() {
                        stats.association_count = arr.len();
                    }
                }
            }

            // Get episode count
            if let Ok(resp) = client.get(&episodes_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(arr) = json.as_array() {
                        stats.episode_count = arr.len();
                    }
                }
            }

            // Get current strategy from meta-learning
            if let Ok(resp) = client.get(&meta_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    stats.strategy = json.get("current_strategy")
                        .and_then(|v| v.as_str())
                        .unwrap_or("exploring")
                        .to_string();
                }
            }

            let _ = learning_tx.send(stats).await;
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    });

    loop {
        // Handle incoming messages
        while let Ok(signal) = msg_rx.try_recv() {
            let expression = signal.to_expression();
            // Extract context from signal label for conversation display
            let context = extract_context(&signal.label);
            visualizer.add_expression(expression, context);
        }

        // Handle stats updates
        while let Ok(stats) = stats_rx.try_recv() {
            visualizer.update(stats);
        }

        // Handle substrate updates (heatmap)
        while let Ok(view) = substrate_rx.try_recv() {
            visualizer.update_substrate(view);
        }

        // Handle learning stats updates
        while let Ok(learn) = learning_rx.try_recv() {
            visualizer.update_learning(learn);
        }

        // Draw UI
        terminal.draw(|f| {
            visualizer.draw(f, &input_buffer);
        })?;

        // Handle input (with timeout)
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match key.code {
                    KeyCode::Esc => break,
                    KeyCode::Tab => {
                        // Cycle between heatmap views: Activity → Tension → Energy
                        visualizer.cycle_view();
                    }
                    KeyCode::Enter => {
                        if !input_buffer.is_empty() {
                            let text = input_buffer.clone();
                            visualizer.add_input(text.clone());

                            let signal = Signal::from_text(&text);
                            let json = serde_json::to_string(&signal)?;
                            write.send(Message::Text(json)).await?;

                            input_buffer.clear();
                        }
                    }
                    KeyCode::Backspace => {
                        input_buffer.pop();
                    }
                    KeyCode::Char(c) => {
                        // Quick feedback shortcuts (only when buffer is empty)
                        if input_buffer.is_empty() {
                            if c == 'y' || c == 'Y' {
                                // Positive feedback
                                visualizer.add_input("✓ Good!".to_string());
                                let signal = Signal::from_text("Bravo!");
                                if let Ok(json) = serde_json::to_string(&signal) {
                                    let _ = write.send(Message::Text(json)).await;
                                }
                                continue;
                            } else if c == 'n' || c == 'N' {
                                // Negative feedback
                                visualizer.add_input("✗ No!".to_string());
                                let signal = Signal::from_text("Non");
                                if let Ok(json) = serde_json::to_string(&signal) {
                                    let _ = write.send(Message::Text(json)).await;
                                }
                                continue;
                            }
                        }
                        input_buffer.push(c);
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, Show)?;

    println!("Goodbye. ARIA continues to live...");

    Ok(())
}
