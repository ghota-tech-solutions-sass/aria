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

#[derive(Clone)]
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
    println!("  Type your message and press Enter. ESC to quit.");
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
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        loop {
            if let Ok(resp) = client.get(&stats_url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let stats = visualizer::BrainStats {
                        tick: json.get("tick").and_then(|v| v.as_u64()).unwrap_or(0),
                        alive_cells: json.get("alive_cells").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        total_energy: json.get("total_energy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        entropy: json.get("entropy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                        active_clusters: json.get("active_clusters").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                        dominant_emotion: json.get("dominant_emotion").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        signals_per_second: json.get("signals_per_second").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
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

    loop {
        // Handle incoming messages
        while let Ok(signal) = msg_rx.try_recv() {
            let expression = signal.to_expression();
            visualizer.add_expression(expression);
        }

        // Handle stats updates
        while let Ok(stats) = stats_rx.try_recv() {
            visualizer.update(stats);
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
