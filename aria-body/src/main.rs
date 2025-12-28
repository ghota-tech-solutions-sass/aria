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
    println!("  She doesn't understand words yet, but she feels your intent.");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Commands:");
    println!("    /quit or /exit  - Exit");
    println!("    /stats          - Show brain statistics");
    println!("    /visual         - Switch to visual mode");
    println!();

    // Channel for expressions from ARIA
    let (expr_tx, mut expr_rx) = mpsc::channel::<String>(100);

    // Task to receive expressions
    let recv_task = tokio::spawn(async move {
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

    // Task to display expressions - throttled to avoid spam
    let display_task = tokio::spawn(async move {
        let mut last_print = std::time::Instant::now();
        let throttle_duration = std::time::Duration::from_millis(500);

        while let Some(expression) = expr_rx.recv().await {
            // Throttle: only print if enough time has passed
            let now = std::time::Instant::now();
            if now.duration_since(last_print) >= throttle_duration {
                println!("\n  \x1B[36mARIA:\x1B[0m {}", expression);
                io::stdout().flush().ok();
                last_print = now;
            }
            // Ignore expressions that come too fast
        }
    });

    // Main input loop
    loop {
        print!("  \x1B[32mYou:\x1B[0m ");
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;

        // EOF reached (e.g., piped input exhausted)
        if bytes_read == 0 {
            break;
        }

        let input = input.trim();

        if input == "/quit" || input == "/exit" {
            break;
        }

        if input == "/stats" {
            let msg = Message::Text(r#"{"type":"stats"}"#.to_string());
            write.send(msg).await?;
            continue;
        }

        if input == "/visual" {
            println!("Switching to visual mode...");
            // Would need to refactor to support mode switching
            continue;
        }

        if input.is_empty() {
            continue;
        }

        // Convert text to signal and send
        let signal = Signal::from_text(input);
        let json = serde_json::to_string(&signal)?;
        write.send(Message::Text(json)).await?;
    }

    println!();
    println!("Goodbye. ARIA continues to live...");
    println!();

    recv_task.abort();
    display_task.abort();

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

    // Receive task
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

    loop {
        // Handle incoming messages
        while let Ok(signal) = msg_rx.try_recv() {
            let expression = signal.to_expression();
            visualizer.add_expression(expression);
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

        // Request stats periodically
        static mut LAST_STATS: u64 = 0;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        unsafe {
            if now - LAST_STATS >= 1 {
                let msg = Message::Text(r#"{"type":"stats"}"#.to_string());
                let _ = write.send(msg).await;
                LAST_STATS = now;
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, Show)?;

    println!("Goodbye. ARIA continues to live...");

    Ok(())
}
