//! Visual TUI for ARIA
//!
//! Provides a rich terminal interface to see ARIA's internal state.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline},
    Frame,
};

/// A message in the conversation (either from user or ARIA)
#[derive(Clone)]
pub enum ChatMessage {
    User(String),
    Aria(String),
}

pub struct AriaVisualizer {
    // History for graphs
    pub energy_history: Vec<u64>,
    pub population_history: Vec<u64>,
    pub entropy_history: Vec<u64>,
    pub activity_history: Vec<u64>,

    // Current state
    pub current_stats: BrainStats,

    // Unified conversation history (chronological order, oldest first)
    pub messages: Vec<ChatMessage>,

    // Throttling for ARIA messages
    last_aria_message: std::time::Instant,
    last_aria_text: String,

    // Connection state
    pub connected: bool,
    pub last_update: std::time::Instant,
}

#[derive(Default, Clone)]
pub struct BrainStats {
    pub tick: u64,
    pub alive_cells: usize,
    pub sleeping_cells: usize,
    pub total_energy: f32,
    pub entropy: f32,
    pub dominant_emotion: String,
    // Emotional state
    pub mood: String,
    pub happiness: f32,
    pub arousal: f32,
    pub curiosity: f32,
}

impl AriaVisualizer {
    pub fn new() -> Self {
        Self {
            energy_history: vec![50; 100],
            population_history: vec![100; 100],
            entropy_history: vec![50; 100],
            activity_history: vec![0; 100],
            current_stats: BrainStats::default(),
            messages: Vec::new(),
            last_aria_message: std::time::Instant::now(),
            last_aria_text: String::new(),
            connected: true,
            last_update: std::time::Instant::now(),
        }
    }

    pub fn update(&mut self, stats: BrainStats) {
        self.current_stats = stats.clone();
        self.last_update = std::time::Instant::now();

        self.energy_history.push((stats.total_energy * 10.0) as u64);
        self.energy_history.remove(0);

        self.population_history.push(stats.alive_cells as u64 / 100);
        self.population_history.remove(0);

        self.entropy_history.push((stats.entropy * 100.0) as u64);
        self.entropy_history.remove(0);

        // Activity = awake cells (alive - sleeping)
        let awake_cells = stats.alive_cells.saturating_sub(stats.sleeping_cells);
        self.activity_history.push(awake_cells as u64);
        self.activity_history.remove(0);
    }

    pub fn add_expression(&mut self, expr: String) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_aria_message);

        // Throttle: skip if same message or too fast (< 500ms)
        if expr == self.last_aria_text || elapsed.as_millis() < 500 {
            return;
        }

        self.messages.push(ChatMessage::Aria(expr.clone()));
        self.last_aria_message = now;
        self.last_aria_text = expr;

        // Keep last 50 messages
        if self.messages.len() > 50 {
            self.messages.remove(0);
        }
    }

    pub fn add_input(&mut self, input: String) {
        self.messages.push(ChatMessage::User(input));
        // Keep last 50 messages
        if self.messages.len() > 50 {
            self.messages.remove(0);
        }
    }

    pub fn draw(&self, frame: &mut Frame, input_buffer: &str) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(5),   // Header
                Constraint::Length(5),   // Stats gauges
                Constraint::Min(10),     // Main area
                Constraint::Length(3),   // Input
            ])
            .split(frame.size());

        // Header
        self.draw_header(frame, chunks[0]);

        // Stats
        self.draw_stats(frame, chunks[1]);

        // Main area (graphs + conversation)
        self.draw_main(frame, chunks[2]);

        // Input
        self.draw_input(frame, chunks[3], input_buffer);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let status_color = if self.connected { Color::Green } else { Color::Red };
        let status_text = if self.connected { "● Connected" } else { "○ Disconnected" };

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("╔═══ ", Style::default().fg(Color::Cyan)),
                Span::styled("A R I A", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled(" ═══╗", Style::default().fg(Color::Cyan)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(format!("  Tick: {} ", self.current_stats.tick), Style::default().fg(Color::White)),
                Span::styled("│", Style::default().fg(Color::DarkGray)),
                Span::styled(format!(" Cells: {} ", self.current_stats.alive_cells), Style::default().fg(Color::White)),
                Span::styled("│", Style::default().fg(Color::DarkGray)),
                Span::styled(format!(" {} ", status_text), Style::default().fg(status_color)),
            ]),
        ])
        .block(Block::default().borders(Borders::NONE));

        frame.render_widget(header, area);
    }

    fn draw_stats(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area);

        // Energy gauge
        let energy_pct = (self.current_stats.total_energy / 100.0).clamp(0.0, 1.0);
        let energy_gauge = Gauge::default()
            .block(Block::default().title(" Energy ").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(energy_pct as f64)
            .label(format!("{:.0}%", energy_pct * 100.0));
        frame.render_widget(energy_gauge, chunks[0]);

        // Entropy gauge
        let entropy_pct = (self.current_stats.entropy / 10.0).clamp(0.0, 1.0);
        let entropy_gauge = Gauge::default()
            .block(Block::default().title(" Entropy ").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Magenta))
            .ratio(entropy_pct as f64)
            .label(format!("{:.2}", self.current_stats.entropy));
        frame.render_widget(entropy_gauge, chunks[1]);

        // Awake cells sparkline
        let activity = Sparkline::default()
            .block(Block::default().title(" Awake ").borders(Borders::ALL))
            .data(&self.activity_history)
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(activity, chunks[2]);

        // Emotion
        let emotion_color = match self.current_stats.dominant_emotion.as_str() {
            "curious" => Color::Cyan,
            "content" => Color::Green,
            "frustrated" => Color::Red,
            "excited" => Color::Yellow,
            "calm" => Color::Blue,
            _ => Color::Gray,
        };

        let emotion_text = if self.current_stats.dominant_emotion.is_empty() {
            "..."
        } else {
            &self.current_stats.dominant_emotion
        };

        let emotion = Paragraph::new(emotion_text.to_string())
            .block(Block::default().title(" State ").borders(Borders::ALL))
            .style(Style::default().fg(emotion_color).add_modifier(Modifier::BOLD));
        frame.render_widget(emotion, chunks[3]);
    }

    fn draw_main(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(35),
                Constraint::Percentage(65),
            ])
            .split(area);

        // Left: Population graph
        let pop_sparkline = Sparkline::default()
            .block(Block::default().title(" Population (x100) ").borders(Borders::ALL))
            .data(&self.population_history)
            .style(Style::default().fg(Color::Blue));
        frame.render_widget(pop_sparkline, chunks[0]);

        // Right: Conversation (chronological order, newest at bottom)
        let max_items = (chunks[1].height as usize).saturating_sub(2);

        // Take only the last max_items messages
        let start_idx = if self.messages.len() > max_items {
            self.messages.len() - max_items
        } else {
            0
        };

        let items: Vec<ListItem> = self.messages[start_idx..]
            .iter()
            .map(|msg| {
                match msg {
                    ChatMessage::User(text) => ListItem::new(Line::from(vec![
                        Span::styled("   You: ", Style::default().fg(Color::Green)),
                        Span::styled(text.as_str(), Style::default().fg(Color::White)),
                    ])),
                    ChatMessage::Aria(text) => ListItem::new(Line::from(vec![
                        Span::styled("  ARIA: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                        Span::raw(text.as_str()),
                    ])),
                }
            })
            .collect();

        let conversation = List::new(items)
            .block(Block::default().title(" Conversation ").borders(Borders::ALL));
        frame.render_widget(conversation, chunks[1]);
    }

    fn draw_input(&self, frame: &mut Frame, area: Rect, input_buffer: &str) {
        let input = Paragraph::new(Line::from(vec![
            Span::styled(" > ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(input_buffer),
            Span::styled("█", Style::default().fg(Color::White)), // Cursor
        ]))
        .block(Block::default()
            .title(Line::from(vec![
                Span::raw(" "),
                Span::styled("y", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw("=Good "),
                Span::styled("n", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw("=No "),
                Span::styled("ESC", Style::default().fg(Color::DarkGray)),
                Span::raw("=Quit "),
            ]))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(input, area);
    }
}

impl Default for AriaVisualizer {
    fn default() -> Self {
        Self::new()
    }
}
