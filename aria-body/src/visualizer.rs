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

pub struct AriaVisualizer {
    // History for graphs
    pub energy_history: Vec<u64>,
    pub population_history: Vec<u64>,
    pub entropy_history: Vec<u64>,
    pub activity_history: Vec<u64>,

    // Current state
    pub current_stats: BrainStats,
    pub recent_expressions: Vec<String>,
    pub recent_inputs: Vec<String>,

    // Connection state
    pub connected: bool,
    pub last_update: std::time::Instant,
}

#[derive(Default, Clone)]
pub struct BrainStats {
    pub tick: u64,
    pub alive_cells: usize,
    pub total_energy: f32,
    pub entropy: f32,
    pub active_clusters: usize,
    pub dominant_emotion: String,
    pub signals_per_second: f32,
}

impl AriaVisualizer {
    pub fn new() -> Self {
        Self {
            energy_history: vec![50; 100],
            population_history: vec![100; 100],
            entropy_history: vec![50; 100],
            activity_history: vec![0; 100],
            current_stats: BrainStats::default(),
            recent_expressions: Vec::new(),
            recent_inputs: Vec::new(),
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

        self.activity_history.push(stats.signals_per_second as u64);
        self.activity_history.remove(0);
    }

    pub fn add_expression(&mut self, expr: String) {
        self.recent_expressions.insert(0, expr);
        if self.recent_expressions.len() > 20 {
            self.recent_expressions.pop();
        }
    }

    pub fn add_input(&mut self, input: String) {
        self.recent_inputs.insert(0, input);
        if self.recent_inputs.len() > 20 {
            self.recent_inputs.pop();
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
            .split(frame.area());

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

        // Activity sparkline
        let activity = Sparkline::default()
            .block(Block::default().title(" Activity ").borders(Borders::ALL))
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

        // Right: Conversation
        let mut items: Vec<ListItem> = Vec::new();

        // Interleave expressions and inputs
        let max_items = (chunks[1].height as usize).saturating_sub(2);
        let mut expr_iter = self.recent_expressions.iter().peekable();
        let mut input_iter = self.recent_inputs.iter().peekable();

        for _ in 0..max_items {
            if let Some(expr) = expr_iter.next() {
                items.push(ListItem::new(Line::from(vec![
                    Span::styled("  ARIA: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::raw(expr),
                ])));
            }

            if items.len() >= max_items {
                break;
            }

            if let Some(input) = input_iter.next() {
                items.push(ListItem::new(Line::from(vec![
                    Span::styled("   You: ", Style::default().fg(Color::Green)),
                    Span::styled(input, Style::default().fg(Color::White)),
                ])));
            }

            if expr_iter.peek().is_none() && input_iter.peek().is_none() {
                break;
            }
        }

        // Reverse to show newest at bottom
        items.reverse();

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
            .title(" Type message (ESC to quit) ")
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
