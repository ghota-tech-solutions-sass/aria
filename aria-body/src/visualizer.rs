//! Visual TUI for ARIA - Redesigned for the new neural substrate
//!
//! Shows: Neural activity, Hebbian connections, health/entropy,
//! emotions, learning progress, and conversation with context.

use chrono::Local;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

/// A message in the conversation with timestamp and context
#[derive(Clone)]
pub struct ChatMessage {
    pub timestamp: String,
    pub is_user: bool,
    pub text: String,
    pub context: Option<String>, // e.g., "[memory: first_cat]" or "[+association]"
}

impl ChatMessage {
    pub fn user(text: String) -> Self {
        Self {
            timestamp: Local::now().format("%H:%M").to_string(),
            is_user: true,
            text,
            context: None,
        }
    }

    pub fn aria(text: String, context: Option<String>) -> Self {
        Self {
            timestamp: Local::now().format("%H:%M").to_string(),
            is_user: false,
            text,
            context,
        }
    }
}

/// Main visualizer state
pub struct AriaVisualizer {
    // History for sparklines
    pub energy_history: Vec<u64>,
    pub entropy_history: Vec<u64>,
    pub health_history: Vec<u64>,
    pub activity_history: Vec<u64>,

    // Current state from brain
    pub current_stats: BrainStats,
    pub substrate_view: SubstrateView,
    pub learning_stats: LearningStats,

    // Conversation
    pub messages: Vec<ChatMessage>,
    last_aria_message: std::time::Instant,
    last_aria_text: String,

    // Connection
    pub connected: bool,
    pub last_update: std::time::Instant,
}

/// Stats from /stats endpoint
#[derive(Default, Clone)]
pub struct BrainStats {
    pub tick: u64,
    pub alive_cells: usize,
    pub sleeping_cells: usize,
    pub total_energy: f32,
    pub entropy: f32,
    pub dominant_emotion: String,
    pub mood: String,
    pub happiness: f32,
    pub arousal: f32,
    pub curiosity: f32,
}

/// Substrate view from /substrate endpoint (updated with new fields)
#[derive(Default, Clone)]
pub struct SubstrateView {
    pub grid_size: usize,
    pub activity_grid: Vec<f32>,
    pub energy_grid: Vec<f32>,
    pub cell_count_grid: Vec<usize>,
    pub total_cells: usize,
    pub alive_cells: usize,
    pub sleeping_cells: usize,
    pub dead_cells: usize,
    pub awake_cells: usize,
    pub energy_histogram: Vec<usize>,
    // New fields
    pub activity_entropy: f32,
    pub system_health: f32,
}

/// Learning progress stats (from /words, /associations, /episodes)
#[derive(Default, Clone)]
pub struct LearningStats {
    pub word_count: usize,
    pub association_count: usize,
    pub episode_count: usize,
    pub recent_words: Vec<String>,
    pub strategy: String,
}

impl AriaVisualizer {
    pub fn new() -> Self {
        Self {
            energy_history: vec![50; 60],
            entropy_history: vec![50; 60],
            health_history: vec![70; 60],
            activity_history: vec![0; 60],
            current_stats: BrainStats::default(),
            substrate_view: SubstrateView::default(),
            learning_stats: LearningStats::default(),
            messages: Vec::new(),
            last_aria_message: std::time::Instant::now(),
            last_aria_text: String::new(),
            connected: true,
            last_update: std::time::Instant::now(),
        }
    }

    pub fn update_substrate(&mut self, view: SubstrateView) {
        // Update history
        self.entropy_history.push((view.activity_entropy * 100.0) as u64);
        self.entropy_history.remove(0);

        self.health_history.push((view.system_health * 100.0) as u64);
        self.health_history.remove(0);

        self.substrate_view = view;
    }

    pub fn update(&mut self, stats: BrainStats) {
        self.current_stats = stats.clone();
        self.last_update = std::time::Instant::now();

        self.energy_history.push((stats.total_energy * 10.0) as u64);
        self.energy_history.remove(0);

        let awake_cells = stats.alive_cells.saturating_sub(stats.sleeping_cells);
        self.activity_history.push((awake_cells / 100) as u64);
        self.activity_history.remove(0);
    }

    pub fn update_learning(&mut self, stats: LearningStats) {
        self.learning_stats = stats;
    }

    pub fn add_expression(&mut self, expr: String, context: Option<String>) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_aria_message);

        // Throttle: skip if same message or too fast
        if expr == self.last_aria_text || elapsed.as_millis() < 500 {
            return;
        }

        self.messages.push(ChatMessage::aria(expr.clone(), context));
        self.last_aria_message = now;
        self.last_aria_text = expr;

        if self.messages.len() > 100 {
            self.messages.remove(0);
        }
    }

    pub fn add_input(&mut self, input: String) {
        self.messages.push(ChatMessage::user(input));
        if self.messages.len() > 100 {
            self.messages.remove(0);
        }
    }

    pub fn draw(&self, frame: &mut Frame, input_buffer: &str) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(0)
            .constraints([
                Constraint::Length(3),  // Header with health bar
                Constraint::Length(8),  // Neural activity + stats
                Constraint::Length(3),  // Emotion bar
                Constraint::Min(8),     // Conversation
                Constraint::Length(3),  // Input
            ])
            .split(frame.size());

        self.draw_header(frame, chunks[0]);
        self.draw_dashboard(frame, chunks[1]);
        self.draw_emotions(frame, chunks[2]);
        self.draw_conversation(frame, chunks[3]);
        self.draw_input(frame, chunks[4], input_buffer);
    }

    fn draw_header(&self, frame: &mut Frame, area: Rect) {
        let status = if self.connected { "●" } else { "○" };
        let status_color = if self.connected { Color::Green } else { Color::Red };

        // Health bar color
        let health = self.substrate_view.system_health;
        let health_color = if health > 0.7 {
            Color::Green
        } else if health > 0.4 {
            Color::Yellow
        } else {
            Color::Red
        };

        let health_pct = (health * 100.0) as u16;
        let health_bar = "█".repeat((health * 20.0) as usize);
        let health_empty = "░".repeat(20 - (health * 20.0) as usize);

        let entropy = self.substrate_view.activity_entropy;
        let entropy_desc = if entropy < 0.3 {
            "ordered"
        } else if entropy < 0.6 {
            "balanced"
        } else {
            "chaotic"
        };

        let header = Paragraph::new(Line::from(vec![
            Span::styled(" 🧒 ARIA ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(status, Style::default().fg(status_color)),
            Span::raw("  │  "),
            Span::styled("Health: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&health_bar, Style::default().fg(health_color)),
            Span::styled(&health_empty, Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {}%", health_pct), Style::default().fg(health_color)),
            Span::raw("  │  "),
            Span::styled("Entropy: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.2}", entropy), Style::default().fg(Color::Magenta)),
            Span::styled(format!(" ({})", entropy_desc), Style::default().fg(Color::DarkGray)),
            Span::raw("  │  "),
            Span::styled(format!("Tick: {}", self.current_stats.tick), Style::default().fg(Color::DarkGray)),
        ]))
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(header, area);
    }

    fn draw_dashboard(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(40),  // Neural activity heatmap
                Constraint::Percentage(30),  // Stats
                Constraint::Percentage(30),  // Learning
            ])
            .split(area);

        self.draw_neural_activity(frame, chunks[0]);
        self.draw_cell_stats(frame, chunks[1]);
        self.draw_learning(frame, chunks[2]);
    }

    fn draw_neural_activity(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;
        let grid_size = if view.grid_size > 0 { view.grid_size } else { 16 };
        let inner_height = area.height.saturating_sub(2) as usize;
        let inner_width = area.width.saturating_sub(2) as usize;

        let scale_y = (grid_size + inner_height - 1) / inner_height.max(1);
        let scale_x = (grid_size + inner_width - 1) / inner_width.max(1);

        let mut lines: Vec<Line> = Vec::new();
        for row in 0..(grid_size / scale_y.max(1)).min(inner_height) {
            let mut spans: Vec<Span> = Vec::new();
            for col in 0..(grid_size / scale_x.max(1)).min(inner_width) {
                let mut activity_sum = 0.0f32;
                let mut count = 0;
                for dy in 0..scale_y {
                    for dx in 0..scale_x {
                        let gy = row * scale_y + dy;
                        let gx = col * scale_x + dx;
                        if gy < grid_size && gx < grid_size {
                            let idx = gy * grid_size + gx;
                            if idx < view.activity_grid.len() {
                                activity_sum += view.activity_grid[idx];
                                count += 1;
                            }
                        }
                    }
                }
                let activity = if count > 0 { activity_sum / count as f32 } else { 0.0 };

                let (ch, color) = if activity < 0.05 {
                    ('·', Color::DarkGray)
                } else if activity < 0.2 {
                    ('░', Color::Blue)
                } else if activity < 0.4 {
                    ('▒', Color::Cyan)
                } else if activity < 0.6 {
                    ('▓', Color::Yellow)
                } else {
                    ('█', Color::Red)
                };

                spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }

        let heatmap = Paragraph::new(lines)
            .block(Block::default()
                .title(Span::styled(" 🧠 Neural Activity ", Style::default().fg(Color::Cyan)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)));
        frame.render_widget(heatmap, area);
    }

    fn draw_cell_stats(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;
        let stats = &self.current_stats;

        let alive = view.alive_cells;
        let awake = view.awake_cells;
        let sleeping = view.sleeping_cells;

        let awake_pct = if alive > 0 { (awake as f32 / alive as f32 * 100.0) as u16 } else { 0 };
        let energy_per_cell = if alive > 0 { stats.total_energy / alive as f32 } else { 0.0 };

        let (energy_icon, energy_color) = if energy_per_cell > 0.6 {
            ("⚡", Color::Green)
        } else if energy_per_cell > 0.3 {
            ("⚡", Color::Yellow)
        } else {
            ("⚠️", Color::Red)
        };

        let content = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(format!(" {} ", energy_icon), Style::default().fg(energy_color)),
                Span::styled(format!("{:.1}", stats.total_energy), Style::default().fg(energy_color).add_modifier(Modifier::BOLD)),
                Span::styled(" total energy", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(format!("{:.3}", energy_per_cell), Style::default().fg(Color::White)),
                Span::styled(" /cell", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(" 🟢 ", Style::default()),
                Span::styled(format!("{}", awake), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" awake ({}%)", awake_pct), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled(" 💤 ", Style::default()),
                Span::styled(format!("{}", sleeping), Style::default().fg(Color::Blue)),
                Span::styled(" sleeping", Style::default().fg(Color::DarkGray)),
            ]),
        ])
        .block(Block::default()
            .title(Span::styled(" Cells ", Style::default().fg(Color::White)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(content, area);
    }

    fn draw_learning(&self, frame: &mut Frame, area: Rect) {
        let learn = &self.learning_stats;

        let content = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(" 📚 ", Style::default()),
                Span::styled(format!("{}", learn.word_count), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled(" words", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled(" 🔗 ", Style::default()),
                Span::styled(format!("{}", learn.association_count), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(" associations", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled(" 💾 ", Style::default()),
                Span::styled(format!("{}", learn.episode_count), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                Span::styled(" memories", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(" Strategy: ", Style::default().fg(Color::DarkGray)),
                Span::styled(&learn.strategy, Style::default().fg(Color::Green)),
            ]),
        ])
        .block(Block::default()
            .title(Span::styled(" Learning ", Style::default().fg(Color::Yellow)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(content, area);
    }

    fn draw_emotions(&self, frame: &mut Frame, area: Rect) {
        let stats = &self.current_stats;

        // Emotion icons and colors
        let emotions = [
            ("😊", "Joy", stats.happiness, Color::Yellow),
            ("🔍", "Curious", stats.curiosity, Color::Cyan),
            ("⚡", "Arousal", stats.arousal, Color::Red),
        ];

        let mut spans: Vec<Span> = vec![Span::raw(" ")];

        for (icon, name, value, color) in emotions {
            let bar_len = (value * 8.0) as usize;
            let bar = "█".repeat(bar_len);
            let empty = "░".repeat(8 - bar_len);

            spans.push(Span::styled(format!(" {} ", icon), Style::default()));
            spans.push(Span::styled(name, Style::default().fg(Color::DarkGray)));
            spans.push(Span::styled(": ", Style::default().fg(Color::DarkGray)));
            spans.push(Span::styled(bar, Style::default().fg(color)));
            spans.push(Span::styled(empty, Style::default().fg(Color::DarkGray)));
            spans.push(Span::raw("  │"));
        }

        // Dominant emotion
        let emotion_color = match stats.dominant_emotion.as_str() {
            "curious" => Color::Cyan,
            "content" | "happy" => Color::Green,
            "frustrated" => Color::Red,
            "excited" => Color::Yellow,
            "calm" => Color::Blue,
            "bored" => Color::DarkGray,
            _ => Color::White,
        };

        spans.push(Span::styled("  Mood: ", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(stats.mood.clone(), Style::default().fg(emotion_color).add_modifier(Modifier::BOLD)));

        let emotions_line = Paragraph::new(Line::from(spans))
            .block(Block::default().borders(Borders::NONE));

        frame.render_widget(emotions_line, area);
    }

    fn draw_conversation(&self, frame: &mut Frame, area: Rect) {
        let max_items = (area.height as usize).saturating_sub(2);
        let start_idx = if self.messages.len() > max_items {
            self.messages.len() - max_items
        } else {
            0
        };

        let items: Vec<ListItem> = self.messages[start_idx..]
            .iter()
            .map(|msg| {
                let (icon, _name_color, text_color) = if msg.is_user {
                    ("👤", Color::Green, Color::White)
                } else {
                    ("🧒", Color::Cyan, Color::White)
                };

                let mut spans = vec![
                    Span::styled(format!(" {} ", msg.timestamp), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{} ", icon), Style::default()),
                    Span::styled(&msg.text, Style::default().fg(text_color)),
                ];

                if let Some(ctx) = &msg.context {
                    spans.push(Span::styled(format!("  {}", ctx), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)));
                }

                ListItem::new(Line::from(spans))
            })
            .collect();

        let conversation = List::new(items)
            .block(Block::default()
                .title(Span::styled(" Conversation ", Style::default().fg(Color::White)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(conversation, area);
    }

    fn draw_input(&self, frame: &mut Frame, area: Rect, input_buffer: &str) {
        let input = Paragraph::new(Line::from(vec![
            Span::styled(" > ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(input_buffer),
            Span::styled("█", Style::default().fg(Color::White)),
        ]))
        .block(Block::default()
            .title(Line::from(vec![
                Span::raw(" "),
                Span::styled("y", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::raw("=Bravo "),
                Span::styled("n", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw("=Non "),
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
