//! Visual TUI for ARIA - Neural Substrate Visualization
//!
//! A thermal scanner for artificial intelligence. When managing millions of cells,
//! raw data becomes unreadable - this visualization acts as a diagnostic window
//! into ARIA's mind.
//!
//! Features:
//! - Heatmap: 2D projection of 16D semantic space with thermal gradient
//! - Health/Entropy graphs: Real-time sparklines of vital metrics
//! - Sparse View: Focus on active cells (sleep% = GPU savings)
//! - Elite Lineage: Genetic strength indicator (evolution pressure)
//! - Tension Map: Physical desire to act (replaces semantic encoding)

use chrono::Local;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline},
    Frame,
};

// ============================================================================
// THERMAL GRADIENT COLORS
// ============================================================================

/// Thermal color palette: black → blue → cyan → green → yellow → red → white
fn thermal_color(value: f32) -> Color {
    let v = value.clamp(0.0, 1.0);
    if v < 0.05 {
        Color::Rgb(20, 20, 30) // Near-black (sleeping)
    } else if v < 0.15 {
        Color::Rgb(30, 30, 80) // Deep blue
    } else if v < 0.25 {
        Color::Rgb(40, 60, 140) // Blue
    } else if v < 0.35 {
        Color::Rgb(50, 100, 180) // Light blue
    } else if v < 0.45 {
        Color::Rgb(60, 160, 160) // Cyan
    } else if v < 0.55 {
        Color::Rgb(80, 180, 100) // Teal-green
    } else if v < 0.65 {
        Color::Rgb(120, 200, 80) // Green
    } else if v < 0.75 {
        Color::Rgb(200, 200, 60) // Yellow
    } else if v < 0.85 {
        Color::Rgb(255, 160, 40) // Orange
    } else if v < 0.95 {
        Color::Rgb(255, 80, 40) // Red
    } else {
        Color::Rgb(255, 220, 220) // White-hot
    }
}

/// Character density for heatmap cells
fn thermal_char(value: f32) -> char {
    let v = value.clamp(0.0, 1.0);
    if v < 0.05 {
        ' '
    } else if v < 0.15 {
        '·'
    } else if v < 0.30 {
        '░'
    } else if v < 0.50 {
        '▒'
    } else if v < 0.70 {
        '▓'
    } else {
        '█'
    }
}

// ============================================================================
// DATA TYPES
// ============================================================================

/// A message in the conversation with timestamp and context
#[derive(Clone)]
pub struct ChatMessage {
    pub timestamp: String,
    pub is_user: bool,
    pub text: String,
    pub context: Option<String>,
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

/// Substrate view from /substrate endpoint (enhanced)
#[derive(Default, Clone)]
pub struct SubstrateView {
    pub grid_size: usize,
    pub activity_grid: Vec<f32>,
    pub energy_grid: Vec<f32>,
    pub tension_grid: Vec<f32>,
    pub cell_count_grid: Vec<usize>,
    pub total_cells: usize,
    pub alive_cells: usize,
    pub sleeping_cells: usize,
    pub dead_cells: usize,
    pub awake_cells: usize,
    pub energy_histogram: Vec<usize>,
    pub activity_entropy: f32,
    pub system_health: f32,
    // Advanced metrics
    pub max_generation: u32,
    pub avg_generation: f32,
    pub elite_count: usize,
    pub sparse_savings_percent: f32,
    pub avg_energy: f32,
    pub avg_tension: f32,
    pub total_tension: f32,
    pub tps: f32,
}

/// Learning progress stats
#[derive(Default, Clone)]
pub struct LearningStats {
    pub word_count: usize,
    pub association_count: usize,
    pub episode_count: usize,
    pub recent_words: Vec<String>,
    pub strategy: String,
}

// ============================================================================
// VISUALIZER STATE
// ============================================================================

/// Main visualizer state
pub struct AriaVisualizer {
    // History for sparklines (60 samples = ~30 seconds at 500ms refresh)
    pub health_history: Vec<u64>,
    pub entropy_history: Vec<u64>,
    pub tension_history: Vec<u64>,
    pub awake_history: Vec<u64>,

    // Current state
    pub current_stats: BrainStats,
    pub substrate_view: SubstrateView,
    pub learning_stats: LearningStats,

    // Conversation
    pub messages: Vec<ChatMessage>,
    last_aria_message: std::time::Instant,
    last_aria_text: String,

    // View mode: 0=Activity, 1=Tension, 2=Energy
    pub view_mode: u8,

    // Connection
    pub connected: bool,
    pub last_update: std::time::Instant,
}

impl AriaVisualizer {
    pub fn new() -> Self {
        Self {
            health_history: vec![70; 60],
            entropy_history: vec![50; 60],
            tension_history: vec![0; 60],
            awake_history: vec![0; 60],
            current_stats: BrainStats::default(),
            substrate_view: SubstrateView::default(),
            learning_stats: LearningStats::default(),
            messages: Vec::new(),
            last_aria_message: std::time::Instant::now(),
            last_aria_text: String::new(),
            view_mode: 0,
            connected: true,
            last_update: std::time::Instant::now(),
        }
    }

    pub fn update_substrate(&mut self, view: SubstrateView) {
        // Update histories
        self.health_history.push((view.system_health * 100.0) as u64);
        self.health_history.remove(0);

        self.entropy_history.push((view.activity_entropy * 100.0) as u64);
        self.entropy_history.remove(0);

        self.tension_history.push((view.avg_tension * 100.0).min(100.0) as u64);
        self.tension_history.remove(0);

        let awake_pct = if view.alive_cells > 0 {
            view.awake_cells as f32 / view.alive_cells as f32 * 100.0
        } else { 0.0 };
        self.awake_history.push(awake_pct as u64);
        self.awake_history.remove(0);

        self.substrate_view = view;
        self.last_update = std::time::Instant::now();
    }

    pub fn update(&mut self, stats: BrainStats) {
        self.current_stats = stats;
    }

    pub fn update_learning(&mut self, stats: LearningStats) {
        self.learning_stats = stats;
    }

    pub fn add_expression(&mut self, expr: String, context: Option<String>) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_aria_message);

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

    pub fn cycle_view(&mut self) {
        self.view_mode = (self.view_mode + 1) % 3;
    }

    // ========================================================================
    // DRAWING
    // ========================================================================

    pub fn draw(&self, frame: &mut Frame, input_buffer: &str) {
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(0)
            .constraints([
                Constraint::Length(1),  // Status bar
                Constraint::Min(10),    // Main content
                Constraint::Length(3),  // Input
            ])
            .split(frame.size());

        self.draw_status_bar(frame, main_chunks[0]);
        self.draw_main_content(frame, main_chunks[1]);
        self.draw_input(frame, main_chunks[2], input_buffer);
    }

    fn draw_status_bar(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;
        let stats = &self.current_stats;

        // Connection indicator
        let conn = if self.connected { "●" } else { "○" };
        let conn_color = if self.connected { Color::Green } else { Color::Red };

        // Health color
        let health = view.system_health;
        let health_color = if health > 0.7 { Color::Green }
            else if health > 0.4 { Color::Yellow }
            else { Color::Red };

        // Entropy description
        let entropy = view.activity_entropy;
        let entropy_desc = if entropy < 0.3 { "ordered" }
            else if entropy < 0.6 { "balanced" }
            else { "chaotic" };

        // Sparse savings (GPU efficiency)
        let sparse = view.sparse_savings_percent;
        let sparse_color = if sparse > 90.0 { Color::Green }
            else if sparse > 50.0 { Color::Yellow }
            else { Color::Red };

        // View mode indicator
        let mode_name = match self.view_mode {
            0 => "ACTIVITY",
            1 => "TENSION",
            _ => "ENERGY",
        };

        let status = Line::from(vec![
            Span::styled(format!(" {} ", conn), Style::default().fg(conn_color)),
            Span::styled("ARIA", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(format!("HP:{:.0}%", health * 100.0), Style::default().fg(health_color)),
            Span::raw("  "),
            Span::styled(format!("E:{:.2}", entropy), Style::default().fg(Color::Magenta)),
            Span::styled(format!("({})", entropy_desc), Style::default().fg(Color::DarkGray)),
            Span::raw("  "),
            Span::styled(format!("GPU:{:.0}%", sparse), Style::default().fg(sparse_color)),
            Span::raw("  "),
            Span::styled(format!("T:{}", stats.tick), Style::default().fg(Color::DarkGray)),
            Span::raw("  "),
            Span::styled(format!("[{}]", mode_name), Style::default().fg(Color::Yellow)),
            Span::styled(" Tab=cycle", Style::default().fg(Color::DarkGray)),
        ]);

        frame.render_widget(Paragraph::new(status), area);
    }

    fn draw_main_content(&self, frame: &mut Frame, area: Rect) {
        // Split: Heatmap (left 60%) | Sidebar (right 40%)
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(60),
                Constraint::Percentage(40),
            ])
            .split(area);

        self.draw_heatmap(frame, chunks[0]);
        self.draw_sidebar(frame, chunks[1]);
    }

    fn draw_heatmap(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;
        let grid_size = if view.grid_size > 0 { view.grid_size } else { 16 };

        // Select grid based on view mode
        let (grid, title) = match self.view_mode {
            0 => (&view.activity_grid, "Neural Activity"),
            1 => (&view.tension_grid, "Tension Field"),
            _ => (&view.energy_grid, "Energy Distribution"),
        };

        let inner_height = area.height.saturating_sub(2) as usize;
        let inner_width = area.width.saturating_sub(2) as usize;

        // Calculate scaling to fit grid in available space
        let scale_y = (grid_size + inner_height.max(1) - 1) / inner_height.max(1);
        let scale_x = (grid_size + inner_width.max(1) - 1) / inner_width.max(1);

        let mut lines: Vec<Line> = Vec::new();
        let rows = (grid_size / scale_y.max(1)).min(inner_height);

        for row in 0..rows {
            let mut spans: Vec<Span> = Vec::new();
            let cols = (grid_size / scale_x.max(1)).min(inner_width);

            for col in 0..cols {
                // Average values in this scaled cell
                let mut sum = 0.0f32;
                let mut count = 0;

                for dy in 0..scale_y {
                    for dx in 0..scale_x {
                        let gy = row * scale_y + dy;
                        let gx = col * scale_x + dx;
                        if gy < grid_size && gx < grid_size {
                            let idx = gy * grid_size + gx;
                            if idx < grid.len() {
                                sum += grid[idx];
                                count += 1;
                            }
                        }
                    }
                }

                let value = if count > 0 { sum / count as f32 } else { 0.0 };
                let ch = thermal_char(value);
                let color = thermal_color(value);

                spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }

        let heatmap = Paragraph::new(lines)
            .block(Block::default()
                .title(Span::styled(
                    format!(" {} ", title),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(heatmap, area);
    }

    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),  // Stats
                Constraint::Length(5),  // Sparklines
                Constraint::Length(5),  // Lineage
                Constraint::Min(5),     // Conversation
            ])
            .split(area);

        self.draw_cell_stats(frame, chunks[0]);
        self.draw_sparklines(frame, chunks[1]);
        self.draw_lineage(frame, chunks[2]);
        self.draw_conversation(frame, chunks[3]);
    }

    fn draw_cell_stats(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;
        let _stats = &self.current_stats;

        let alive = view.alive_cells;
        let awake = view.awake_cells;
        let sleeping = view.sleeping_cells;

        let awake_pct = if alive > 0 { awake as f32 / alive as f32 * 100.0 } else { 0.0 };

        // Energy indicator
        let (energy_icon, energy_color) = if view.avg_energy > 0.6 {
            ("⚡", Color::Green)
        } else if view.avg_energy > 0.3 {
            ("⚡", Color::Yellow)
        } else {
            ("⚠", Color::Red)
        };

        // Tension indicator (physical intelligence)
        let tension_bar_len = (view.avg_tension * 8.0).min(8.0) as usize;
        let tension_bar = "█".repeat(tension_bar_len);
        let tension_empty = "░".repeat(8 - tension_bar_len);
        let tension_color = if view.avg_tension > 0.5 { Color::Red }
            else if view.avg_tension > 0.2 { Color::Yellow }
            else { Color::Green };

        let content = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(format!(" {} ", energy_icon), Style::default().fg(energy_color)),
                Span::styled(format!("{:.2}", view.avg_energy), Style::default().fg(energy_color).add_modifier(Modifier::BOLD)),
                Span::styled(" avg energy", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::raw("   "),
                Span::styled(&tension_bar, Style::default().fg(tension_color)),
                Span::styled(&tension_empty, Style::default().fg(Color::DarkGray)),
                Span::styled(format!(" {:.2}", view.avg_tension), Style::default().fg(tension_color)),
                Span::styled(" tension", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::styled(" ● ", Style::default().fg(Color::Green)),
                Span::styled(format!("{:>5}", awake), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" awake ({:.0}%)", awake_pct), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled(" ◐ ", Style::default().fg(Color::Blue)),
                Span::styled(format!("{:>5}", sleeping), Style::default().fg(Color::Blue)),
                Span::styled(" sleeping", Style::default().fg(Color::DarkGray)),
            ]),
        ])
        .block(Block::default()
            .title(Span::styled(" Cells ", Style::default().fg(Color::White)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(content, area);
    }

    fn draw_sparklines(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Health sparkline
        let health_spark = Sparkline::default()
            .block(Block::default()
                .title(Span::styled(" HP ", Style::default().fg(Color::Green)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)))
            .data(&self.health_history)
            .max(100)
            .style(Style::default().fg(Color::Green));

        // Entropy sparkline
        let entropy_spark = Sparkline::default()
            .block(Block::default()
                .title(Span::styled(" Entropy ", Style::default().fg(Color::Magenta)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)))
            .data(&self.entropy_history)
            .max(100)
            .style(Style::default().fg(Color::Magenta));

        frame.render_widget(health_spark, chunks[0]);
        frame.render_widget(entropy_spark, chunks[1]);
    }

    fn draw_lineage(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;
        let learn = &self.learning_stats;

        // Elite bar visualization
        let elite_ratio = if view.alive_cells > 0 {
            view.elite_count as f32 / view.alive_cells as f32
        } else { 0.0 };
        let elite_bar_len = (elite_ratio * 10.0).min(10.0) as usize;
        let elite_bar = "█".repeat(elite_bar_len);
        let elite_empty = "░".repeat(10 - elite_bar_len);

        let content = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(" 🧬 Gen:", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{}", view.max_generation), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" (avg:{:.1})", view.avg_generation), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled(" 👑 ", Style::default()),
                Span::styled(&elite_bar, Style::default().fg(Color::Yellow)),
                Span::styled(&elite_empty, Style::default().fg(Color::DarkGray)),
                Span::styled(format!(" {} elite", view.elite_count), Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled(" 📚 ", Style::default()),
                Span::styled(format!("{}", learn.word_count), Style::default().fg(Color::Cyan)),
                Span::styled(" words ", Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{}", learn.association_count), Style::default().fg(Color::Green)),
                Span::styled(" links", Style::default().fg(Color::DarkGray)),
            ]),
        ])
        .block(Block::default()
            .title(Span::styled(" Lineage ", Style::default().fg(Color::Yellow)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)));

        frame.render_widget(content, area);
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
                let (icon, text_color) = if msg.is_user {
                    ("▶", Color::Green)
                } else {
                    ("◀", Color::Cyan)
                };

                let mut spans = vec![
                    Span::styled(format!("{} ", icon), Style::default().fg(text_color)),
                    Span::styled(&msg.text, Style::default().fg(Color::White)),
                ];

                if let Some(ctx) = &msg.context {
                    spans.push(Span::styled(
                        format!(" [{}]", ctx),
                        Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC)
                    ));
                }

                ListItem::new(Line::from(spans))
            })
            .collect();

        let conversation = List::new(items)
            .block(Block::default()
                .title(Span::styled(" Chat ", Style::default().fg(Color::White)))
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
                Span::raw("=Good "),
                Span::styled("n", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
                Span::raw("=Bad "),
                Span::styled("Tab", Style::default().fg(Color::Yellow)),
                Span::raw("=View "),
                Span::styled("Esc", Style::default().fg(Color::DarkGray)),
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
