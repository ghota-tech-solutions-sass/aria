//! Visual TUI for ARIA
//!
//! Provides a rich terminal interface to see ARIA's internal state,
//! including a spatial heatmap of the substrate activity.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Bar, BarChart, BarGroup, Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline},
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

    // Substrate spatial view
    pub substrate_view: SubstrateView,

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
#[allow(dead_code)]
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

/// Spatial view of the substrate (from /substrate endpoint)
#[derive(Default, Clone)]
#[allow(dead_code)]
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
}

impl AriaVisualizer {
    pub fn new() -> Self {
        Self {
            energy_history: vec![50; 100],
            population_history: vec![100; 100],
            entropy_history: vec![50; 100],
            activity_history: vec![0; 100],
            current_stats: BrainStats::default(),
            substrate_view: SubstrateView::default(),
            messages: Vec::new(),
            last_aria_message: std::time::Instant::now(),
            last_aria_text: String::new(),
            connected: true,
            last_update: std::time::Instant::now(),
        }
    }

    /// Update substrate spatial view
    pub fn update_substrate(&mut self, view: SubstrateView) {
        self.substrate_view = view;
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

        // Calculate hunger level for advice
        let cells = self.current_stats.alive_cells.max(1) as f32;
        let energy_per_cell = self.current_stats.total_energy / cells;

        // Advice based on hunger state
        let advice = if energy_per_cell > 0.8 {
            ("", Color::DarkGray) // No advice needed
        } else if energy_per_cell > 0.5 {
            ("💡 Bientôt l'heure de parler...", Color::DarkGray)
        } else if energy_per_cell > 0.3 {
            ("💬 Parle-lui maintenant!", Color::Yellow)
        } else if energy_per_cell > 0.15 {
            ("⚠️  PARLE VITE! Elle a faim!", Color::LightRed)
        } else {
            ("🚨 URGENCE! Nourris-la avec des mots!", Color::Red)
        };

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("╔═══ ", Style::default().fg(Color::Cyan)),
                Span::styled("A R I A", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                Span::styled(" ═══╗", Style::default().fg(Color::Cyan)),
                Span::raw("  "),
                Span::styled(advice.0, Style::default().fg(advice.1).add_modifier(Modifier::BOLD)),
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
                Constraint::Percentage(30),  // More space for hunger indicator
                Constraint::Percentage(20),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(area);

        // Energy gauge with HUNGER INDICATOR
        let cells = self.current_stats.alive_cells.max(1) as f32;
        let energy_per_cell = self.current_stats.total_energy / cells;
        let energy_pct = (energy_per_cell / 1.5).clamp(0.0, 1.0); // 1.5 = energy_cap

        // Determine hunger state and color
        let (hunger_status, energy_color) = if energy_per_cell > 0.8 {
            ("😊 Rassasiée", Color::Green)
        } else if energy_per_cell > 0.5 {
            ("🙂 Bien", Color::LightGreen)
        } else if energy_per_cell > 0.3 {
            ("😐 A faim", Color::Yellow)
        } else if energy_per_cell > 0.15 {
            ("😟 FAIM!", Color::LightRed)
        } else {
            ("🆘 AFFAMÉE!", Color::Red)
        };

        let energy_gauge = Gauge::default()
            .block(Block::default()
                .title(Span::styled(
                    format!(" {} ", hunger_status),
                    Style::default().fg(energy_color).add_modifier(Modifier::BOLD)
                ))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(energy_color)))
            .gauge_style(Style::default().fg(energy_color).bg(Color::DarkGray))
            .ratio(energy_pct as f64)
            .label(Span::styled(
                format!("{:.2}/cell", energy_per_cell),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            ));
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
                Constraint::Percentage(40),
                Constraint::Percentage(60),
            ])
            .split(area);

        // Left: Substrate view (activity heatmap + energy heatmap + population stats)
        let left_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9),   // Activity heatmap
                Constraint::Length(9),   // Energy heatmap
                Constraint::Min(4),      // Population stats
            ])
            .split(chunks[0]);

        // Draw activity heatmap
        self.draw_substrate_heatmap(frame, left_chunks[0]);

        // Draw energy heatmap
        self.draw_energy_heatmap(frame, left_chunks[1]);

        // Draw population bar chart
        self.draw_population_stats(frame, left_chunks[2]);

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

    /// Draw substrate heatmap showing activity distribution
    fn draw_substrate_heatmap(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;

        // Build heatmap lines using Unicode block characters
        // Intensity: ░ (0.0-0.25) ▒ (0.25-0.5) ▓ (0.5-0.75) █ (0.75-1.0)
        let grid_size = if view.grid_size > 0 { view.grid_size } else { 16 };
        let inner_height = area.height.saturating_sub(2) as usize;
        let inner_width = area.width.saturating_sub(2) as usize;

        // Scale grid to fit available space
        let scale_y = (grid_size + inner_height - 1) / inner_height.max(1);
        let scale_x = (grid_size + inner_width - 1) / inner_width.max(1);

        let mut lines: Vec<Line> = Vec::new();
        for row in 0..(grid_size / scale_y.max(1)).min(inner_height) {
            let mut spans: Vec<Span> = Vec::new();
            for col in 0..(grid_size / scale_x.max(1)).min(inner_width) {
                // Average activity in this scaled region
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

                // Choose character and color based on activity
                let (ch, color) = if activity < 0.1 {
                    (' ', Color::DarkGray)
                } else if activity < 0.25 {
                    ('░', Color::DarkGray)
                } else if activity < 0.5 {
                    ('▒', Color::Blue)
                } else if activity < 0.75 {
                    ('▓', Color::Cyan)
                } else {
                    ('█', Color::Yellow)
                };

                spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }

        let title = format!(" Substrate {}x{} ", grid_size, grid_size);
        let heatmap = Paragraph::new(lines)
            .block(Block::default().title(title).borders(Borders::ALL))
            .style(Style::default());
        frame.render_widget(heatmap, area);
    }

    /// Draw energy heatmap showing energy distribution
    fn draw_energy_heatmap(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;

        // Build heatmap lines using Unicode block characters
        // Energy: ░ (low) ▒ (medium) ▓ (high) █ (full)
        let grid_size = if view.grid_size > 0 { view.grid_size } else { 16 };
        let inner_height = area.height.saturating_sub(2) as usize;
        let inner_width = area.width.saturating_sub(2) as usize;

        // Scale grid to fit available space
        let scale_y = (grid_size + inner_height - 1) / inner_height.max(1);
        let scale_x = (grid_size + inner_width - 1) / inner_width.max(1);

        let mut lines: Vec<Line> = Vec::new();
        for row in 0..(grid_size / scale_y.max(1)).min(inner_height) {
            let mut spans: Vec<Span> = Vec::new();
            for col in 0..(grid_size / scale_x.max(1)).min(inner_width) {
                // Average energy in this scaled region
                let mut energy_sum = 0.0f32;
                let mut count = 0;
                for dy in 0..scale_y {
                    for dx in 0..scale_x {
                        let gy = row * scale_y + dy;
                        let gx = col * scale_x + dx;
                        if gy < grid_size && gx < grid_size {
                            let idx = gy * grid_size + gx;
                            if idx < view.energy_grid.len() {
                                energy_sum += view.energy_grid[idx];
                                count += 1;
                            }
                        }
                    }
                }
                // Normalize energy (assume max ~1.5 from config.energy_cap)
                let energy = if count > 0 { (energy_sum / count as f32) / 1.5 } else { 0.0 };

                // Choose character and color based on energy
                // Green = high energy (well fed), Red = low energy (starving)
                let (ch, color) = if energy < 0.1 {
                    (' ', Color::DarkGray)
                } else if energy < 0.25 {
                    ('░', Color::Red)          // Starving
                } else if energy < 0.5 {
                    ('▒', Color::Yellow)       // Low energy
                } else if energy < 0.75 {
                    ('▓', Color::LightGreen)   // Good energy
                } else {
                    ('█', Color::Green)        // Full energy
                };

                spans.push(Span::styled(ch.to_string(), Style::default().fg(color)));
            }
            lines.push(Line::from(spans));
        }

        let title = format!(" Energy {}x{} ", grid_size, grid_size);
        let heatmap = Paragraph::new(lines)
            .block(Block::default().title(title).borders(Borders::ALL))
            .style(Style::default());
        frame.render_widget(heatmap, area);
    }

    /// Draw population breakdown as bar chart
    fn draw_population_stats(&self, frame: &mut Frame, area: Rect) {
        let view = &self.substrate_view;

        // Population breakdown
        let awake = view.awake_cells.max(1) as u64;
        let sleeping = view.sleeping_cells as u64;
        let dead = view.dead_cells as u64;

        let bars = vec![
            Bar::default()
                .value(awake)
                .label(Line::from(format!("Awake {}", awake)))
                .style(Style::default().fg(Color::Green)),
            Bar::default()
                .value(sleeping)
                .label(Line::from(format!("Sleep {}", sleeping)))
                .style(Style::default().fg(Color::Blue)),
            Bar::default()
                .value(dead)
                .label(Line::from(format!("Dead {}", dead)))
                .style(Style::default().fg(Color::Red)),
        ];

        let chart = BarChart::default()
            .block(Block::default().title(" Population ").borders(Borders::ALL))
            .bar_width(8)
            .bar_gap(1)
            .group_gap(2)
            .data(BarGroup::default().bars(&bars))
            .bar_style(Style::default());

        frame.render_widget(chart, area);
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
