//! Adaptive Auto-Training Module for ARIA
//!
//! Intelligent trainer that adapts to ARIA's state:
//! - Interval based on population health
//! - Phase based on generation maturity
//! - Intensity based on energy levels
//!
//! Toggle with 'a' key in visual mode.

use rand::seq::SliceRandom;
use rand::Rng;
use std::time::{Duration, Instant};

/// Training intensity levels
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Intensity {
    /// Light touch - cells are thriving
    Gentle,
    /// Normal stimulation
    Normal,
    /// Aggressive - cells are struggling
    Urgent,
    /// Critical - population crashing
    Critical,
}

impl Intensity {
    pub fn symbol(&self) -> &'static str {
        match self {
            Intensity::Gentle => "◦",
            Intensity::Normal => "●",
            Intensity::Urgent => "◉",
            Intensity::Critical => "⚠",
        }
    }
}

/// Training mode based on ARIA's maturity
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingMode {
    /// Newborn: simple, frequent signals to establish energy flow
    Nurture,
    /// Growing: vocabulary and associations
    Teach,
    /// Mature: sequences and predictions
    Challenge,
    /// Elite: complex conversations
    Converse,
}

impl TrainingMode {
    pub fn name(&self) -> &'static str {
        match self {
            TrainingMode::Nurture => "Nurture",
            TrainingMode::Teach => "Teach",
            TrainingMode::Challenge => "Challenge",
            TrainingMode::Converse => "Converse",
        }
    }

    /// Select mode based on max generation
    pub fn from_generation(max_gen: u32, avg_gen: f32) -> Self {
        if max_gen < 3 || avg_gen < 1.0 {
            TrainingMode::Nurture
        } else if max_gen < 8 || avg_gen < 3.0 {
            TrainingMode::Teach
        } else if max_gen < 15 || avg_gen < 6.0 {
            TrainingMode::Challenge
        } else {
            TrainingMode::Converse
        }
    }
}

/// Substrate metrics for adaptive decisions (only fields actually used)
#[derive(Clone, Debug, Default)]
struct SubstrateMetrics {
    avg_energy: f32,
    system_health: f32,
}

/// Adaptive auto-trainer
pub struct Trainer {
    /// Is auto-training enabled?
    pub enabled: bool,

    /// Current training mode (auto-selected)
    pub mode: TrainingMode,

    /// Current intensity (auto-selected)
    pub intensity: Intensity,

    /// Last substrate metrics
    metrics: SubstrateMetrics,

    /// Previous population (to detect growth/decline)
    prev_population: usize,

    /// Population trend: positive = growing, negative = declining
    population_trend: i32,

    /// Last time we sent a stimulus
    last_stimulus: Instant,

    /// Dynamic interval (adapts to health)
    interval: Duration,

    /// Last stimulus sent
    last_sent: Option<String>,

    /// Sequence state
    sequence_position: usize,
    current_sequence: usize,

    /// Stats
    pub total_sent: usize,
    pub total_responses: usize,

    /// Auto-start threshold (start training if health drops below)
    auto_start_threshold: f32,

    // === Vocabulary pools ===

    /// Simple words for nurturing
    simple_words: Vec<&'static str>,

    /// Word pairs for teaching
    word_pairs: Vec<(&'static str, &'static str)>,

    /// Sequences for challenging
    sequences: Vec<Vec<&'static str>>,

    /// Conversation phrases
    conversations: Vec<&'static str>,

    /// Emotional expressions
    emotions: Vec<&'static str>,
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            enabled: false,
            mode: TrainingMode::Nurture,
            intensity: Intensity::Normal,
            metrics: SubstrateMetrics::default(),
            prev_population: 0,
            population_trend: 0,
            last_stimulus: Instant::now(),
            interval: Duration::from_secs(3),
            last_sent: None,
            sequence_position: 0,
            current_sequence: 0,
            total_sent: 0,
            total_responses: 0,
            auto_start_threshold: 0.3,

            // Simple, high-energy words for nurturing
            simple_words: vec![
                "bonjour", "salut", "oui", "non", "merci",
                "moka", "chat", "bien", "content", "là",
                "eau", "soleil", "jouer", "manger", "dormir",
            ],

            // Word pairs for teaching associations
            word_pairs: vec![
                ("chat", "moka"), ("joli", "chat"), ("petit", "moka"),
                ("bon", "matin"), ("bonne", "nuit"), ("chat", "dort"),
                ("soleil", "chaud"), ("eau", "fraiche"), ("chat", "joue"),
                ("content", "moka"), ("bonjour", "ARIA"), ("merci", "beaucoup"),
            ],

            // Predictable sequences for prediction law
            sequences: vec![
                vec!["un", "deux", "trois"],
                vec!["un", "deux", "trois", "quatre", "cinq"],
                vec!["A", "B", "C", "D"],
                vec!["matin", "midi", "soir", "nuit"],
                vec!["petit", "moyen", "grand"],
                vec!["lundi", "mardi", "mercredi", "jeudi", "vendredi"],
                vec!["dort", "réveille", "mange", "joue", "dort"],
            ],

            // Complex conversations
            conversations: vec![
                "Comment vas-tu?", "Tu es là?", "Qu'est-ce que tu fais?",
                "Je suis content de te voir", "Tu me reconnais?",
                "Raconte-moi quelque chose", "À quoi tu penses?",
                "Tu as faim?", "On joue?", "Bonne nuit ARIA",
            ],

            // Emotional expressions
            emotions: vec![
                "Bravo!", "Super!", "Bien joué!", "Parfait!",
                "Oui!", "C'est ça!", "Continue!", "Encore!",
                "Non", "Pas ça", "Essaie encore",
            ],
        }
    }

    /// Update with latest substrate metrics
    pub fn update_metrics(
        &mut self,
        alive_cells: usize,
        _awake_cells: usize,
        avg_energy: f32,
        _avg_tension: f32,
        max_generation: u32,
        avg_generation: f32,
        system_health: f32,
    ) {
        // Track population trend
        if self.prev_population > 0 {
            let diff = alive_cells as i32 - self.prev_population as i32;
            // Smooth the trend
            self.population_trend = (self.population_trend * 3 + diff.signum()) / 4;
        }
        self.prev_population = alive_cells;

        self.metrics = SubstrateMetrics {
            avg_energy,
            system_health,
        };

        // Auto-select mode based on generation
        self.mode = TrainingMode::from_generation(max_generation, avg_generation);

        // Auto-select intensity based on health
        self.intensity = self.calculate_intensity();

        // Auto-adjust interval based on intensity
        self.interval = self.calculate_interval();

        // Auto-start if health is critical and not enabled
        if !self.enabled && system_health < self.auto_start_threshold && alive_cells > 1000 {
            self.enabled = true;
            self.last_stimulus = Instant::now();
        }
    }

    fn calculate_intensity(&self) -> Intensity {
        let m = &self.metrics;

        // Critical: population crashing or very low energy
        if m.avg_energy < 0.2 || self.population_trend < -2 {
            return Intensity::Critical;
        }

        // Urgent: struggling
        if m.avg_energy < 0.35 || m.system_health < 0.4 {
            return Intensity::Urgent;
        }

        // Gentle: thriving
        if m.avg_energy > 0.6 && m.system_health > 0.7 && self.population_trend > 0 {
            return Intensity::Gentle;
        }

        Intensity::Normal
    }

    fn calculate_interval(&self) -> Duration {
        match self.intensity {
            Intensity::Critical => Duration::from_millis(500),  // Very fast
            Intensity::Urgent => Duration::from_secs(1),        // Fast
            Intensity::Normal => Duration::from_secs(3),        // Normal
            Intensity::Gentle => Duration::from_secs(6),        // Slow, let them rest
        }
    }

    /// Toggle auto-training on/off
    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        if self.enabled {
            self.last_stimulus = Instant::now();
        }
    }

    /// Notify trainer that ARIA responded
    pub fn on_response(&mut self, _response: &str) {
        self.total_responses += 1;
    }

    /// Get the next stimulus to send (if any)
    pub fn tick(&mut self) -> Option<String> {
        if !self.enabled {
            return None;
        }

        let now = Instant::now();
        if now.duration_since(self.last_stimulus) < self.interval {
            return None;
        }

        self.last_stimulus = now;
        self.total_sent += 1;

        let stimulus = self.generate_stimulus();
        self.last_sent = Some(stimulus.clone());
        Some(stimulus)
    }

    fn generate_stimulus(&mut self) -> String {
        let mut rng = rand::thread_rng();

        match self.mode {
            TrainingMode::Nurture => {
                // Simple, frequent words to establish energy flow
                // Occasionally repeat for reinforcement
                if rng.gen_bool(0.3) && self.last_sent.is_some() {
                    // Repeat last word for reinforcement
                    return self.last_sent.clone().unwrap();
                }
                self.simple_words
                    .choose(&mut rng)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "bonjour".to_string())
            }

            TrainingMode::Teach => {
                // Mix of single words and pairs
                if rng.gen_bool(0.6) {
                    // Word pair
                    if let Some((a, b)) = self.word_pairs.choose(&mut rng) {
                        format!("{} {}", a, b)
                    } else {
                        "chat moka".to_string()
                    }
                } else {
                    // Single word
                    self.simple_words
                        .choose(&mut rng)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "moka".to_string())
                }
            }

            TrainingMode::Challenge => {
                // Sequences for prediction law training
                if self.sequences.is_empty() {
                    return "un".to_string();
                }

                let seq = &self.sequences[self.current_sequence % self.sequences.len()];
                let word = seq[self.sequence_position % seq.len()].to_string();

                self.sequence_position += 1;
                if self.sequence_position >= seq.len() {
                    self.sequence_position = 0;
                    self.current_sequence = (self.current_sequence + 1) % self.sequences.len();
                }

                word
            }

            TrainingMode::Converse => {
                // Mix of conversations and emotional responses
                if rng.gen_bool(0.3) {
                    self.emotions
                        .choose(&mut rng)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "Bravo!".to_string())
                } else {
                    self.conversations
                        .choose(&mut rng)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "bonjour".to_string())
                }
            }
        }
    }

    /// Get status string for display
    pub fn status(&self) -> String {
        if self.enabled {
            let trend = if self.population_trend > 0 { "↑" }
                else if self.population_trend < 0 { "↓" }
                else { "→" };

            format!(
                "AUTO {} {} | {}s | Pop{} | Sent:{} Resp:{}",
                self.intensity.symbol(),
                self.mode.name(),
                self.interval.as_secs(),
                trend,
                self.total_sent,
                self.total_responses
            )
        } else {
            "AUTO: Off ('a' to start)".to_string()
        }
    }
}
