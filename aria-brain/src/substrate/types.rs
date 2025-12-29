//! Types and constants for the Substrate module

use aria_core::SIGNAL_DIMS;
use serde::{Deserialize, Serialize};

/// Minimum ticks between emissions (anti-spam)
/// At ~4ms/tick (actual observed rate), 75 ticks ≈ 300ms between responses
pub const EMISSION_COOLDOWN_TICKS: u64 = 25;  // ~100ms between emissions (faster response)

/// Words that are too common - ARIA focuses on meaningful words
pub const STOP_WORDS: &[&str] = &[
    // French
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "leur", "en", "y",
    "est", "suis", "es", "sont", "sommes", "êtes",
    "ai", "as", "a", "ont", "avons", "avez",
    "fait", "faire", "vais", "vas", "va", "vont",
    "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "chez",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    // English
    "the", "a", "an", "is", "are", "am", "was", "were",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "and", "or", "but", "so", "if", "then", "to", "of", "in", "on", "at",
    "be", "have", "has", "had", "do", "does", "did",
    "si", "ne", "pas", "plus", "très", "bien",
];

/// Adaptive parameters that ARIA modifies herself
/// These evolve through feedback - no hardcoded rules, just emergence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveParams {
    /// Coherence threshold to emit a response (0.0 - 1.0)
    /// Higher = more selective, lower = more talkative
    pub emission_threshold: f32,

    /// Probability of responding when she could (0.0 - 1.0)
    /// Higher = more responsive, lower = more contemplative
    pub response_probability: f32,

    /// How fast associations are learned (0.0 - 1.0)
    /// Higher = faster learning, lower = more stable
    pub learning_rate: f32,

    /// Tendency to speak spontaneously (0.0 - 1.0)
    /// Higher = more spontaneous, lower = waits for input
    pub spontaneity: f32,

    /// How long to wait before sleeping (in ticks)
    pub idle_ticks_to_sleep: u64,

    /// Parameters at last positive feedback (for reinforcement)
    pub(crate) last_success_params: Option<Box<AdaptiveParamsSnapshot>>,

    /// Count of positive and negative feedback
    pub(crate) positive_count: u64,
    pub(crate) negative_count: u64,
}

/// Snapshot of params for reinforcement learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveParamsSnapshot {
    pub emission_threshold: f32,
    pub response_probability: f32,
    pub learning_rate: f32,
    pub spontaneity: f32,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            emission_threshold: 0.15,
            response_probability: 0.8,
            learning_rate: 0.3,
            spontaneity: 0.05,
            idle_ticks_to_sleep: 100,
            last_success_params: None,
            positive_count: 0,
            negative_count: 0,
        }
    }
}

impl AdaptiveParams {
    /// Called when ARIA receives positive feedback
    pub fn reinforce_positive(&mut self) {
        self.positive_count += 1;

        // Save current params as "what worked"
        self.last_success_params = Some(Box::new(AdaptiveParamsSnapshot {
            emission_threshold: self.emission_threshold,
            response_probability: self.response_probability,
            learning_rate: self.learning_rate,
            spontaneity: self.spontaneity,
        }));

        // Slight exploration towards what worked (become slightly more like this)
        // But also add tiny random mutation for exploration
        let mut rng = rand::thread_rng();
        use rand::Rng;
        self.spontaneity = (self.spontaneity + 0.01).min(0.3);  // Encouraged to be more spontaneous
        self.response_probability = (self.response_probability + 0.02).min(1.0);

        // Small random exploration
        self.emission_threshold += (rng.gen::<f32>() - 0.5) * 0.02;
        self.emission_threshold = self.emission_threshold.clamp(0.05, 0.5);

        tracing::info!("🧬 ADAPTED (positive): emission={:.2}, response={:.2}, spontaneity={:.2}",
            self.emission_threshold, self.response_probability, self.spontaneity);
    }

    /// Called when ARIA receives negative feedback
    pub fn reinforce_negative(&mut self) {
        self.negative_count += 1;

        // Move away from current params, towards last success if available
        if let Some(ref success) = self.last_success_params {
            // Drift towards what worked before
            self.emission_threshold += (success.emission_threshold - self.emission_threshold) * 0.1;
            self.response_probability += (success.response_probability - self.response_probability) * 0.1;
            self.spontaneity += (success.spontaneity - self.spontaneity) * 0.1;
        } else {
            // No success yet - become more conservative
            self.emission_threshold = (self.emission_threshold + 0.02).min(0.5);
            self.response_probability = (self.response_probability - 0.05).max(0.3);
        }

        tracing::info!("🧬 ADAPTED (negative): emission={:.2}, response={:.2}, spontaneity={:.2}",
            self.emission_threshold, self.response_probability, self.spontaneity);
    }

    /// Random exploration - called periodically to try new things
    pub fn explore(&mut self) {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        // Small random mutations
        self.emission_threshold += (rng.gen::<f32>() - 0.5) * 0.01;
        self.response_probability += (rng.gen::<f32>() - 0.5) * 0.01;
        self.learning_rate += (rng.gen::<f32>() - 0.5) * 0.01;
        self.spontaneity += (rng.gen::<f32>() - 0.5) * 0.005;

        // Keep in bounds
        self.emission_threshold = self.emission_threshold.clamp(0.05, 0.5);
        self.response_probability = self.response_probability.clamp(0.3, 1.0);
        self.learning_rate = self.learning_rate.clamp(0.1, 0.8);
        self.spontaneity = self.spontaneity.clamp(0.01, 0.3);
    }

    /// Get current params for logging
    pub fn summary(&self) -> String {
        format!("emit={:.2} resp={:.2} learn={:.2} spont={:.2} (+{}/-{})",
            self.emission_threshold, self.response_probability,
            self.learning_rate, self.spontaneity,
            self.positive_count, self.negative_count)
    }

    /// Get positive feedback count
    pub fn positive_count(&self) -> u64 {
        self.positive_count
    }

    /// Get negative feedback count
    pub fn negative_count(&self) -> u64 {
        self.negative_count
    }
}

/// A recently heard word for imitation
#[derive(Clone, Debug)]
pub struct RecentWord {
    pub word: String,
    pub vector: [f32; SIGNAL_DIMS],
    pub heard_at: u64,
}

/// Spatial inhibitor for adaptive regional thresholds (Gemini optimization)
///
/// Divides semantic space into regions and tracks activity per region.
/// Recently active regions have higher inhibition thresholds (refractory period).
#[derive(Clone, Debug)]
pub struct SpatialInhibitor {
    /// Activity levels per region (grid_size^2 regions)
    region_activity: Vec<f32>,
    /// Last activation tick per region
    region_last_active: Vec<u64>,
    /// Grid size (number of divisions per dimension)
    grid_size: usize,
    /// Base threshold (from adaptive params)
    base_threshold: f32,
    /// Maximum threshold multiplier (for very active regions)
    max_multiplier: f32,
    /// Decay rate per tick (how fast refractory period fades)
    decay_rate: f32,
}

impl Default for SpatialInhibitor {
    fn default() -> Self {
        let grid_size = 8; // 8x8 = 64 regions
        let region_count = grid_size * grid_size;
        Self {
            region_activity: vec![0.0; region_count],
            region_last_active: vec![0; region_count],
            grid_size,
            base_threshold: 0.15,
            max_multiplier: 3.0,
            decay_rate: 0.02,
        }
    }
}

#[allow(dead_code)]
impl SpatialInhibitor {
    /// Create with custom grid size
    pub fn new(grid_size: usize) -> Self {
        let region_count = grid_size * grid_size;
        Self {
            region_activity: vec![0.0; region_count],
            region_last_active: vec![0; region_count],
            grid_size,
            base_threshold: 0.15,
            max_multiplier: 3.0,
            decay_rate: 0.02,
        }
    }

    /// Update base threshold from adaptive params
    pub fn set_base_threshold(&mut self, threshold: f32) {
        self.base_threshold = threshold;
    }

    /// Map a position in semantic space to a region index
    /// Position values are typically in [-10, 10] range
    fn position_to_region(&self, position: &[f32]) -> usize {
        // Use first 2 dimensions for 2D grid
        let x = ((position.get(0).unwrap_or(&0.0) + 10.0) / 20.0 * self.grid_size as f32)
            .clamp(0.0, (self.grid_size - 1) as f32) as usize;
        let y = ((position.get(1).unwrap_or(&0.0) + 10.0) / 20.0 * self.grid_size as f32)
            .clamp(0.0, (self.grid_size - 1) as f32) as usize;
        y * self.grid_size + x
    }

    /// Record activity at a position
    pub fn record_activity(&mut self, position: &[f32], intensity: f32, tick: u64) {
        let region = self.position_to_region(position);
        if region < self.region_activity.len() {
            self.region_activity[region] = (self.region_activity[region] + intensity).min(1.0);
            self.region_last_active[region] = tick;
        }
    }

    /// Decay all regions (call once per tick)
    pub fn decay(&mut self, current_tick: u64) {
        for i in 0..self.region_activity.len() {
            // Decay based on time since last activity
            let ticks_since_active = current_tick.saturating_sub(self.region_last_active[i]);
            if ticks_since_active > 10 {
                self.region_activity[i] *= 1.0 - self.decay_rate;
            }
        }
    }

    /// Get adaptive threshold for a position
    ///
    /// Returns higher threshold for recently active regions (refractory period)
    pub fn get_threshold(&self, position: &[f32]) -> f32 {
        let region = self.position_to_region(position);
        if region < self.region_activity.len() {
            let activity = self.region_activity[region];
            // Scale threshold: more active = higher threshold
            let multiplier = 1.0 + activity * (self.max_multiplier - 1.0);
            self.base_threshold * multiplier
        } else {
            self.base_threshold
        }
    }

    /// Get average activity across all regions
    pub fn average_activity(&self) -> f32 {
        if self.region_activity.is_empty() {
            0.0
        } else {
            self.region_activity.iter().sum::<f32>() / self.region_activity.len() as f32
        }
    }

    /// Get number of highly active regions (activity > 0.5)
    pub fn active_region_count(&self) -> usize {
        self.region_activity.iter().filter(|&&a| a > 0.5).count()
    }
}

/// Statistics about the substrate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubstrateStats {
    pub tick: u64,
    pub alive_cells: usize,
    pub total_energy: f32,
    pub entropy: f32,
    pub active_clusters: usize,
    pub dominant_emotion: String,
    pub signals_per_second: f32,
    pub oldest_cell_age: u64,
    pub average_connections: f32,
    pub mood: String,
    pub happiness: f32,
    pub arousal: f32,
    pub curiosity: f32,
    // New V2 stats
    pub sleeping_cells: usize,
    pub cpu_savings_percent: f32,
    pub backend_name: String,
    // Adaptive params (self-modification)
    pub adaptive_emission_threshold: f32,
    pub adaptive_response_probability: f32,
    pub adaptive_spontaneity: f32,
    pub adaptive_feedback_positive: u64,
    pub adaptive_feedback_negative: u64,
    // Boredom level
    pub boredom: f32,
}
