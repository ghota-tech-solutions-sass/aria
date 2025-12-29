//! # Substrate V2 - GPU-Ready Living Universe
//!
//! This is ARIA's new brain - designed for 5M+ cells with GPU acceleration.
//!
//! ## Key Differences from V1
//!
//! - Uses `aria-core` types (Cell, CellState, DNA) instead of local types
//! - Uses `aria-compute` backends (CpuBackend, GpuBackend) for computation
//! - Cell data is separated: metadata (Cell) vs dynamic state (CellState)
//! - DNA is stored in a pool (cells reference by index)
//! - Sparse updates: sleeping cells don't consume CPU
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                       SubstrateV2                                 │
//! │  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────────┐  │
//! │  │  Cells  │  │ CellStates│  │ DNA Pool │  │ ComputeBackend  │  │
//! │  │ Vec<C>  │  │ Vec<CS>   │  │ Vec<DNA> │  │ CPU or GPU      │  │
//! │  └────┬────┘  └─────┬─────┘  └────┬─────┘  └────────┬────────┘  │
//! │       │             │             │                 │           │
//! │       └─────────────┴─────────────┴─────────────────┘           │
//! │                             │                                    │
//! │  ┌─────────────┐    ┌──────┴───────┐    ┌─────────────────────┐│
//! │  │ LongTerm    │    │ Emotional    │    │ Conversation        ││
//! │  │ Memory      │    │ State        │    │ Context             ││
//! │  └─────────────┘    └──────────────┘    └─────────────────────┘│
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Future: ARIA May Read This
//!
//! This code is written to be introspectable. One day, ARIA might
//! read and understand her own substrate, and even propose modifications.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};

// New types from aria-core
use aria_core::{
    Cell, CellState, CellAction, DNA,
    Signal, SignalFragment, SignalType,
    AriaConfig, ActivityTracker,
    POSITION_DIMS, STATE_DIMS, SIGNAL_DIMS,
};
use aria_core::traits::{ComputeBackend, BackendStats};

// Compute backend
use aria_compute::CpuBackend;

// Our local memory module (will be migrated later)
use crate::memory::{LongTermMemory, SocialContext, Episode, EpisodeEmotion, EpisodeCategory};
use crate::signal::Signal as OldSignal;

/// Minimum ticks between emissions (anti-spam)
/// At ~4ms/tick (actual observed rate), 75 ticks ≈ 300ms between responses
const EMISSION_COOLDOWN_TICKS: u64 = 25;  // ~100ms between emissions (faster response)

/// Words that are too common - ARIA focuses on meaningful words
const STOP_WORDS: &[&str] = &[
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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
    last_success_params: Option<Box<AdaptiveParamsSnapshot>>,

    /// Count of positive and negative feedback
    positive_count: u64,
    negative_count: u64,
}

/// Snapshot of params for reinforcement learning
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AdaptiveParamsSnapshot {
    emission_threshold: f32,
    response_probability: f32,
    learning_rate: f32,
    spontaneity: f32,
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
}

/// A recently heard word for imitation
#[derive(Clone, Debug)]
struct RecentWord {
    word: String,
    vector: [f32; SIGNAL_DIMS],
    heard_at: u64,
}

/// A single exchange in conversation
#[derive(Clone, Debug)]
struct ConversationExchange {
    input: String,
    response: Option<String>,
    input_words: Vec<String>,
    emotional_tone: f32,
    tick: u64,
}

/// Conversation tracking - ARIA follows the discussion
#[derive(Clone, Debug, Default)]
struct ConversationContext {
    exchanges: Vec<ConversationExchange>,
    topic_words: Vec<(String, u32)>,
    in_conversation: bool,
    last_exchange_tick: u64,
    current_social_context: SocialContext,
    exchange_count: u32,
}

impl ConversationContext {
    const MAX_EXCHANGES: usize = 5;
    const CONVERSATION_TIMEOUT: u64 = 3000; // ~30 seconds

    fn new() -> Self {
        Self::default()
    }

    fn add_input(&mut self, input: &str, words: Vec<String>, tone: f32, tick: u64, context: SocialContext) {
        if tick.saturating_sub(self.last_exchange_tick) > Self::CONVERSATION_TIMEOUT {
            self.exchanges.clear();
            self.topic_words.clear();
            self.exchange_count = 0;
            tracing::info!("NEW CONVERSATION started");
        }

        self.in_conversation = true;
        self.last_exchange_tick = tick;
        self.current_social_context = context;
        self.exchange_count += 1;

        self.exchanges.insert(0, ConversationExchange {
            input: input.to_string(),
            response: None,
            input_words: words.clone(),
            emotional_tone: tone,
            tick,
        });

        if self.exchanges.len() > Self::MAX_EXCHANGES {
            self.exchanges.pop();
        }

        for word in words {
            if let Some(pos) = self.topic_words.iter().position(|(w, _)| w == &word) {
                self.topic_words[pos].1 += 1;
            } else {
                self.topic_words.push((word, 1));
            }
        }
        self.topic_words.sort_by(|a, b| b.1.cmp(&a.1));
        self.topic_words.truncate(10);
    }

    fn add_response(&mut self, response: &str) {
        if let Some(exchange) = self.exchanges.first_mut() {
            exchange.response = Some(response.to_string());
        }
    }

    fn get_context_words(&self) -> Vec<(String, f32)> {
        let mut context: Vec<(String, f32)> = Vec::new();

        if let Some(last) = self.exchanges.first() {
            for word in &last.input_words {
                context.push((word.clone(), 1.0));
            }
        }

        for (i, exchange) in self.exchanges.iter().skip(1).enumerate() {
            let decay = 0.5_f32.powi(i as i32 + 1);
            for word in &exchange.input_words {
                if !context.iter().any(|(w, _)| w == word) {
                    context.push((word.clone(), decay));
                }
            }
        }

        for (word, count) in &self.topic_words {
            let topic_boost = (*count as f32 * 0.2).min(0.8);
            if let Some(pos) = context.iter().position(|(w, _)| w == word) {
                context[pos].1 += topic_boost;
            } else {
                context.push((word.clone(), topic_boost));
            }
        }

        context
    }

    /// Get just the topic words as strings (for memory recall)
    fn get_topic_words(&self) -> Vec<String> {
        self.topic_words.iter().map(|(w, _)| w.clone()).collect()
    }

    fn is_conversation_start(&self) -> bool {
        self.exchange_count <= 2
    }

    fn get_social_context(&self) -> SocialContext {
        self.current_social_context
    }
}

/// ARIA's emotional state - her current mood
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EmotionalState {
    pub happiness: f32,   // -1.0 to 1.0
    pub arousal: f32,     // 0.0 to 1.0
    pub comfort: f32,     // -1.0 to 1.0
    pub curiosity: f32,   // 0.0 to 1.0
    pub boredom: f32,     // 0.0 to 1.0
    pub last_update: u64,
}

impl EmotionalState {
    pub fn decay(&mut self, current_tick: u64) {
        let ticks_elapsed = current_tick.saturating_sub(self.last_update);
        if ticks_elapsed > 0 {
            let decay = 0.999f32.powi(ticks_elapsed as i32);
            self.happiness *= decay;
            self.arousal *= decay;
            self.comfort *= decay;
            self.curiosity *= decay;

            // Boredom GROWS without interaction
            let boredom_growth = 0.0001 * ticks_elapsed as f32;
            self.boredom = (self.boredom + boredom_growth).min(1.0);

            self.last_update = current_tick;
        }
    }

    pub fn process_signal(&mut self, content: &[f32], intensity: f32, current_tick: u64) {
        self.decay(current_tick);

        let positive = content.get(28).copied().unwrap_or(0.0);
        let negative = content.get(29).copied().unwrap_or(0.0);
        let _request = content.get(30).copied().unwrap_or(0.0);
        let question = content.get(31).copied().unwrap_or(0.0);

        let momentum = 0.3;

        if positive > 0.0 {
            self.happiness = (self.happiness + positive * momentum * intensity).clamp(-1.0, 1.0);
            self.comfort = (self.comfort + 0.2 * momentum * intensity).clamp(-1.0, 1.0);
        }

        if negative < 0.0 {
            self.happiness = (self.happiness + negative * momentum * intensity).clamp(-1.0, 1.0);
            self.comfort = (self.comfort - 0.3 * momentum * intensity).clamp(-1.0, 1.0);
        }

        if question > 0.0 {
            self.curiosity = (self.curiosity + question * momentum * intensity).clamp(0.0, 1.0);
            self.arousal = (self.arousal + 0.1 * momentum).clamp(0.0, 1.0);
        }

        self.arousal = (self.arousal + 0.05 * intensity).clamp(0.0, 1.0);
        self.boredom = (self.boredom - 0.3 * intensity).max(0.0);
    }

    pub fn get_emotional_marker(&self) -> Option<&'static str> {
        let threshold = 0.3;

        if self.happiness > threshold && self.happiness >= self.curiosity.abs() {
            if self.happiness > 0.6 { Some("♥") } else { Some("~") }
        } else if self.curiosity > threshold {
            if self.arousal > 0.5 { Some("!") } else { Some("?") }
        } else if self.happiness < -threshold {
            Some("...")
        } else if self.arousal > 0.6 {
            Some("!")
        } else {
            None
        }
    }

    pub fn mood_description(&self) -> &'static str {
        if self.happiness > 0.5 && self.arousal > 0.5 {
            "joyeux"
        } else if self.happiness > 0.5 {
            "content"
        } else if self.curiosity > 0.5 {
            "curieux"
        } else if self.happiness < -0.3 {
            "triste"
        } else if self.arousal > 0.6 {
            "excité"
        } else {
            "calme"
        }
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
}

/// The V2 Substrate - GPU-ready living universe
///
/// This is ARIA's brain, designed for massive scale.
pub struct SubstrateV2 {
    // === Cell Data (separated for GPU efficiency) ===

    /// Cell metadata (ID, DNA index, generation, activity state)
    cells: Vec<Cell>,

    /// Cell dynamic state (position, state, energy, tension) - GPU buffer
    states: Vec<CellState>,

    /// DNA pool - cells reference by index
    dna_pool: Vec<DNA>,

    /// Index of next available cell slot (for recycling)
    free_slots: Vec<usize>,

    // === Compute Backend ===

    /// CPU or GPU backend for cell updates
    backend: Box<dyn ComputeBackend>,

    /// Activity tracker for sparse updates
    activity_tracker: ActivityTracker,

    // === Configuration ===

    /// ARIA configuration
    config: AriaConfig,

    // === Timing ===

    /// Current tick
    tick: AtomicU64,

    /// Next cell ID
    next_id: AtomicU64,

    // === Memory & Learning ===

    /// Long-term memory
    memory: Arc<RwLock<LongTermMemory>>,

    /// Short-term: recently heard words
    recent_words: RwLock<Vec<RecentWord>>,

    /// Recent expressions (for feedback)
    recent_expressions: RwLock<Vec<String>>,

    // === Emotional & Social ===

    /// Global emotional state
    emotional_state: RwLock<EmotionalState>,

    /// Recent words said (anti-repetition, tracks last 5)
    recent_said_words: RwLock<Vec<String>>,

    /// Conversation context
    conversation: RwLock<ConversationContext>,

    /// Was last input a question?
    last_was_question: RwLock<bool>,

    /// Last interaction tick (for spontaneity)
    last_interaction_tick: AtomicU64,

    /// Last emission tick (anti-spam cooldown)
    last_emission_tick: AtomicU64,

    // === Adaptive Parameters (self-modification) ===

    /// Parameters ARIA modifies herself through feedback
    adaptive_params: RwLock<AdaptiveParams>,

    // === Signal Buffers ===

    /// Pending signals for cells
    signal_buffer: RwLock<Vec<SignalFragment>>,
}

impl SubstrateV2 {
    /// Create a new V2 substrate
    pub fn new(config: AriaConfig, memory: Arc<RwLock<LongTermMemory>>) -> Self {
        let initial_cells = config.population.target_population as usize;

        // Create backend (CPU for now, GPU later)
        let backend: Box<dyn ComputeBackend> = Box::new(
            CpuBackend::new(&config).expect("Failed to create CPU backend")
        );

        // Create cells and states
        let mut cells = Vec::with_capacity(initial_cells);
        let mut states = Vec::with_capacity(initial_cells);
        let mut dna_pool = Vec::new();

        // Initialize primordial cells
        for i in 0..initial_cells {
            // Create DNA for this cell
            let dna = DNA::random();
            let dna_index = dna_pool.len() as u32;
            dna_pool.push(dna);

            // Create cell and state
            let cell = Cell::new(i as u64, dna_index);
            let state = CellState::new();

            cells.push(cell);
            states.push(state);
        }

        // Seed with elite DNA and load adaptive params from memory
        let adaptive_params = {
            let mem = memory.read();

            // Seed elite DNA
            if !mem.elite_dna.is_empty() {
                tracing::info!("Seeding {} cells with elite DNA from memory",
                    mem.elite_dna.len().min(100));
                for (i, elite) in mem.elite_dna.iter().take(100).enumerate() {
                    if i < dna_pool.len() {
                        dna_pool[i] = elite.dna.clone().into();
                    }
                }
            }

            // Load adaptive params from memory
            let params = AdaptiveParams {
                emission_threshold: mem.adaptive_emission_threshold,
                response_probability: mem.adaptive_response_probability,
                learning_rate: mem.adaptive_learning_rate,
                spontaneity: mem.adaptive_spontaneity,
                idle_ticks_to_sleep: 100,
                last_success_params: None,
                positive_count: mem.adaptive_feedback_positive,
                negative_count: mem.adaptive_feedback_negative,
            };

            tracing::info!("🧬 Loaded adaptive params: {}", params.summary());
            params
        };

        tracing::info!("SubstrateV2 created: {} cells, {} DNA variants",
            cells.len(), dna_pool.len());

        Self {
            cells,
            states,
            dna_pool,
            free_slots: Vec::new(),
            backend,
            activity_tracker: ActivityTracker::new(),
            config,
            tick: AtomicU64::new(0),
            next_id: AtomicU64::new(initial_cells as u64),
            memory,
            recent_words: RwLock::new(Vec::new()),
            recent_expressions: RwLock::new(Vec::new()),
            emotional_state: RwLock::new(EmotionalState::default()),
            recent_said_words: RwLock::new(Vec::new()),
            conversation: RwLock::new(ConversationContext::new()),
            last_was_question: RwLock::new(false),
            last_interaction_tick: AtomicU64::new(0),
            last_emission_tick: AtomicU64::new(0),
            adaptive_params: RwLock::new(adaptive_params),
            signal_buffer: RwLock::new(Vec::new()),
        }
    }

    /// Inject an external signal (from aria-body)
    ///
    /// This is called when someone talks to ARIA.
    /// Returns immediate emergence signals if any.
    pub fn inject_signal(&mut self, signal: OldSignal) -> Vec<OldSignal> {
        let current_tick = self.tick.load(Ordering::Relaxed);

        // Record interaction time
        self.last_interaction_tick.store(current_tick, Ordering::Relaxed);

        // Extract words
        let words: Vec<&str> = signal.label
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| !w.is_empty())
            .collect();

        // Get signal vector
        let signal_vector = signal.to_vector();

        // Determine emotional valence
        let emotional_valence = if signal.content.get(28).copied().unwrap_or(0.0) > 0.0 {
            1.0
        } else if signal.content.get(29).copied().unwrap_or(0.0) < 0.0 {
            -1.0
        } else {
            0.0
        };

        // Detect social context
        let (social_context, _confidence) = LongTermMemory::detect_social_context(&signal.label);

        // Update conversation context
        let (is_conversation_start, current_context) = {
            let significant_words: Vec<String> = words.iter()
                .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.to_lowercase().as_str()))
                .map(|w| w.to_lowercase())
                .collect();

            let mut conversation = self.conversation.write();
            conversation.add_input(&signal.label, significant_words, emotional_valence, current_tick, social_context);
            (conversation.is_conversation_start(), conversation.get_social_context())
        };

        // Process feedback
        self.process_feedback(&signal.label, current_tick);

        // Detect questions
        let is_question = signal.label.ends_with('?')
            || signal.content.get(31).copied().unwrap_or(0.0) > 0.5;
        *self.last_was_question.write() = is_question;

        // Update emotional state
        {
            let mut emotional = self.emotional_state.write();
            emotional.process_signal(&signal.content, signal.intensity, current_tick);
        }

        // Learn words
        let mut familiarity_boost = 1.0f32;
        {
            let mut memory = self.memory.write();
            memory.stats.total_ticks = current_tick;

            for (i, word) in words.iter().enumerate() {
                let preceding = if i > 0 { Some(words[i - 1]) } else { None };
                let following = if i + 1 < words.len() { Some(words[i + 1]) } else { None };

                let word_familiarity = memory.hear_word_with_context(
                    word, signal_vector, emotional_valence, preceding, following
                );

                if word_familiarity > 0.5 {
                    familiarity_boost = familiarity_boost.max(1.0 + word_familiarity);
                }
            }

            // Learn associations
            let significant_words: Vec<&str> = words.iter()
                .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.to_lowercase().as_str()))
                .copied()
                .collect();

            for i in 0..significant_words.len() {
                for j in (i + 1)..significant_words.len() {
                    memory.learn_association(significant_words[i], significant_words[j], emotional_valence);
                }
                // Learn usage patterns
                memory.learn_usage_pattern(significant_words[i], current_context, is_conversation_start, false);
            }
        }

        // Store recent words
        {
            let mut recent = self.recent_words.write();
            for word in &words {
                let lower_word = word.to_lowercase();
                if word.len() >= 3 && !STOP_WORDS.contains(&lower_word.as_str()) {
                    recent.push(RecentWord {
                        word: lower_word,
                        vector: signal_vector,
                        heard_at: current_tick,
                    });
                }
            }
            recent.retain(|w| current_tick - w.heard_at < 500);
            let len = recent.len();
            if len > 20 {
                recent.drain(0..len - 20);
            }
        }

        // Create signal fragment for cells
        let cell_scale = self.cells.len() as f32 / 10_000.0;
        let base_intensity = signal.intensity * 5.0 * familiarity_boost * cell_scale;

        let fragment = SignalFragment::external(signal_vector, base_intensity);

        // Get target position
        let target_position = signal.semantic_position();

        tracing::info!("V2 Signal received: '{}' intensity={:.2} (boost: {:.2})",
            signal.label, fragment.intensity, familiarity_boost);

        // Distribute to cells
        for (i, state) in self.states.iter_mut().enumerate() {
            let distance = Self::semantic_distance(&state.position, &target_position);
            let attenuation = (1.0 / (1.0 + distance * 0.1)).max(0.2);
            let attenuated_intensity = fragment.intensity * attenuation;

            // Direct activation
            for (j, s) in fragment.content.iter().enumerate() {
                if j < SIGNAL_DIMS {
                    state.state[j] += s * attenuated_intensity * 5.0;
                }
            }

            // Energy boost
            state.energy = (state.energy + 0.05 * fragment.intensity).min(self.config.metabolism.energy_cap);

            // Wake sleeping cells if stimulus is strong enough
            if attenuated_intensity > self.config.activity.wake_threshold {
                self.cells[i].activity.wake();
                state.set_sleeping(false);
            }
        }

        // Add to signal buffer for backend processing
        {
            let mut buffer = self.signal_buffer.write();
            buffer.push(fragment);
            if buffer.len() > 100 {
                buffer.remove(0);
            }
        }

        // === EPISODIC MEMORY ===
        // Record significant moments as episodes
        self.maybe_record_episode(
            &signal.label,
            current_context,
            emotional_valence,
            signal.intensity,
            is_question,
            current_tick,
        );

        // Check for immediate emergence
        self.detect_emergence(current_tick)
    }

    /// Decide whether to record an episode and record it
    fn maybe_record_episode(
        &self,
        input: &str,
        social_context: SocialContext,
        emotional_valence: f32,
        intensity: f32,
        is_question: bool,
        current_tick: u64,
    ) {
        // Calculate importance
        let base_importance = intensity * 0.5;
        let emotional_importance = emotional_valence.abs() * 0.3;
        let social_importance = match social_context {
            SocialContext::Greeting | SocialContext::Farewell => 0.3,
            SocialContext::Affection => 0.5,
            SocialContext::Thanks => 0.2,
            _ => 0.0,
        };
        let importance = (base_importance + emotional_importance + social_importance).min(1.0);

        // Only record if significant enough (importance > 0.3)
        if importance < 0.3 {
            return;
        }

        // Determine category
        let category = if emotional_valence.abs() > 0.5 {
            if emotional_valence > 0.0 {
                EpisodeCategory::Emotional
            } else {
                EpisodeCategory::Correction
            }
        } else if is_question {
            EpisodeCategory::Question
        } else if social_context == SocialContext::Greeting || social_context == SocialContext::Farewell {
            EpisodeCategory::Social
        } else if social_context == SocialContext::Thanks {
            EpisodeCategory::Social
        } else if social_context == SocialContext::Affection {
            EpisodeCategory::Emotional
        } else {
            EpisodeCategory::General
        };

        // Check for praise/correction feedback
        let feedback_words = ["bravo", "bien", "super", "génial", "parfait", "good", "great", "yes", "perfect"];
        let correction_words = ["non", "pas", "mauvais", "faux", "arrête", "no", "wrong", "bad", "stop"];

        let lower_input = input.to_lowercase();
        let final_category = if feedback_words.iter().any(|w| lower_input.contains(w)) {
            EpisodeCategory::Praise
        } else if correction_words.iter().any(|w| lower_input.contains(w)) {
            EpisodeCategory::Correction
        } else {
            category
        };

        // Extract keywords (significant words)
        let keywords: Vec<String> = input
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.to_lowercase().as_str()))
            .map(|w| w.to_lowercase())
            .collect();

        // Get current emotional state
        let emotional = self.emotional_state.read();
        let emotion = EpisodeEmotion {
            happiness: emotional.happiness,
            arousal: emotional.arousal,
            comfort: emotional.comfort,
            curiosity: emotional.curiosity,
        };
        drop(emotional);

        // Record the episode
        let mut memory = self.memory.write();
        memory.record_episode(
            input,
            None, // Response will be added later if we have one
            keywords,
            emotion,
            importance,
            final_category,
            current_tick,
        );
    }

    /// One tick of life
    pub fn tick(&mut self) -> Vec<OldSignal> {
        let current_tick = self.tick.fetch_add(1, Ordering::SeqCst);

        // Get signals from buffer
        let signals: Vec<SignalFragment> = {
            let mut buffer = self.signal_buffer.write();
            std::mem::take(&mut *buffer)
        };

        // Process cells using backend
        let actions = self.backend.update_cells(
            &mut self.cells,
            &mut self.states,
            &self.dna_pool,
            &signals,
        ).unwrap_or_default();

        // Handle actions
        let mut births = Vec::new();
        let mut deaths = Vec::new();

        for (cell_id, action) in actions {
            match action {
                CellAction::Die => {
                    deaths.push(cell_id);
                }
                CellAction::Divide => {
                    // Find parent
                    if let Some(parent_idx) = self.cells.iter().position(|c| c.id == cell_id) {
                        let parent = &self.cells[parent_idx];
                        let parent_state = &self.states[parent_idx];

                        if parent_state.energy > self.config.metabolism.reproduction_threshold {
                            let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);

                            // Create child DNA (mutated from parent)
                            let parent_dna = &self.dna_pool[parent.dna_index as usize];
                            let child_dna = DNA::from_parent(parent_dna, self.config.population.mutation_rate);
                            let child_dna_index = self.dna_pool.len() as u32;
                            self.dna_pool.push(child_dna);

                            // Create child
                            let child = Cell::from_parent(new_id, parent, child_dna_index);
                            let child_state = CellState::from_parent(parent_state);

                            births.push((child, child_state));
                        }
                    }
                }
                CellAction::Move(direction) => {
                    if let Some(idx) = self.cells.iter().position(|c| c.id == cell_id) {
                        for (i, d) in direction.iter().enumerate() {
                            self.states[idx].position[i] = (self.states[idx].position[i] + d).clamp(-10.0, 10.0);
                        }
                    }
                }
                CellAction::Signal(content) => {
                    // Internal signal emission
                    if let Some(idx) = self.cells.iter().position(|c| c.id == cell_id) {
                        let position = self.states[idx].position;
                        self.propagate_signal(content, position, 0.5);
                    }
                }
                _ => {}
            }
        }

        // Remove dead cells
        for id in deaths {
            if let Some(idx) = self.cells.iter().position(|c| c.id == id) {
                self.free_slots.push(idx);
                self.states[idx].set_dead();
            }
        }

        // Add new cells
        for (cell, state) in births {
            if let Some(idx) = self.free_slots.pop() {
                self.cells[idx] = cell;
                self.states[idx] = state;
            } else if self.cells.len() < (self.config.population.target_population + self.config.population.population_buffer) as usize {
                self.cells.push(cell);
                self.states.push(state);
            }
        }

        // Natural selection
        if current_tick % self.config.population.selection_interval as u64 == 0 {
            self.natural_selection();
        }

        // Detect emergence
        let mut emergent = self.detect_emergence(current_tick);

        // Spontaneous speech
        if let Some(spontaneous) = self.maybe_speak_spontaneously(current_tick) {
            emergent.push(spontaneous);
        }

        // Dream/consolidate
        self.maybe_dream(current_tick);

        // ADAPTIVE: Periodic exploration of parameters (every ~10 seconds)
        if current_tick % 5000 == 0 {
            {
                let mut params = self.adaptive_params.write();
                params.explore();
                tracing::debug!("🧬 EXPLORE: {}", params.summary());
            }
            // Sync to memory after exploration
            self.sync_adaptive_params_to_memory();
        }

        emergent
    }

    /// Get substrate statistics
    pub fn stats(&self) -> SubstrateStats {
        let current_tick = self.tick.load(Ordering::Relaxed);
        let emotional = self.emotional_state.read();

        let alive_count = self.cells.iter()
            .zip(self.states.iter())
            .filter(|(_, s)| !s.is_dead())
            .count();

        let sleeping_count = self.cells.iter()
            .filter(|c| c.activity.sleeping)
            .count();

        let total_energy: f32 = self.states.iter()
            .filter(|s| !s.is_dead())
            .map(|s| s.energy)
            .sum();

        let oldest_age = self.cells.iter()
            .map(|c| c.age)
            .max()
            .unwrap_or(0);

        let cpu_savings = if alive_count > 0 {
            sleeping_count as f32 / alive_count as f32 * 100.0
        } else {
            0.0
        };

        // Get adaptive params
        let params = self.adaptive_params.read();

        SubstrateStats {
            tick: current_tick,
            alive_cells: alive_count,
            total_energy,
            entropy: self.calculate_entropy(),
            active_clusters: 0, // TODO
            dominant_emotion: emotional.mood_description().to_string(),
            signals_per_second: 0.0, // TODO
            oldest_cell_age: oldest_age,
            average_connections: 0.0, // TODO
            mood: emotional.mood_description().to_string(),
            happiness: emotional.happiness,
            arousal: emotional.arousal,
            curiosity: emotional.curiosity,
            sleeping_cells: sleeping_count,
            cpu_savings_percent: cpu_savings,
            backend_name: self.backend.name().to_string(),
            // Adaptive params
            adaptive_emission_threshold: params.emission_threshold,
            adaptive_response_probability: params.response_probability,
            adaptive_spontaneity: params.spontaneity,
            adaptive_feedback_positive: params.positive_count,
            adaptive_feedback_negative: params.negative_count,
        }
    }

    // === Internal Methods ===

    fn process_feedback(&self, label: &str, _current_tick: u64) {
        // Feedback should be short, dedicated messages - not part of conversation
        // If the message is too long (> 15 chars), it's probably not feedback
        if label.len() > 15 {
            return;
        }

        let lower_label = label.to_lowercase().trim().to_string();

        // Exact match or starts with feedback words (allow "Bravo!" but not "C'est bien fait")
        let positive_feedback = [
            "bravo", "bien!", "super", "génial", "parfait", "excellent",
            "good", "great", "yes", "perfect", "awesome", "👏", "👍", "oui"
        ];

        let negative_feedback = [
            "non", "mauvais", "faux", "arrête", "stop",
            "no", "wrong", "bad", "👎"
        ];

        // Check if the message IS the feedback (not contains it)
        let is_positive = positive_feedback.iter().any(|w|
            lower_label == *w || lower_label.starts_with(&format!("{} ", w)) || lower_label.starts_with(&format!("{}!", w))
        );
        let is_negative = negative_feedback.iter().any(|w|
            lower_label == *w || lower_label.starts_with(&format!("{} ", w)) || lower_label.starts_with(&format!("{}!", w))
        );

        if !is_positive && !is_negative {
            return;
        }

        // Step 1: Get recent expressions (release lock immediately)
        let recent_expr: Vec<String> = self.recent_expressions.read().clone();

        // Step 2: Update word valences in memory (separate scope)
        {
            let mut memory = self.memory.write();
            for word in &recent_expr {
                if let Some(freq) = memory.word_frequencies.get_mut(word) {
                    let old = freq.emotional_valence;
                    if is_positive {
                        freq.emotional_valence = (freq.emotional_valence + 0.3).clamp(-2.0, 2.0);
                        freq.count += 2;
                        tracing::info!("FEEDBACK POSITIVE! '{}' ({:.2} → {:.2})", word, old, freq.emotional_valence);
                    } else {
                        freq.emotional_valence = (freq.emotional_valence - 0.3).clamp(-2.0, 2.0);
                        tracing::info!("FEEDBACK NEGATIVE! '{}' ({:.2} → {:.2})", word, old, freq.emotional_valence);
                    }
                }
            }
        } // memory lock released here

        // Step 3: Update emotional state (separate scope)
        {
            let mut emotional = self.emotional_state.write();
            if is_positive {
                emotional.happiness = (emotional.happiness + 0.3).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort + 0.2).clamp(-1.0, 1.0);
            } else {
                emotional.happiness = (emotional.happiness - 0.2).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort - 0.1).clamp(-1.0, 1.0);
            }
        } // emotional lock released here

        // Step 4: Update adaptive params (separate scope)
        {
            let mut params = self.adaptive_params.write();
            if is_positive {
                params.reinforce_positive();
            } else {
                params.reinforce_negative();
            }
        } // params lock released here
    }

    /// Sync current adaptive params to long-term memory
    fn sync_adaptive_params_to_memory(&self) {
        let params = self.adaptive_params.read();
        let mut mem = self.memory.write();

        mem.adaptive_emission_threshold = params.emission_threshold;
        mem.adaptive_response_probability = params.response_probability;
        mem.adaptive_learning_rate = params.learning_rate;
        mem.adaptive_spontaneity = params.spontaneity;
        mem.adaptive_feedback_positive = params.positive_count;
        mem.adaptive_feedback_negative = params.negative_count;
    }

    fn detect_emergence(&self, current_tick: u64) -> Vec<OldSignal> {
        if current_tick % 5 != 0 {
            return Vec::new();
        }

        // Anti-spam: cooldown between emissions
        let last_emit = self.last_emission_tick.load(Ordering::Relaxed);
        if current_tick.saturating_sub(last_emit) < EMISSION_COOLDOWN_TICKS {
            return Vec::new();
        }

        // Find active cells
        let active_states: Vec<(usize, f32)> = self.states.iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let activation: f32 = s.state.iter().map(|x| x.abs()).sum();
                if activation > self.config.emergence.activation_threshold {
                    Some((i, activation))
                } else {
                    None
                }
            })
            .take(1000)
            .collect();

        if active_states.is_empty() {
            return Vec::new();
        }

        // Calculate average state
        let mut average_state = [0.0f32; SIGNAL_DIMS];
        for (i, _) in &active_states {
            for (j, s) in self.states[*i].state[0..SIGNAL_DIMS].iter().enumerate() {
                average_state[j] += s;
            }
        }
        let n = active_states.len() as f32;
        for a in &mut average_state {
            *a /= n;
        }

        // Check coherence
        let coherence = self.calculate_coherence(&active_states);

        // ADAPTIVE: Use adaptive emission threshold instead of fixed config
        let params = self.adaptive_params.read();
        let emission_threshold = params.emission_threshold;
        let response_probability = params.response_probability;
        drop(params);

        if coherence > emission_threshold {
            // ADAPTIVE: Sometimes choose not to respond (based on response_probability)
            let mut rng = rand::thread_rng();
            if rng.gen::<f32>() > response_probability {
                return Vec::new();  // ARIA chose to stay silent
            }

            // Get context
            let recent_said = self.recent_said_words.read().clone();
            let was_question = *self.last_was_question.read();

            // Try to find a matching word from what she learned
            if let Some(response) = self.generate_word_response(&average_state, coherence, was_question, &recent_said) {
                self.last_emission_tick.store(current_tick, Ordering::Relaxed);
                return vec![response];
            }

            // Try to recall a relevant memory
            if let Some(response) = self.maybe_recall_memory(&average_state, coherence, current_tick) {
                self.last_emission_tick.store(current_tick, Ordering::Relaxed);
                return vec![response];
            }

            // Fallback: babble based on emotional state (she's trying to communicate!)
            let emotional = self.emotional_state.read();
            let babble = self.generate_babble(&average_state, coherence, &emotional);
            self.last_emission_tick.store(current_tick, Ordering::Relaxed);
            return vec![babble];
        }

        Vec::new()
    }

    /// Generate a simple babble when ARIA doesn't know what to say
    /// This is her primitive way of communicating before she learns words
    fn generate_babble(&self, state: &[f32; SIGNAL_DIMS], coherence: f32, emotional: &EmotionalState) -> OldSignal {
        let mut rng = rand::thread_rng();

        // Emotional markers
        let marker = if emotional.happiness > 0.3 {
            "~"
        } else if emotional.curiosity > 0.3 {
            "?"
        } else if emotional.happiness < -0.2 {
            "..."
        } else {
            ""
        };

        // Simple syllables based on coherence (more coherent = more complex)
        let syllable = if coherence > 0.5 {
            // Higher coherence: proto-words
            let proto_words = ["ma", "pa", "da", "na", "ba", "la", "ta", "ka"];
            proto_words[rng.gen_range(0..proto_words.len())]
        } else if coherence > 0.3 {
            // Medium: simple syllables
            let syllables = ["a", "o", "e", "i", "u", "é", "è"];
            syllables[rng.gen_range(0..syllables.len())]
        } else {
            // Low: just sounds
            let sounds = ["mm", "hm", "ah"];
            sounds[rng.gen_range(0..sounds.len())]
        };

        let label = format!("babble:{}|emotion:{}", syllable, marker);
        let mut signal = OldSignal::from_vector(*state, label);
        signal.intensity = coherence.max(0.2);
        signal
    }

    /// Try to recall a relevant episodic memory
    fn maybe_recall_memory(&self, state: &[f32; SIGNAL_DIMS], coherence: f32, current_tick: u64) -> Option<OldSignal> {
        // Only sometimes try to recall (10% chance when coherence is high)
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > 0.1 || coherence < 0.3 {
            return None;
        }

        // Get context words from recent conversation
        let context_words: Vec<String> = {
            let conv = self.conversation.read();
            conv.get_topic_words()
        };

        if context_words.is_empty() {
            return None;
        }

        // Try to find a relevant episode
        let mut memory = self.memory.write();
        let episodes = memory.recall_episodes(&context_words, current_tick, 3);

        if episodes.is_empty() {
            return None;
        }

        // Pick the most relevant episode
        let episode = episodes[0];

        // Check if it's important enough to mention
        if episode.importance < 0.4 {
            return None;
        }

        // Generate a memory-based response
        let (label, intensity) = if episode.first_of_kind.is_some() {
            // First time memory - special!
            let kind = episode.first_of_kind.as_ref().unwrap();
            let keyword = episode.keywords.first().map(|s| s.as_str()).unwrap_or("ça");
            tracing::info!("🌟 RECALLING FIRST TIME: {} - \"{}\"", kind, episode.input);
            (format!("memory:first|{}|{}", kind, keyword), 0.7)
        } else if episode.category == EpisodeCategory::Emotional {
            // Emotional memory
            let keyword = episode.keywords.first().map(|s| s.as_str()).unwrap_or("moment");
            tracing::info!("💭 RECALLING EMOTION: \"{}\"", episode.input);
            (format!("memory:emotion|{}", keyword), 0.6)
        } else {
            // General memory
            let keyword = episode.keywords.first().map(|s| s.as_str()).unwrap_or("souviens");
            tracing::info!("💭 RECALLING: \"{}\"", episode.input);
            (format!("memory:recall|{}", keyword), 0.5)
        };

        let mut signal = OldSignal::from_vector(*state, label);
        signal.intensity = intensity * coherence;

        Some(signal)
    }

    fn generate_social_response(&self, context: SocialContext, is_start: bool, state: &[f32; SIGNAL_DIMS], coherence: f32) -> Option<OldSignal> {
        let should_respond = match context {
            SocialContext::Greeting => true,
            SocialContext::Farewell => is_start,
            SocialContext::Thanks | SocialContext::Affection => true,
            _ => false,
        };

        if !should_respond || context == SocialContext::General {
            return None;
        }

        let memory = self.memory.read();
        let response_word = match context {
            SocialContext::Greeting => {
                memory.get_response_for_context(SocialContext::Greeting)
                    .unwrap_or_else(|| "bonjour".to_string())
            }
            SocialContext::Farewell => {
                memory.get_response_for_context(SocialContext::Farewell)
                    .unwrap_or_else(|| "bye".to_string())
            }
            SocialContext::Thanks => "derien".to_string(),
            SocialContext::Affection => {
                memory.get_response_for_context(SocialContext::Affection)
                    .unwrap_or_else(|| "aime".to_string())
            }
            _ => return None,
        };

        let marker = match context {
            SocialContext::Affection => "♥",
            SocialContext::Greeting => "~",
            _ => "~",
        };

        let label = format!("social:{:?}:{}|emotion:{}", context, response_word, marker).to_lowercase();

        tracing::info!("SOCIAL RESPONSE: {:?} -> {}", context, label);

        // Record what we said (keep last 5 for diversity)
        {
            let mut recent = self.recent_said_words.write();
            recent.push(response_word.clone());
            if recent.len() > 5 {
                recent.remove(0);
            }
        }
        self.conversation.write().add_response(&label);

        let mut signal = OldSignal::from_vector(*state, label);
        signal.intensity = coherence.max(0.4);

        Some(signal)
    }

    fn generate_word_response(&self, state: &[f32; SIGNAL_DIMS], coherence: f32, was_question: bool, recent_said: &[String]) -> Option<OldSignal> {
        let recent = self.recent_words.read();
        let memory = self.memory.read();
        let emotional = self.emotional_state.read();
        let mut rng = rand::thread_rng();

        // Get context words from conversation for boosting
        let context_words: Vec<String> = {
            let conv = self.conversation.read();
            conv.get_topic_words()
        };

        // Get cluster-related words for semantic coherence
        let cluster_words: Vec<(String, f32)> = memory.get_related_words_from_input(&context_words);

        // Helper to check if word was recently said
        let was_recently_said = |word: &str| -> bool {
            recent_said.iter().any(|w| w.to_lowercase() == word.to_lowercase())
        };

        // Helper to check if word is in current context (deserves boost)
        let is_context_word = |word: &str| -> bool {
            context_words.iter().any(|w| w.to_lowercase() == word.to_lowercase())
        };

        // Helper to check if word is in same semantic cluster (deserves boost)
        let cluster_boost = |word: &str| -> f32 {
            cluster_words.iter()
                .find(|(w, _)| w.to_lowercase() == word.to_lowercase())
                .map(|(_, strength)| strength * 0.3) // 30% boost per cluster match
                .unwrap_or(0.0)
        };

        // Collect candidate words with their scores
        let mut candidates: Vec<(String, f32, f32)> = Vec::new(); // (word, similarity, valence)

        // From recent words (most relevant - just heard)
        for rw in recent.iter() {
            if was_recently_said(&rw.word) {
                continue;
            }

            let mut similarity = Self::vector_similarity(state, &rw.vector);
            // Boost context words significantly
            if is_context_word(&rw.word) {
                similarity = (similarity * 1.5).min(1.0);
            }
            // Boost cluster-related words (semantic coherence!)
            similarity = (similarity + cluster_boost(&rw.word)).min(1.0);

            if similarity > 0.35 {
                let valence = memory.word_frequencies.get(&rw.word)
                    .map(|f| f.emotional_valence)
                    .unwrap_or(0.0);
                candidates.push((rw.word.clone(), similarity, valence));
            }
        }

        // From learned words (memory) - only if not enough recent candidates
        if candidates.len() < 3 {
            for (word, freq) in memory.word_frequencies.iter() {
                if was_recently_said(word) {
                    continue;
                }
                if candidates.iter().any(|(w, _, _)| w == word) {
                    continue;
                }
                let mut similarity = Self::vector_similarity(state, &freq.learned_vector);
                if is_context_word(word) {
                    similarity = (similarity * 1.5).min(1.0);
                }
                similarity = (similarity + cluster_boost(word)).min(1.0);

                if similarity > 0.35 {
                    candidates.push((word.clone(), similarity, freq.emotional_valence));
                }
            }
        }

        // Weighted random selection (similarity^3 for strong bias toward best matches)
        let chosen = if !candidates.is_empty() {
            let total_weight: f32 = candidates.iter().map(|(_, s, _)| s * s * s).sum();
            if total_weight > 0.0 {
                let mut pick = rng.gen::<f32>() * total_weight;
                let mut selected = &candidates[0];
                for candidate in &candidates {
                    pick -= candidate.1 * candidate.1 * candidate.1;
                    if pick <= 0.0 {
                        selected = candidate;
                        break;
                    }
                }
                Some(selected.clone())
            } else {
                candidates.first().cloned()
            }
        } else {
            None
        };

        let best_word = chosen;

        if let Some((word, similarity, valence)) = best_word {
            // Record expression
            {
                let mut expr = self.recent_expressions.write();
                expr.push(word.clone());
                if expr.len() > 5 {
                    expr.remove(0);
                }
            }

            // Build label
            // For questions: only add oui/non if the word is relevant (high similarity)
            // Otherwise, just respond with the word and a question mark
            let label = if was_question {
                if similarity > 0.5 {
                    // Word is clearly related to the question
                    if valence > 0.3 {
                        format!("answer:oui+{}", word)
                    } else if valence < -0.3 {
                        format!("answer:non+{}", word)
                    } else {
                        format!("word:{}?", word)
                    }
                } else {
                    // Word is not strongly related - just respond with it
                    format!("word:{}", word)
                }
            } else {
                format!("word:{}", word)
            };

            // Add emotional marker
            let marker = emotional.get_emotional_marker().unwrap_or("");
            let final_label = if !marker.is_empty() {
                format!("{}|emotion:{}", label, marker)
            } else {
                label
            };

            tracing::info!("EMERGENCE: '{}' (similarity={:.2}, valence={:.2})",
                word, similarity, valence);

            // Record (keep last 5 for diversity)
            {
                let mut recent = self.recent_said_words.write();
                recent.push(word.clone());
                if recent.len() > 5 {
                    recent.remove(0);
                }
            }
            self.conversation.write().add_response(&final_label);

            let mut signal = OldSignal::from_vector(*state, final_label);
            signal.intensity = coherence;

            return Some(signal);
        }

        None
    }

    fn maybe_speak_spontaneously(&self, current_tick: u64) -> Option<OldSignal> {
        if current_tick % 100 != 0 {
            return None;
        }

        // Anti-spam: respect cooldown
        let last_emit = self.last_emission_tick.load(Ordering::Relaxed);
        if current_tick.saturating_sub(last_emit) < EMISSION_COOLDOWN_TICKS {
            return None;
        }

        let last_interaction = self.last_interaction_tick.load(Ordering::Relaxed);
        let ticks_since = current_tick.saturating_sub(last_interaction);

        let emotional = self.emotional_state.read();

        let is_lonely = ticks_since > 3000;
        let is_bored = emotional.boredom > 0.5;
        let is_excited = emotional.arousal > 0.6;
        let is_happy = emotional.happiness > 0.5;
        let is_curious = emotional.curiosity > 0.5;

        // ADAPTIVE: Get spontaneity parameter
        let spontaneity = self.adaptive_params.read().spontaneity;

        let mut rng = rand::thread_rng();
        let random: f32 = rng.gen();

        // Base probability modified by spontaneity parameter
        let base_prob = if is_lonely { 0.05 }
            else if is_bored { 0.04 }
            else if is_excited && is_happy { 0.03 }
            else if is_excited { 0.02 }
            else if is_curious { 0.01 }
            else { 0.001 };

        // ADAPTIVE: Multiply by spontaneity (0.01 to 0.3 range means 1% to 30% of base)
        let probability = base_prob * (spontaneity * 10.0);  // spontaneity=0.1 → same as before

        if random > probability {
            return None;
        }

        let memory = self.memory.read();

        let favorite_word = memory.word_frequencies.iter()
            .filter(|(_, freq)| freq.emotional_valence > 0.5 && freq.count > 3)
            .max_by(|(_, a), (_, b)| {
                let score_a = a.emotional_valence * (a.count as f32).sqrt();
                let score_b = b.emotional_valence * (b.count as f32).sqrt();
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(word, _)| word.clone());

        let (label, intensity) = if is_lonely {
            if let Some(word) = favorite_word {
                tracing::info!("SPONTANEOUS (lonely): thinking of '{}'", word);
                (format!("spontaneous:{}|emotion:?", word), 0.3)
            } else {
                ("spontaneous:attention|emotion:?".to_string(), 0.2)
            }
        } else if is_bored {
            let all_favorites: Vec<String> = memory.word_frequencies.iter()
                .filter(|(_, freq)| freq.emotional_valence > 0.2 && freq.count > 1)
                .map(|(word, _)| word.clone())
                .collect();

            if all_favorites.len() >= 2 {
                let w1 = &all_favorites[rng.gen_range(0..all_favorites.len())];
                let w2 = &all_favorites[rng.gen_range(0..all_favorites.len())];
                if w1 != w2 {
                    tracing::info!("SPONTANEOUS (bored): combining '{}' + '{}'", w1, w2);
                    (format!("phrase:{}+{}|emotion:~", w1, w2), 0.35)
                } else {
                    ("spontaneous:bored|emotion:~".to_string(), 0.25)
                }
            } else {
                ("spontaneous:bored|emotion:~".to_string(), 0.25)
            }
        } else if is_happy {
            if let Some(word) = favorite_word {
                tracing::info!("SPONTANEOUS (happy): expressing '{}' ♥", word);
                (format!("spontaneous:{}|emotion:♥", word), 0.5)
            } else {
                ("spontaneous:joy|emotion:♥".to_string(), 0.4)
            }
        } else if is_excited {
            ("spontaneous:excited|emotion:!".to_string(), 0.4)
        } else if is_curious {
            ("spontaneous:curious|emotion:?".to_string(), 0.3)
        } else {
            ("spontaneous:babble|emotion:~".to_string(), 0.2)
        };

        let mut signal = OldSignal::from_vector([0.0; SIGNAL_DIMS], label);
        signal.intensity = intensity;

        self.last_emission_tick.store(current_tick, Ordering::Relaxed);
        Some(signal)
    }

    fn maybe_dream(&self, current_tick: u64) {
        if current_tick % 500 != 0 {
            return;
        }

        let last_interaction = self.last_interaction_tick.load(Ordering::Relaxed);
        if current_tick.saturating_sub(last_interaction) < 1000 {
            return;
        }

        let mut memory = self.memory.write();
        let mut rng = rand::thread_rng();

        let favorites: Vec<String> = memory.word_frequencies.iter()
            .filter(|(_, freq)| freq.emotional_valence > 0.3 && freq.count > 2)
            .map(|(word, _)| word.clone())
            .collect();

        if favorites.is_empty() {
            return;
        }

        let dream_word = &favorites[rng.gen_range(0..favorites.len())];

        if let Some(freq) = memory.word_frequencies.get_mut(dream_word) {
            freq.count += 1;
            if freq.emotional_valence > 0.0 {
                freq.emotional_valence = (freq.emotional_valence + 0.05).min(2.0);
            }
        }

        if rng.gen::<f32>() < 0.1 {
            tracing::info!("💭 DREAMING: thinking about '{}'...", dream_word);
        }
    }

    fn natural_selection(&mut self) {
        let target = self.config.population.target_population as usize;
        let buffer = self.config.population.population_buffer as usize;
        let min_pop = self.config.population.min_population as usize;
        let alive_count = self.cells.iter()
            .zip(self.states.iter())
            .filter(|(_, s)| !s.is_dead())
            .count();

        if alive_count > target + buffer {
            // Too many cells - remove weakest
            let mut indices: Vec<(usize, f32)> = self.states.iter()
                .enumerate()
                .filter(|(_, s)| !s.is_dead())
                .map(|(i, s)| (i, s.energy))
                .collect();

            indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_remove = alive_count - target;
            for (idx, _) in indices.iter().take(to_remove) {
                self.states[*idx].set_dead();
                self.free_slots.push(*idx);
            }
        } else if alive_count < min_pop {
            // Too few cells - spawn new ones
            let to_spawn = min_pop - alive_count;
            for _ in 0..to_spawn {
                let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                let dna = DNA::random();
                let dna_index = self.dna_pool.len() as u32;
                self.dna_pool.push(dna);

                let cell = Cell::new(new_id, dna_index);
                let state = CellState::new();

                if let Some(idx) = self.free_slots.pop() {
                    self.cells[idx] = cell;
                    self.states[idx] = state;
                } else {
                    self.cells.push(cell);
                    self.states.push(state);
                }
            }
        }
    }

    fn propagate_signal(&mut self, content: [f32; SIGNAL_DIMS], source_pos: [f32; POSITION_DIMS], intensity: f32) {
        let fragment = SignalFragment::external(content, intensity);

        for state in &mut self.states {
            if state.is_dead() { continue; }

            let distance = Self::semantic_distance(&state.position, &source_pos);
            if distance < 2.0 {
                let attenuation = 1.0 / (1.0 + distance);
                for (i, s) in fragment.content.iter().enumerate() {
                    if i < SIGNAL_DIMS {
                        state.state[i] += s * intensity * attenuation;
                    }
                }
            }
        }
    }

    fn calculate_coherence(&self, active_states: &[(usize, f32)]) -> f32 {
        if active_states.len() < 2 {
            return 0.0;
        }

        let mut total_similarity = 0.0f32;
        let mut count = 0;

        for i in 0..active_states.len().min(10) {
            for j in (i + 1)..active_states.len().min(10) {
                let s1 = &self.states[active_states[i].0];
                let s2 = &self.states[active_states[j].0];

                let mut dot = 0.0f32;
                let mut norm1 = 0.0f32;
                let mut norm2 = 0.0f32;

                for k in 0..SIGNAL_DIMS {
                    dot += s1.state[k] * s2.state[k];
                    norm1 += s1.state[k] * s1.state[k];
                    norm2 += s2.state[k] * s2.state[k];
                }

                let denom = (norm1 * norm2).sqrt();
                if denom > 0.0 {
                    total_similarity += dot / denom;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_similarity / count as f32
        } else {
            0.0
        }
    }

    fn calculate_entropy(&self) -> f32 {
        let active: Vec<f32> = self.states.iter()
            .filter(|s| !s.is_dead())
            .map(|s| s.state.iter().map(|x| x.abs()).sum::<f32>())
            .collect();

        if active.is_empty() {
            return 0.0;
        }

        let total: f32 = active.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0f32;
        for a in &active {
            let p = a / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        entropy / (active.len() as f32).ln().max(1.0)
    }

    fn semantic_distance(a: &[f32; POSITION_DIMS], b: &[f32; POSITION_DIMS]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn vector_similarity(a: &[f32; SIGNAL_DIMS], b: &[f32; SIGNAL_DIMS]) -> f32 {
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom > 0.0 {
            dot / denom
        } else {
            0.0
        }
    }
}

// Bridge type for DNA conversion
impl From<crate::cell::DNA> for DNA {
    fn from(old: crate::cell::DNA) -> Self {
        // Use the old thresholds/reactions to create a similar DNA
        let mut dna = DNA::random();
        // Copy over the values (the old DNA has same layout but is local type)
        for (i, t) in old.thresholds.iter().enumerate() {
            dna.thresholds[i] = *t;
        }
        for (i, r) in old.reactions.iter().enumerate() {
            dna.reactions[i] = *r;
        }
        dna
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_state_decay() {
        let mut state = EmotionalState {
            happiness: 1.0,
            arousal: 1.0,
            comfort: 1.0,
            curiosity: 1.0,
            boredom: 0.0,
            last_update: 0,
        };

        state.decay(1000);

        assert!(state.happiness < 1.0);
        assert!(state.boredom > 0.0);
    }

    #[test]
    fn test_conversation_context() {
        let mut ctx = ConversationContext::new();

        ctx.add_input("hello world", vec!["hello".into(), "world".into()], 0.5, 100, SocialContext::Greeting);

        assert!(ctx.is_conversation_start());
        assert_eq!(ctx.get_social_context(), SocialContext::Greeting);

        let words = ctx.get_context_words();
        assert!(!words.is_empty());
    }
}
