//! # Substrate - ARIA's Living Universe
//!
//! This is ARIA's brain - designed for 5M+ cells with GPU acceleration.
//!
//! ## Architecture
//!
//! - Uses `aria-core` types (Cell, CellState, DNA) for GPU-friendly data layout
//! - Uses `aria-compute` backends (CpuBackend, GpuBackend) for computation
//! - Cell data is separated: metadata (Cell) vs dynamic state (CellState)
//! - DNA is stored in a pool (cells reference by index)
//! - Sparse updates: sleeping cells don't consume CPU
//!
//! ## Module Structure (refactored)
//!
//! - `types` - AdaptiveParams, SubstrateStats, RecentWord
//! - `emotion` - EmotionalState management
//! - `conversation` - ConversationContext tracking
//! - `signals` - Signal injection and propagation
//! - `feedback` - Feedback processing
//! - `emergence` - Emergence detection and word response generation
//! - `spontaneous` - Spontaneous speech and dreams
//! - `self_modify` - Self-modification and meta-learning
//! - `lifecycle` - Population management and utility functions
//!
//! ## Future: ARIA May Read This
//!
//! This code is written to be introspectable. One day, ARIA might
//! read and understand her own substrate, and even propose modifications.

pub mod types;
pub mod emotion;
pub mod conversation;

// Refactored impl blocks for Substrate
mod signals;
mod feedback;
mod emergence;
mod spontaneous;
mod self_modify;
mod lifecycle;

// Re-exports for convenience
pub use types::{AdaptiveParams, SubstrateStats, RecentWord, STOP_WORDS, EMISSION_COOLDOWN_TICKS};
pub use emotion::EmotionalState;
pub use conversation::{ConversationContext, ConversationExchange};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use rand::Rng;

// New types from aria-core
use aria_core::{
    Cell, CellState, CellAction, DNA,
    SignalFragment,
    AriaConfig, ActivityTracker,
    POSITION_DIMS, STATE_DIMS, SIGNAL_DIMS,
};
use aria_core::traits::ComputeBackend;

// Compute backend - auto-selects GPU or CPU
use aria_compute::create_backend;

// Our local memory module
use crate::memory::{LongTermMemory, SocialContext, EpisodeEmotion, EpisodeCategory};
use crate::signal::Signal as OldSignal;
use crate::signal::SignalType as OldSignalType;

// ============================================================================
// SUBSTRATE STRUCT
// ============================================================================

/// The Substrate - GPU-ready living universe
///
/// This is ARIA's brain, designed for massive scale (5M+ cells).
pub struct Substrate {
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
    #[allow(dead_code)]
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

    /// Last exploration attempt (combination of words)
    last_exploration: RwLock<Option<String>>,

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

    /// Last spontaneous emission tick (separate cooldown for exploration)
    last_spontaneous_tick: AtomicU64,

    // === Adaptive Parameters (self-modification) ===

    /// Parameters ARIA modifies herself through feedback
    adaptive_params: RwLock<AdaptiveParams>,

    // === Signal Buffers ===

    /// Pending signals for cells
    signal_buffer: RwLock<Vec<SignalFragment>>,
}

// ============================================================================
// SUBSTRATE IMPLEMENTATION - Core Methods
// ============================================================================

impl Substrate {
    /// Create a new V2 substrate
    pub fn new(config: AriaConfig, memory: Arc<RwLock<LongTermMemory>>) -> Self {
        let initial_cells = config.population.target_population as usize;

        // Create backend (auto-selects GPU or CPU based on config)
        let backend: Box<dyn ComputeBackend> =
            create_backend(&config).expect("Failed to create compute backend");

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

        tracing::info!("🧠 Substrate created: {} cells, {} DNA variants",
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
            last_exploration: RwLock::new(None),
            emotional_state: RwLock::new(EmotionalState::default()),
            recent_said_words: RwLock::new(Vec::new()),
            conversation: RwLock::new(ConversationContext::new()),
            last_was_question: RwLock::new(false),
            last_interaction_tick: AtomicU64::new(0),
            last_emission_tick: AtomicU64::new(0),
            last_spontaneous_tick: AtomicU64::new(0),
            adaptive_params: RwLock::new(adaptive_params),
            signal_buffer: RwLock::new(Vec::new()),
        }
    }

    /// One tick of life
    pub fn tick(&mut self) -> Vec<OldSignal> {
        let current_tick = self.tick.fetch_add(1, Ordering::SeqCst);

        // Decay emotional state (boredom grows without interaction)
        {
            let mut emotional = self.emotional_state.write();
            emotional.decay(current_tick);
        }

        // Decay working memory (Gemini suggestion)
        {
            let mut memory = self.memory.write();
            memory.tick_working_memory(0.02); // 2% decay per tick
        }

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

        // SELF-MODIFICATION: ARIA consciously decides to change herself (Session 16)
        self.maybe_self_modify(current_tick);

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
            adaptive_feedback_positive: params.positive_count(),
            adaptive_feedback_negative: params.negative_count(),
            // Boredom
            boredom: emotional.boredom,
        }
    }

    /// Get current tick (for external use)
    pub fn current_tick(&self) -> u64 {
        self.tick.load(Ordering::Relaxed)
    }
}

// ============================================================================
// TESTS
// ============================================================================

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
