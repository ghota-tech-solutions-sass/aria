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
//! - `types` - AdaptiveParams, SubstrateStats, SpatialInhibitor
//! - `emotion` - EmotionalState management
//! - `conversation` - ConversationContext (legacy, kept for reference)
//! - `signals` - Signal injection and tension propagation
//! - `feedback` - Feedback processing (emotional state updates)
//! - `emergence` - Emergence detection and tension response generation
//! - `spontaneous` - Spontaneous tension pulses and dreams
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
mod shadow_brain;

// Re-exports for convenience
// NOTE: RecentWord, STOP_WORDS, ConversationContext removed in Session 20 (Physical Intelligence)
pub use types::{AdaptiveParams, SubstrateStats, SpatialInhibitor, EMISSION_COOLDOWN_TICKS};
pub use emotion::EmotionalState;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use rand::Rng;

// New types from aria-core
use aria_core::{
    Cell, CellState, CellAction, DNA, MutationContext,
    SignalFragment,
    AriaConfig, ActivityTracker,
    POSITION_DIMS, SIGNAL_DIMS,
};
use aria_core::traits::ComputeBackend;
use aria_core::error::AriaResult;

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

    // NOTE: recent_words, recent_expressions, recent_said_words, conversation,
    // last_was_question removed in Session 20 (Physical Intelligence)
    // ARIA no longer tracks words - only tension patterns

    /// Last exploration attempt (combination of tensions)
    last_exploration: RwLock<Option<String>>,

    // === Emotional & Social ===

    /// Global emotional state
    emotional_state: RwLock<EmotionalState>,

    /// Last interaction tick (for spontaneity)
    last_interaction_tick: AtomicU64,

    /// Last emission tick (anti-spam cooldown)
    last_emission_tick: AtomicU64,

    /// Cells that participated in last emission (for feedback loop - Gemini suggestion)
    last_emission_cells: RwLock<Vec<usize>>,

    /// Last spontaneous emission tick (separate cooldown for exploration)
    last_spontaneous_tick: AtomicU64,

    // === Adaptive Parameters (self-modification) ===

    /// Parameters ARIA modifies herself through feedback
    adaptive_params: RwLock<AdaptiveParams>,

    // === Signal Buffers ===

    /// Pending signals for cells
    signal_buffer: RwLock<Vec<SignalFragment>>,

    // === Spatial Inhibition (Gemini optimization) ===

    /// Spatial inhibitor for adaptive regional thresholds
    spatial_inhibitor: RwLock<SpatialInhibitor>,

    // === Reflexivity (Axe 3 - Genesis) ===

    /// Last emergent tension (ARIA's internal thought state)
    last_emergent_tension: RwLock<[f32; SIGNAL_DIMS]>,

    /// Current structural checksum active in the backend
    structural_checksum: u64,

    /// Shadow brain for testing mutations before applying them
    shadow_brain: Option<shadow_brain::ShadowBrain>,
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
            config: config.clone(), // Use clone for first pass
            tick: AtomicU64::new(0),
            next_id: AtomicU64::new(initial_cells as u64),
            memory,
            last_exploration: RwLock::new(None),
            emotional_state: RwLock::new(EmotionalState::default()),
            last_interaction_tick: AtomicU64::new(0),
            last_emission_tick: AtomicU64::new(0),
            last_emission_cells: RwLock::new(Vec::new()),
            last_spontaneous_tick: AtomicU64::new(0),
            adaptive_params: RwLock::new(adaptive_params),
            signal_buffer: RwLock::new(Vec::new()),
            spatial_inhibitor: RwLock::new(SpatialInhibitor::default()),
            last_emergent_tension: RwLock::new([0.0f32; SIGNAL_DIMS]),
            structural_checksum: 0,
            shadow_brain: shadow_brain::ShadowBrain::new(&config).ok(),
        }
    }

    /// Check if the dominant structural DNA has changed and recompile if needed
    fn check_for_structural_evolution(&mut self) -> AriaResult<()> {
        if let Some(first_dna) = self.dna_pool.first() {
            let new_checksum = first_dna.structural_checksum;
            if new_checksum != self.structural_checksum {
                // Use shadow brain to evaluate if we have one
                let should_apply = if let Some(shadow) = &mut self.shadow_brain {
                    match shadow.evaluate_candidate(new_checksum) {
                        Ok(score) => {
                            if score > 0.8 {
                                tracing::info!("✅ SHADOW BRAIN: Candidate {} accepted (score: {:.2})", new_checksum, score);
                                {
                                    let mut mem = self.memory.write();
                                    let current_tick = self.tick.load(std::sync::atomic::Ordering::SeqCst);
                                    mem.save_elite_structural_code(new_checksum, score, current_tick);
                                }
                                true
                            } else {
                                tracing::warn!("❌ SHADOW BRAIN: Candidate {} rejected (score: {:.2})", new_checksum, score);
                                false
                            }
                        }
                        Err(e) => {
                            tracing::error!("❌ SHADOW BRAIN: Evaluation failed: {}", e);
                            false // Safety first: don't apply if evaluation fails
                        }
                    }
                } else {
                    // No shadow brain, apply anyway (legacy behavior)
                    true
                };

                if should_apply {
                    tracing::info!("🧬 STRUCTURAL EVOLUTION APPLIED: Checksum {} -> {}",
                        self.structural_checksum, new_checksum);
                    self.structural_checksum = new_checksum;
                    self.backend.recompile(new_checksum)?;
                }
            }
        }
        Ok(())
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

        // Check for structural evolution every 1000 ticks
        if current_tick % 1000 == 0 {
            let _ = self.check_for_structural_evolution();
        }

        // Decay spatial inhibition (Gemini optimization)
        {
            let mut inhibitor = self.spatial_inhibitor.write();
            inhibitor.decay(current_tick);
            // Sync base threshold with adaptive params
            let params = self.adaptive_params.read();
            inhibitor.set_base_threshold(params.emission_threshold);
        }

        // Get external signals from buffer
        let mut signals: Vec<SignalFragment> = {
            let mut buffer = self.signal_buffer.write();
            std::mem::take(&mut *buffer)
        };

        // === BACKGROUND NOISE: Prevent entropy=0 freeze ===
        // Inject random perturbations to keep the system alive
        // This acts like "neural noise" in biological brains
        // Also provides baseline energy to prevent mass starvation
        if current_tick % 20 == 0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Touch 1% of cells with random noise (increased from 0.1%)
            // This provides enough background activity to prevent complete freeze
            let num_to_touch = (self.cells.len() / 100).max(50);
            for _ in 0..num_to_touch {
                let idx = rng.gen_range(0..self.states.len());
                if !self.states[idx].is_dead() {
                    // Generate small random noise signal
                    let mut noise = [0.0f32; SIGNAL_DIMS];
                    for n in noise.iter_mut() {
                        *n = rng.gen_range(-0.1..0.1);
                    }

                    // Inject tiny tension (to create activity, not energy!)
                    for i in 0..SIGNAL_DIMS {
                        self.states[idx].state[i] += noise[i] * 0.3;
                    }
                    self.states[idx].tension += 0.02;

                    // NO FREE ENERGY! "La Vraie Faim" - cells must earn energy through resonance
                    // Background noise creates activity (tension) but doesn't feed cells
                    // Only resonant signals provide energy (handled in backend)
                }
            }
        }

        // === REFLEXIVITY: ARIA hears her own thoughts (Axe 3 - Genesis) ===
        // This creates a recursive loop of consciousness
        if current_tick % 10 == 0 {
            self.inject_self_signal();
        }

        // === Multi-pass recurrent processing (Gemini optimization) ===
        let passes = if self.config.recurrent.enabled {
            self.config.recurrent.passes_per_tick.max(1)
        } else {
            1
        };

        let mut all_actions = Vec::new();

        for pass in 0..passes {
            // Process cells using backend
            let actions = self.backend.update_cells(
                &mut self.cells,
                &mut self.states,
                &self.dna_pool,
                &signals,
            ).unwrap_or_default();

            all_actions.extend(actions);

            // Generate internal signals for next pass (recurrent feedback)
            if pass < passes - 1 && self.config.recurrent.enabled {
                signals = self.generate_internal_signals();
            }
        }

        // === Sync GPU flags to CPU activity state ===
        // Only sync every 100 ticks (matches GPU download frequency)
        // The GPU is the source of truth - CPU only needs periodic updates
        if current_tick % 100 == 0 {
            for (cell, state) in self.cells.iter_mut().zip(self.states.iter()) {
                let gpu_sleeping = state.is_sleeping();
                if cell.activity.sleeping != gpu_sleeping {
                    if gpu_sleeping {
                        cell.activity.sleep();
                    } else {
                        cell.activity.wake();
                    }
                }
            }
        }

        let actions = all_actions;

        // === CLUSTER MAINTENANCE & HYSTERESIS (Phase 6 - Axe 2) ===
        if current_tick % 50 == 0 {
            // In a real implementation with millions of cells, this would be partially
            // delegated to GPU or optimized. Here we do a statistical sampling.
            let mut cluster_activity = std::collections::HashMap::new();
            let mut cluster_counts = std::collections::HashMap::new();

            // Calculate cluster stability
            for state in &self.states {
                if state.cluster_id > 0 && !state.is_dead() {
                    *cluster_activity.entry(state.cluster_id).or_insert(0.0) += state.activity_level;
                    *cluster_counts.entry(state.cluster_id).or_insert(0) += 1;
                }
            }

            // Update hysteresis based on cluster activity
            for state in &mut self.states {
                if state.cluster_id > 0 {
                    let avg_activity = cluster_activity.get(&state.cluster_id).cloned().unwrap_or(0.0)
                        / cluster_counts.get(&state.cluster_id).cloned().unwrap_or(1) as f32;

                    if avg_activity > 0.6 {
                        // Stable & Active: lock it in!
                        state.hysteresis = (state.hysteresis + 0.05).min(1.0);
                    } else if avg_activity < 0.2 {
                        // Fading out: release bond
                        state.hysteresis = (state.hysteresis - 0.02).max(0.0);
                    }
                } else {
                    // No cluster: decay hysteresis faster
                    state.hysteresis = (state.hysteresis - 0.1).max(0.0);
                }
            }
        }

        // Increment age less frequently to reduce CPU overhead
        if current_tick % 100 == 0 {
            for cell in &mut self.cells {
                cell.age += 100;
            }
        }

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

                            // Check if parent DNA is elite (Gemini neuroplasticity)
                            let is_elite = {
                                let memory = self.memory.read();
                                memory.elite_dna.iter().any(|e| e.dna.signature == self.dna_pool[parent.dna_index as usize].signature)
                            };

                            // Build mutation context (Gemini adaptive neuroplasticity)
                            let mutation_ctx = MutationContext {
                                age: parent.age,
                                fitness: parent_state.energy / self.config.metabolism.energy_cap, // Fitness from energy
                                activity: parent_state.activity_level,
                                exploring: !parent.activity.sleeping, // Active cells are exploring
                                is_elite,
                                hysteresis: parent_state.hysteresis,
                            };

                            // Create child DNA with adaptive mutation
                            let parent_dna = &self.dna_pool[parent.dna_index as usize];
                            let child_dna = DNA::from_parent_adaptive(
                                parent_dna,
                                self.config.population.mutation_rate,
                                mutation_ctx,
                            );
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

        // Compact dead cells FIRST (La Vraie Faim cleanup)
        // Must happen BEFORE natural_selection to avoid 100k+ element vectors
        // Every 1000 ticks to reduce freezes (was 100)
        if current_tick % 1000 == 0 {
            self.compact_dead_cells();
        }

        // Natural selection (repopulate if needed)
        if current_tick % self.config.population.selection_interval as u64 == 0 {
            self.natural_selection();
        }

        // Detect emergence (Axe 3 - Genesis & Phase 6)
        self.conceptualize(current_tick);
        let mut emergent = self.detect_emergence(current_tick);

        // Spontaneous speech - NOW INJECTED INTO SUBSTRATE to wake cells!
        if let Some(spontaneous) = self.maybe_speak_spontaneously(current_tick) {
            // Convert to SignalFragment and add to signal buffer for next tick
            // This will wake sleeping cells that resonate with the spontaneous signal
            let tension_8d: [f32; SIGNAL_DIMS] = std::array::from_fn(|i| {
                spontaneous.content.get(i).copied().unwrap_or(0.0)
            });
            let position_8d = aria_core::tension::tension_to_position(&tension_8d);
            let fragment = SignalFragment::external_at(tension_8d, position_8d, spontaneous.intensity * 3.0);

            // Add to signal buffer so it propagates to cells
            {
                let mut buffer = self.signal_buffer.write();
                buffer.push(fragment);
            }

            tracing::debug!("⚡ SPONTANEOUS INJECTED: intensity={:.2}", spontaneous.intensity * 3.0);
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

    /// Generate internal signals for recurrent processing (Gemini multi-pass)
    ///
    /// Active cells emit internal signals that influence their neighbors,
    /// creating feedback loops and richer internal dynamics.
    fn generate_internal_signals(&self) -> Vec<SignalFragment> {
        let threshold = self.config.recurrent.internal_signal_threshold;
        let decay = self.config.recurrent.internal_signal_decay;

        let mut signals = Vec::new();

        // Collect signals from active cells
        for (cell, state) in self.cells.iter().zip(self.states.iter()) {
            // Skip sleeping, dead, or low-activity cells
            // Read from CellState.flags (GPU source of truth)
            if state.is_sleeping() || state.is_dead() {
                continue;
            }

            // Calculate activation level
            let activation: f32 = state.state.iter().map(|x| x.abs()).sum::<f32>() / state.state.len() as f32;

            if activation > threshold {
                // Generate internal signal from this cell's state
                let content: [f32; SIGNAL_DIMS] = std::array::from_fn(|j| {
                    state.state[j] * decay // Apply decay to internal signals
                });

                // Use cell's position (first 8D) as signal position
                let position: [f32; SIGNAL_DIMS] = std::array::from_fn(|j| {
                    state.position[j]
                });

                signals.push(SignalFragment::new(
                    cell.id,
                    content,
                    position,
                    activation * decay,
                ));
            }

            // Limit internal signals to prevent explosion
            if signals.len() >= 100 {
                break;
            }
        }

        if !signals.is_empty() {
            tracing::trace!(
                "🔄 Recurrent: {} internal signals generated",
                signals.len()
            );
        }

        signals
    }

    /// Get substrate statistics
    pub fn stats(&self) -> SubstrateStats {
        let current_tick = self.tick.load(Ordering::Relaxed);
        let emotional = self.emotional_state.read();

        let alive_count = self.cells.iter()
            .zip(self.states.iter())
            .filter(|(_, s)| !s.is_dead())
            .count();

        // Read sleeping from CellState.flags (GPU source of truth)
        // NOT from Cell.activity.sleeping (CPU state that may be out of sync)
        let sleeping_count = self.states.iter()
            .filter(|s| s.is_sleeping() && !s.is_dead())
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

    /// Get spatial activity data for visualization
    ///
    /// Returns a 16x16 grid of activity levels and energy,
    /// plus population breakdown for the substrate view.
    pub fn spatial_view(&self) -> SubstrateView {
        const GRID_SIZE: usize = 16;
        let mut activity_grid = vec![0.0f32; GRID_SIZE * GRID_SIZE];
        let mut energy_grid = vec![0.0f32; GRID_SIZE * GRID_SIZE];
        let mut tension_grid = vec![0.0f32; GRID_SIZE * GRID_SIZE];
        let mut cell_count_grid = vec![0usize; GRID_SIZE * GRID_SIZE];

        // Lineage tracking
        let mut max_generation: u32 = 0;
        let mut total_generation: u64 = 0;
        let mut elite_count: usize = 0;
        let mut total_energy: f32 = 0.0;
        let mut total_tension: f32 = 0.0;

        // Count cells in each grid position
        // Using first 2 dimensions of position (typically in -10..10 range)
        for (cell, state) in self.cells.iter().zip(self.states.iter()) {
            if state.is_dead() {
                continue;
            }

            // Track lineage
            if cell.generation > max_generation {
                max_generation = cell.generation;
            }
            total_generation += cell.generation as u64;
            if cell.generation > 10 {
                elite_count += 1;
            }

            // Track totals
            total_energy += state.energy;
            total_tension += state.tension;

            // Map position to grid (position is typically -10..10)
            let x = ((state.position[0] + 10.0) / 20.0 * GRID_SIZE as f32)
                .clamp(0.0, (GRID_SIZE - 1) as f32) as usize;
            let y = ((state.position[1] + 10.0) / 20.0 * GRID_SIZE as f32)
                .clamp(0.0, (GRID_SIZE - 1) as f32) as usize;
            let idx = y * GRID_SIZE + x;

            cell_count_grid[idx] += 1;
            energy_grid[idx] += state.energy;
            tension_grid[idx] += state.tension;

            // Activity: awake cells show full activation, sleeping show potential (energy)
            // This ensures the heatmap always shows something even when cells sleep
            if !state.is_sleeping() {
                let activation: f32 = state.state.iter().map(|x| x.abs()).sum();
                activity_grid[idx] += activation;
            } else {
                // Sleeping cells contribute their energy as "potential activity"
                // Dimmed (0.2x) so awake cells stand out
                activity_grid[idx] += state.energy * 0.2;
            }
        }

        // Normalize grids
        let max_activity = activity_grid.iter().cloned().fold(0.0f32, f32::max).max(1.0);
        let max_energy_grid = energy_grid.iter().cloned().fold(0.0f32, f32::max).max(1.0);
        let max_tension = tension_grid.iter().cloned().fold(0.0f32, f32::max).max(0.1);

        for i in 0..activity_grid.len() {
            activity_grid[i] /= max_activity;
            energy_grid[i] /= max_energy_grid;
            tension_grid[i] /= max_tension;
        }

        // Population breakdown
        let total = self.cells.len();
        let alive = self.states.iter().filter(|s| !s.is_dead()).count();
        // Read from CellState.flags (GPU source of truth)
        let sleeping = self.states.iter().filter(|s| s.is_sleeping() && !s.is_dead()).count();
        let dead = total - alive;
        let awake = alive - sleeping;

        // Calculate averages
        let avg_energy = if alive > 0 { total_energy / alive as f32 } else { 0.0 };
        let avg_tension = if alive > 0 { total_tension / alive as f32 } else { 0.0 };
        let avg_generation = if alive > 0 { total_generation as f32 / alive as f32 } else { 0.0 };
        let sparse_savings_percent = if alive > 0 { sleeping as f32 / alive as f32 * 100.0 } else { 0.0 };

        // Energy distribution (histogram)
        let mut energy_histogram = [0usize; 10];
        for state in &self.states {
            if !state.is_dead() {
                let bucket = ((state.energy / 1.5) * 9.0).clamp(0.0, 9.0) as usize;
                energy_histogram[bucket] += 1;
            }
        }

        // Calculate activity entropy (Shannon entropy normalized to 0-1)
        // Entropy = -sum(p * log2(p)) / log2(n)
        // Where p is the probability of activity in each cell
        let activity_sum: f32 = activity_grid.iter().sum();
        let activity_entropy = if activity_sum > 0.001 {
            let mut entropy = 0.0f32;
            for &activity in &activity_grid {
                if activity > 0.001 {
                    let p = activity / activity_sum;
                    entropy -= p * p.log2();
                }
            }
            // Normalize by max entropy (log2 of grid size)
            let max_entropy = (GRID_SIZE * GRID_SIZE) as f32;
            (entropy / max_entropy.log2()).clamp(0.0, 1.0)
        } else {
            0.0 // No activity = no entropy
        };

        // Calculate system health (composite metric)
        // Healthy system: moderate entropy, good awake ratio, stable energy
        let awake_ratio = if alive > 0 { awake as f32 / alive as f32 } else { 0.0 };
        let alive_ratio = if total > 0 { alive as f32 / total as f32 } else { 0.0 };

        // Health is optimal when:
        // - entropy is moderate (not too chaotic, not dead)
        // - awake_ratio is moderate (not all sleeping, not all awake)
        // - alive_ratio is high (not dying)
        let entropy_health = 1.0 - (activity_entropy - 0.5).abs() * 2.0; // Peak at 0.5
        let activity_health = 1.0 - (awake_ratio - 0.3).abs() * 2.0; // Peak at 30% awake
        let survival_health = alive_ratio;

        let system_health = (entropy_health * 0.3 + activity_health * 0.3 + survival_health * 0.4)
            .clamp(0.0, 1.0);

        SubstrateView {
            grid_size: GRID_SIZE,
            activity_grid,
            energy_grid,
            tension_grid,
            cell_count_grid,
            total_cells: total,
            alive_cells: alive,
            sleeping_cells: sleeping,
            dead_cells: dead,
            awake_cells: awake,
            energy_histogram: energy_histogram.to_vec(),
            activity_entropy,
            system_health,
            // Advanced metrics
            max_generation,
            avg_generation,
            elite_count,
            sparse_savings_percent,
            avg_energy,
            avg_tension,
            total_tension,
            tps: 0.0, // TPS computed in main loop, updated externally
        }
    }

    /// Inject ARIA's last thoughts back into her substrate
    ///
    /// This is the "Self-Input Loop" that enables reflexivity.
    /// Cells with high reflexivity gain will be influenced by these thoughts.
    fn inject_self_signal(&self) {
        let last_thought = self.last_emergent_tension.read().clone();

        // Only inject if there is some activity
        let intensity: f32 = last_thought.iter().map(|x| x.abs()).sum::<f32>() / SIGNAL_DIMS as f32;
        if intensity < 0.05 {
            return;
        }

        // Position of the thought is its own semantic representation
        let position = aria_core::tension::tension_to_position(&last_thought);

        // Self-signal is slightly weaker and has a specific intensity
        // We use a high amplification because it's "internal"
        // source_id = u64::MAX (internal thought)
        let fragment = SignalFragment::new(
            u64::MAX, // Special ID for ARIA's internal thoughts
            last_thought,
            position,
            intensity * 2.0 // Strong enough to be felt
        );

        let mut buffer = self.signal_buffer.write();
        buffer.push(fragment);

        tracing::trace!("🧠 REFLEXIVITY: Self-signal injected (intensity={:.2})", intensity);
    }
}

/// Spatial view of the substrate for visualization
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubstrateView {
    /// Grid size (16x16)
    pub grid_size: usize,
    /// Activity level per grid cell (0.0-1.0)
    pub activity_grid: Vec<f32>,
    /// Energy level per grid cell (normalized)
    pub energy_grid: Vec<f32>,
    /// Tension level per grid cell (normalized)
    pub tension_grid: Vec<f32>,
    /// Number of cells per grid position
    pub cell_count_grid: Vec<usize>,
    /// Total cells (including dead)
    pub total_cells: usize,
    /// Alive cells count
    pub alive_cells: usize,
    /// Sleeping cells count
    pub sleeping_cells: usize,
    /// Dead cells count
    pub dead_cells: usize,
    /// Awake cells count
    pub awake_cells: usize,
    /// Energy distribution histogram (10 buckets)
    pub energy_histogram: Vec<usize>,
    /// Activity entropy (0.0 = structured, 1.0 = chaotic)
    pub activity_entropy: f32,
    /// System health (0.0 = dying, 1.0 = thriving)
    pub system_health: f32,
    // === Advanced metrics for visualization ===
    /// Maximum generation (lineage depth) - elite lineage indicator
    pub max_generation: u32,
    /// Average generation across alive cells
    pub avg_generation: f32,
    /// Elite cell count (generation > 10)
    pub elite_count: usize,
    /// Sparse dispatch savings (% of cells sleeping)
    pub sparse_savings_percent: f32,
    /// Average energy per cell
    pub avg_energy: f32,
    /// Average tension per cell
    pub avg_tension: f32,
    /// Total tension in the system (indicates desire to act)
    pub total_tension: f32,
    /// Tick per second (TPS) - computed externally but stored here
    pub tps: f32,
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

    // NOTE: test_conversation_context removed in Session 20 (Physical Intelligence)
    // ConversationContext is no longer used in Substrate
}
