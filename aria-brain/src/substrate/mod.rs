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
pub use types::{AdaptiveParams, SubstrateStats, SubstrateView, SpatialInhibitor, EMISSION_COOLDOWN_TICKS};
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

    /// Pending signals for cells (ring buffer for constant throughput - Session 35)
    /// Capacity: 256 signals ≈ 4 ticks of buffering at max trainer rate
    signal_buffer: RwLock<types::SignalRingBuffer>,

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

        // Create cells and states in parallel (Session 32)
        // At 5M cells, sequential creation takes minutes - parallel takes seconds
        use rayon::prelude::*;

        tracing::info!("🧬 Creating {} primordial cells (parallel)...", initial_cells);
        let start = std::time::Instant::now();

        // Parallel creation of (Cell, CellState, DNA) tuples
        let cell_data: Vec<(Cell, CellState, DNA)> = (0..initial_cells)
            .into_par_iter()
            .map(|i| {
                let dna = DNA::random();
                let cell = Cell::new(i as u64, i as u32); // dna_index = i (1:1 mapping initially)
                let state = CellState::new();
                (cell, state, dna)
            })
            .collect();

        // Unzip into separate Vecs (sequential but fast - just moving data)
        let mut cells = Vec::with_capacity(initial_cells);
        let mut states = Vec::with_capacity(initial_cells);
        let mut dna_pool = Vec::with_capacity(initial_cells);

        for (cell, state, dna) in cell_data {
            cells.push(cell);
            states.push(state);
            dna_pool.push(dna);
        }

        tracing::info!("🧬 Created {} cells in {:.2}s", initial_cells, start.elapsed().as_secs_f32());

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
            signal_buffer: RwLock::new(types::SignalRingBuffer::new(256)), // Session 35: Ring buffer
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

        // Save elite DNA every 10000 ticks (before any crash/restart)
        if current_tick % 10000 == 0 && current_tick > 0 {
            self.save_elite_dna_periodic();
        }

        // Decay spatial inhibition (Gemini optimization)
        {
            let mut inhibitor = self.spatial_inhibitor.write();
            inhibitor.decay(current_tick);
            // Sync base threshold with adaptive params
            let params = self.adaptive_params.read();
            inhibitor.set_base_threshold(params.emission_threshold);
        }

        // Get external signals from buffer (limited per tick - Session 35)
        // Ring buffer prevents accumulation; drain() limits GPU burst
        let mut signals: Vec<SignalFragment> = {
            let mut buffer = self.signal_buffer.write();
            buffer.drain(types::MAX_SIGNALS_PER_TICK)
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
        // REDUCED: Was every 10 ticks, now every 100 ticks
        // Too much reflexivity = self-feeding loop that prevents starvation
        if current_tick % 100 == 0 {
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
        // Session 32: REMOVED O(n) loop - GPU is source of truth, CPU doesn't need sync.
        // The activity state in Cell is legacy and can be derived from CellState.flags.

        let actions = all_actions;

        // === CLUSTER MAINTENANCE & HYSTERESIS (Phase 6 - Axe 2) ===
        // NOTE: Now handled by GPU CLUSTER_STATS_SHADER + CLUSTER_HYSTERESIS_SHADER (gpu_soa.rs)

        // Session 32: Age increment REMOVED - cell.age can be computed from (current_tick - birth_tick)
        // if needed. Saving O(n) loop every 100 ticks.

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
        // Session 32 Part 11: Every 5000 ticks + sampled check to reduce freezes
        if current_tick % 5000 == 0 {
            // Quick sample to check if compacting is even needed (>50% dead)
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let sample_size = 500.min(self.states.len());
            let dead_sample = (0..sample_size)
                .filter(|_| {
                    let idx = rng.gen_range(0..self.states.len());
                    self.states[idx].is_dead()
                })
                .count();
            let dead_ratio = dead_sample as f32 / sample_size as f32;

            if dead_ratio > 0.5 {
                self.compact_dead_cells();
            }
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

        // === LAW OF ASSOCIATION (Hebb's Law) ===
        // "Fire together, wire together" implemented as spatial attraction.
        // Active cells move towards their shared center of gravity.
        // NOTE: Now handled by GPU HEBBIAN_CENTROID_SHADER + HEBBIAN_ATTRACTION_SHADER (gpu_soa.rs)

        // === LAW OF COMPRESSION (Predictive Physics) ===
        // "Surprise costs energy."
        // Cells bet on their future state.
        // NOTE: Now handled by GPU PREDICTION_EVALUATE_SHADER (gpu_soa.rs)

        emergent
    }

    /// Generate internal signals for recurrent processing (Gemini multi-pass)
    ///
    /// Active cells emit internal signals that influence their neighbors,
    /// creating feedback loops and richer internal dynamics.
    fn generate_internal_signals(&self) -> Vec<SignalFragment> {
        // Skip if recurrent processing is disabled (Session 32 - avoid O(n) loop)
        if !self.config.recurrent.enabled {
            return Vec::new();
        }

        let threshold = self.config.recurrent.internal_signal_threshold;
        let decay = self.config.recurrent.internal_signal_decay;

        let mut signals = Vec::new();

        // Sample-based internal signal generation (avoid O(n) loop)
        let total_cells = self.cells.len();
        let sample_size = 1000.min(total_cells);
        let step = total_cells / sample_size.max(1);

        for i in 0..sample_size {
            let idx = i * step;
            if idx >= total_cells {
                break;
            }
            let cell = &self.cells[idx];
            let state = &self.states[idx];
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

    /// Get substrate statistics (using sampling for O(1) instead of O(n))
    pub fn stats(&self) -> SubstrateStats {
        let current_tick = self.tick.load(Ordering::Relaxed);
        let emotional = self.emotional_state.read();

        // Sample-based statistics for O(1) complexity at scale
        let total_cells = self.cells.len();
        let sample_size = 5000.min(total_cells);

        if sample_size == 0 {
            let params = self.adaptive_params.read();
            return SubstrateStats {
                tick: current_tick,
                alive_cells: 0,
                total_energy: 0.0,
                entropy: 0.0,
                active_clusters: 0,
                dominant_emotion: emotional.mood_description().to_string(),
                signals_per_second: 0.0,
                oldest_cell_age: 0,
                average_connections: 0.0,
                mood: emotional.mood_description().to_string(),
                happiness: emotional.happiness,
                arousal: emotional.arousal,
                curiosity: emotional.curiosity,
                sleeping_cells: 0,
                cpu_savings_percent: 0.0,
                backend_name: self.backend.name().to_string(),
                adaptive_emission_threshold: params.emission_threshold,
                adaptive_response_probability: params.response_probability,
                adaptive_spontaneity: params.spontaneity,
                adaptive_feedback_positive: params.positive_count(),
                adaptive_feedback_negative: params.negative_count(),
                boredom: emotional.boredom,
            };
        }

        // Stratified sampling for better coverage
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut alive_sample = 0usize;
        let mut sleeping_sample = 0usize;
        let mut energy_sum = 0.0f32;
        let mut max_age = 0u64;

        let step = total_cells / sample_size;
        for i in 0..sample_size {
            let idx = (i * step + rng.gen_range(0..step.max(1))) % total_cells;
            let state = &self.states[idx];
            let cell = &self.cells[idx];

            if !state.is_dead() {
                alive_sample += 1;
                energy_sum += state.energy;
                if state.is_sleeping() {
                    sleeping_sample += 1;
                }
                if cell.age > max_age {
                    max_age = cell.age;
                }
            }
        }

        // Extrapolate from sample to population
        let sample_alive_ratio = alive_sample as f32 / sample_size as f32;
        let alive_count = (sample_alive_ratio * total_cells as f32) as usize;

        let sample_sleeping_ratio = if alive_sample > 0 {
            sleeping_sample as f32 / alive_sample as f32
        } else {
            0.0
        };
        let sleeping_count = (sample_sleeping_ratio * alive_count as f32) as usize;

        // Extrapolate total energy
        let avg_sample_energy = if alive_sample > 0 {
            energy_sum / alive_sample as f32
        } else {
            0.0
        };
        let total_energy = avg_sample_energy * alive_count as f32;

        let cpu_savings = sample_sleeping_ratio * 100.0;

        let params = self.adaptive_params.read();

        SubstrateStats {
            tick: current_tick,
            alive_cells: alive_count,
            total_energy,
            entropy: self.calculate_entropy_sampled(),
            active_clusters: 0,
            dominant_emotion: emotional.mood_description().to_string(),
            signals_per_second: 0.0,
            oldest_cell_age: max_age,
            average_connections: 0.0,
            mood: emotional.mood_description().to_string(),
            happiness: emotional.happiness,
            arousal: emotional.arousal,
            curiosity: emotional.curiosity,
            sleeping_cells: sleeping_count,
            cpu_savings_percent: cpu_savings,
            backend_name: self.backend.name().to_string(),
            adaptive_emission_threshold: params.emission_threshold,
            adaptive_response_probability: params.response_probability,
            adaptive_spontaneity: params.spontaneity,
            adaptive_feedback_positive: params.positive_count(),
            adaptive_feedback_negative: params.negative_count(),
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
    ///
    /// **OPTIMIZED (Session 31)**: Uses sampling for grids (10k max) instead of O(n).
    /// At 1M cells, this is 100x faster.
    pub fn spatial_view(&self) -> SubstrateView {
        use rand::Rng;
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

        // === SAMPLING for grids (Session 31) ===
        // Sample up to 10k cells for visualization grids.
        // Statistically sufficient for 16x16 heatmap.
        let sample_size = 10_000.min(self.cells.len());
        let mut rng = rand::thread_rng();
        let mut sampled_alive = 0usize;
        let mut sampled_sleeping = 0usize;

        for _ in 0..sample_size {
            let idx = rng.gen_range(0..self.cells.len());
            let cell = &self.cells[idx];
            let state = &self.states[idx];

            if state.is_dead() {
                continue;
            }
            sampled_alive += 1;
            if state.is_sleeping() {
                sampled_sleeping += 1;
            }

            // Track lineage (sampling gives good approximation for max)
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
            let grid_idx = y * GRID_SIZE + x;

            cell_count_grid[grid_idx] += 1;
            energy_grid[grid_idx] += state.energy;
            tension_grid[grid_idx] += state.tension;

            // Activity: awake cells show full activation, sleeping show potential (energy)
            if !state.is_sleeping() {
                let activation: f32 = state.state.iter().map(|x| x.abs()).sum();
                activity_grid[grid_idx] += activation;
            } else {
                activity_grid[grid_idx] += state.energy * 0.2;
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

        // Population breakdown - extrapolate from sample
        let total = self.cells.len();
        let scale = if sampled_alive > 0 { self.cells.len() as f32 / sample_size as f32 } else { 1.0 };
        let alive = (sampled_alive as f32 * scale) as usize;
        let sleeping = (sampled_sleeping as f32 * scale) as usize;
        let dead = total.saturating_sub(alive);
        let awake = alive.saturating_sub(sleeping);

        // Calculate averages (from sample)
        let avg_energy = if sampled_alive > 0 { total_energy / sampled_alive as f32 } else { 0.0 };
        let avg_tension = if sampled_alive > 0 { total_tension / sampled_alive as f32 } else { 0.0 };
        let avg_generation = if sampled_alive > 0 { total_generation as f32 / sampled_alive as f32 } else { 0.0 };
        let sparse_savings_percent = if alive > 0 { sleeping as f32 / alive as f32 * 100.0 } else { 0.0 };

        // Energy histogram from sample (extrapolated)
        let mut energy_histogram = [0usize; 10];
        // Already sampled above, just use the sample counts with scale
        for grid_val in &energy_grid {
            let bucket = ((grid_val * 9.0).clamp(0.0, 9.0)) as usize;
            energy_histogram[bucket] += 1;
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

        // Self-signal is WEAK to prevent self-feeding loop
        // Reduced from 2.0 to 0.3 - ARIA needs external input to thrive
        let fragment = SignalFragment::new(
            u64::MAX, // Special ID for ARIA's internal thoughts
            last_thought,
            position,
            intensity * 0.3 // Weak - prevents self-sustaining loop
        );

        // Session 35: Ring buffer auto-discards oldest when full
        let mut buffer = self.signal_buffer.write();
        buffer.push(fragment);

        tracing::trace!("🧠 REFLEXIVITY: Self-signal injected (intensity={:.2})", intensity);
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

    // NOTE: test_conversation_context removed in Session 20 (Physical Intelligence)
    // ConversationContext is no longer used in Substrate
}
