//! Lifecycle & Utility Methods for ARIA's Substrate
//!
//! Population management, coherence calculation, and helper functions.

use super::*;

impl Substrate {
    /// Decide whether to record an episode and record it
    /// NOTE: Not used in Physical Intelligence (Session 20) but kept for future memory features
    #[allow(dead_code)]
    pub(super) fn maybe_record_episode(
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
        // NOTE: STOP_WORDS removed in Session 20 (Physical Intelligence)
        // Just filter by length - no semantic word filtering
        let keywords: Vec<String> = input
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| w.len() >= 4) // Slightly longer filter since no stop words
            .map(|w| w.to_lowercase())
            .take(5) // Limit to 5 keywords
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

    /// Natural selection - maintain population
    ///
    /// OPTIMIZED: Uses partial sort (select_nth_unstable) instead of full sort.
    /// O(n) instead of O(n log n) - critical for 100k+ cells.
    ///
    /// CONTINUOUS REPRODUCTION (Session 27):
    /// Cells with energy > reproduction_threshold can divide regardless of population.
    /// This creates real lineage evolution without waiting for extinction.
    /// Natural selection - maintain population
    ///
    /// **OPTIMIZED (Session 32)**: Sampling instead of O(n) full scans.
    /// - Stats computed from 5k sample
    /// - Only higher generations (Gen2+) tracked for reproduction
    /// - Gen0 drain moved to GPU (SLEEPING_DRAIN handles it)
    pub(super) fn natural_selection(&mut self) {
        use rand::Rng;
        let target = self.config.population.target_population as usize;
        let buffer = self.config.population.population_buffer as usize;
        let min_pop = self.config.population.min_population as usize;
        let reproduction_threshold = self.config.metabolism.reproduction_threshold;
        let child_energy = self.config.metabolism.child_energy;
        let divide_cost = self.config.metabolism.cost_divide;

        // === SAMPLED alive_count (Session 32) ===
        // Instead of O(n) count, sample 5k cells and extrapolate
        let mut rng = rand::thread_rng();
        let sample_size = 5000.min(self.cells.len());
        let mut sampled_alive = 0usize;
        let mut max_energy = 0.0f32;
        let mut sum_energy = 0.0f32;

        for _ in 0..sample_size {
            let idx = rng.gen_range(0..self.states.len());
            let state = &self.states[idx];
            if !state.is_dead() {
                sampled_alive += 1;
                sum_energy += state.energy;
                if state.energy > max_energy { max_energy = state.energy; }
            }
        }

        let scale = self.cells.len() as f32 / sample_size as f32;
        let alive_count = (sampled_alive as f32 * scale) as usize;
        let avg_energy = if sampled_alive > 0 { sum_energy / sampled_alive as f32 } else { 0.0 };

        // === CONTINUOUS REPRODUCTION (Law of Expansion) ===
        // Cap population at GPU's max_capacity to prevent VRAM exhaustion and "Device lost" errors
        let backend_max = self.backend.stats().max_capacity;
        let safety_cap = if backend_max > 0 {
            (target * 2).min(backend_max)
        } else {
            target * 2
        };

        if alive_count < safety_cap {
            // === SAMPLED reproduction candidates (Session 32) ===
            // Only sample 10k cells to find reproduction candidates
            // Prioritize higher generations by weighted sampling
            const MAX_GEN_BUCKETS: usize = 32;
            let mut gen_buckets: [Vec<(usize, u32)>; MAX_GEN_BUCKETS] = Default::default();

            let repro_sample_size = 10_000.min(self.cells.len());
            for _ in 0..repro_sample_size {
                let i = rng.gen_range(0..self.cells.len());
                let cell = &self.cells[i];
                let state = &self.states[i];
                if !state.is_dead() && state.energy > reproduction_threshold {
                    let gen = (cell.generation as usize).min(MAX_GEN_BUCKETS - 1);
                    // Avoid duplicates - check if already in bucket
                    if gen_buckets[gen].len() < 1000 {
                        gen_buckets[gen].push((i, cell.dna_index));
                    }
                }
            }

            // Flatten buckets from highest generation down
            // Session 32 Part 11: Removed artificial 500 cap - let economy self-regulate
            // Technical limit of 2000 prevents massive allocation spikes
            const TECHNICAL_MAX_BIRTHS: usize = 2000;
            let mut ready_to_divide: Vec<(usize, u32, u32)> = Vec::with_capacity(TECHNICAL_MAX_BIRTHS);
            for gen in (0..MAX_GEN_BUCKETS).rev() {
                for (idx, dna_idx) in gen_buckets[gen].iter() {
                    ready_to_divide.push((*idx, *dna_idx, gen as u32));
                    if ready_to_divide.len() >= TECHNICAL_MAX_BIRTHS { break; }
                }
                if ready_to_divide.len() >= TECHNICAL_MAX_BIRTHS { break; }
            }

            let room = safety_cap.saturating_sub(alive_count);
            let max_births = room.min(ready_to_divide.len()); // No artificial cap!

            if ready_to_divide.is_empty() {
                tracing::info!("🧬 LINEAGE: 0 cells ready (threshold={:.2}, max_energy={:.2}, avg={:.2}, pop={})",
                    reproduction_threshold, max_energy, avg_energy, alive_count);
            } else if max_births > 0 {
                let gen0_ready = gen_buckets[0].len();
                let gen1_ready = gen_buckets[1].len();
                let gen2plus_ready: usize = gen_buckets[2..].iter().map(|b| b.len()).sum();

                tracing::info!("🧬 EXPANSION: ~{} ready (Gen0:{}, Gen1:{}, Gen2+:{}), {} reproducing, pop:{}",
                    gen0_ready + gen1_ready + gen2plus_ready, gen0_ready, gen1_ready, gen2plus_ready,
                    max_births, alive_count);

                // Session 32: Reserve headroom if approaching capacity
                // This prevents per-push reallocations by adding chunks of capacity
                let current_cap = self.cells.capacity();
                let needed = self.cells.len() + max_births;
                if needed > current_cap {
                    // Add 10% headroom when we need to grow
                    let extra = (current_cap / 10).max(1000);
                    self.cells.reserve(extra);
                    self.states.reserve(extra);
                    self.dna_pool.reserve(extra);
                    tracing::debug!("📈 VEC RESERVE: +{} capacity (total: {})", extra, self.cells.capacity());
                }

                for (parent_idx, parent_dna_idx, parent_generation) in ready_to_divide.into_iter().take(max_births) {
                    self.states[parent_idx].energy -= divide_cost;

                    let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                    let parent_dna = &self.dna_pool[parent_dna_idx as usize];

                    let mutation_ctx = MutationContext {
                        age: 0,
                        fitness: self.states[parent_idx].energy / self.config.metabolism.energy_cap,
                        activity: self.states[parent_idx].activity_level,
                        exploring: false,
                        is_elite: parent_generation > 5,
                        hysteresis: 0.0,
                    };

                    let child_dna = DNA::from_parent_adaptive(
                        parent_dna,
                        self.config.population.mutation_rate,
                        mutation_ctx,
                    );
                    let dna_index = self.dna_pool.len() as u32;
                    self.dna_pool.push(child_dna);

                    let mut cell = Cell::new(new_id, dna_index);
                    cell.generation = parent_generation + 1;

                    let mut state = CellState::new();
                    state.energy = child_energy;

                    if let Some(idx) = self.free_slots.pop() {
                        self.cells[idx] = cell;
                        self.states[idx] = state;
                    } else {
                        self.cells.push(cell);
                        self.states.push(state);
                    }
                }
            }

            // === GEN0 DRAIN moved to GPU (Session 32) ===
            // The SLEEPING_DRAIN_SHADER now handles Gen0 energy drain.
            // CPU only logs periodically for debugging.
            let gen0_count = gen_buckets[0].len();
            if gen0_count > 10_000 {
                tracing::debug!("🧹 GEN0: ~{} cells in sample (GPU handles drain)", gen0_count);
            }
        }

        // === POPULATION CONTROL (Session 32: Sampling) ===
        if alive_count > target + buffer {
            // Too many cells - SAMPLE weakest instead of full scan
            let to_remove = (alive_count - target).min(500);
            let sample_size = 5000.min(self.states.len());
            let mut weak_cells: Vec<(usize, f32)> = Vec::with_capacity(to_remove);

            for _ in 0..sample_size {
                let idx = rng.gen_range(0..self.states.len());
                let state = &self.states[idx];
                if !state.is_dead() && state.energy < 0.3 {
                    // Only consider weak cells (energy < 0.3)
                    if weak_cells.len() < to_remove {
                        weak_cells.push((idx, state.energy));
                    }
                }
                if weak_cells.len() >= to_remove { break; }
            }

            for (idx, _) in weak_cells.iter() {
                self.states[*idx].set_dead();
                self.free_slots.push(*idx);
            }
            if weak_cells.len() > 0 {
                tracing::debug!("🗑️ CULL: {} weak cells removed", weak_cells.len());
            }
        } else if alive_count < min_pop {
            // Too few cells - SAMPLE survivors
            let to_spawn = min_pop - alive_count;

            // Sample to find elite survivors instead of O(n) scan
            const ELITE_COUNT: usize = 10;
            let mut top_survivors: Vec<(usize, f32, u32, u32)> = Vec::with_capacity(ELITE_COUNT);
            let sample_size = 1000.min(self.cells.len());

            for _ in 0..sample_size {
                let i = rng.gen_range(0..self.cells.len());
                let cell = &self.cells[i];
                let state = &self.states[i];
                if state.is_dead() { continue; }

                let entry = (i, state.energy, cell.dna_index, cell.generation);
                if top_survivors.len() < ELITE_COUNT {
                    top_survivors.push(entry);
                    top_survivors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else if state.energy > top_survivors.last().map(|x| x.1).unwrap_or(0.0) {
                    top_survivors.pop();
                    top_survivors.push(entry);
                    top_survivors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            }

            let survivors = top_survivors;

            if survivors.is_empty() {
                // Total extinction - try to use elite_dna from memory first!
                // Elite DNA stores (dna, generation) for lineage continuity
                let elite_with_gen: Vec<(DNA, u64)> = {
                    let memory = self.memory.read();
                    memory.elite_dna.iter().map(|e| (e.dna.clone(), e.generation)).collect()
                };

                if elite_with_gen.is_empty() {
                    // True extinction - no elite DNA saved, must use random
                    tracing::warn!("⚠️ TOTAL EXTINCTION - no elite DNA, spawning random cells");
                    for _ in 0..to_spawn {
                        let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                        let dna = DNA::random();
                        let dna_index = self.dna_pool.len() as u32;
                        self.dna_pool.push(dna);
                        let cell = Cell::new(new_id, dna_index);
                        let mut state = CellState::new();
                        // Give new cells high energy and tension to survive until training
                        state.energy = self.config.metabolism.energy_cap * 0.8;
                        state.tension = 0.5;
                        self.cells.push(cell);
                        self.states.push(state);
                    }
                } else {
                    // RESURRECTION from elite DNA!
                    tracing::info!("🧬 RESURRECTION: {} new cells from {} elite DNA", to_spawn, elite_with_gen.len());
                    for i in 0..to_spawn {
                        let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                        let (parent_dna, parent_gen) = &elite_with_gen[i % elite_with_gen.len()];

                        // Mutate from elite
                        let mutation_ctx = MutationContext {
                            age: 0,
                            fitness: 0.8, // Elite are fit
                            activity: 0.5,
                            exploring: true,
                            is_elite: true,
                            hysteresis: 0.0,
                        };
                        let child_dna = DNA::from_parent_adaptive(
                            parent_dna,
                            self.config.population.mutation_rate,
                            mutation_ctx,
                        );
                        let dna_index = self.dna_pool.len() as u32;
                        self.dna_pool.push(child_dna);

                        // LINEAGE: Resurrect with parent's generation + 1
                        let mut cell = Cell::new(new_id, dna_index);
                        cell.generation = (*parent_gen as u32) + 1;

                        let mut state = CellState::new();
                        // Resurrected cells get FULL energy (not child_energy)
                        // They need to survive long enough to receive training signals
                        state.energy = self.config.metabolism.energy_cap * 0.8;
                        // Give them initial tension to stay awake longer
                        state.tension = 0.5;

                        if let Some(idx) = self.free_slots.pop() {
                            self.cells[idx] = cell;
                            self.states[idx] = state;
                        } else {
                            self.cells.push(cell);
                            self.states.push(state);
                        }
                    }
                }
            } else {
                // EVOLUTION: New cells inherit from survivors with mutation
                // THIS IS WHERE LINEAGE IS BUILT - children inherit parent's generation + 1
                tracing::info!("🧬 REPOPULATING: {} new cells from {} survivors", to_spawn, survivors.len());

                for i in 0..to_spawn {
                    let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);

                    // Pick parent from survivors (round-robin through best ones)
                    let parent_idx = i % survivors.len();
                    let (_, parent_energy, parent_dna_idx, parent_generation) = survivors[parent_idx];
                    let parent_dna = &self.dna_pool[parent_dna_idx as usize];

                    // Create child DNA with mutation (survivors are fit, so lower mutation)
                    let mutation_ctx = MutationContext {
                        age: 0,
                        fitness: parent_energy / self.config.metabolism.energy_cap,
                        activity: 0.5,
                        exploring: true,
                        is_elite: parent_idx < 3, // Top 3 are elite
                        hysteresis: 0.0,
                    };
                    let child_dna = DNA::from_parent_adaptive(
                        parent_dna,
                        self.config.population.mutation_rate,
                        mutation_ctx,
                    );
                    let dna_index = self.dna_pool.len() as u32;
                    self.dna_pool.push(child_dna);

                    // LINEAGE: Child inherits parent's generation + 1
                    let mut cell = Cell::new(new_id, dna_index);
                    cell.generation = parent_generation + 1;

                    let mut state = CellState::new();
                    state.energy = self.config.metabolism.child_energy; // Start weak

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
    }

    /// Compact dead cells - actually remove them from vectors
    ///
    /// Called when dead cells exceed 50% of total to prevent iteration overhead.
    /// "La Vraie Faim" can cause mass extinction, so we need this cleanup.
    /// IMPORTANT: Saves elite DNA before compacting to preserve genetic heritage!
    pub(super) fn compact_dead_cells(&mut self) {
        let total = self.cells.len();
        let dead_count = self.states.iter().filter(|s| s.is_dead()).count();

        // Only compact if >50% dead (expensive operation)
        if dead_count < total / 2 {
            return;
        }

        tracing::info!("🧹 COMPACTING: {} dead / {} total cells", dead_count, total);

        // SAVE ELITE DNA BEFORE EXTINCTION!
        // Collect top performers by energy (before they're all gone)
        let mut performers: Vec<(f32, &Cell, &CellState)> = self.cells.iter()
            .zip(self.states.iter())
            .filter(|(_, s)| !s.is_dead() && s.energy > 0.1)
            .map(|(c, s)| (s.energy, c, s))
            .collect();
        performers.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Save top 10 to elite_dna
        let _current_tick = self.tick.load(Ordering::Relaxed);
        {
            let mut memory = self.memory.write();
            for (energy, cell, _) in performers.iter().take(10) {
                if let Some(dna) = self.dna_pool.get(cell.dna_index as usize) {
                    memory.preserve_elite(
                        dna.clone(),
                        *energy / self.config.metabolism.energy_cap,
                        cell.age,
                        "survivor",
                    );
                }
            }
        }

        if !performers.is_empty() {
            tracing::info!("🧬 PRESERVED: {} elite DNA before compaction", performers.len().min(10));
        }

        // Keep only living cells AND compact DNA pool
        // Add 10% headroom to prevent immediate reallocations (Session 32)
        let alive_estimate = total - dead_count;
        let compacted_capacity = alive_estimate + alive_estimate / 10;
        let mut new_cells = Vec::with_capacity(compacted_capacity);
        let mut new_states = Vec::with_capacity(compacted_capacity);
        let mut new_dna_pool = Vec::with_capacity(compacted_capacity);
        let mut dna_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

        for (cell, state) in self.cells.iter().zip(self.states.iter()) {
            if !state.is_dead() {
                let mut new_cell = cell.clone();

                // Remap DNA index to new compacted pool
                let old_dna_idx = cell.dna_index;
                let new_dna_idx = if let Some(&idx) = dna_map.get(&old_dna_idx) {
                    idx
                } else {
                    let idx = new_dna_pool.len() as u32;
                    if let Some(dna) = self.dna_pool.get(old_dna_idx as usize) {
                        new_dna_pool.push(dna.clone());
                        dna_map.insert(old_dna_idx, idx);
                    }
                    idx
                };
                new_cell.dna_index = new_dna_idx;

                new_cells.push(new_cell);
                new_states.push(state.clone());
            }
        }

        let alive = new_cells.len();
        let old_dna_count = self.dna_pool.len();
        let new_dna_count = new_dna_pool.len();

        self.cells = new_cells;
        self.states = new_states;
        self.dna_pool = new_dna_pool;
        self.free_slots.clear(); // Reset - no holes anymore

        tracing::info!("🧹 COMPACTED: {} alive cells, DNA pool {} → {} (freed {})",
            alive, old_dna_count, new_dna_count, old_dna_count - new_dna_count);
    }

    /// Periodically save elite DNA (cells with gen > 10)
    /// Called every 10000 ticks to preserve genetic heritage without mass extinction
    pub fn save_elite_dna_periodic(&mut self) {
        // Collect elite cells (gen > 10) with good energy
        let mut elite_cells: Vec<(f32, &Cell, u32)> = self.cells.iter()
            .zip(self.states.iter())
            .filter(|(c, s)| !s.is_dead() && c.generation > 10 && s.energy > 0.2)
            .map(|(c, s)| (s.energy, c, c.generation))
            .collect();

        if elite_cells.is_empty() {
            return;
        }

        // Sort by generation DESC, then energy DESC
        elite_cells.sort_by(|a, b| {
            b.2.cmp(&a.2).then_with(|| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Save top 20 to memory
        let mut saved = 0;
        {
            let mut memory = self.memory.write();
            for (energy, cell, gen) in elite_cells.iter().take(20) {
                if let Some(dna) = self.dna_pool.get(cell.dna_index as usize) {
                    memory.preserve_elite(
                        dna.clone(),
                        *energy / self.config.metabolism.energy_cap,
                        *gen as u64,
                        "elite",
                    );
                    saved += 1;
                }
            }
        }

        if saved > 0 {
            tracing::info!("💾 ELITE SAVED: {} DNA (top gen: {})", saved, elite_cells[0].2);
        }
    }

    /// Calculate coherence among active cells
    pub(super) fn calculate_coherence(&self, active_states: &[(usize, f32)]) -> f32 {
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

    /// Calculate entropy of the substrate (sampled for O(1) at scale)
    pub(super) fn calculate_entropy_sampled(&self) -> f32 {
        let total_cells = self.states.len();
        let sample_size = 2000.min(total_cells);

        if sample_size == 0 {
            return 0.0;
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;

        let step = total_cells / sample_size;
        let mut active: Vec<f32> = Vec::with_capacity(sample_size);

        for i in 0..sample_size {
            let idx = (i * step + rng.gen_range(0..step.max(1))) % total_cells;
            let state = &self.states[idx];
            if !state.is_dead() {
                let activation: f32 = state.state.iter().map(|x| x.abs()).sum();
                active.push(activation);
            }
        }

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

    // NOTE: semantic_distance() removed in Session 31 - GPU handles distance calculations

    /// Calculate vector similarity (cosine similarity)
    /// NOTE: Utility function kept for potential future use
    #[allow(dead_code)]
    pub(super) fn vector_similarity(a: &[f32; SIGNAL_DIMS], b: &[f32; SIGNAL_DIMS]) -> f32 {
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
