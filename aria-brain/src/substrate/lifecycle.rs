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
    pub(super) fn natural_selection(&mut self) {
        let target = self.config.population.target_population as usize;
        let buffer = self.config.population.population_buffer as usize;
        let min_pop = self.config.population.min_population as usize;
        let reproduction_threshold = self.config.metabolism.reproduction_threshold;
        let child_energy = self.config.metabolism.child_energy;
        let divide_cost = self.config.metabolism.cost_divide;

        let alive_count = self.cells.iter()
            .zip(self.states.iter())
            .filter(|(_, s)| !s.is_dead())
            .count();

        // === CONTINUOUS REPRODUCTION (Law of Expansion) ===
        // Life expands to fill available energy.
        // We only cap at a high "Physics Limit" to prevent OOM/crash.
        // Otherwise, if cells have energy to reproduce, they should.
        let safety_cap = target * 2; // Hard limit for system stability

        if alive_count < safety_cap {
            // Find cells ready to divide (energy > threshold)
            let ready_to_divide: Vec<(usize, u32, u32)> = self.cells.iter()
                .zip(self.states.iter())
                .enumerate()
                .filter(|(_, (_, s))| !s.is_dead() && s.energy > reproduction_threshold)
                .map(|(i, (c, _))| (i, c.dna_index, c.generation))
                .collect();

            // Limit births per tick to avoid explosion (still biological pacing)
            // But allow significantly more than before if energy is abundant.
            let room = safety_cap.saturating_sub(alive_count);
            let max_births = room.min(ready_to_divide.len()).min(500); // Increased cap from 100

            if max_births > 0 {
                tracing::debug!("🧬 EXPANSION: {} cells ready, {} dividing (pop: {})",
                    ready_to_divide.len(), max_births, alive_count);

                for (parent_idx, parent_dna_idx, parent_generation) in ready_to_divide.into_iter().take(max_births) {
                    // Parent pays the cost
                    self.states[parent_idx].energy -= divide_cost;

                    // Create child with mutation
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

                    // LINEAGE: Child = parent.generation + 1
                    let mut cell = Cell::new(new_id, dna_index);
                    cell.generation = parent_generation + 1;

                    let mut state = CellState::new();
                    state.energy = child_energy;

                    // Use free slot or append
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

        // === POPULATION CONTROL ===
        if alive_count > target + buffer {
            // Too many cells - remove weakest using PARTIAL SORT
            let mut indices: Vec<(usize, f32)> = self.states.iter()
                .enumerate()
                .filter(|(_, s)| !s.is_dead())
                .map(|(i, s)| (i, s.energy))
                .collect();

            let to_remove = alive_count - target;
            if to_remove > 0 && to_remove < indices.len() {
                // Partial sort: only find the `to_remove` weakest - O(n) instead of O(n log n)
                indices.select_nth_unstable_by(to_remove, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            for (idx, _) in indices.iter().take(to_remove) {
                self.states[*idx].set_dead();
                self.free_slots.push(*idx);
            }
        } else if alive_count < min_pop {
            // Too few cells - spawn from SURVIVORS (La Vraie Faim evolution)
            // Don't use random DNA - inherit from the elite survivors!
            let to_spawn = min_pop - alive_count;

            // Find TOP 10 survivors without sorting everything - O(n) instead of O(n log n)
            // Tuple: (cell_idx, energy, dna_index, generation) - generation for lineage tracking!
            const ELITE_COUNT: usize = 10;
            let mut top_survivors: Vec<(usize, f32, u32, u32)> = Vec::with_capacity(ELITE_COUNT);

            for (i, (cell, state)) in self.cells.iter().zip(self.states.iter()).enumerate() {
                if state.is_dead() {
                    continue;
                }
                let entry = (i, state.energy, cell.dna_index, cell.generation);

                if top_survivors.len() < ELITE_COUNT {
                    top_survivors.push(entry);
                    // Keep sorted (small vec, fast)
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
                        let state = CellState::new();
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
                        state.energy = self.config.metabolism.child_energy;

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
        let mut new_cells = Vec::with_capacity(total - dead_count + 100);
        let mut new_states = Vec::with_capacity(total - dead_count + 100);
        let mut new_dna_pool = Vec::with_capacity(total - dead_count + 100);
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

    /// Calculate entropy of the substrate
    pub(super) fn calculate_entropy(&self) -> f32 {
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

    /// Calculate semantic distance between two positions
    pub(super) fn semantic_distance(a: &[f32; POSITION_DIMS], b: &[f32; POSITION_DIMS]) -> f32 {
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

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
