//! Lifecycle & Utility Methods for ARIA's Substrate
//!
//! Population management, coherence calculation, and helper functions.

use super::*;

impl Substrate {
    /// Decide whether to record an episode and record it
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

    /// Natural selection - maintain population
    pub(super) fn natural_selection(&mut self) {
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
