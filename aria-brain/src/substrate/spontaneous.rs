//! Spontaneous Speech & Dreams for ARIA's Substrate
//!
//! Handles ARIA's autonomous behavior when not being addressed.

use super::*;

impl Substrate {
    /// Maybe speak spontaneously (when bored, lonely, etc.)
    pub(super) fn maybe_speak_spontaneously(&self, current_tick: u64) -> Option<OldSignal> {
        if current_tick % 100 != 0 {
            return None;
        }

        // Use separate cooldown for spontaneous speech (500 ticks = ~2 seconds)
        let last_spont = self.last_spontaneous_tick.load(Ordering::Relaxed);
        if current_tick.saturating_sub(last_spont) < 500 {
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

        // Check bored FIRST (exploration has priority over lonely thinking)
        let (label, intensity) = if is_bored {
            // META-LEARNING DRIVEN EXPLORATION (Session 14)
            // ARIA selects her exploration strategy and evaluates herself
            let all_favorites: Vec<String> = memory.word_frequencies.iter()
                .filter(|(_, freq)| freq.emotional_valence > 0.2 && freq.count > 1)
                .map(|(word, _)| word.clone())
                .collect();

            if all_favorites.len() >= 2 {
                // Get word pair first (needs read lock)
                let word_pair = memory.get_novel_combination(&all_favorites, current_tick);
                drop(memory); // Release read lock

                // Step 1: Select exploration strategy using MetaLearner (needs write lock)
                let strategy = {
                    let mut memory_write = self.memory.write();
                    memory_write.meta_learner.select_strategy(current_tick)
                };
                tracing::info!("🧠 META: Selected strategy '{}'", strategy.name());

                // Step 2: Get word pair based on strategy (if not already found)
                // Note: For now we use novel combination; strategy-specific will be enhanced later
                let final_word_pair = word_pair;

                if let Some((w1, w2)) = final_word_pair {
                    let combination = format!("{}+{}", w1, w2);
                    tracing::info!("🔍 EXPLORING ({}): trying '{}'", strategy.name(), combination);

                    // Log exploration and store for feedback
                    {
                        let mut memory_write = self.memory.write();
                        memory_write.log_exploration(&combination, current_tick, 0.35);
                    }
                    {
                        let mut last_expl = self.last_exploration.write();
                        *last_expl = Some(combination.clone());
                    }

                    (format!("explore:{}+{}|strategy:{}|emotion:~", w1, w2, strategy.name()), 0.35)
                } else {
                    // Fallback to random if exploration fails
                    let w1 = &all_favorites[rng.gen_range(0..all_favorites.len())];
                    let w2 = &all_favorites[rng.gen_range(0..all_favorites.len())];
                    if w1 != w2 {
                        tracing::info!("SPONTANEOUS (bored): combining '{}' + '{}'", w1, w2);
                        (format!("phrase:{}+{}|emotion:~", w1, w2), 0.35)
                    } else {
                        ("spontaneous:bored|emotion:~".to_string(), 0.25)
                    }
                }
            } else {
                ("spontaneous:bored|emotion:~".to_string(), 0.25)
            }
        } else if is_lonely {
            // Lonely but not bored - think about favorite words
            if let Some(word) = favorite_word.clone() {
                tracing::info!("SPONTANEOUS (lonely): thinking of '{}'", word);
                (format!("spontaneous:{}|emotion:?", word), 0.3)
            } else {
                ("spontaneous:attention|emotion:?".to_string(), 0.2)
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

        // Update spontaneous tick (separate from regular emission cooldown)
        self.last_spontaneous_tick.store(current_tick, Ordering::Relaxed);
        Some(signal)
    }

    /// Dream and consolidate memories when inactive
    pub(super) fn maybe_dream(&self, current_tick: u64) {
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

        // Consolidate short-term memory (Gemini suggestion)
        memory.consolidate(current_tick);

        if rng.gen::<f32>() < 0.1 {
            tracing::info!("💭 DREAMING: thinking about '{}'...", dream_word);
        }
    }
}
