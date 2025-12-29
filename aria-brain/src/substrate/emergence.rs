//! Emergence Detection for ARIA's Substrate
//!
//! Detects when cells align coherently and generates appropriate responses.

use super::*;

impl Substrate {
    /// Detect emergence and generate responses
    pub(super) fn detect_emergence(&self, current_tick: u64) -> Vec<OldSignal> {
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
        let _base_threshold = params.emission_threshold; // Used by spatial_inhibitor
        let response_probability = params.response_probability;
        drop(params);

        // SPATIAL INHIBITION: Get threshold based on average position of active cells (Gemini)
        // This creates regional refractory periods - recently active regions have higher thresholds
        let avg_position: Vec<f32> = if !active_states.is_empty() {
            let mut avg = vec![0.0f32; POSITION_DIMS];
            for (i, _) in &active_states {
                for (j, p) in self.states[*i].position.iter().enumerate() {
                    avg[j] += p;
                }
            }
            let n = active_states.len() as f32;
            for v in &mut avg {
                *v /= n;
            }
            avg
        } else {
            vec![0.0f32; POSITION_DIMS]
        };

        let emission_threshold = {
            let inhibitor = self.spatial_inhibitor.read();
            inhibitor.get_threshold(&avg_position)
        };

        if coherence > emission_threshold {
            // Record activity in this region for future inhibition
            {
                let mut inhibitor = self.spatial_inhibitor.write();
                inhibitor.record_activity(&avg_position, coherence, current_tick);
            }
            // ADAPTIVE: Sometimes choose not to respond (based on response_probability)
            let mut rng = rand::thread_rng();
            if rng.gen::<f32>() > response_probability {
                return Vec::new();  // ARIA chose to stay silent
            }

            // META-LEARNING: Evaluate this response (Session 14)
            // ARIA learns from ALL interactions, not just explorations
            let intensity = coherence; // Use coherence as intensity proxy
            self.evaluate_response(coherence, intensity, current_tick);

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
    pub(super) fn generate_babble(&self, state: &[f32; SIGNAL_DIMS], coherence: f32, emotional: &EmotionalState) -> OldSignal {
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
    pub(super) fn maybe_recall_memory(&self, state: &[f32; SIGNAL_DIMS], coherence: f32, current_tick: u64) -> Option<OldSignal> {
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

    /// Generate a word-based response
    pub(super) fn generate_word_response(&self, state: &[f32; SIGNAL_DIMS], coherence: f32, was_question: bool, recent_said: &[String]) -> Option<OldSignal> {
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
}
