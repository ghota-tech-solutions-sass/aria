//! Signal Processing for ARIA's Substrate
//!
//! Handles external signal injection and internal signal propagation.

use super::*;

impl Substrate {
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
        // La Vraie Faim: Don't scale down intensity for small populations!
        // Minimum scale of 1.0 ensures cells can be fed even with few cells
        let cell_scale = (self.cells.len() as f32 / 10_000.0).max(1.0);
        let base_intensity = signal.intensity * 5.0 * familiarity_boost * cell_scale;

        // Get target position and convert to cell space [-10, 10]
        // Signal content values are typically in [0, 1], cells are in [-10, 10]
        let mut target_position_8d = [0.0f32; 8];
        let mut target_position_16d = [0.0f32; 16];
        for i in 0..16 {
            // Scale from [0, 1] to [-10, 10]
            let v = signal_vector.get(i).copied().unwrap_or(0.0);
            let scaled = v * 20.0 - 10.0;
            target_position_16d[i] = scaled;
            if i < 8 {
                target_position_8d[i] = scaled;
            }
        }

        // Create fragment with scaled position for GPU spatial hashing
        let fragment = SignalFragment::external_at(signal_vector, target_position_8d, base_intensity);

        // Keep 16D for CPU-side processing
        let target_position = target_position_16d;

        tracing::info!("V2 Signal received: '{}' intensity={:.2} (boost: {:.2})",
            signal.label, fragment.intensity, familiarity_boost);

        // Distribute to cells - OPTIMIZED: skip dead/sleeping cells
        // Only wake sleeping cells, don't process them
        let mut processed_count = 0u32;
        for (i, state) in self.states.iter_mut().enumerate() {
            // Skip dead cells entirely
            if state.is_dead() {
                continue;
            }

            let distance = Self::semantic_distance(&state.position, &target_position);
            let attenuation = (1.0 / (1.0 + distance * 0.1)).max(0.2);
            let attenuated_intensity = fragment.intensity * attenuation;

            // Wake sleeping cells if stimulus is strong enough
            if state.is_sleeping() {
                if attenuated_intensity > self.config.activity.wake_threshold {
                    self.cells[i].activity.wake();
                    state.set_sleeping(false);
                } else {
                    // Skip sleeping cells that don't wake
                    continue;
                }
            }

            // Only process awake cells (includes just-woken)
            // Direct activation
            for (j, s) in fragment.content.iter().enumerate() {
                if j < SIGNAL_DIMS {
                    state.state[j] += s * attenuated_intensity * 5.0;
                }
            }

            // LA VRAIE FAIM: No direct energy boost!
            // Cells must earn energy through resonance (handled by backend)

            processed_count += 1;
            // Limit processing to avoid lag on large populations
            if processed_count > 10_000 {
                break;
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

        // === VISUAL-LINGUISTIC RESPONSE ===
        // If this is a visual signal, check if ARIA recognizes it and can name it
        let visual_emergences = self.process_visual_signal(&signal, current_tick);

        // Check for immediate emergence
        let mut emergences = self.detect_emergence(current_tick);
        emergences.extend(visual_emergences);
        emergences
    }

    /// Process visual signals - ARIA speaks what she sees
    pub(super) fn process_visual_signal(&self, signal: &OldSignal, current_tick: u64) -> Vec<OldSignal> {
        // Only process Visual signals
        if signal.signal_type != OldSignalType::Visual {
            return Vec::new();
        }

        // Extract visual signature from signal content
        let mut signature = [0.0f32; 32];
        for (i, v) in signal.content.iter().take(32).enumerate() {
            signature[i] = *v;
        }

        // Check memory for visual-word links
        let memory = self.memory.read();
        let suggested_words = memory.visual_to_words(&signature);

        if suggested_words.is_empty() {
            return Vec::new();
        }

        // Get the top word (highest confidence)
        let (top_word, confidence) = &suggested_words[0];

        // Only speak if confidence is high enough
        if *confidence < 0.5 {
            return Vec::new();
        }

        // Random chance based on response probability
        let params = self.adaptive_params.read();
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > params.response_probability * 0.5 {
            return Vec::new();
        }

        // Create an expression signal with the recognized word
        let mut expression_content = vec![0.0f32; 32];
        // Copy some of the visual features
        for i in 0..8 {
            expression_content[i] = signature[i] * 0.5;
        }

        let expression = OldSignal {
            content: expression_content,
            intensity: confidence * 0.8,
            label: format!("word:{}", top_word),
            signal_type: OldSignalType::Expression,
            timestamp: current_tick,
        };

        tracing::info!("👁️→💬 VISUAL RECOGNITION: ARIA sees '{}' (confidence: {:.2})",
            top_word, confidence);

        vec![expression]
    }

    /// Propagate a signal through the substrate
    pub(super) fn propagate_signal(&mut self, content: [f32; SIGNAL_DIMS], source_pos: [f32; POSITION_DIMS], intensity: f32) {
        // Convert 16D source position to 8D for fragment
        let mut pos_8d = [0.0f32; SIGNAL_DIMS];
        for i in 0..SIGNAL_DIMS {
            pos_8d[i] = source_pos[i];
        }
        let fragment = SignalFragment::external_at(content, pos_8d, intensity);

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
}
