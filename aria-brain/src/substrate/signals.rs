//! Signal Processing for ARIA's Substrate
//!
//! ## Physical Intelligence (Session 20)
//!
//! ARIA no longer learns words or vocabulary.
//! Text is converted to pure TENSION vectors.
//! Cells resonate with tension patterns - survival through physics.

use super::*;

impl Substrate {
    /// Inject an external signal (from aria-body)
    ///
    /// **Physical Intelligence**: No word learning, no semantic encoding.
    /// The signal is pure tension that propagates through the substrate.
    /// Cells that resonate gain energy; cells that don't, starve.
    pub fn inject_signal(&mut self, signal: OldSignal) -> Vec<OldSignal> {
        let current_tick = self.tick.load(Ordering::Relaxed);

        // Record interaction time
        self.last_interaction_tick.store(current_tick, Ordering::Relaxed);

        // Update stats
        {
            let mut memory = self.memory.write();
            memory.stats.total_ticks = current_tick;
            memory.stats.total_interactions += 1;
        }

        // Get tension vector from signal (already computed in Signal::from_text)
        // The first 8 dimensions are the tension values
        let tension_vector = signal.to_vector();

        // Extract emotional valence from tension (index 1 = valence)
        let emotional_valence = signal.content.get(1).copied().unwrap_or(0.0);

        // Process feedback (still useful for reinforcement learning)
        self.process_feedback(&signal.label, current_tick);

        // Update emotional state from tension
        {
            let mut emotional = self.emotional_state.write();
            emotional.process_signal(&signal.content, signal.intensity, current_tick);
        }

        // === PURE TENSION INJECTION ===
        // No word learning, no familiarity boost - just raw energy injection
        let cell_scale = (self.cells.len() as f32 / 10_000.0).max(1.0);
        let base_intensity = signal.intensity * 5.0 * cell_scale;

        // Use tension-based position (already calculated in Signal::from_text)
        // Tension values map directly to cell space
        let mut target_position_8d = [0.0f32; 8];
        let mut target_position_16d = [0.0f32; 16];

        for i in 0..8 {
            // Tension is in [-1, 1] for valence, [0, 1] for others
            // Scale to cell space [-10, 10]
            let t = tension_vector.get(i).copied().unwrap_or(0.0);
            let scaled = if i == 1 {
                t * 10.0  // valence: -1..1 → -10..10
            } else {
                t * 20.0 - 10.0  // others: 0..1 → -10..10
            };
            target_position_8d[i] = scaled;
            target_position_16d[i] = scaled;
        }

        // Create fragment with tension-based position
        let fragment = SignalFragment::external_at(tension_vector, target_position_8d, base_intensity);
        let target_position = target_position_16d;

        // Log tension injection (not word reception)
        tracing::info!("⚡ TENSION: '{}' intensity={:.2} valence={:.2}",
            signal.label, fragment.intensity, emotional_valence);

        // Distribute tension to cells - wake those in resonance range
        let mut processed_count = 0u32;
        for (i, state) in self.states.iter_mut().enumerate() {
            if state.is_dead() {
                continue;
            }

            let distance = Self::semantic_distance(&state.position, &target_position);
            let attenuation = (1.0 / (1.0 + distance * 0.1)).max(0.2);
            let attenuated_intensity = fragment.intensity * attenuation;

            // Wake sleeping cells if tension is strong enough
            if state.is_sleeping() {
                if attenuated_intensity > self.config.activity.wake_threshold {
                    self.cells[i].activity.wake();
                    state.set_sleeping(false);
                } else {
                    continue;
                }
            }

            // Inject tension into cell state
            // This modifies the cell's internal vibration
            for (j, t) in fragment.content.iter().enumerate() {
                if j < SIGNAL_DIMS {
                    state.state[j] += t * attenuated_intensity * 5.0;
                }
            }

            // LA VRAIE FAIM: No direct energy!
            // Energy comes ONLY from resonance in the GPU backend

            processed_count += 1;
            if processed_count > 10_000 {
                break;
            }
        }

        // Add to signal buffer for GPU processing
        {
            let mut buffer = self.signal_buffer.write();
            buffer.push(fragment);
            if buffer.len() > 100 {
                buffer.remove(0);
            }
        }

        // Visual processing still works (images → tension)
        let visual_emergences = self.process_visual_signal(&signal, current_tick);

        // Check for emergence
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
