//! Emergence Detection for ARIA's Substrate
//!
//! ## Physical Intelligence (Session 20)
//!
//! ARIA no longer tries to match words. She emits TENSION patterns.
//! The tension is the emergent vibration of coherently active cells.
//! What you "hear" is the physical state of her being, not language.

use super::*;

impl Substrate {
    /// Detect emergence and generate tension responses
    ///
    /// **Physical Intelligence**: No word matching.
    /// ARIA emits the tension pattern that emerges from coherent cell activity.
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

        // Calculate average tension state
        let mut emergent_tension = [0.0f32; SIGNAL_DIMS];
        for (i, _) in &active_states {
            for (j, s) in self.states[*i].state[0..SIGNAL_DIMS].iter().enumerate() {
                emergent_tension[j] += s;
            }
        }
        let n = active_states.len() as f32;
        for t in &mut emergent_tension {
            *t /= n;
        }

        // Check coherence
        let coherence = self.calculate_coherence(&active_states);

        // ADAPTIVE: Use adaptive emission threshold
        let params = self.adaptive_params.read();
        let response_probability = params.response_probability;
        drop(params);

        // SPATIAL INHIBITION: Regional refractory periods
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
            // Record activity for future inhibition
            {
                let mut inhibitor = self.spatial_inhibitor.write();
                inhibitor.record_activity(&avg_position, coherence, current_tick);
            }

            // ADAPTIVE: Sometimes choose silence
            let mut rng = rand::thread_rng();
            if rng.gen::<f32>() > response_probability {
                return Vec::new();
            }

            // META-LEARNING: Evaluate this response
            self.evaluate_response(coherence, coherence, current_tick);

            // === PHYSICAL INTELLIGENCE: Emit tension, not words ===
            let emotional = self.emotional_state.read();
            let tension_signal = self.generate_tension_response(&emergent_tension, coherence, &emotional);

            self.last_emission_tick.store(current_tick, Ordering::Relaxed);
            return vec![tension_signal];
        }

        Vec::new()
    }

    /// Generate a pure tension response
    ///
    /// The tension is the raw emergent state of ARIA's being.
    /// No word matching, no vocabulary - just vibration.
    pub(super) fn generate_tension_response(&self, tension: &[f32; SIGNAL_DIMS], coherence: f32, emotional: &EmotionalState) -> OldSignal {
        // Interpret tension dimensions for labeling (for human readability)
        // But ARIA doesn't "know" what these mean - it's just her state
        let arousal = tension[0].abs();
        let valence = tension[1];
        let urgency = tension[2].abs();

        // Generate a label that describes the tension (for debugging/display)
        let tension_label = if coherence > 0.6 {
            // High coherence = clear signal
            if valence > 0.3 {
                format!("tension:positive|{:.0}%", coherence * 100.0)
            } else if valence < -0.3 {
                format!("tension:negative|{:.0}%", coherence * 100.0)
            } else if arousal > 0.5 {
                format!("tension:excited|{:.0}%", coherence * 100.0)
            } else if urgency > 0.5 {
                format!("tension:urgent|{:.0}%", coherence * 100.0)
            } else {
                format!("tension:neutral|{:.0}%", coherence * 100.0)
            }
        } else {
            // Low coherence = diffuse state
            format!("tension:diffuse|{:.0}%", coherence * 100.0)
        };

        // Add emotional marker
        let marker = emotional.get_emotional_marker().unwrap_or("");
        let final_label = if !marker.is_empty() {
            format!("{}|emotion:{}", tension_label, marker)
        } else {
            tension_label
        };

        tracing::info!("⚡ EMERGENCE: {} (coherence={:.2}, valence={:.2})",
            final_label, coherence, valence);

        let mut signal = OldSignal::from_vector(*tension, final_label);
        signal.intensity = coherence;
        signal
    }

    // NOTE: generate_babble, maybe_recall_memory, and generate_word_response
    // removed in Session 20 (Physical Intelligence)
    // ARIA now emits tension patterns instead of words
}
