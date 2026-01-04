//! Spontaneous Activity & Dreams for ARIA's Substrate
//!
//! ## Physical Intelligence (Session 20)
//!
//! ARIA no longer "speaks words" spontaneously.
//! She emits spontaneous TENSION pulses based on her internal state.
//! These are raw physical vibrations, not language.

use super::*;

impl Substrate {
    /// Maybe emit spontaneous tension (when bored, lonely, etc.)
    ///
    /// **Physical Intelligence**: No words - just tension pulses.
    /// ARIA's spontaneous activity is a physical vibration, not speech.
    pub(super) fn maybe_speak_spontaneously(&self, current_tick: u64) -> Option<OldSignal> {
        if current_tick % 100 != 0 {
            return None;
        }

        // Cooldown between spontaneous emissions (~5 seconds)
        let last_spont = self.last_spontaneous_tick.load(Ordering::Relaxed);
        if current_tick.saturating_sub(last_spont) < 5000 {
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

        let probability = base_prob * (spontaneity * 10.0);

        if random > probability {
            return None;
        }

        // === PHYSICAL INTELLIGENCE: Generate spontaneous tension ===
        // Build tension vector from emotional state
        let mut tension = [0.0f32; SIGNAL_DIMS];

        // Map emotional state to tension dimensions
        tension[0] = emotional.arousal;           // Arousal
        tension[1] = emotional.happiness;         // Valence
        tension[2] = if is_lonely { 0.7 } else { emotional.boredom }; // Urgency
        tension[3] = if is_excited { 0.8 } else { 0.3 };  // Intensity
        tension[4] = if is_happy { 0.7 } else { 0.3 };    // Rhythm (flowing)
        tension[5] = emotional.comfort;           // Weight
        tension[6] = emotional.curiosity;         // Complexity
        tension[7] = if is_bored { 0.3 } else { 0.6 };    // Novelty

        // Generate label for debugging
        let state_name = if is_bored {
            "restless"
        } else if is_lonely {
            "seeking"
        } else if is_happy && is_excited {
            "joyful"
        } else if is_excited {
            "active"
        } else if is_curious {
            "exploring"
        } else {
            "ambient"
        };

        let label = format!("spontaneous:{}|arousal:{:.1}|valence:{:.1}",
            state_name, emotional.arousal, emotional.happiness);

        tracing::info!("⚡ SPONTANEOUS: {} (bored={:.2}, lonely={})",
            state_name, emotional.boredom, is_lonely);

        let intensity = if is_excited { 0.5 }
            else if is_bored || is_lonely { 0.35 }
            else if is_happy { 0.4 }
            else { 0.25 };

        let mut signal = OldSignal::from_vector(tension, label);
        signal.intensity = intensity;

        self.last_spontaneous_tick.store(current_tick, Ordering::Relaxed);
        Some(signal)
    }

    /// Dream and consolidate memories when inactive
    ///
    /// **Physical Intelligence**: Dreams strengthen resonance patterns.
    /// No word rehearsal - just pattern consolidation.
    pub(super) fn maybe_dream(&self, current_tick: u64) {
        if current_tick % 500 != 0 {
            return;
        }

        let last_interaction = self.last_interaction_tick.load(Ordering::Relaxed);
        if current_tick.saturating_sub(last_interaction) < 1000 {
            return;
        }

        // Consolidate memory (Hebbian patterns, not words)
        {
            let mut memory = self.memory.write();
            memory.consolidate(current_tick);
        }

        // Occasionally log dreaming
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < 0.1 {
            tracing::info!("💭 DREAMING: consolidating resonance patterns...");
        }
    }
}
