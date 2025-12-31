//! # Tension - Physical Signal Encoding
//!
//! Instead of encoding text semantically (character codes, word meanings),
//! we encode the PHYSICAL TENSION of the message.
//!
//! ARIA doesn't understand language - she feels the vibration of communication.
//!
//! ## 8D Tension Vector
//!
//! Each dimension represents a physical quality of the input:
//!
//! 0. **Arousal**: calm (0) ↔ excited (1)
//! 1. **Valence**: negative (-1) ↔ positive (1)
//! 2. **Urgency**: patient (0) ↔ urgent (1)
//! 3. **Intensity**: whisper (0) ↔ shout (1)
//! 4. **Rhythm**: choppy (0) ↔ flowing (1)
//! 5. **Weight**: light (0) ↔ heavy (1)
//! 6. **Complexity**: simple (0) ↔ complex (1)
//! 7. **Novelty**: repetitive (0) ↔ varied (1)
//!
//! ## Stochastic Variation (Gemini suggestion)
//!
//! The same word "Moka" should NOT produce the exact same vector every time.
//! A small phase noise forces cells to develop GENERALIZATION (a resonance zone)
//! rather than dead mathematical precision.
//!
//! ## Why?
//!
//! A baby doesn't understand "I love you" semantically.
//! But she FEELS the warmth, the softness, the rhythm.
//! ARIA should be the same - resonating with vibration, not meaning.

use crate::SIGNAL_DIMS;
use rand::Rng;

/// 8D Tension vector - the physical qualities of a message
pub type TensionVector = [f32; SIGNAL_DIMS];

/// Convert text to a tension vector (8D)
///
/// This captures the PHYSICAL quality of the text, not its semantic meaning.
pub fn text_to_tension(text: &str) -> TensionVector {
    let mut tension = [0.0f32; SIGNAL_DIMS];

    if text.is_empty() {
        return tension;
    }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len() as f32;
    let lower = text.to_lowercase();

    // === 0. AROUSAL (calm ↔ excited) ===
    // Exclamation marks, caps, punctuation density
    let exclamations = chars.iter().filter(|c| **c == '!').count() as f32;
    let caps_ratio = chars.iter().filter(|c| c.is_uppercase()).count() as f32 / len.max(1.0);
    let punct_ratio = chars.iter().filter(|c| c.is_ascii_punctuation()).count() as f32 / len.max(1.0);
    tension[0] = (exclamations * 0.3 + caps_ratio * 0.4 + punct_ratio * 0.3).min(1.0);

    // === 1. VALENCE (negative ↔ positive) ===
    // Positive words push up, negative push down
    let positive_markers = ["<3", ":)", "love", "aime", "bien", "oui", "yes", "merci",
                           "thank", "happy", "joy", "bon", "good", "bravo", "super"];
    let negative_markers = [":(", "hate", "no", "non", "bad", "mal", "triste", "sad",
                           "peur", "fear", "colère", "angry", "hurt", "never", "jamais"];

    let pos_count = positive_markers.iter().filter(|m| lower.contains(*m)).count() as f32;
    let neg_count = negative_markers.iter().filter(|m| lower.contains(*m)).count() as f32;
    let valence_raw = (pos_count - neg_count) / (pos_count + neg_count + 1.0);
    tension[1] = valence_raw.clamp(-1.0, 1.0);

    // === 2. URGENCY (patient ↔ urgent) ===
    // Short messages, repeated punctuation, questions
    let question_marks = chars.iter().filter(|c| **c == '?').count() as f32;
    let short_msg = if len < 10.0 { 0.5 } else if len < 20.0 { 0.3 } else { 0.1 };
    let repeated_punct = text.contains("!!") || text.contains("??") || text.contains("...");
    tension[2] = (question_marks * 0.2 + short_msg + if repeated_punct { 0.3 } else { 0.0 }).min(1.0);

    // === 3. INTENSITY (whisper ↔ shout) ===
    // All caps = shouting, long messages = more intensity
    let all_caps = chars.iter().all(|c| !c.is_alphabetic() || c.is_uppercase());
    let length_intensity = (len / 100.0).min(0.5);
    tension[3] = if all_caps && len > 3.0 { 0.9 } else { length_intensity + caps_ratio * 0.3 };

    // === 4. RHYTHM (choppy ↔ flowing) ===
    // Word length variance, punctuation frequency
    let words: Vec<&str> = text.split_whitespace().collect();
    let avg_word_len = if words.is_empty() { 0.0 } else {
        words.iter().map(|w| w.len() as f32).sum::<f32>() / words.len() as f32
    };
    let word_count = words.len() as f32;
    let space_ratio = chars.iter().filter(|c| c.is_whitespace()).count() as f32 / len.max(1.0);

    // Flowing = good spacing, moderate word length
    let flow = space_ratio * 0.5 + (avg_word_len.clamp(3.0, 8.0) - 3.0) / 10.0;
    tension[4] = flow.clamp(0.0, 1.0);

    // === 5. WEIGHT (light ↔ heavy) ===
    // Long words = heavy, short messages = light
    let heavy_words = words.iter().filter(|w| w.len() > 8).count() as f32;
    let weight = (heavy_words / word_count.max(1.0)) * 0.5 + (len / 200.0).min(0.5);
    tension[5] = weight.clamp(0.0, 1.0);

    // === 6. COMPLEXITY (simple ↔ complex) ===
    // Unique chars, punctuation variety, numbers
    let unique_chars = {
        let mut seen = std::collections::HashSet::new();
        for c in chars.iter().filter(|c| c.is_alphanumeric()) {
            seen.insert(c.to_lowercase().next().unwrap_or(*c));
        }
        seen.len() as f32
    };
    let has_numbers = chars.iter().any(|c| c.is_numeric());
    let punct_variety = chars.iter()
        .filter(|c| c.is_ascii_punctuation())
        .collect::<std::collections::HashSet<_>>()
        .len() as f32;

    tension[6] = ((unique_chars / 26.0).min(1.0) * 0.4
                 + punct_variety * 0.1
                 + if has_numbers { 0.2 } else { 0.0 })
                 .min(1.0);

    // === 7. NOVELTY (repetitive ↔ varied) ===
    // Character repetition = low novelty
    let char_repetition = {
        let mut max_repeat = 0;
        let mut current_repeat = 1;
        let mut last_char = ' ';
        for c in chars.iter() {
            if *c == last_char {
                current_repeat += 1;
                max_repeat = max_repeat.max(current_repeat);
            } else {
                current_repeat = 1;
            }
            last_char = *c;
        }
        max_repeat as f32
    };

    let repetition_penalty = (char_repetition - 1.0).max(0.0) * 0.1;
    let variety = (unique_chars / len.max(1.0)).clamp(0.0, 1.0);
    tension[7] = (variety - repetition_penalty).clamp(0.0, 1.0);

    // === STOCHASTIC VARIATION (Gemini suggestion) ===
    // Add small random noise to prevent "dead mathematical precision"
    // The same "Moka" at morning should vibrate slightly differently than at evening
    // This forces cells to develop GENERALIZATION rather than exact matching
    let mut rng = rand::thread_rng();
    let noise_scale = 0.1; // 10% variation
    for t in tension.iter_mut() {
        let noise = rng.gen_range(-noise_scale..noise_scale);
        *t = (*t + noise).clamp(-1.0, 1.0);
    }

    tension
}

/// Calculate the overall intensity from a tension vector
pub fn tension_intensity(tension: &TensionVector) -> f32 {
    let arousal = tension[0];
    let urgency = tension[2];
    let intensity_dim = tension[3];

    // Intensity is a combination of arousal, urgency, and the intensity dimension
    ((arousal + urgency + intensity_dim) / 3.0).clamp(0.1, 1.0)
}

/// Convert tension to cell-space position [-10, 10]
pub fn tension_to_position(tension: &TensionVector) -> [f32; SIGNAL_DIMS] {
    let mut position = [0.0f32; SIGNAL_DIMS];
    for i in 0..SIGNAL_DIMS {
        // Scale from [-1, 1] to [-10, 10]
        // But tension[1] (valence) is in [-1, 1], others in [0, 1]
        if i == 1 {
            position[i] = tension[i] * 10.0; // valence: -1..1 → -10..10
        } else {
            position[i] = tension[i] * 20.0 - 10.0; // 0..1 → -10..10
        }
    }
    position
}

/// Expand 8D tension to 16D with harmonics (Gemini suggestion)
///
/// The 8D tension vector is too "flat" for a 16D cell space.
/// Harmonics create a spectral signature that resonates differently
/// at different positions in the 16D substrate.
///
/// Dimensions 0-7: Base signal (tension direct)
/// Dimensions 8-15: Harmonics (cross-products and modulations)
pub fn tension_to_harmonics_16d(tension: &TensionVector) -> [f32; 16] {
    let mut harmonics = [0.0f32; 16];

    // First 8 dimensions: direct tension values (scaled to cell space)
    for i in 0..SIGNAL_DIMS {
        if i == 1 {
            harmonics[i] = tension[i] * 10.0; // valence: -1..1 → -10..10
        } else {
            harmonics[i] = tension[i] * 20.0 - 10.0; // 0..1 → -10..10
        }
    }

    // Dimensions 8-15: Harmonic combinations
    // These create "overtones" that differentiate signals spatially

    // Arousal × Valence interaction (emotional intensity direction)
    harmonics[8] = tension[0] * tension[1] * 10.0;

    // Urgency × Intensity (pressure)
    harmonics[9] = tension[2] * tension[3] * 10.0;

    // Rhythm × Weight (physical texture)
    harmonics[10] = tension[4] * tension[5] * 10.0;

    // Complexity × Novelty (information density)
    harmonics[11] = tension[6] * tension[7] * 10.0;

    // Second-order harmonics: differences
    harmonics[12] = (tension[0] - tension[3]).abs() * 10.0 - 5.0; // Arousal vs Intensity
    harmonics[13] = (tension[1] - tension[2]).abs() * 10.0 - 5.0; // Valence vs Urgency

    // Phase harmonics: sine-like modulations
    harmonics[14] = (tension[0] * std::f32::consts::PI).sin() * tension[4] * 10.0;
    harmonics[15] = (tension[1] * std::f32::consts::PI).cos() * tension[5] * 10.0;

    harmonics
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calm_message() {
        let tension = text_to_tension("bonjour");
        assert!(tension[0] < 0.3, "calm message should have low arousal");
        assert!(tension[3] < 0.5, "calm message should have low intensity");
    }

    #[test]
    fn test_excited_message() {
        let tension = text_to_tension("WOW AMAZING!!!");
        assert!(tension[0] > 0.5, "excited message should have high arousal");
        assert!(tension[3] > 0.5, "excited message should have high intensity");
    }

    #[test]
    fn test_positive_valence() {
        let tension = text_to_tension("je t'aime <3");
        assert!(tension[1] > 0.0, "loving message should have positive valence");
    }

    #[test]
    fn test_negative_valence() {
        let tension = text_to_tension("je suis triste :(");
        assert!(tension[1] < 0.0, "sad message should have negative valence");
    }

    #[test]
    fn test_urgent_message() {
        let tension = text_to_tension("HELP!!!");
        assert!(tension[2] > 0.5, "urgent message should have high urgency");
    }
}
