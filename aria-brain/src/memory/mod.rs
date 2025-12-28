//! Long-term memory for ARIA
//!
//! This persists between sessions, allowing ARIA to remember
//! and learn over time.

use crate::cell::DNA;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use std::collections::HashMap;

/// Long-term memory - persisted to disk
#[derive(Serialize, Deserialize)]
pub struct LongTermMemory {
    /// Format version (for future migrations)
    pub version: u32,

    /// Elite DNA - the genetic heritage
    pub elite_dna: Vec<EliteDNA>,

    /// Learned patterns (sequences that repeat)
    pub learned_patterns: Vec<Pattern>,

    /// Stimulus-response associations
    pub associations: Vec<Association>,

    /// Important memories (high emotional moments)
    pub memories: Vec<Memory>,

    /// Global statistics
    pub stats: GlobalStats,

    /// Vocabulary learned (emergent language)
    #[serde(default)]
    pub vocabulary: HashMap<String, WordMeaning>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EliteDNA {
    pub dna: DNA,
    pub fitness_score: f32,
    pub generation: u64,
    pub specialization: String,
    pub preserved_at: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Pattern {
    /// The pattern itself (sequence of vectors)
    pub sequence: Vec<[f32; 8]>,

    /// How many times observed
    pub frequency: u64,

    /// What typically follows this pattern
    pub typical_response: [f32; 8],

    /// Emotional valence
    pub valence: f32,

    /// When first learned
    pub first_seen: u64,

    /// When last seen
    pub last_seen: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Association {
    pub stimulus: [f32; 16],
    pub response: [f32; 16],
    pub strength: f32,
    pub last_reinforced: u64,
    pub times_reinforced: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Memory {
    pub timestamp: u64,
    pub trigger: String,
    pub internal_state: [f32; 32],
    pub emotional_intensity: f32,
    pub outcome: Outcome,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Outcome {
    Positive(f32),
    Negative(f32),
    Neutral,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct GlobalStats {
    pub total_ticks: u64,
    pub total_interactions: u64,
    pub total_births: u64,
    pub total_deaths: u64,
    pub peak_population: usize,
    pub longest_lineage: u64,
    pub total_patterns_learned: u64,
    pub total_memories: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WordMeaning {
    /// The vector representation
    pub vector: [f32; 8],
    /// How confident we are in this meaning
    pub confidence: f32,
    /// Examples of usage
    pub examples: Vec<String>,
    /// Times encountered
    pub frequency: u64,
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            version: 1,
            elite_dna: Vec::new(),
            learned_patterns: Vec::new(),
            associations: Vec::new(),
            memories: Vec::new(),
            stats: GlobalStats::default(),
            vocabulary: HashMap::new(),
        }
    }

    pub fn load_or_create(path: &Path) -> Self {
        if path.exists() {
            match fs::read(path) {
                Ok(data) => {
                    match bincode::deserialize(&data) {
                        Ok(memory) => {
                            let mem: LongTermMemory = memory;
                            tracing::info!(
                                "Memory loaded: {} memories, {} patterns, {} elite DNA",
                                mem.memories.len(),
                                mem.learned_patterns.len(),
                                mem.elite_dna.len()
                            );
                            return mem;
                        }
                        Err(e) => {
                            tracing::warn!("Memory corrupted, starting fresh: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Cannot read memory file: {}", e);
                }
            }
        }

        tracing::info!("Creating new memory");
        Self::new()
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let data = bincode::serialize(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Remember an important moment
    pub fn remember(&mut self, trigger: &str, state: [f32; 32], intensity: f32, outcome: Outcome) {
        // Only remember significant moments
        if intensity > 0.3 {
            self.memories.push(Memory {
                timestamp: self.stats.total_ticks,
                trigger: trigger.to_string(),
                internal_state: state,
                emotional_intensity: intensity,
                outcome,
            });

            self.stats.total_memories += 1;

            // Keep only the most intense memories
            if self.memories.len() > 10_000 {
                self.memories.sort_by(|a, b| {
                    b.emotional_intensity.partial_cmp(&a.emotional_intensity).unwrap()
                });
                self.memories.truncate(5_000);
            }
        }
    }

    /// Learn a pattern
    pub fn learn_pattern(&mut self, sequence: Vec<[f32; 8]>, response: [f32; 8], valence: f32) {
        let current_tick = self.stats.total_ticks;

        // Check if pattern exists
        for pattern in &mut self.learned_patterns {
            if Self::sequences_similar(&pattern.sequence, &sequence) {
                pattern.frequency += 1;
                pattern.last_seen = current_tick;

                // Update response with moving average
                for (i, r) in response.iter().enumerate() {
                    pattern.typical_response[i] =
                        pattern.typical_response[i] * 0.9 + r * 0.1;
                }
                pattern.valence = pattern.valence * 0.9 + valence * 0.1;
                return;
            }
        }

        // New pattern
        self.learned_patterns.push(Pattern {
            sequence,
            frequency: 1,
            typical_response: response,
            valence,
            first_seen: current_tick,
            last_seen: current_tick,
        });

        self.stats.total_patterns_learned += 1;

        // Prune old, infrequent patterns
        if self.learned_patterns.len() > 5_000 {
            self.learned_patterns.retain(|p| {
                p.frequency > 5 || (current_tick - p.last_seen) < 100_000
            });
        }
    }

    fn sequences_similar(a: &[[f32; 8]], b: &[[f32; 8]]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let total_distance: f32 = a.iter().zip(b.iter())
            .map(|(va, vb)| {
                va.iter().zip(vb.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .sum();

        total_distance / a.len() as f32 < 0.5
    }

    /// Preserve elite DNA
    pub fn preserve_elite(&mut self, dna: DNA, fitness: f32, generation: u64, spec: &str) {
        self.elite_dna.push(EliteDNA {
            dna,
            fitness_score: fitness,
            generation,
            specialization: spec.to_string(),
            preserved_at: self.stats.total_ticks,
        });

        // Keep top 100 by fitness
        self.elite_dna.sort_by(|a, b| {
            b.fitness_score.partial_cmp(&a.fitness_score).unwrap()
        });
        self.elite_dna.truncate(100);

        // Track longest lineage
        if generation > self.stats.longest_lineage {
            self.stats.longest_lineage = generation;
        }
    }

    /// Create an association between stimulus and response
    pub fn associate(&mut self, stimulus: [f32; 16], response: [f32; 16], strength: f32) {
        let current_tick = self.stats.total_ticks;

        // Check if association exists
        for assoc in &mut self.associations {
            let stim_sim = Self::vector_similarity_16(&assoc.stimulus, &stimulus);
            if stim_sim > 0.9 {
                // Reinforce existing association
                assoc.strength = (assoc.strength + strength) / 2.0;
                assoc.last_reinforced = current_tick;
                assoc.times_reinforced += 1;

                // Update response with moving average
                for (i, r) in response.iter().enumerate() {
                    assoc.response[i] = assoc.response[i] * 0.8 + r * 0.2;
                }
                return;
            }
        }

        // New association
        self.associations.push(Association {
            stimulus,
            response,
            strength,
            last_reinforced: current_tick,
            times_reinforced: 1,
        });

        // Prune weak associations
        if self.associations.len() > 1_000 {
            self.associations.retain(|a| a.strength > 0.1 || a.times_reinforced > 10);
        }
    }

    fn vector_similarity_16(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < 0.001 || mag_b < 0.001 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    /// Learn a word meaning
    pub fn learn_word(&mut self, word: &str, vector: [f32; 8], example: &str) {
        if let Some(meaning) = self.vocabulary.get_mut(word) {
            // Update existing meaning
            meaning.frequency += 1;
            meaning.confidence = (meaning.confidence + 0.1).min(1.0);

            // Blend vectors
            for (i, v) in vector.iter().enumerate() {
                meaning.vector[i] = meaning.vector[i] * 0.9 + v * 0.1;
            }

            // Add example
            if meaning.examples.len() < 10 {
                meaning.examples.push(example.to_string());
            }
        } else {
            // New word
            self.vocabulary.insert(word.to_string(), WordMeaning {
                vector,
                confidence: 0.1,
                examples: vec![example.to_string()],
                frequency: 1,
            });
        }
    }

    /// Find response for a similar stimulus
    pub fn recall(&self, stimulus: &[f32; 16]) -> Option<([f32; 16], f32)> {
        let mut best_match: Option<(&Association, f32)> = None;

        for assoc in &self.associations {
            let sim = Self::vector_similarity_16(&assoc.stimulus, stimulus);
            if sim > 0.5 {
                match best_match {
                    Some((_, best_sim)) if sim > best_sim => {
                        best_match = Some((assoc, sim));
                    }
                    None => {
                        best_match = Some((assoc, sim));
                    }
                    _ => {}
                }
            }
        }

        best_match.map(|(assoc, sim)| (assoc.response, sim * assoc.strength))
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_creation() {
        let memory = LongTermMemory::new();
        assert_eq!(memory.version, 1);
        assert!(memory.memories.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.memory");

        let mut memory = LongTermMemory::new();
        memory.remember("test", [0.0; 32], 0.5, Outcome::Positive(1.0));

        memory.save(&path).unwrap();

        let loaded = LongTermMemory::load_or_create(&path);
        assert_eq!(loaded.memories.len(), 1);
    }

    #[test]
    fn test_pattern_learning() {
        let mut memory = LongTermMemory::new();

        let seq = vec![[0.5; 8]];
        memory.learn_pattern(seq.clone(), [0.5; 8], 0.5);
        memory.learn_pattern(seq.clone(), [0.5; 8], 0.6);

        assert_eq!(memory.learned_patterns.len(), 1);
        assert_eq!(memory.learned_patterns[0].frequency, 2);
    }
}
