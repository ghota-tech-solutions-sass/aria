//! # DNA - The Genetic Code of Cells
//!
//! DNA defines how a cell behaves: when it acts, how it reacts,
//! and how it connects to others.
//!
//! ## Structure (64 bytes total, GPU-aligned)
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │ thresholds [8 × f32] - When to trigger actions             │
//! │ reactions  [8 × f32] - How strongly to react to signals    │
//! │ signature  [u64]     - Unique genetic fingerprint          │
//! └────────────────────────────────────────────────────────────┘
//! ```

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::DNA_DIMS;

/// Computational DNA - defines the "character" of a cell
///
/// This structure is `repr(C)` and implements `Pod` for direct GPU transfer.
/// Total size: 72 bytes (8*4 + 8*4 + 8 = 72, padded to 80 for alignment)
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct DNA {
    /// Genes controlling thresholds (when to act)
    /// - [0]: Action threshold (tension needed to act)
    /// - [1]: Division threshold (energy needed to reproduce)
    /// - [2]: Connection threshold (when to form new connections)
    /// - [3]: Signal threshold (when to emit signals)
    /// - [4]: Movement threshold (when to move in semantic space)
    /// - [5]: Sleep threshold (when to enter dormant state)
    /// - [6]: Wake threshold (stimulus needed to wake up)
    /// - [7]: Reserved for future evolution
    pub thresholds: [f32; DNA_DIMS],

    /// Genes controlling reactions (how to respond)
    /// - [0-7]: Multipliers for signal dimensions during processing
    pub reactions: [f32; DNA_DIMS],

    /// Unique signature - genetic fingerprint
    pub signature: u64,

    /// Padding for GPU alignment (80 bytes total)
    _pad: u64,
}

impl DNA {
    /// Create random DNA for a new cell
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            thresholds: std::array::from_fn(|_| rng.gen()),
            reactions: std::array::from_fn(|_| rng.gen()),
            signature: rng.gen(),
            _pad: 0,
        }
    }

    /// Create DNA from a parent with mutations
    pub fn from_parent(parent: &DNA, mutation_rate: f32) -> Self {
        let mut child = *parent;
        child.mutate(mutation_rate);
        child.signature = rand::thread_rng().gen();
        child
    }

    /// Mutate this DNA in place
    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();

        for t in &mut self.thresholds {
            if rng.gen::<f32>() < rate {
                *t = (*t + rng.gen::<f32>() * 0.2 - 0.1).clamp(0.0, 1.0);
            }
        }

        for r in &mut self.reactions {
            if rng.gen::<f32>() < rate {
                *r = (*r + rng.gen::<f32>() * 0.2 - 0.1).clamp(0.0, 1.0);
            }
        }
    }

    /// Sexual reproduction: crossover with another DNA
    pub fn crossover(&self, other: &DNA) -> DNA {
        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(0..DNA_DIMS);

        let mut child = *self;

        for i in crossover_point..DNA_DIMS {
            child.thresholds[i] = other.thresholds[i];
            child.reactions[i] = other.reactions[i];
        }

        child.signature = rng.gen();
        child.mutate(0.05); // Slight mutation on reproduction
        child
    }

    /// Calculate similarity with another DNA (0.0 = different, 1.0 = identical)
    pub fn similarity(&self, other: &DNA) -> f32 {
        let threshold_sim: f32 = self
            .thresholds
            .iter()
            .zip(other.thresholds.iter())
            .map(|(a, b)| 1.0 - (a - b).abs())
            .sum::<f32>()
            / DNA_DIMS as f32;

        let reaction_sim: f32 = self
            .reactions
            .iter()
            .zip(other.reactions.iter())
            .map(|(a, b)| 1.0 - (a - b).abs())
            .sum::<f32>()
            / DNA_DIMS as f32;

        (threshold_sim + reaction_sim) / 2.0
    }

    /// Get the action threshold
    #[inline]
    pub fn action_threshold(&self) -> f32 {
        self.thresholds[0] * 2.0 + 0.5
    }

    /// Get the division threshold
    #[inline]
    pub fn division_threshold(&self) -> f32 {
        self.thresholds[1]
    }

    /// Get the sleep threshold (below this activity level, cell sleeps)
    #[inline]
    pub fn sleep_threshold(&self) -> f32 {
        self.thresholds[5] * 0.01 // Very small changes trigger sleep
    }

    /// Get the wake threshold (stimulus needed to wake)
    #[inline]
    pub fn wake_threshold(&self) -> f32 {
        self.thresholds[6] * 0.1
    }
}

impl Default for DNA {
    fn default() -> Self {
        Self::random()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dna_size() {
        assert_eq!(std::mem::size_of::<DNA>(), 80);
    }

    #[test]
    fn test_dna_mutation() {
        let mut dna = DNA::random();
        let original = dna;
        dna.mutate(1.0);
        // With 100% mutation rate, something should change
        assert!(dna.thresholds != original.thresholds || dna.reactions != original.reactions);
    }

    #[test]
    fn test_dna_crossover() {
        let parent1 = DNA::random();
        let parent2 = DNA::random();
        let child = parent1.crossover(&parent2);
        assert_ne!(child.signature, parent1.signature);
        assert_ne!(child.signature, parent2.signature);
    }
}
