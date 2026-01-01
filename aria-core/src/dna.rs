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
//!
//! ## Adaptive Neuroplasticity (Gemini optimization)
//!
//! Mutation rates are now **adaptive** based on:
//! - Cell age (older = more stable, less mutation)
//! - Fitness (successful DNA mutates less to preserve good traits)
//! - Activity (more active = more exploration = more mutation)
//! - Exploration bonus (explicit exploration gets higher mutation)

use bytemuck::{Pod, Zeroable};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::DNA_DIMS;

/// Context for adaptive mutation (Gemini neuroplasticity)
///
/// This allows mutation rates to vary based on the cell's history and context.
#[derive(Clone, Copy, Debug, Default)]
pub struct MutationContext {
    /// Cell age in ticks (older = more stable)
    pub age: u64,
    /// Fitness score 0.0-1.0 (higher = better performing)
    pub fitness: f32,
    /// Recent activity level 0.0-1.0 (higher = more active)
    pub activity: f32,
    /// Is this cell actively exploring? (boosts mutation)
    pub exploring: bool,
    /// Is this DNA from the elite pool? (reduces mutation)
    pub is_elite: bool,
}

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
    /// - [7]: Reflexivity gain (how sensitive the cell is to ARIA's own thoughts)
    pub thresholds: [f32; DNA_DIMS],

    /// Genes controlling reactions (how to respond)
    /// - [0-7]: Multipliers for signal dimensions during processing
    pub reactions: [f32; DNA_DIMS],

    /// Unique signature - genetic fingerprint
    pub signature: u64,

    /// Structural checksum - validates code integrity for auto-evolution
    pub structural_checksum: u64,
}

impl DNA {
    /// Create random DNA for a new cell
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            thresholds: std::array::from_fn(|_| rng.gen()),
            reactions: std::array::from_fn(|_| rng.gen()),
            signature: rng.gen(),
            structural_checksum: 0, // Initial structural code state
        }
    }

    /// Create DNA from a parent with mutations
    pub fn from_parent(parent: &DNA, mutation_rate: f32) -> Self {
        let mut child = *parent;
        child.mutate(mutation_rate);
        child.signature = rand::thread_rng().gen();
        child
    }

    /// Create DNA from parent with adaptive mutation (Gemini neuroplasticity)
    ///
    /// The mutation rate is computed based on the cell's context:
    /// - Older cells: lower mutation (stability)
    /// - Higher fitness: lower mutation (preserve good traits)
    /// - Higher activity: higher mutation (exploration)
    /// - Exploring flag: bonus mutation
    /// - Elite DNA: minimal mutation
    pub fn from_parent_adaptive(parent: &DNA, base_rate: f32, ctx: MutationContext) -> Self {
        let mut child = *parent;
        let (rate, magnitude) = Self::compute_adaptive_mutation(base_rate, ctx);
        child.mutate_adaptive(rate, magnitude);
        child.signature = rand::thread_rng().gen();
        child
    }

    /// Compute adaptive mutation rate based on context (Gemini neuroplasticity)
    ///
    /// Returns (rate, magnitude) tuple where:
    /// - rate: probability of mutation per gene (0.0-1.0)
    /// - magnitude: size of mutation when it occurs (0.0-0.3)
    pub fn compute_adaptive_mutation(base_rate: f32, ctx: MutationContext) -> (f32, f32) {
        // Age factor: older cells mutate less (stability)
        // Reaches 0.5x at 10000 ticks, 0.25x at 20000 ticks
        let age_factor = 1.0 / (1.0 + (ctx.age as f32 / 10000.0));

        // Fitness factor: successful DNA mutates less
        // 0.0 fitness = 1.5x mutation, 1.0 fitness = 0.2x mutation
        let fitness_factor = 1.5 - ctx.fitness * 1.3;

        // Activity factor: more active = more mutation (exploration)
        // 0.0 activity = 0.5x, 1.0 activity = 1.5x
        let activity_factor = 0.5 + ctx.activity;

        // Exploration bonus: explicit exploration gets 2x mutation
        let exploration_bonus = if ctx.exploring { 2.0 } else { 1.0 };

        // Elite protection: elite DNA mutates at 20% rate
        let elite_factor = if ctx.is_elite { 0.2 } else { 1.0 };

        // Combine all factors
        let rate = (base_rate * age_factor * fitness_factor * activity_factor
            * exploration_bonus
            * elite_factor)
            .clamp(0.01, 0.5); // Min 1%, max 50% mutation rate

        // Magnitude also varies: younger/exploring cells have larger mutations
        // Young cells (age_factor ~1.0) get larger mutations
        // Old cells (age_factor ~0.16 at 50k ticks) get smaller mutations
        let base_magnitude = 0.1;
        let magnitude = (base_magnitude * age_factor * activity_factor * exploration_bonus
            / elite_factor.max(0.5)) // Elite gets smaller mutations
            .clamp(0.05, 0.3);

        (rate, magnitude)
    }

    /// Mutate this DNA in place
    pub fn mutate(&mut self, rate: f32) {
        self.mutate_adaptive(rate, 0.1); // Default magnitude 0.1
    }

    /// Mutate this DNA with adaptive magnitude (Gemini neuroplasticity)
    pub fn mutate_adaptive(&mut self, rate: f32, magnitude: f32) {
        let mut rng = rand::thread_rng();

        for t in &mut self.thresholds {
            if rng.gen::<f32>() < rate {
                // Random mutation within [-magnitude, +magnitude]
                let delta = rng.gen::<f32>() * magnitude * 2.0 - magnitude;
                *t = (*t + delta).clamp(0.0, 1.0);
            }
        }

        for r in &mut self.reactions {
            if rng.gen::<f32>() < rate {
                let delta = rng.gen::<f32>() * magnitude * 2.0 - magnitude;
                *r = (*r + delta).clamp(0.0, 1.0);
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

    /// Get the reflexivity gain (how much self-thoughts influence this cell)
    #[inline]
    pub fn reflexivity_gain(&self) -> f32 {
        self.thresholds[7]
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

    #[test]
    fn test_adaptive_mutation_factors() {
        let base_rate = 0.1;

        // Young, active, exploring cell should have high mutation
        let young_exploring = MutationContext {
            age: 100,
            fitness: 0.2,
            activity: 0.8,
            exploring: true,
            is_elite: false,
        };
        let (rate1, mag1) = DNA::compute_adaptive_mutation(base_rate, young_exploring);

        // Old, elite, stable cell should have low mutation
        let old_elite = MutationContext {
            age: 50000,
            fitness: 0.9,
            activity: 0.1,
            exploring: false,
            is_elite: true,
        };
        let (rate2, mag2) = DNA::compute_adaptive_mutation(base_rate, old_elite);

        // Young exploring should mutate more
        assert!(rate1 > rate2, "Young exploring ({}) should mutate more than old elite ({})", rate1, rate2);
        assert!(mag1 > mag2, "Young exploring magnitude ({}) should be larger than old elite ({})", mag1, mag2);

        // Check bounds
        assert!(rate1 <= 0.5 && rate1 >= 0.01, "Rate {} out of bounds", rate1);
        assert!(rate2 <= 0.5 && rate2 >= 0.01, "Rate {} out of bounds", rate2);
    }

    #[test]
    fn test_adaptive_mutation_changes_dna() {
        let parent = DNA::random();
        let ctx = MutationContext {
            age: 0,
            fitness: 0.0,
            activity: 1.0,
            exploring: true,
            is_elite: false,
        };

        let child = DNA::from_parent_adaptive(&parent, 1.0, ctx);

        // With high mutation context, DNA should change
        assert_ne!(parent.signature, child.signature);
        // Some genes should have changed (probabilistic but very likely with rate=1.0)
        assert!(parent.thresholds != child.thresholds || parent.reactions != child.reactions,
            "Adaptive mutation should change DNA");
    }
}
