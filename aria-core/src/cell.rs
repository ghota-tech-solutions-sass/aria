//! # Cell - The Fundamental Unit of Life in ARIA
//!
//! A cell is not a neuron. It's a **living entity** with:
//! - DNA that defines its behavior
//! - Energy that it needs to survive
//! - Tension that drives it to act
//! - An activity state for sparse updates
//!
//! ## Memory Layout (GPU-optimized)
//!
//! The cell is split into two parts for efficiency:
//! - `Cell`: Core data (ID, DNA reference, activity) - 32 bytes
//! - `CellState`: Dynamic state (position, state, energy) - 256 bytes
//!
//! This separation allows:
//! - Static data to stay in CPU memory
//! - Dynamic state to live in GPU buffers
//! - Sparse updates (only transfer active cells)

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::activity::ActivityState;
use crate::{POSITION_DIMS, STATE_DIMS};

/// Core cell metadata (CPU-side, 32 bytes)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    /// Unique identifier
    pub id: u64,

    /// Index into DNA pool (cells can share DNA lineages)
    pub dna_index: u32,

    /// Generation (depth in lineage tree)
    pub generation: u32,

    /// Age in ticks
    pub age: u64,

    /// Activity state for sparse updates
    pub activity: ActivityState,
}

/// Dynamic cell state (GPU-transferable, 256 bytes)
///
/// This structure lives in GPU memory and is updated by compute shaders.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct CellState {
    /// Position in semantic space (16D)
    pub position: [f32; POSITION_DIMS],

    /// Internal activation state (32D)
    pub state: [f32; STATE_DIMS],

    /// Vital energy (0.0 = death)
    pub energy: f32,

    /// Tension - desire to act (builds up over time)
    pub tension: f32,

    /// Activity level (for sparse updates)
    pub activity_level: f32,

    /// Flags (packed booleans)
    /// - bit 0: is_sleeping
    /// - bit 1: wants_to_divide
    /// - bit 2: wants_to_signal
    /// - bit 3: wants_to_connect
    /// - bit 4: wants_to_move
    /// - bit 5: is_dead
    pub flags: u32,

    /// Cluster ID (Phase 6 - Semantic Synthesis)
    pub cluster_id: u32,

    /// Mutation hysteresis (Phase 6 - Structural Stability)
    /// 0.0 = fully mutable, 1.0 = structurally locked
    pub hysteresis: f32,

    /// Reserved for alignment and future use (Total struct size = 256 bytes)
    _reserved: [f32; 10],
}

impl CellState {
    /// Create a new cell state with initial values
    ///
    /// "La Vraie Faim": Energy is randomized (0.5-1.5) to stagger deaths.
    /// Cells don't all die at the same tick - creates natural selection windows.
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        Self {
            // Position in semantic space: -10..10 (matches spatial_view grid)
            position: std::array::from_fn(|_| rng.gen::<f32>() * 20.0 - 10.0),
            state: [0.0; STATE_DIMS],
            // Randomize energy: 0.5 to 1.5 (average 1.0)
            // This staggers deaths over ~1000 ticks instead of all at once
            energy: 0.5 + rng.gen::<f32>(),
            tension: 0.0,
            activity_level: 1.0, // Start awake
            flags: 0,
            cluster_id: 0,
            hysteresis: 0.0,
            _reserved: [0.0; 10],
        }
    }

    /// Create state for a child cell near parent
    pub fn from_parent(parent: &CellState) -> Self {
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut child = Self::new();

        // Position near parent with small variation
        for (i, p) in parent.position.iter().enumerate() {
            child.position[i] = p + rng.gen::<f32>() * 0.2 - 0.1;
        }

        child.energy = 0.5; // Child starts with half energy
        child
    }

    /// Check if cell is sleeping
    #[inline]
    pub fn is_sleeping(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Set sleeping state
    #[inline]
    pub fn set_sleeping(&mut self, sleeping: bool) {
        if sleeping {
            self.flags |= 1;
        } else {
            self.flags &= !1;
        }
    }

    /// Check if cell is dead
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.flags & 32 != 0 || self.energy <= 0.0
    }

    /// Mark cell as dead
    #[inline]
    pub fn set_dead(&mut self) {
        self.flags |= 32;
    }

    /// Get the dominant emotion from state
    pub fn emotion(&self) -> Emotion {
        let activation: f32 = self.state[0..4].iter().sum();
        let valence: f32 = self.state[4..8].iter().sum();

        if activation.abs() < 0.2 {
            Emotion::Calm
        } else if activation > 0.5 && valence > 0.0 {
            Emotion::Excited
        } else if activation > 0.5 && valence < 0.0 {
            Emotion::Frustrated
        } else if valence > 0.3 {
            Emotion::Content
        } else if valence < -0.3 {
            Emotion::Distressed
        } else {
            Emotion::Curious
        }
    }

    /// Calculate semantic distance to a position
    pub fn distance_to(&self, other: &[f32; POSITION_DIMS]) -> f32 {
        self.position
            .iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize the internal state (prevent explosion)
    pub fn normalize_state(&mut self, cap: f32) {
        let norm: f32 = self.state.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > cap {
            let scale = cap / norm;
            for s in &mut self.state {
                *s *= scale;
            }
        }
    }
}

impl Default for CellState {
    fn default() -> Self {
        Self::new()
    }
}

impl Cell {
    /// Create a new cell
    pub fn new(id: u64, dna_index: u32) -> Self {
        Self {
            id,
            dna_index,
            generation: 0,
            age: 0,
            activity: ActivityState::new(),
        }
    }

    /// Create a child cell from parent
    pub fn from_parent(id: u64, parent: &Cell, new_dna_index: u32) -> Self {
        Self {
            id,
            dna_index: new_dna_index,
            generation: parent.generation + 1,
            age: 0,
            activity: ActivityState::new(),
        }
    }
}

/// What a cell can do
#[derive(Clone, Debug, PartialEq)]
pub enum CellAction {
    /// Do nothing
    Rest,
    /// Die (no more energy)
    Die,
    /// Reproduce (divide)
    Divide,
    /// Create a connection to another cell
    Connect(u64),
    /// Emit a signal
    Signal([f32; 8]),
    /// Move in semantic space
    Move([f32; POSITION_DIMS]),
}

/// Emotional state derived from cell activity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Emotion {
    Calm,
    Curious,
    Excited,
    Content,
    Frustrated,
    Distressed,
}

impl std::fmt::Display for Emotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Emotion::Calm => write!(f, "calm"),
            Emotion::Curious => write!(f, "curious"),
            Emotion::Excited => write!(f, "excited"),
            Emotion::Content => write!(f, "content"),
            Emotion::Frustrated => write!(f, "frustrated"),
            Emotion::Distressed => write!(f, "distressed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_state_size() {
        // Verify GPU-friendly size (multiple of 16 bytes)
        let size = std::mem::size_of::<CellState>();
        assert!(size % 16 == 0, "CellState size {} not aligned", size);
    }

    #[test]
    fn test_cell_state_flags() {
        let mut state = CellState::new();
        assert!(!state.is_sleeping());
        assert!(!state.is_dead());

        state.set_sleeping(true);
        assert!(state.is_sleeping());

        state.set_dead();
        assert!(state.is_dead());
    }

    #[test]
    fn test_cell_state_normalize() {
        let mut state = CellState::new();
        for s in &mut state.state {
            *s = 10.0; // Very high values
        }

        state.normalize_state(5.0);

        let norm: f32 = state.state.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 5.1, "Norm {} exceeds cap", norm);
    }
}
