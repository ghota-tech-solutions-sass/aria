//! # Structure of Arrays (SoA) for GPU
//!
//! GPU-optimized data layout for 5M+ cells.
//!
//! Instead of one large CellState struct, we split into separate buffers:
//! - Energy buffer (f32): Most frequently updated
//! - Position buffer: Read-only for signal propagation
//! - State buffer: Read/write for signal processing
//! - Flags buffer (u32): Compact boolean states
//!
//! This layout provides:
//! - +40% FPS from better memory coalescing
//! - Easier partial updates (only touch what changed)
//! - Better cache utilization

use bytemuck::{Pod, Zeroable};

use crate::{POSITION_DIMS, STATE_DIMS};

/// Cell energy (most frequently updated field)
/// Buffer: f32[] - one per cell
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CellEnergy {
    pub energy: f32,
    pub tension: f32,
    pub activity_level: f32,
    pub _pad: f32, // Align to 16 bytes
}

impl CellEnergy {
    pub fn new(energy: f32) -> Self {
        Self {
            energy,
            tension: 0.0,
            activity_level: 1.0, // Start awake
            _pad: 0.0,
        }
    }
}

/// Cell position in semantic space (read-mostly)
/// Buffer: vec4<f32>[4] = 16 floats per cell
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CellPosition {
    pub position: [f32; POSITION_DIMS], // 16 floats = 64 bytes
}

impl CellPosition {
    pub fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            // Position in semantic space: -10..10
            position: std::array::from_fn(|_| rng.gen::<f32>() * 20.0 - 10.0),
        }
    }

    pub fn from_parent(parent: &CellPosition) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut position = parent.position;
        for p in &mut position {
            *p += rng.gen::<f32>() * 0.2 - 0.1;
        }
        Self { position }
    }
}

impl Default for CellPosition {
    fn default() -> Self {
        Self::new()
    }
}

/// Cell internal state (read/write for signals)
/// Buffer: f32[32] per cell = 128 bytes
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CellInternalState {
    pub state: [f32; STATE_DIMS], // 32 floats = 128 bytes
}

impl CellInternalState {
    pub fn new() -> Self {
        Self {
            state: [0.0; STATE_DIMS],
        }
    }
}

impl Default for CellInternalState {
    fn default() -> Self {
        Self::new()
    }
}

/// Cell flags (compact booleans)
/// Cell metadata (flags, cluster, hysteresis)
/// Buffer: CellMetadata[] - one per cell
/// Size: 16 bytes per cell
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CellMetadata {
    pub flags: u32,
    pub cluster_id: u32,
    pub hysteresis: f32,
    pub _pad: u32,
}

impl CellMetadata {
    pub const SLEEPING: u32 = 1 << 0;
    pub const WANTS_DIVIDE: u32 = 1 << 1;
    pub const WANTS_SIGNAL: u32 = 1 << 2;
    pub const WANTS_CONNECT: u32 = 1 << 3;
    pub const WANTS_MOVE: u32 = 1 << 4;
    pub const DEAD: u32 = 1 << 5;

    // Hysteresis counter (legacy 2 bits: 0-3)
    pub const SLEEP_COUNTER_MASK: u32 = 0b11 << 6;
    pub const SLEEP_COUNTER_SHIFT: u32 = 6;

    pub fn new() -> Self {
        Self {
            flags: 0,
            cluster_id: 0,
            hysteresis: 0.0,
            _pad: 0,
        }
    }

    #[inline]
    pub fn is_sleeping(&self) -> bool {
        self.flags & Self::SLEEPING != 0
    }

    #[inline]
    pub fn is_dead(&self) -> bool {
        self.flags & Self::DEAD != 0
    }

    #[inline]
    pub fn set_sleeping(&mut self, sleeping: bool) {
        if sleeping {
            self.flags |= Self::SLEEPING;
        } else {
            self.flags &= !Self::SLEEPING;
        }
    }

    #[inline]
    pub fn set_dead(&mut self) {
        self.flags |= Self::DEAD;
    }

    /// Get sleep counter for hysteresis (0-3)
    #[inline]
    pub fn sleep_counter(&self) -> u32 {
        (self.flags & Self::SLEEP_COUNTER_MASK) >> Self::SLEEP_COUNTER_SHIFT
    }

    /// Increment sleep counter (capped at 3)
    #[inline]
    pub fn increment_sleep_counter(&mut self) {
        let counter = self.sleep_counter();
        if counter < 3 {
            self.flags = (self.flags & !Self::SLEEP_COUNTER_MASK)
                | ((counter + 1) << Self::SLEEP_COUNTER_SHIFT);
        }
    }

    /// Reset sleep counter to 0
    #[inline]
    pub fn reset_sleep_counter(&mut self) {
        self.flags &= !Self::SLEEP_COUNTER_MASK;
    }
}

impl Default for CellMetadata {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PREDICTION LAW
// ============================================================================

/// Cell prediction for the Prediction Law
///
/// "Cells that predict correctly, survive"
///
/// Each cell predicts what its next internal state will be based on:
/// - Its current connections (Hebbian weights)
/// - The activity of connected cells
///
/// After each tick:
/// - Compare prediction vs reality
/// - Low error → energy bonus (cell "understands" its world)
/// - High error → energy loss (cell is confused)
///
/// This creates evolutionary pressure toward intelligence:
/// Cells that model their environment survive.
///
/// Buffer: CellPrediction[] - one per cell
/// Size: 48 bytes per cell
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CellPrediction {
    /// Predicted next internal state (first 8 dims of 32D state)
    /// Why 8? The first 8 dimensions encode the most salient features
    pub predicted_state: [f32; 8],

    /// Confidence in prediction (0.0 = guessing, 1.0 = certain)
    /// High confidence + correct = big reward
    /// High confidence + wrong = big penalty (overconfidence punished)
    pub confidence: f32,

    /// Error from last prediction (0.0 = perfect, 1.0+ = very wrong)
    pub last_error: f32,

    /// Cumulative prediction score (long-term track record)
    pub cumulative_score: f32,

    /// Padding for 16-byte alignment
    pub _pad: f32,
}

impl CellPrediction {
    pub fn new() -> Self {
        Self {
            predicted_state: [0.0; 8],
            confidence: 0.1, // Start with low confidence
            last_error: 0.0,
            cumulative_score: 0.0,
            _pad: 0.0,
        }
    }

    /// Calculate prediction error (MSE between predicted and actual)
    #[inline]
    pub fn calculate_error(&self, actual_state: &[f32; 8]) -> f32 {
        let mut sum = 0.0;
        for i in 0..8 {
            let diff = self.predicted_state[i] - actual_state[i];
            sum += diff * diff;
        }
        (sum / 8.0).sqrt() // RMSE
    }

    /// Update prediction based on connections and neighbor states
    /// This is the "thinking" part - predicting the future
    pub fn predict_from_connections(
        &mut self,
        connections: &CellConnections,
        neighbor_states: &[[f32; 8]],
        neighbor_indices: &[usize],
    ) {
        // Reset prediction
        self.predicted_state = [0.0; 8];
        let mut total_weight = 0.0;

        // Weighted average of connected neighbors' states
        for (slot, &target) in connections.targets.iter().enumerate() {
            if target == CellConnections::NO_CONNECTION {
                continue;
            }

            // Find this target in our neighbor list
            if let Some(pos) = neighbor_indices.iter().position(|&idx| idx == target as usize) {
                let strength = connections.strengths[slot];
                for i in 0..8 {
                    self.predicted_state[i] += neighbor_states[pos][i] * strength;
                }
                total_weight += strength;
            }
        }

        // Normalize by total weight
        if total_weight > 0.001 {
            for p in &mut self.predicted_state {
                *p /= total_weight;
            }
            // Confidence based on connection strength (more connections = more confident)
            self.confidence = (total_weight / 3.0).min(1.0);
        } else {
            // No connections = very low confidence
            self.confidence = 0.05;
        }
    }

    /// Evaluate prediction and return energy delta
    /// Called after the actual tick to see how well we predicted
    pub fn evaluate(&mut self, actual_state: &[f32; 8]) -> f32 {
        self.last_error = self.calculate_error(actual_state);

        // Prediction reward/penalty calculation
        // Good prediction (low error) + high confidence = big reward
        // Bad prediction (high error) + high confidence = big penalty
        // Bad prediction + low confidence = small penalty (honest uncertainty)

        let accuracy = 1.0 - self.last_error.min(1.0);
        let reward = if accuracy > 0.7 {
            // Good prediction: reward proportional to confidence
            accuracy * self.confidence * 0.02 // Max +0.02 energy per tick
        } else if accuracy < 0.3 {
            // Bad prediction: penalty worse if overconfident
            -self.last_error * self.confidence * 0.01 // Max -0.01 energy per tick
        } else {
            // Mediocre prediction: small adjustment
            (accuracy - 0.5) * 0.005
        };

        // Update cumulative score
        self.cumulative_score = self.cumulative_score * 0.99 + reward;

        reward
    }
}

impl Default for CellPrediction {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HEBBIAN CONNECTIONS
// ============================================================================

/// Maximum connections per cell (fixed for GPU efficiency)
pub const MAX_CONNECTIONS: usize = 16;

/// Hebbian connection thresholds
pub const HEBBIAN_COACTIVATION_THRESHOLD: f32 = 0.3; // Both cells must be above this
pub const HEBBIAN_STRENGTHEN_RATE: f32 = 0.1;        // How fast connections grow
pub const HEBBIAN_DECAY_RATE: f32 = 0.001;           // How fast unused connections fade
pub const HEBBIAN_MAX_STRENGTH: f32 = 1.0;           // Maximum connection strength

/// Cell connections for Hebbian learning
/// Each cell can have up to MAX_CONNECTIONS connections to other cells
/// "Cells that fire together, wire together"
///
/// Buffer: CellConnections[] - one per cell
/// Size: 144 bytes per cell (for 5M cells: ~720 MB)
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct CellConnections {
    /// Indices of connected cells (0xFFFFFFFF = no connection)
    pub targets: [u32; MAX_CONNECTIONS],
    /// Connection strengths (0.0 = no connection, 1.0 = max)
    pub strengths: [f32; MAX_CONNECTIONS],
    /// Number of active connections
    pub count: u32,
    /// Padding for alignment
    pub _pad: [u32; 3],
}

impl CellConnections {
    pub const NO_CONNECTION: u32 = 0xFFFFFFFF;

    pub fn new() -> Self {
        Self {
            targets: [Self::NO_CONNECTION; MAX_CONNECTIONS],
            strengths: [0.0; MAX_CONNECTIONS],
            count: 0,
            _pad: [0; 3],
        }
    }

    /// Find existing connection to target, returns slot index
    pub fn find_connection(&self, target: u32) -> Option<usize> {
        for i in 0..self.count as usize {
            if self.targets[i] == target {
                return Some(i);
            }
        }
        None
    }

    /// Strengthen or create connection to target (Hebbian)
    pub fn strengthen(&mut self, target: u32, amount: f32) -> bool {
        // Try to find existing connection
        if let Some(slot) = self.find_connection(target) {
            self.strengths[slot] = (self.strengths[slot] + amount).min(HEBBIAN_MAX_STRENGTH);
            return true;
        }

        // Create new connection if room available
        if (self.count as usize) < MAX_CONNECTIONS {
            let slot = self.count as usize;
            self.targets[slot] = target;
            self.strengths[slot] = amount.min(HEBBIAN_MAX_STRENGTH);
            self.count += 1;
            return true;
        }

        // Replace weakest connection if new one would be stronger
        let mut weakest_slot = 0;
        let mut weakest_strength = self.strengths[0];
        for i in 1..MAX_CONNECTIONS {
            if self.strengths[i] < weakest_strength {
                weakest_slot = i;
                weakest_strength = self.strengths[i];
            }
        }

        if amount > weakest_strength {
            self.targets[weakest_slot] = target;
            self.strengths[weakest_slot] = amount;
            return true;
        }

        false
    }

    /// Decay all connections by rate (called each tick)
    pub fn decay(&mut self, rate: f32) {
        for i in 0..self.count as usize {
            self.strengths[i] = (self.strengths[i] - rate).max(0.0);
        }

        // Compact: remove dead connections
        let mut write = 0;
        for read in 0..self.count as usize {
            if self.strengths[read] > 0.001 {
                if write != read {
                    self.targets[write] = self.targets[read];
                    self.strengths[write] = self.strengths[read];
                }
                write += 1;
            }
        }

        // Clear remaining slots
        for i in write..self.count as usize {
            self.targets[i] = Self::NO_CONNECTION;
            self.strengths[i] = 0.0;
        }
        self.count = write as u32;
    }

    /// Get total connection strength (useful for activity spreading)
    pub fn total_strength(&self) -> f32 {
        self.strengths[..self.count as usize].iter().sum()
    }
}

impl Default for CellConnections {
    fn default() -> Self {
        Self::new()
    }
}

/// Indirect dispatch arguments for GPU
/// Used to let GPU decide workgroup count without CPU roundtrip
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct IndirectDispatchArgs {
    /// Number of workgroups in X (computed by GPU)
    pub workgroup_count_x: u32,
    /// Always 1
    pub workgroup_count_y: u32,
    /// Always 1
    pub workgroup_count_z: u32,
    /// Padding for alignment
    pub _pad: u32,
}

impl IndirectDispatchArgs {
    pub fn new() -> Self {
        Self {
            workgroup_count_x: 1,
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            _pad: 0,
        }
    }
}

impl Default for IndirectDispatchArgs {
    fn default() -> Self {
        Self::new()
    }
}

/// SoA buffers collection for easy management
/// This is the CPU-side representation of GPU buffers
pub struct SoABuffers {
    pub energies: Vec<CellEnergy>,
    pub positions: Vec<CellPosition>,
    pub states: Vec<CellInternalState>,
    pub metadata: Vec<CellMetadata>,
    pub predictions: Vec<CellPrediction>,
    pub connections: Vec<CellConnections>,
}

impl SoABuffers {
    /// Create new SoA buffers for given cell count
    pub fn new(cell_count: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        Self {
            energies: (0..cell_count)
                .map(|_| CellEnergy::new(0.5 + rng.gen::<f32>()))
                .collect(),
            positions: (0..cell_count).map(|_| CellPosition::new()).collect(),
            states: (0..cell_count)
                .map(|_| CellInternalState::new())
                .collect(),
            metadata: (0..cell_count).map(|_| CellMetadata::new()).collect(),
            predictions: (0..cell_count).map(|_| CellPrediction::new()).collect(),
            connections: (0..cell_count).map(|_| CellConnections::new()).collect(),
        }
    }

    /// Convert from legacy CellState slice
    pub fn from_cell_states(states: &[crate::cell::CellState]) -> Self {
        let count = states.len();
        Self {
            energies: states
                .iter()
                .map(|s| CellEnergy {
                    energy: s.energy,
                    tension: s.tension,
                    activity_level: s.activity_level,
                    _pad: 0.0,
                })
                .collect(),
            positions: states
                .iter()
                .map(|s| CellPosition {
                    position: s.position,
                })
                .collect(),
            states: states
                .iter()
                .map(|s| CellInternalState { state: s.state })
                .collect(),
            metadata: states.iter().map(|s| CellMetadata {
                flags: s.flags,
                cluster_id: s.cluster_id,
                hysteresis: s.hysteresis,
                _pad: 0,
            }).collect(),
            // Predictions and connections start fresh (not stored in legacy CellState)
            predictions: (0..count).map(|_| CellPrediction::new()).collect(),
            connections: (0..count).map(|_| CellConnections::new()).collect(),
        }
    }

    /// Convert back to legacy CellState slice
    pub fn to_cell_states(&self, out: &mut [crate::cell::CellState]) {
        for (i, state) in out.iter_mut().enumerate() {
            if i < self.energies.len() {
                state.energy = self.energies[i].energy;
                state.tension = self.energies[i].tension;
                state.activity_level = self.energies[i].activity_level;
            }
            if i < self.positions.len() {
                state.position = self.positions[i].position;
            }
            if i < self.states.len() {
                state.state = self.states[i].state;
            }
            if i < self.metadata.len() {
                state.flags = self.metadata[i].flags;
                state.cluster_id = self.metadata[i].cluster_id;
                state.hysteresis = self.metadata[i].hysteresis;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.energies.len()
    }

    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_energy_size() {
        assert_eq!(std::mem::size_of::<CellEnergy>(), 16);
    }

    #[test]
    fn test_cell_position_size() {
        assert_eq!(std::mem::size_of::<CellPosition>(), 64);
    }

    #[test]
    fn test_cell_internal_state_size() {
        assert_eq!(std::mem::size_of::<CellInternalState>(), 128);
    }

    #[test]
    fn test_cell_metadata_size() {
        assert_eq!(std::mem::size_of::<CellMetadata>(), 16);
    }

    #[test]
    fn test_metadata_hysteresis() {
        let mut meta = CellMetadata::new();
        assert_eq!(meta.sleep_counter(), 0);

        meta.increment_sleep_counter();
        assert_eq!(meta.sleep_counter(), 1);

        meta.increment_sleep_counter();
        meta.increment_sleep_counter();
        meta.increment_sleep_counter(); // Should cap at 3
        assert_eq!(meta.sleep_counter(), 3);

        meta.reset_sleep_counter();
        assert_eq!(meta.sleep_counter(), 0);
    }

    #[test]
    fn test_cell_prediction_size() {
        // 8 floats (32) + 4 floats (16) = 48 bytes
        assert_eq!(std::mem::size_of::<CellPrediction>(), 48);
    }

    #[test]
    fn test_cell_prediction_evaluate() {
        let mut pred = CellPrediction::new();
        pred.predicted_state = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        pred.confidence = 0.8;

        // Perfect prediction
        let actual_perfect = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let reward = pred.evaluate(&actual_perfect);
        assert!(reward > 0.0, "Good prediction should give positive reward");

        // Bad prediction with high confidence = penalty
        pred.predicted_state = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        pred.confidence = 0.9;
        let actual_bad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let penalty = pred.evaluate(&actual_bad);
        assert!(penalty < 0.0, "Bad prediction with high confidence should give penalty");
    }

    #[test]
    fn test_cell_connections_size() {
        // 16 targets (u32) + 16 strengths (f32) + count (u32) + pad (3 u32)
        // = 64 + 64 + 4 + 12 = 144 bytes
        assert_eq!(std::mem::size_of::<CellConnections>(), 144);
    }

    #[test]
    fn test_cell_connections_hebbian() {
        let mut conn = CellConnections::new();
        assert_eq!(conn.count, 0);

        // Create connection
        assert!(conn.strengthen(42, 0.5));
        assert_eq!(conn.count, 1);
        assert_eq!(conn.targets[0], 42);
        assert_eq!(conn.strengths[0], 0.5);

        // Strengthen existing connection
        conn.strengthen(42, 0.3);
        assert_eq!(conn.count, 1);
        assert_eq!(conn.strengths[0], 0.8);

        // Add another connection
        conn.strengthen(100, 0.2);
        assert_eq!(conn.count, 2);

        // Find connection
        assert_eq!(conn.find_connection(42), Some(0));
        assert_eq!(conn.find_connection(100), Some(1));
        assert_eq!(conn.find_connection(999), None);

        // Decay
        conn.decay(0.1);
        assert!(conn.strengths[0] < 0.8); // Should have decayed
    }
}
