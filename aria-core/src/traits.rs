//! # Traits - Abstractions for ARIA's Components
//!
//! These traits define the interfaces that different implementations
//! must follow. This allows ARIA to run on CPU, GPU, or distributed
//! across multiple machines.
//!
//! ## Key Traits
//!
//! - `ComputeBackend`: Abstraction over CPU/GPU computation
//! - `Substrate`: The living environment where cells exist
//! - `MemoryStore`: Persistent storage for learning

use crate::cell::{Cell, CellAction, CellState};
use crate::config::AriaConfig;
use crate::dna::DNA;
use crate::error::AriaResult;
use crate::signal::{Signal, SignalFragment};

/// Compute backend trait - abstraction over CPU/GPU
///
/// Implementations of this trait handle the heavy computation:
/// - Updating cell states
/// - Propagating signals
/// - Detecting emergence
pub trait ComputeBackend: Send + Sync {
    /// Initialize the backend with configuration
    fn init(&mut self, config: &AriaConfig) -> AriaResult<()>;

    /// Update all active cells for one tick
    ///
    /// Returns the actions cells want to take.
    fn update_cells(
        &mut self,
        cells: &mut [Cell],
        states: &mut [CellState],
        dna_pool: &[DNA],
        signals: &[SignalFragment],
    ) -> AriaResult<Vec<(u64, CellAction)>>;

    /// Propagate signals through the substrate
    ///
    /// Takes outgoing signals from cells and distributes them
    /// to nearby cells based on semantic distance.
    fn propagate_signals(
        &mut self,
        states: &[CellState],
        signals: Vec<SignalFragment>,
    ) -> AriaResult<Vec<(usize, SignalFragment)>>;

    /// Detect emergent patterns
    ///
    /// Looks for synchronized cell clusters and generates
    /// expression signals.
    fn detect_emergence(
        &self,
        cells: &[Cell],
        states: &[CellState],
        config: &AriaConfig,
    ) -> AriaResult<Vec<Signal>>;

    /// Get statistics about the current state
    fn stats(&self) -> BackendStats;

    /// Synchronize state (for GPU backends that buffer)
    fn sync(&mut self) -> AriaResult<()>;

    /// Recompile dynamic logic if supported by backend
    fn recompile(&mut self, structural_checksum: u64) -> AriaResult<()> {
        let _ = structural_checksum;
        Ok(())
    }

    /// Name of this backend (for logging)
    fn name(&self) -> &'static str;
}

/// Statistics from the compute backend
#[derive(Clone, Debug, Default)]
pub struct BackendStats {
    /// Cells processed this tick
    pub cells_processed: u64,

    /// Cells skipped (sleeping)
    pub cells_sleeping: u64,

    /// Signals propagated
    pub signals_propagated: u64,

    /// Emergence events detected
    pub emergences_detected: u64,

    /// Time spent on computation (microseconds)
    pub compute_time_us: u64,

    /// GPU memory used (if applicable)
    pub gpu_memory_bytes: u64,
}

/// Substrate trait - the living environment
///
/// The substrate is where cells live, signals travel,
/// and intelligence emerges.
pub trait Substrate: Send + Sync {
    /// Inject an external signal into the substrate
    fn inject_signal(&mut self, signal: Signal) -> AriaResult<Vec<Signal>>;

    /// Run one tick of the simulation
    fn tick(&mut self) -> AriaResult<Vec<Signal>>;

    /// Get current statistics
    fn stats(&self) -> SubstrateStats;

    /// Get the current tick number
    fn current_tick(&self) -> u64;

    /// Spawn new cells
    fn spawn_cells(&mut self, count: usize) -> AriaResult<()>;

    /// Remove dead cells
    fn cleanup_dead(&mut self) -> AriaResult<usize>;

    /// Save state to persistent storage
    fn save(&self) -> AriaResult<Vec<u8>>;

    /// Load state from persistent storage
    fn load(&mut self, data: &[u8]) -> AriaResult<()>;
}

/// Statistics from the substrate
#[derive(Clone, Debug, Default)]
pub struct SubstrateStats {
    /// Current tick
    pub tick: u64,

    /// Number of living cells
    pub alive_cells: u64,

    /// Total energy in system
    pub total_energy: f32,

    /// System entropy (chaos level)
    pub entropy: f32,

    /// Number of active clusters
    pub active_clusters: usize,

    /// Average connections per cell
    pub avg_connections: f32,

    /// Oldest cell age
    pub oldest_cell_age: u64,

    /// Current emotional state
    pub dominant_emotion: String,

    /// Cells awake (for sparse updates)
    pub cells_awake: u64,

    /// CPU savings from sparse updates
    pub sparse_savings_percent: f32,
}

/// Memory store trait - persistent learning
///
/// Handles storage and retrieval of:
/// - Learned words and associations
/// - Elite DNA from successful cells
/// - Episodic memories
pub trait MemoryStore: Send + Sync {
    /// Store a learned word
    fn learn_word(&mut self, word: &str, vector: &[f32], valence: f32) -> AriaResult<()>;

    /// Find a word matching a vector
    fn find_word(&self, vector: &[f32], threshold: f32) -> Option<(String, f32)>;

    /// Store an association between words
    fn learn_association(&mut self, word1: &str, word2: &str, strength: f32) -> AriaResult<()>;

    /// Get associations for a word
    fn get_associations(&self, word: &str) -> Vec<(String, f32)>;

    /// Store elite DNA
    fn store_elite_dna(&mut self, dna: &DNA, fitness: f32) -> AriaResult<()>;

    /// Get best DNA for spawning
    fn get_elite_dna(&self) -> Option<DNA>;

    /// Save memory to bytes
    fn serialize(&self) -> AriaResult<Vec<u8>>;

    /// Load memory from bytes
    fn deserialize(&mut self, data: &[u8]) -> AriaResult<()>;
}

/// Cluster node trait - for distributed ARIA
///
/// When ARIA runs across multiple machines, this trait
/// handles synchronization.
pub trait ClusterNode: Send + Sync {
    /// Connect to cluster peers
    fn connect(&mut self, peers: &[String]) -> AriaResult<()>;

    /// Broadcast emergence to peers
    fn broadcast_emergence(&self, signal: &Signal) -> AriaResult<()>;

    /// Receive emergences from peers
    fn receive_emergences(&mut self) -> AriaResult<Vec<Signal>>;

    /// Synchronize DNA pool with peers
    fn sync_dna(&mut self, dna_pool: &[DNA]) -> AriaResult<Vec<DNA>>;

    /// Get cluster status
    fn status(&self) -> ClusterStatus;
}

/// Status of cluster connection
#[derive(Clone, Debug, Default)]
pub struct ClusterStatus {
    /// Number of connected peers
    pub connected_peers: usize,

    /// Total cells across cluster
    pub total_cells: u64,

    /// Is this the primary node?
    pub is_primary: bool,

    /// Last sync time
    pub last_sync_tick: u64,
}

/// Introspection trait - for self-understanding
///
/// In the future, ARIA may use this to understand
/// and modify her own code.
pub trait Introspectable {
    /// Get a description of this component
    fn describe(&self) -> String;

    /// Get the source code location (if available)
    fn source_location(&self) -> Option<&'static str>;

    /// Get configurable parameters
    fn parameters(&self) -> Vec<Parameter>;
}

/// A configurable parameter
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Parameter name
    pub name: String,

    /// Current value
    pub value: f64,

    /// Minimum allowed value
    pub min: f64,

    /// Maximum allowed value
    pub max: f64,

    /// Description
    pub description: String,
}
