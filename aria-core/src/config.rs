//! # Configuration - ARIA's Vital Parameters
//!
//! These parameters define how ARIA lives, learns, and evolves.
//! In the future, ARIA may learn to modify these herself.

use serde::{Deserialize, Serialize};

use crate::activity::SleepConfig;

/// Master configuration for ARIA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AriaConfig {
    /// Population settings
    pub population: PopulationConfig,

    /// Metabolism settings
    pub metabolism: MetabolismConfig,

    /// Emergence detection settings
    pub emergence: EmergenceConfig,

    /// Signal processing settings
    pub signals: SignalConfig,

    /// Sparse update settings
    pub activity: SleepConfig,

    /// Compute backend preference
    pub compute: ComputeConfig,

    /// Network/cluster settings
    pub network: NetworkConfig,

    /// Recurrent processing (Gemini multi-pass)
    pub recurrent: RecurrentConfig,
}

impl Default for AriaConfig {
    fn default() -> Self {
        Self {
            population: PopulationConfig::default(),
            metabolism: MetabolismConfig::default(),
            emergence: EmergenceConfig::default(),
            signals: SignalConfig::default(),
            activity: SleepConfig::default(),
            compute: ComputeConfig::default(),
            network: NetworkConfig::default(),
            recurrent: RecurrentConfig::default(),
        }
    }
}

/// Population management
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopulationConfig {
    /// Target number of cells
    pub target_population: u64,

    /// Buffer before culling (target + buffer = max)
    pub population_buffer: u64,

    /// Minimum population (always maintain)
    pub min_population: u64,

    /// How often to run natural selection (ticks)
    pub selection_interval: u64,

    /// Mutation rate on reproduction
    pub mutation_rate: f32,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            target_population: 10_000,
            population_buffer: 2_000,
            min_population: 10,   // Allow near-extinction for evolution pressure
            selection_interval: 100,  // Was 10 - less frequent to avoid spawn/death loop
            mutation_rate: 0.1,
        }
    }
}

/// Metabolism (energy flow)
///
/// "La Vraie Faim" - Cells must struggle to survive.
/// No free energy, actions cost, only resonance feeds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolismConfig {
    /// Base energy consumed per tick (just breathing costs energy)
    pub energy_consumption: f32,

    /// Passive energy gain per tick - SET TO 0 FOR TRUE LIFE
    /// In nature, nothing is free.
    pub energy_gain: f32,

    /// Maximum energy a cell can hold
    pub energy_cap: f32,

    /// Energy needed to reproduce
    pub reproduction_threshold: f32,

    /// Energy given to child on division
    pub child_energy: f32,

    // === ACTION COSTS (La Vraie Faim) ===

    /// Cost to emit a signal (speaking is expensive!)
    pub cost_signal: f32,

    /// Cost to divide (creating life is exhausting)
    pub cost_divide: f32,

    /// Cost to move in semantic space
    pub cost_move: f32,

    /// Cost to just rest (breathing)
    pub cost_rest: f32,

    // === SIGNAL ENERGY ===

    /// Base energy gain from signals (reduced from 0.05)
    pub signal_energy_base: f32,

    /// Resonance multiplier (coherent signals give more energy)
    pub signal_resonance_factor: f32,
}

impl Default for MetabolismConfig {
    fn default() -> Self {
        Self {
            // Base metabolism - NO FREE LUNCH
            energy_consumption: 0.0,    // Replaced by action costs
            energy_gain: 0.0,           // NO PASSIVE GAIN - ARIA must earn energy through resonance!
            energy_cap: 1.5,
            // Session 32 Part 12: Lowered threshold to match actual energy levels (~0.30 avg)
            // Cells need to reproduce to create generational evolution!
            reproduction_threshold: 0.28,
            child_energy: 0.24,           // 85% of threshold - small gap to earn

            // Action costs - "La Vraie Faim v3" (BRUTAL - real evolutionary pressure)
            cost_signal: 0.005,  // Speaking is EXPENSIVE
            cost_divide: 0.12,   // Creating life costs but parent survives (Session 32 Part 12)
            cost_move: 0.002,    // Moving costs energy
            // Session 32: Tuned for balance between survival and pressure
            cost_rest: 0.0002,   // From tuning: "Balanced drain" scored best

            // Signal energy - must exceed cost_rest for NET POSITIVE growth
            // Session 32 Part 11: Stronger feeding for wave propagation
            signal_energy_base: 0.60,       // 6x original - compensate wave attenuation
            signal_resonance_factor: 3.0,   // Good differentiation for selection
        }
    }
}

/// Emergence detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmergenceConfig {
    /// Minimum activation to be considered "active"
    pub activation_threshold: f32,

    /// Minimum coherence to emit an expression
    pub coherence_threshold: f32,

    /// Minimum intensity to send to client
    pub expression_threshold: f32,

    /// How often to check for emergence (ticks)
    pub check_interval: u64,

    /// Minimum cells needed for emergence
    pub min_active_cells: usize,
}

impl Default for EmergenceConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.01,
            coherence_threshold: 0.1,
            expression_threshold: 0.01,
            check_interval: 5,
            min_active_cells: 5,
        }
    }
}

/// Signal processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Base amplification for external signals
    pub external_amplification: f32,

    /// Amplification during cell reaction
    pub reaction_amplification: f32,

    /// Direct activation multiplier
    pub immediate_activation: f32,

    /// State normalization cap
    pub state_cap: f32,

    /// How far signals travel (semantic distance)
    pub signal_radius: f32,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            external_amplification: 5.0,
            reaction_amplification: 10.0,
            immediate_activation: 5.0,
            state_cap: 5.0,
            // Signal radius in 8D semantic space [-10..10]
            // Expected distance between random points: ~23
            // 30.0 = covers most cells with strong attenuation at edges
            // Wave propagation: cells near signal source get more energy
            signal_radius: 30.0,
        }
    }
}

/// Recurrent processing configuration (Gemini multi-pass)
///
/// Enables internal "thinking" passes where cells influence each other
/// before emergence detection. This creates richer internal dynamics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecurrentConfig {
    /// Number of internal passes per tick (1 = single-pass, 2+ = multi-pass)
    /// Higher values = deeper processing but more compute
    pub passes_per_tick: u32,

    /// Decay factor for internal signals between passes (0.0-1.0)
    /// Higher = signals persist longer across passes
    pub internal_signal_decay: f32,

    /// Minimum activation to generate internal signals
    /// Cells below this threshold don't propagate to neighbors
    pub internal_signal_threshold: f32,

    /// Enable recurrent processing (can be disabled for performance)
    pub enabled: bool,

    /// Internal signal radius (usually smaller than external)
    pub internal_radius: f32,
}

impl Default for RecurrentConfig {
    fn default() -> Self {
        Self {
            passes_per_tick: 1,               // Was 2 - disable multi-pass for now
            internal_signal_decay: 0.7,       // 30% decay between passes
            internal_signal_threshold: 0.3,   // Was 0.1 - higher threshold = fewer internal signals
            enabled: false,                   // DISABLED - was causing 0% sparse savings
            internal_radius: 1.0,             // Smaller than external (2.0)
        }
    }
}

/// Compute backend configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComputeConfig {
    /// Preferred backend
    pub backend: ComputeBackendType,

    /// Maximum cells for CPU backend (switch to GPU above this)
    pub cpu_max_cells: u64,

    /// GPU workgroup size
    pub gpu_workgroup_size: u32,

    /// Enable sparse updates
    pub sparse_updates: bool,

    /// Enable spatial hashing
    pub spatial_hashing: bool,

    /// Spatial grid resolution
    pub grid_resolution: u32,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            backend: ComputeBackendType::Auto,
            cpu_max_cells: 100_000,
            gpu_workgroup_size: 256,
            sparse_updates: true,
            spatial_hashing: true,
            grid_resolution: 64,
        }
    }
}

/// Available compute backends
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeBackendType {
    /// Automatically choose best backend
    Auto,
    /// CPU with Rayon parallelism
    Cpu,
    /// GPU with wgpu (legacy AoS layout)
    Gpu,
    /// GPU with wgpu (optimized SoA layout for 5M+ cells)
    GpuSoA,
}

/// Network/cluster configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// WebSocket bind address
    pub bind_address: String,

    /// WebSocket port
    pub port: u16,

    /// Is this node a cluster member?
    pub cluster_enabled: bool,

    /// Cluster peers (for multi-brain)
    pub cluster_peers: Vec<String>,

    /// Role in cluster
    pub cluster_role: ClusterRole,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8765,
            cluster_enabled: false,
            cluster_peers: Vec::new(),
            cluster_role: ClusterRole::Primary,
        }
    }
}

/// Role in a multi-brain cluster
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterRole {
    /// Main brain (handles interaction)
    Primary,
    /// Secondary brain (handles deep processing)
    Secondary,
    /// Archive brain (handles long-term memory)
    Archive,
}

impl AriaConfig {
    /// Load configuration from file
    pub fn load(path: &str) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save configuration to file
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, content)
    }

    /// Create config for high-performance GPU setup
    pub fn gpu_optimized(target_cells: u64) -> Self {
        let mut config = Self::default();
        config.population.target_population = target_cells;
        config.population.min_population = 10;   // Allow near-extinction (La Vraie Faim)
        config.population.selection_interval = 100;  // Less frequent selection
        config.compute.backend = ComputeBackendType::Gpu;
        config.compute.sparse_updates = true;
        config.compute.spatial_hashing = true;
        config
    }

    /// Create config for CPU-only development
    pub fn cpu_dev() -> Self {
        let mut config = Self::default();
        config.population.target_population = 10_000;
        config.compute.backend = ComputeBackendType::Cpu;
        config
    }

    /// Create config for high-performance CPU (more cells, sparse updates)
    pub fn cpu_high_performance(target_cells: u64) -> Self {
        let mut config = Self::default();
        config.population.target_population = target_cells;
        config.population.population_buffer = (target_cells / 5) as u64; // 20% buffer
        config.population.min_population = 10;   // Allow near-extinction (La Vraie Faim)
        config.population.selection_interval = 100;  // Less frequent selection
        config.compute.backend = ComputeBackendType::Cpu;
        config.compute.sparse_updates = true;   // Sleep inactive cells
        config.compute.spatial_hashing = true;  // O(1) neighbor lookup
        config
    }

    /// Create config from environment variables
    ///
    /// Reads:
    /// - ARIA_CELLS: Target population (default: 50000)
    /// - ARIA_BACKEND: "cpu" or "gpu" (default: cpu)
    /// - ARIA_COST_REST: Base metabolism cost (default: 0.0003)
    /// - ARIA_COST_SIGNAL: Signal emission cost (default: 0.005)
    /// - ARIA_SIGNAL_ENERGY_BASE: Base energy from signals (default: 0.10)
    /// - ARIA_CHILD_ENERGY: Energy given to children (default: 0.50)
    /// - ARIA_SLEEPING_DRAIN_MULT: Sleeping drain multiplier (default: 1.0)
    pub fn from_env() -> Self {
        let target_cells = std::env::var("ARIA_CELLS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50_000);  // Default: 50k cells

        let backend = std::env::var("ARIA_BACKEND")
            .map(|s| s.to_lowercase())
            .ok();

        let mut config = match backend.as_deref() {
            Some("gpu") => Self::gpu_optimized(target_cells),
            _ => Self::cpu_high_performance(target_cells),
        };

        // Economy tuning parameters
        if let Ok(val) = std::env::var("ARIA_COST_REST") {
            if let Ok(v) = val.parse() {
                config.metabolism.cost_rest = v;
            }
        }
        if let Ok(val) = std::env::var("ARIA_COST_SIGNAL") {
            if let Ok(v) = val.parse() {
                config.metabolism.cost_signal = v;
            }
        }
        if let Ok(val) = std::env::var("ARIA_SIGNAL_ENERGY_BASE") {
            if let Ok(v) = val.parse() {
                config.metabolism.signal_energy_base = v;
            }
        }
        if let Ok(val) = std::env::var("ARIA_CHILD_ENERGY") {
            if let Ok(v) = val.parse() {
                config.metabolism.child_energy = v;
            }
        }

        config
    }

    /// Get sleeping drain multiplier from environment
    /// Returns multiplier for how much sleeping cells drain compared to awake cells
    pub fn sleeping_drain_multiplier() -> f32 {
        std::env::var("ARIA_SLEEPING_DRAIN_MULT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)  // Default: same drain as awake (La Vraie Faim)
    }
}
