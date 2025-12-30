//! # GPU Spatial Hashing
//!
//! Efficient neighbor lookup on GPU using a grid-based spatial hash.
//!
//! ## How It Works
//!
//! 1. **Grid Buffer**: A 3D grid where each cell contains indices of cells in that region
//! 2. **Scatter Pass**: Each cell writes its index to the grid based on position
//! 3. **Gather Pass**: When processing signals, only check cells in nearby grid regions
//!
//! ## Performance Impact
//!
//! For 5M cells with 1024 signals:
//! - Without spatial hash: 5M * 1024 = 5B distance checks
//! - With spatial hash (64x64x64 grid, ~20 cells/region): 1024 * 27 * 20 = 552K checks
//! - **9000x reduction in distance calculations**
//!
//! ## Grid Design
//!
//! Using first 3 dimensions of 16D semantic space for spatial partitioning.
//! Grid resolution adapts to population density.

use bytemuck::{Pod, Zeroable};

/// Maximum cells per grid region
/// If exceeded, extra cells are ignored (acceptable for sparse signals)
pub const MAX_CELLS_PER_REGION: usize = 64;

/// Grid dimensions (64x64x64 = 262,144 regions)
pub const GRID_SIZE: usize = 64;

/// Total number of grid regions
pub const TOTAL_REGIONS: usize = GRID_SIZE * GRID_SIZE * GRID_SIZE;

/// GPU-compatible grid region
/// Each region stores up to MAX_CELLS_PER_REGION cell indices
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GridRegion {
    /// Number of cells in this region
    pub count: u32,
    /// Cell indices (padded to fixed size)
    pub cell_indices: [u32; MAX_CELLS_PER_REGION],
    /// Padding for alignment
    pub _pad: [u32; 3],
}

impl GridRegion {
    pub fn new() -> Self {
        Self {
            count: 0,
            cell_indices: [0; MAX_CELLS_PER_REGION],
            _pad: [0; 3],
        }
    }
}

impl Default for GridRegion {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU configuration for spatial hash
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SpatialHashConfig {
    /// Grid dimensions (x, y, z)
    pub grid_size: [u32; 3],
    /// Maximum cells per region
    pub max_cells_per_region: u32,
    /// Minimum position in semantic space
    pub min_pos: [f32; 3],
    /// Region size in semantic space
    pub region_size: f32,
    /// Total number of cells
    pub cell_count: u32,
    /// Padding for alignment
    pub _pad: [u32; 3],
}

impl SpatialHashConfig {
    /// Create config for given bounds and resolution
    pub fn new(min_pos: [f32; 3], max_pos: [f32; 3], cell_count: usize) -> Self {
        let range = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];
        let max_range = range.iter().cloned().fold(0.0f32, f32::max);
        let region_size = max_range / GRID_SIZE as f32;

        Self {
            grid_size: [GRID_SIZE as u32, GRID_SIZE as u32, GRID_SIZE as u32],
            max_cells_per_region: MAX_CELLS_PER_REGION as u32,
            min_pos,
            region_size,
            cell_count: cell_count as u32,
            _pad: [0; 3],
        }
    }

    /// Default config for ARIA's semantic space (-10..10)
    pub fn default_aria(cell_count: usize) -> Self {
        Self::new([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], cell_count)
    }
}

/// WGSL shader for building the spatial hash grid
pub const BUILD_GRID_SHADER: &str = r#"
// Build spatial hash grid from cell positions

struct GridRegion {
    count: atomic<u32>,
    cell_indices: array<u32, 64>,  // MAX_CELLS_PER_REGION
    _pad: array<u32, 3>,
}

struct SpatialConfig {
    grid_size: vec3<u32>,
    max_cells_per_region: u32,
    min_pos: vec3<f32>,
    region_size: f32,
    cell_count: u32,
    _pad: array<u32, 3>,
}

struct CellPosition {
    position: array<f32, 16>,
}

@group(0) @binding(0) var<storage, read> positions: array<CellPosition>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridRegion>;
@group(0) @binding(2) var<uniform> config: SpatialConfig;

fn position_to_grid(pos: vec3<f32>) -> vec3<u32> {
    let normalized = (pos - config.min_pos) / config.region_size;
    return vec3<u32>(
        clamp(u32(normalized.x), 0u, config.grid_size.x - 1u),
        clamp(u32(normalized.y), 0u, config.grid_size.y - 1u),
        clamp(u32(normalized.z), 0u, config.grid_size.z - 1u)
    );
}

fn grid_index(coord: vec3<u32>) -> u32 {
    return coord.x + coord.y * config.grid_size.x + coord.z * config.grid_size.x * config.grid_size.y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= config.cell_count {
        return;
    }

    // Get cell position (first 3 dimensions)
    let cell_pos = positions[cell_idx];
    let pos = vec3<f32>(cell_pos.position[0], cell_pos.position[1], cell_pos.position[2]);

    // Find grid region
    let grid_coord = position_to_grid(pos);
    let region_idx = grid_index(grid_coord);

    // Atomically add to region
    let slot = atomicAdd(&grid[region_idx].count, 1u);
    if slot < config.max_cells_per_region {
        grid[region_idx].cell_indices[slot] = cell_idx;
    }
}
"#;

/// WGSL shader for clearing the grid (run before build)
pub const CLEAR_GRID_SHADER: &str = r#"
// Clear spatial hash grid

struct GridRegion {
    count: atomic<u32>,
    cell_indices: array<u32, 64>,
    _pad: array<u32, 3>,
}

@group(0) @binding(0) var<storage, read_write> grid: array<GridRegion>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let region_idx = id.x;
    if region_idx >= arrayLength(&grid) {
        return;
    }
    atomicStore(&grid[region_idx].count, 0u);
}
"#;

/// WGSL shader for signal propagation using spatial hash + Hebbian connections
/// This replaces the O(cells * signals) loop with O(signals * neighbors)
/// AND propagates through learned connections for "neural pathway" behavior
pub const SIGNAL_WITH_SPATIAL_HASH_SHADER: &str = r#"
// Signal propagation with spatial hash lookup + Hebbian connections

struct CellEnergy {
    energy: f32,
    tension: f32,
    activity_level: f32,
    _pad: f32,
}

struct CellPosition {
    position: array<f32, 16>,
}

struct SignalFragment {
    source_id_low: u32,
    source_id_high: u32,
    content: array<f32, 8>,
    intensity: f32,
    _pad: f32,
}

struct GridRegion {
    count: u32,  // Not atomic for reading
    cell_indices: array<u32, 64>,
    _pad: array<u32, 3>,
}

struct SpatialConfig {
    grid_size: vec3<u32>,
    max_cells_per_region: u32,
    min_pos: vec3<f32>,
    region_size: f32,
    cell_count: u32,
    _pad: array<u32, 3>,
}

struct Config {
    energy_cap: f32,
    reaction_amplification: f32,
    state_cap: f32,
    signal_radius: f32,
    cost_rest: f32,
    cost_signal: f32,
    cost_move: f32,
    cost_divide: f32,
    signal_energy_base: f32,
    signal_resonance_factor: f32,
    energy_gain: f32,
    tick: u32,
    cell_count: u32,
    workgroup_size: u32,
    _pad: vec2<u32>,
}

// Hebbian connections structure
struct CellConnections {
    targets: array<u32, 16>,
    strengths: array<f32, 16>,
    count: u32,
    _pad: array<u32, 3>,
}

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
const NO_CONNECTION: u32 = 0xFFFFFFFFu;
const CONNECTION_SIGNAL_FACTOR: f32 = 0.5;  // Signals via connections are 50% weaker

@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> positions: array<CellPosition>;
@group(0) @binding(2) var<storage, read_write> states: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;
@group(0) @binding(4) var<storage, read> signals: array<SignalFragment>;
@group(0) @binding(5) var<storage, read> grid: array<GridRegion>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<uniform> spatial_config: SpatialConfig;
@group(0) @binding(8) var<storage, read> connections: array<CellConnections>;

fn position_to_grid(pos: vec3<f32>) -> vec3<i32> {
    let normalized = (pos - spatial_config.min_pos) / spatial_config.region_size;
    return vec3<i32>(
        i32(normalized.x),
        i32(normalized.y),
        i32(normalized.z)
    );
}

fn grid_index(coord: vec3<i32>) -> u32 {
    let clamped = vec3<u32>(
        u32(clamp(coord.x, 0, i32(spatial_config.grid_size.x) - 1)),
        u32(clamp(coord.y, 0, i32(spatial_config.grid_size.y) - 1)),
        u32(clamp(coord.z, 0, i32(spatial_config.grid_size.z) - 1))
    );
    return clamped.x + clamped.y * spatial_config.grid_size.x + clamped.z * spatial_config.grid_size.x * spatial_config.grid_size.y;
}

fn calculate_resonance(signal_content: array<f32, 8>, cell_state_0: vec4<f32>, cell_state_1: vec4<f32>) -> f32 {
    var dot: f32 = 0.0;
    var norm_sig: f32 = 0.0;
    var norm_state: f32 = 0.0;

    for (var i = 0u; i < 4u; i++) {
        dot += signal_content[i] * cell_state_0[i];
        norm_sig += signal_content[i] * signal_content[i];
        norm_state += cell_state_0[i] * cell_state_0[i];
    }
    for (var i = 0u; i < 4u; i++) {
        dot += signal_content[i + 4u] * cell_state_1[i];
        norm_sig += signal_content[i + 4u] * signal_content[i + 4u];
        norm_state += cell_state_1[i] * cell_state_1[i];
    }

    let denom = sqrt(norm_sig * norm_state);
    if denom > 0.001 {
        return (dot / denom + 1.0) * 0.5;
    }
    return 0.5;
}

// This shader is dispatched PER SIGNAL (not per cell)
// Each workgroup handles one signal and iterates over nearby grid regions
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let signal_idx = id.x;
    if signal_idx >= arrayLength(&signals) {
        return;
    }

    let signal = signals[signal_idx];
    if signal.intensity < 0.001 {
        return;
    }

    // Get signal position (from content, first 3 dimensions)
    let signal_pos = vec3<f32>(signal.content[0], signal.content[1], signal.content[2]);
    let signal_grid = position_to_grid(signal_pos);

    // Search radius in grid cells
    let grid_radius = i32(ceil(config.signal_radius / spatial_config.region_size));

    // Iterate over nearby grid regions (3D neighborhood)
    for (var dx = -grid_radius; dx <= grid_radius; dx++) {
        for (var dy = -grid_radius; dy <= grid_radius; dy++) {
            for (var dz = -grid_radius; dz <= grid_radius; dz++) {
                let neighbor_coord = signal_grid + vec3<i32>(dx, dy, dz);

                // Bounds check
                if neighbor_coord.x < 0 || neighbor_coord.x >= i32(spatial_config.grid_size.x) ||
                   neighbor_coord.y < 0 || neighbor_coord.y >= i32(spatial_config.grid_size.y) ||
                   neighbor_coord.z < 0 || neighbor_coord.z >= i32(spatial_config.grid_size.z) {
                    continue;
                }

                let region_idx = grid_index(neighbor_coord);
                let region = grid[region_idx];

                // Process cells in this region
                for (var i = 0u; i < min(region.count, spatial_config.max_cells_per_region); i++) {
                    let cell_idx = region.cell_indices[i];
                    if cell_idx >= config.cell_count {
                        continue;
                    }

                    var cell_energy = energies[cell_idx];
                    var cell_flags = flags[cell_idx];

                    let is_sleeping = (cell_flags & FLAG_SLEEPING) != 0u;
                    let is_dead = (cell_flags & FLAG_DEAD) != 0u;

                    if is_dead {
                        continue;
                    }

                    // Distance check in semantic space (first 8 dimensions)
                    let cell_pos = positions[cell_idx];
                    var dist_sq: f32 = 0.0;
                    for (var d = 0u; d < 8u; d++) {
                        let diff = cell_pos.position[d] - signal.content[d];
                        dist_sq += diff * diff;
                    }
                    let dist = sqrt(dist_sq);

                    if dist >= config.signal_radius {
                        continue;
                    }

                    let attenuation = 1.0 - (dist / config.signal_radius);
                    let intensity = signal.intensity * attenuation * config.reaction_amplification;

                    // Wake sleeping cells
                    if is_sleeping && intensity > 0.1 {
                        cell_flags = cell_flags & ~FLAG_SLEEPING;
                        cell_energy.activity_level = 0.5;
                        cell_energy.tension = 0.2;
                    }

                    // Process signal if awake
                    if (cell_flags & FLAG_SLEEPING) == 0u {
                        var state0 = states[cell_idx * 8u];
                        var state1 = states[cell_idx * 8u + 1u];

                        for (var j = 0u; j < 4u; j++) {
                            state0[j] += signal.content[j] * intensity;
                        }
                        for (var j = 0u; j < 4u; j++) {
                            state1[j] += signal.content[j + 4u] * intensity;
                        }

                        states[cell_idx * 8u] = state0;
                        states[cell_idx * 8u + 1u] = state1;

                        // Resonance-based energy
                        let resonance = calculate_resonance(signal.content, state0, state1);
                        if resonance > 0.3 {
                            let understanding = (resonance - 0.3) / 0.7;
                            let energy_gain = config.signal_energy_base
                                * intensity
                                * understanding
                                * (1.0 + resonance * config.signal_resonance_factor);
                            cell_energy.energy = min(cell_energy.energy + energy_gain, config.energy_cap);
                        }

                        cell_energy.activity_level += intensity;
                    }

                    energies[cell_idx] = cell_energy;
                    flags[cell_idx] = cell_flags;

                    // HEBBIAN PROPAGATION: Forward signal through connections
                    // "Neural pathways" - signals travel faster along learned connections
                    if (cell_flags & FLAG_SLEEPING) == 0u {
                        let conn = connections[cell_idx];
                        for (var c = 0u; c < conn.count; c++) {
                            let target_idx = conn.targets[c];
                            if target_idx == NO_CONNECTION || target_idx >= config.cell_count {
                                continue;
                            }

                            let conn_strength = conn.strengths[c];
                            if conn_strength < 0.1 {
                                continue;  // Weak connections don't propagate
                            }

                            var target_energy = energies[target_idx];
                            var target_flags = flags[target_idx];

                            let target_sleeping = (target_flags & FLAG_SLEEPING) != 0u;
                            let target_dead = (target_flags & FLAG_DEAD) != 0u;

                            if target_dead {
                                continue;
                            }

                            // Signal strength through connection
                            let conn_intensity = intensity * conn_strength * CONNECTION_SIGNAL_FACTOR;

                            // Wake sleeping connected cells
                            if target_sleeping && conn_intensity > 0.05 {
                                target_flags = target_flags & ~FLAG_SLEEPING;
                                target_energy.activity_level = 0.3;
                            }

                            // Apply signal to connected cell
                            if (target_flags & FLAG_SLEEPING) == 0u {
                                var t_state0 = states[target_idx * 8u];
                                var t_state1 = states[target_idx * 8u + 1u];

                                for (var j = 0u; j < 4u; j++) {
                                    t_state0[j] += signal.content[j] * conn_intensity;
                                }
                                for (var j = 0u; j < 4u; j++) {
                                    t_state1[j] += signal.content[j + 4u] * conn_intensity;
                                }

                                states[target_idx * 8u] = t_state0;
                                states[target_idx * 8u + 1u] = t_state1;

                                // Energy from connection-propagated signal
                                let t_resonance = calculate_resonance(signal.content, t_state0, t_state1);
                                if t_resonance > 0.3 {
                                    let understanding = (t_resonance - 0.3) / 0.7;
                                    let e_gain = config.signal_energy_base
                                        * conn_intensity
                                        * understanding
                                        * (1.0 + t_resonance * config.signal_resonance_factor);
                                    target_energy.energy = min(target_energy.energy + e_gain, config.energy_cap);
                                }

                                target_energy.activity_level += conn_intensity;
                            }

                            energies[target_idx] = target_energy;
                            flags[target_idx] = target_flags;
                        }
                    }
                }
            }
        }
    }
}
"#;

/// Calculate grid buffer size in bytes
pub fn grid_buffer_size() -> usize {
    std::mem::size_of::<GridRegion>() * TOTAL_REGIONS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_region_size() {
        // GridRegion: 4 (count) + 64*4 (indices) + 12 (pad) = 272 bytes
        let expected = 4 + MAX_CELLS_PER_REGION * 4 + 12;
        assert_eq!(std::mem::size_of::<GridRegion>(), expected);
    }

    #[test]
    fn test_grid_buffer_size() {
        // 64^3 regions * 272 bytes = ~71 MB
        let size_mb = grid_buffer_size() / 1024 / 1024;
        println!("Grid buffer size: {} MB", size_mb);
        assert!(size_mb < 100); // Should be under 100 MB
    }

    #[test]
    fn test_spatial_config() {
        let config = SpatialHashConfig::default_aria(5_000_000);
        assert_eq!(config.grid_size, [64, 64, 64]);
        assert!(config.region_size > 0.0);
    }
}
