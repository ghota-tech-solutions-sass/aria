//! Hebbian learning shaders - "Cells that fire together, wire together"

/// Hebbian learning shader: strengthens connections between co-active cells
/// This is the foundation for the Prediction Law - cells need connections to predict
pub const HEBBIAN_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}

struct CellConnections {
    targets: array<u32, 16>,
    strengths: array<f32, 16>,
    count: u32,
    _pad: array<u32, 3>,
}

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
const NO_CONNECTION: u32 = 0xFFFFFFFFu;

// Hebbian learning constants
const COACTIVATION_THRESHOLD: f32 = 0.3;  // Both cells must be above this
const STRENGTHEN_RATE: f32 = 0.1;         // How fast connections grow
const DECAY_RATE: f32 = 0.001;            // How fast unused connections fade
const MAX_STRENGTH: f32 = 1.0;            // Maximum connection strength
const MAX_CONNECTIONS: u32 = 16u;

struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read_write> connections: array<CellConnections>;
@group(0) @binding(3) var<uniform> config: Config;

// Find an existing connection to target_id, returns slot index or MAX_CONNECTIONS if not found
fn find_connection(conn: CellConnections, target_id: u32) -> u32 {
    for (var i = 0u; i < conn.count; i = i + 1u) {
        if conn.targets[i] == target_id {
            return i;
        }
    }
    return MAX_CONNECTIONS;
}

// Find the weakest connection slot (for replacement)
fn find_weakest_slot(conn: CellConnections) -> u32 {
    var weakest_slot = 0u;
    var weakest_strength = conn.strengths[0];
    for (var i = 1u; i < MAX_CONNECTIONS; i = i + 1u) {
        if conn.strengths[i] < weakest_strength {
            weakest_slot = i;
            weakest_strength = conn.strengths[i];
        }
    }
    return weakest_slot;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }

    let cell_meta = metadata[idx];
    if (cell_meta.flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { return; }

    let cell_energy = energies[idx];

    // Only active cells participate in Hebbian learning
    if cell_energy.activity_level < COACTIVATION_THRESHOLD { return; }

    var conn = connections[idx];

    // 1. DECAY: All connections decay slightly each tick
    for (var i = 0u; i < conn.count; i = i + 1u) {
        conn.strengths[i] = max(0.0, conn.strengths[i] - DECAY_RATE);
    }

    // 2. STRENGTHEN: Look for co-active neighbors and strengthen connections
    // We check a subset of cells each tick to avoid O(N²) complexity
    // Using tick modulo to spread the work across frames
    let check_start = (config.tick * 256u + idx) % config.cell_count;
    let check_count = min(64u, config.cell_count); // Check up to 64 cells per tick

    for (var i = 0u; i < check_count; i = i + 1u) {
        let other_idx = (check_start + i) % config.cell_count;
        if other_idx == idx { continue; }

        let other_meta = metadata[other_idx];
        if (other_meta.flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { continue; }

        let other_energy = energies[other_idx];

        // Both cells must be active (co-activation)
        if other_energy.activity_level < COACTIVATION_THRESHOLD { continue; }

        // "Fire together, wire together" - strengthen connection
        let existing_slot = find_connection(conn, other_idx);

        if existing_slot < MAX_CONNECTIONS {
            // Strengthen existing connection
            conn.strengths[existing_slot] = min(MAX_STRENGTH,
                conn.strengths[existing_slot] + STRENGTHEN_RATE);
        } else if conn.count < MAX_CONNECTIONS {
            // Create new connection
            let slot = conn.count;
            conn.targets[slot] = other_idx;
            conn.strengths[slot] = STRENGTHEN_RATE;
            conn.count = conn.count + 1u;
        } else {
            // Replace weakest connection if new one would be stronger
            let weakest = find_weakest_slot(conn);
            if conn.strengths[weakest] < STRENGTHEN_RATE {
                conn.targets[weakest] = other_idx;
                conn.strengths[weakest] = STRENGTHEN_RATE;
            }
        }
    }

    // 3. COMPACT: Remove dead connections (strength near zero)
    var write_idx = 0u;
    for (var read_idx = 0u; read_idx < conn.count; read_idx = read_idx + 1u) {
        if conn.strengths[read_idx] > 0.001 {
            if write_idx != read_idx {
                conn.targets[write_idx] = conn.targets[read_idx];
                conn.strengths[write_idx] = conn.strengths[read_idx];
            }
            write_idx = write_idx + 1u;
        }
    }

    // Clear removed slots
    for (var i = write_idx; i < conn.count; i = i + 1u) {
        conn.targets[i] = NO_CONNECTION;
        conn.strengths[i] = 0.0;
    }
    conn.count = write_idx;

    connections[idx] = conn;
}
"#;

/// HEBBIAN_CENTROID_SHADER: Pass 1 - Calculate weighted centroid of active cells
/// Uses fixed-point i32 atomics since WGSL doesn't have atomicAdd for f32.
/// Scale factor: × 1000 for 0.001 precision.
pub const HEBBIAN_CENTROID_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct CellPosition { position: array<f32, 16> }
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

// Centroid buffer: 16 weighted positions + total_mass + count
// Uses fixed-point i32 (×1000) for atomic accumulation
struct CentroidData {
    weighted_pos: array<atomic<i32>, 16>,  // Fixed-point weighted position sum
    total_mass: atomic<u32>,               // Fixed-point total mass (×1000)
    count: atomic<u32>,                    // Number of contributing cells
}

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
const ACTIVITY_THRESHOLD: f32 = 0.1;  // Minimum activity to contribute
const FP_SCALE: f32 = 1000.0;         // Fixed-point scale factor

@group(0) @binding(0) var<storage, read> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> positions: array<CellPosition>;
@group(0) @binding(2) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(3) var<storage, read_write> centroid: CentroidData;
@group(0) @binding(4) var<uniform> cell_count: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= cell_count { return; }

    let flags = metadata[idx].flags;
    if (flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { return; }

    let cell_energy = energies[idx];
    let mass = cell_energy.activity_level * cell_energy.energy;

    // Only active cells contribute
    if mass < ACTIVITY_THRESHOLD { return; }

    let pos = positions[idx];

    // Convert mass to fixed-point
    let mass_fp = u32(mass * FP_SCALE);

    // Accumulate weighted position (fixed-point)
    for (var i = 0u; i < 16u; i = i + 1u) {
        // pos × mass in fixed-point: (pos × 1000) × mass_fp / 1000
        let weighted_fp = i32(pos.position[i] * FP_SCALE) * i32(mass_fp) / 1000;
        atomicAdd(&centroid.weighted_pos[i], weighted_fp);
    }

    // Accumulate total mass and count
    atomicAdd(&centroid.total_mass, mass_fp);
    atomicAdd(&centroid.count, 1u);
}
"#;

/// HEBBIAN_ATTRACTION_SHADER: Pass 2 - Move active cells towards centroid
/// Reads normalized centroid and applies attraction force.
pub const HEBBIAN_ATTRACTION_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct CellPosition { position: array<f32, 16> }
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

// Centroid buffer (read the accumulated values)
struct CentroidData {
    weighted_pos: array<atomic<i32>, 16>,
    total_mass: atomic<u32>,
    count: atomic<u32>,
}

struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
const ACTIVITY_THRESHOLD: f32 = 0.1;
const FP_SCALE: f32 = 1000.0;
const PLASTICITY: f32 = 0.01;  // Base attraction rate

@group(0) @binding(0) var<storage, read_write> positions: array<CellPosition>;
@group(0) @binding(1) var<storage, read> energies: array<CellEnergy>;
@group(0) @binding(2) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(3) var<storage, read> centroid: CentroidData;
@group(0) @binding(4) var<uniform> config: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }

    let flags = metadata[idx].flags;
    if (flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { return; }

    let cell_energy = energies[idx];
    if cell_energy.activity_level < ACTIVITY_THRESHOLD { return; }

    // Read total mass (fixed-point)
    let total_mass_fp = atomicLoad(&centroid.total_mass);
    if total_mass_fp < 1000u { return; }  // No significant centroid (< 1.0 mass)

    var pos = positions[idx];

    // Calculate attraction for each dimension
    for (var i = 0u; i < 16u; i = i + 1u) {
        // Get centroid position (normalized from fixed-point)
        let weighted_fp = atomicLoad(&centroid.weighted_pos[i]);
        let center = f32(weighted_fp) / f32(total_mass_fp);

        // Calculate distance to center
        let dist = center - pos.position[i];

        // Force proportional to distance × activity × plasticity
        let move_amount = dist * cell_energy.activity_level * PLASTICITY;

        // Apply movement (clamped to world bounds)
        pos.position[i] = clamp(pos.position[i] + move_amount, -10.0, 10.0);
    }

    positions[idx] = pos;
}
"#;
