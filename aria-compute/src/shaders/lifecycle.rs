//! Lifecycle shaders - Birth, death, and counter management

/// Death shader: marks dead cells and returns their slots to the free list
///
/// Called each tick to:
/// 1. Check if cell energy <= 0
/// 2. If not already dead, mark as dead
/// 3. Push slot index to free_list (atomic)
/// 4. Decrement alive_count, increment deaths_this_tick
pub const DEATH_SHADER: &str = r#"
struct CellEnergy {
    energy: f32,
    tension: f32,
    activity_level: f32,
    _pad: f32,
}

struct CellMetadata {
    flags: u32,
    cluster_id: u32,
    hysteresis: f32,
    _pad: u32,
}

struct LifecycleCounters {
    free_count: atomic<u32>,
    alive_count: atomic<u32>,
    births_this_tick: atomic<u32>,
    deaths_this_tick: atomic<u32>,
    max_births_per_tick: u32,
    max_capacity: u32,
    reproduction_threshold_u32: u32,
    child_energy_u32: u32,
}

struct CellPosition {
    position: array<f32, 16>,
}

struct DNA {
    genes: array<f32, 16>,
    lineage_id_low: u32,
    lineage_id_high: u32,
    generation: u32,
    structural_checksum: u32,
}

const FLAG_DEAD: u32 = 32u; // 1 << 5

// Match lifecycle bind group layout (same as birth shader)
// All buffers are read_write to match the unified layout
@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read_write> positions: array<CellPosition>;  // unused
@group(0) @binding(3) var<storage, read_write> dna_pool: array<DNA>;           // unused
@group(0) @binding(4) var<storage, read_write> dna_indices: array<u32>;        // unused
@group(0) @binding(5) var<storage, read_write> free_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: LifecycleCounters;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let max_cells = counters.max_capacity;
    if idx >= max_cells { return; }

    let cell_meta = metadata[idx];

    // Already dead - skip
    if (cell_meta.flags & FLAG_DEAD) != 0u { return; }

    let energy = energies[idx].energy;

    // Cell dies when energy <= 0
    if energy <= 0.0 {
        // Mark as dead (atomic read-modify-write pattern for flags)
        metadata[idx].flags = cell_meta.flags | FLAG_DEAD;

        // Push slot to free list
        let free_idx = atomicAdd(&counters.free_count, 1u);
        if free_idx < max_cells {
            free_list[free_idx] = idx;
        }

        // Update counters
        atomicSub(&counters.alive_count, 1u);
        atomicAdd(&counters.deaths_this_tick, 1u);
    }
}
"#;

/// Birth shader: creates new cells from ready parents
///
/// Called each tick after death shader to:
/// 1. Check if cell has enough energy to reproduce
/// 2. Try to claim a birth slot (atomic counter)
/// 3. Pop a slot from free_list (atomic)
/// 4. Initialize child cell with mutated DNA
/// 5. Reduce parent energy
pub const BIRTH_SHADER: &str = r#"
struct CellEnergy {
    energy: f32,
    tension: f32,
    activity_level: f32,
    _pad: f32,
}

struct CellMetadata {
    flags: u32,
    cluster_id: u32,
    hysteresis: f32,
    _pad: u32,
}

struct CellPosition {
    position: array<f32, 16>,
}

struct DNA {
    genes: array<f32, 16>,
    lineage_id_low: u32,
    lineage_id_high: u32,
    generation: u32,
    structural_checksum: u32,
}

struct LifecycleCounters {
    free_count: atomic<u32>,
    alive_count: atomic<u32>,
    births_this_tick: atomic<u32>,
    deaths_this_tick: atomic<u32>,
    max_births_per_tick: u32,
    max_capacity: u32,
    reproduction_threshold_u32: u32,
    child_energy_u32: u32,
}

const FLAG_DEAD: u32 = 32u;
const FLAG_SLEEPING: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read_write> positions: array<CellPosition>;
@group(0) @binding(3) var<storage, read_write> dna_pool: array<DNA>;
@group(0) @binding(4) var<storage, read_write> dna_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> free_list: array<u32>;
@group(0) @binding(6) var<storage, read_write> counters: LifecycleCounters;

// Simple hash function for stochastic mutation
fn hash(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x & 0xFFFFu) / 65535.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let parent_idx = id.x;
    let max_cells = counters.max_capacity;
    if parent_idx >= max_cells { return; }

    let parent_meta = metadata[parent_idx];

    // Skip dead or sleeping cells
    if (parent_meta.flags & FLAG_DEAD) != 0u { return; }
    if (parent_meta.flags & FLAG_SLEEPING) != 0u { return; }

    let parent_energy = energies[parent_idx].energy;
    let reproduction_threshold = bitcast<f32>(counters.reproduction_threshold_u32);
    let child_energy_config = bitcast<f32>(counters.child_energy_u32);

    // Check if ready to reproduce
    if parent_energy < reproduction_threshold { return; }

    // Try to claim a birth slot (limited per tick)
    let birth_claim = atomicAdd(&counters.births_this_tick, 1u);
    if birth_claim >= counters.max_births_per_tick { return; }

    // Try to get a free slot
    let free_count = atomicSub(&counters.free_count, 1u);
    if free_count == 0u {
        // No free slots available, restore counter
        atomicAdd(&counters.free_count, 1u);
        return;
    }

    // Pop slot from free list (free_count was pre-decremented, so use free_count - 1)
    let child_idx = free_list[free_count - 1u];

    // Sanity check
    if child_idx >= max_cells {
        atomicAdd(&counters.free_count, 1u);
        return;
    }

    // === Initialize child cell ===

    // Energy: child gets child_energy, parent pays the cost
    let child_energy = child_energy_config;
    energies[child_idx] = CellEnergy(child_energy, 0.0, 1.0, 0.0);
    energies[parent_idx].energy = parent_energy - child_energy;

    // Metadata: child starts alive and awake
    metadata[child_idx] = CellMetadata(0u, 0u, 0.0, 0u);

    // Position: child near parent with small offset
    var child_pos: CellPosition;
    let parent_pos = positions[parent_idx];
    let noise_seed = parent_idx * 16u + counters.births_this_tick;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let noise = (hash(noise_seed + i) - 0.5) * 0.2; // ±0.1
        child_pos.position[i] = parent_pos.position[i] + noise;
    }
    positions[child_idx] = child_pos;

    // DNA: copy parent DNA with mutation
    let parent_dna_idx = dna_indices[parent_idx];
    let parent_dna = dna_pool[parent_dna_idx];

    var child_dna: DNA;
    // Mutate genes slightly
    for (var i = 0u; i < 16u; i = i + 1u) {
        let mutation = (hash(noise_seed + 100u + i) - 0.5) * 0.1; // ±5% mutation
        child_dna.genes[i] = clamp(parent_dna.genes[i] + mutation, 0.0, 1.0);
    }
    child_dna.lineage_id_low = parent_dna.lineage_id_low;
    child_dna.lineage_id_high = parent_dna.lineage_id_high;
    child_dna.generation = parent_dna.generation + 1u;
    child_dna.structural_checksum = parent_dna.structural_checksum; // Could mutate too

    // Store child DNA (reuse same index for simplicity, or could allocate new)
    // For now, child uses its own slot index as DNA index (1:1 mapping)
    dna_pool[child_idx] = child_dna;
    dna_indices[child_idx] = child_idx;

    // Update alive count
    atomicAdd(&counters.alive_count, 1u);
}
"#;

/// Reset lifecycle counters: called at the start of each tick
///
/// Resets per-tick counters (births_this_tick, deaths_this_tick) to 0
pub const RESET_LIFECYCLE_COUNTERS_SHADER: &str = r#"
struct LifecycleCounters {
    free_count: atomic<u32>,
    alive_count: atomic<u32>,
    births_this_tick: atomic<u32>,
    deaths_this_tick: atomic<u32>,
    max_births_per_tick: u32,
    max_capacity: u32,
    reproduction_threshold_u32: u32,
    child_energy_u32: u32,
}

@group(0) @binding(0) var<storage, read_write> counters: LifecycleCounters;

@compute @workgroup_size(1)
fn main() {
    atomicStore(&counters.births_this_tick, 0u);
    atomicStore(&counters.deaths_this_tick, 0u);
}
"#;
