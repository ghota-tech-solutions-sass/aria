//! # ARIA JIT Compiler
//!
//! Dynamic shader generation and compilation for ARIA's GPU backend.

pub struct ShaderCompiler {}

impl ShaderCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn generate_shader(&self, template: &str, dna_logic: &str) -> String {
        template.replace("// [DYNAMIC_LOGIC]", dna_logic)
    }

    /// Translates structural DNA (checksum) into WGSL logic snippets
    pub fn generate_dna_logic(&self, checksum: u64) -> String {
        let mut logic = String::new();

        // 1. Metabolic strategy (Bits 0-7)
        let metabolism_type = checksum & 0x03;
        let metabolism_gain_mod = ((checksum >> 2) & 0x0F) as f32 / 8.0; // Range: 0.0 to 1.875

        match metabolism_type {
            0 => { // Linear (standard)
                logic.push_str(&format!("    cell_energy.energy = cell_energy.energy + config.energy_gain * {:.4};\n", metabolism_gain_mod));
            }
            1 => { // Logistic (saturation)
                logic.push_str(&format!("    cell_energy.energy = cell_energy.energy + config.energy_gain * {:.4} * (1.0 - cell_energy.energy / config.energy_cap);\n", metabolism_gain_mod));
            }
            2 => { // Pulsed (oscillatory)
                logic.push_str(&format!("    let pulse = 0.5 + 0.5 * sin(f32(config.tick) * 0.1);\n    cell_energy.energy = cell_energy.energy + config.energy_gain * {:.4} * pulse;\n", metabolism_gain_mod));
            }
            _ => { // Tension-regulated
                logic.push_str(&format!("    cell_energy.energy = cell_energy.energy + config.energy_gain * {:.4} * (1.0 - cell_energy.tension);\n", metabolism_gain_mod));
            }
        }

        // 2. Activity decay strategy (Bits 8-15)
        // TUNED: Lower decay rate for faster cooling (was 0.8-0.9875, now 0.5-0.75)
        // This allows cells to sleep more easily when not actively stimulated
        let decay_rate = 0.5 + ((checksum >> 8) & 0x0F) as f32 * 0.0167; // Range: 0.5 to 0.75
        let decay_nonlinear = (checksum >> 12) & 0x01;

        if decay_nonlinear != 0 {
            logic.push_str(&format!("    cell_energy.activity_level = cell_energy.activity_level * ({:.4} + 0.1 * (cell_energy.energy / config.energy_cap));\n", decay_rate - 0.1));
        } else {
            logic.push_str(&format!("    cell_energy.activity_level = cell_energy.activity_level * {:.4};\n", decay_rate));
        }

        // 3. Reflexivity processing (Bits 16-23)
        let reflex_pow = 0.5 + ((checksum >> 16) & 0x0F) as f32 * 0.125; // Range: 0.5 to 2.375
        logic.push_str(&format!("    let raw_reflex = pow(max(0.001, reflexivity_gain), {:.4});\n", reflex_pow));

        // 4. Selective Attention (Axe 3 - Genesis)
        logic.push_str("    let attention_boost = 0.5 + attention_focus * 1.5;\n");
        logic.push_str("    let semantic_threshold = semantic_filter * 0.2;\n");

        // 5. Structural Hysteresis (Phase 6)
        logic.push_str("    let hysteresis_val = cell_meta.hysteresis;\n");
        logic.push_str("    let reflexive_boost = raw_reflex * (0.5 + 0.5 * hysteresis_val);\n");

        logic
    }

    pub fn get_spatial_signal_template(&self) -> &str { SPATIAL_SIGNAL_TEMPLATE }

    pub fn get_cell_update_template(&self) -> &str { CELL_UPDATE_TEMPLATE }
    pub fn get_signal_template(&self) -> &str { SIGNAL_TEMPLATE }
    pub fn get_compact_shader(&self) -> &str { COMPACT_SHADER }
    pub fn get_prepare_dispatch_shader(&self) -> &str { PREPARE_DISPATCH_SHADER }
    pub fn get_sleeping_drain_shader(&self) -> &str { SLEEPING_DRAIN_SHADER }
    pub fn get_hebbian_shader(&self) -> &str { HEBBIAN_SHADER }

    // Prediction Law shaders
    pub fn get_prediction_generate_shader(&self) -> &str { PREDICTION_GENERATE_SHADER }
    pub fn get_prediction_evaluate_shader(&self) -> &str { PREDICTION_EVALUATE_SHADER }

    // Hebbian Spatial Attraction shaders (GPU migration)
    pub fn get_hebbian_centroid_shader(&self) -> &str { HEBBIAN_CENTROID_SHADER }
    pub fn get_hebbian_attraction_shader(&self) -> &str { HEBBIAN_ATTRACTION_SHADER }

    // Cluster Hysteresis shaders (GPU migration)
    pub fn get_cluster_stats_shader(&self) -> &str { CLUSTER_STATS_SHADER }
    pub fn get_cluster_hysteresis_shader(&self) -> &str { CLUSTER_HYSTERESIS_SHADER }
}

// ============================================================================
// WGSL Templates & Shaders
// ============================================================================

const CELL_UPDATE_TEMPLATE: &str = r#"
struct CellEnergy {
    energy: f32,
    tension: f32,
    activity_level: f32,
    _pad: f32,
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

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
const SLEEP_COUNTER_MASK: u32 = 192u;
const SLEEP_COUNTER_SHIFT: u32 = 6u;
const SLEEP_COUNTER_MAX: u32 = 3u;

struct CellMetadata {
    flags: u32,
    cluster_id: u32,
    hysteresis: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> states: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(4) var<storage, read> dna_pool: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> signals: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<storage, read> dna_indices: array<u32>;

fn get_sleep_counter(f: u32) -> u32 { return (f & SLEEP_COUNTER_MASK) >> SLEEP_COUNTER_SHIFT; }
fn set_sleep_counter(f: u32, counter: u32) -> u32 {
    return (f & ~SLEEP_COUNTER_MASK) | ((counter & 3u) << SLEEP_COUNTER_SHIFT);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }
    var cell_energy = energies[idx];
    var cell_meta = metadata[idx];
    if (cell_meta.flags & FLAG_DEAD) != 0u { return; }
    let dna_idx = dna_indices[idx];
    let dna_base = dna_idx * 5u;
    // TUNED: Sleep/wake thresholds for better sparse savings
    // gene_sleep: cells sleep when activity < this (higher = sleep easier) range 0.15-0.4
    // gene_wake: cells wake when activity > this (lower = wake easier) range 0.1-0.3
    let raw_sleep = dna_pool[dna_base+1u].y;  // thresholds[5] in 0-1
    let raw_wake = dna_pool[dna_base+1u].z;   // thresholds[6] in 0-1
    let gene_sleep = 0.15 + raw_sleep * 0.25; // Map to 0.15-0.4
    let gene_wake = 0.1 + raw_wake * 0.2;     // Map to 0.1-0.3
    if (cell_meta.flags & FLAG_SLEEPING) != 0u {
        if cell_energy.activity_level > gene_wake {
            cell_meta.flags = cell_meta.flags & ~FLAG_SLEEPING;
            cell_meta.flags = set_sleep_counter(cell_meta.flags, 0u);
        } else {
            // Sleeping cells still breathe! Same drain as awake (La Vraie Faim)
            cell_energy.energy -= config.cost_rest;
            if cell_energy.energy <= 0.0 { cell_meta.flags = cell_meta.flags | FLAG_DEAD; }
            energies[idx] = cell_energy; metadata[idx] = cell_meta; return;
        }
    }

    cell_energy.energy -= config.cost_rest;

    // DNA GENES for Laws
    let gene_decay = dna_pool[dna_base+1u].z;       // thresholds[6] (Tension Decay)
    let reflexivity_gain = dna_pool[dna_base+1u].w; // thresholds[7]
    let attention_focus = dna_pool[dna_base+3u].z;  // reactions[6]
    let semantic_filter = dna_pool[dna_base+3u].w;  // reactions[7]

    // LAW: Tension Decay (Inertia)
    // Map [0,1] -> [0.5, 0.95]
    let tension_decay = 0.5 + gene_decay * 0.45;

    // Apply decay if activity is low (prevent infinite tension)
    if (abs(cell_energy.activity_level) < 0.1) {
        cell_energy.tension = cell_energy.tension * tension_decay;
        cell_energy.tension = max(0.0, cell_energy.tension - 0.01);
    }

    // [DYNAMIC_LOGIC]
    if cell_energy.activity_level < gene_sleep {
        let counter = get_sleep_counter(cell_meta.flags);
        if counter >= SLEEP_COUNTER_MAX {
            cell_meta.flags = cell_meta.flags | FLAG_SLEEPING;
            cell_meta.flags = set_sleep_counter(cell_meta.flags, 0u);
        } else {
            cell_meta.flags = set_sleep_counter(cell_meta.flags, counter + 1u);
        }
    } else {
        cell_meta.flags = set_sleep_counter(cell_meta.flags, 0u);
    }
    cell_energy.energy = clamp(cell_energy.energy, 0.0, config.energy_cap);
    energies[idx] = cell_energy; metadata[idx] = cell_meta;
}
"#;

const SIGNAL_TEMPLATE: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}
const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> states: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(4) var<storage, read> dna_pool: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> signals: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<storage, read> dna_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }
    let cell_meta = metadata[idx];
    if (cell_meta.flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { return; }
    var cell_energy = energies[idx];
    let dna_idx = dna_indices[idx];
    let dna_base = dna_idx * 5u;
    let reflexivity_gain = dna_pool[dna_base+1u].w; // thresholds[7]
    let attention_focus = dna_pool[dna_base+3u].z;  // reactions[6]
    let semantic_filter = dna_pool[dna_base+3u].w;  // reactions[7]

    // [DYNAMIC_LOGIC]

    if cell_energy.tension > 0.8 {
        cell_energy.activity_level += 0.5 * reflexive_boost;
        cell_energy.energy -= config.cost_signal;
    }
    energies[idx] = cell_energy;
}
"#;

const COMPACT_SHADER: &str = r#"
struct SparseDispatch { counter: atomic<u32>, _pad: array<u32, 3>, indices: array<u32> }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}
const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(1) var<storage, read_write> dispatch: SparseDispatch;
@group(0) @binding(2) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(3) var<uniform> config: Config;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }
    let f = metadata[idx].flags;
    if (f & (FLAG_SLEEPING | FLAG_DEAD)) == 0u {
        let write_idx = atomicAdd(&dispatch.counter, 1u);
        if write_idx < arrayLength(&dispatch.indices) { dispatch.indices[write_idx] = idx; }
    }
}
"#;

const PREPARE_DISPATCH_SHADER: &str = r#"
struct SparseDispatch { counter: atomic<u32>, _pad: array<u32, 3>, indices: array<u32> }
struct IndirectArgs { x: u32, y: u32, z: u32, _pad: u32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(1) var<storage, read_write> dispatch: SparseDispatch;
@group(0) @binding(2) var<storage, read_write> indirect: IndirectArgs;
@group(0) @binding(3) var<uniform> config: Config;
@compute @workgroup_size(1)
fn main() {
    let active_count = atomicLoad(&dispatch.counter);
    indirect.x = (active_count + config.workgroup_size - 1u) / config.workgroup_size;
    indirect.y = 1u; indirect.z = 1u;
}
"#;

const SLEEPING_DRAIN_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}
const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(2) var<uniform> config: Config;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }
    var cell_meta = metadata[idx];
    if (cell_meta.flags & FLAG_SLEEPING) != 0u {
        var cell_energy = energies[idx];
        // Session 32: Reduced from 2.0 to 0.5 - sleeping cells conserve energy
        // This prevents death loop after resurrection (cells need time to receive signals)
        cell_energy.energy = cell_energy.energy - config.cost_rest * 0.5;
        if cell_energy.energy <= 0.0 { cell_meta.flags = cell_meta.flags | FLAG_DEAD; }
        energies[idx] = cell_energy;
        metadata[idx] = cell_meta;
    }
}
"#;

// ============================================================================
// HEBBIAN LEARNING - "Cells that fire together, wire together"
// ============================================================================

/// Hebbian learning shader: strengthens connections between co-active cells
/// This is the foundation for the Prediction Law - cells need connections to predict
const HEBBIAN_SHADER: &str = r#"
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

// ============================================================================
// PREDICTION LAW - "Cells that predict correctly, survive"
// ============================================================================

/// Prediction phase: Each cell predicts its next state based on connections
/// Run BEFORE the main tick to generate predictions
const PREDICTION_GENERATE_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}

// CellPrediction struct (48 bytes)
// predicted_state[8] + confidence + last_error + cumulative_score + _pad
struct CellPrediction {
    predicted_state: array<f32, 8>,
    confidence: f32,
    last_error: f32,
    cumulative_score: f32,
    _pad: f32,
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

struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read> states: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read> connections: array<CellConnections>;
@group(0) @binding(3) var<storage, read_write> predictions: array<CellPrediction>;
@group(0) @binding(4) var<uniform> config: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }

    let cell_meta = metadata[idx];
    if (cell_meta.flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { return; }

    var pred = predictions[idx];
    let conn = connections[idx];

    // Reset prediction
    for (var i = 0u; i < 8u; i = i + 1u) {
        pred.predicted_state[i] = 0.0;
    }

    var total_weight = 0.0;

    // Weighted average of connected neighbors' states
    // "What will my state be, given what my neighbors are doing?"
    for (var slot = 0u; slot < 16u; slot = slot + 1u) {
        let target_idx = conn.targets[slot];
        if target_idx == NO_CONNECTION || target_idx >= config.cell_count { continue; }

        let strength = conn.strengths[slot];
        if strength < 0.001 { continue; }

        // Read neighbor's current state (first 8 dims stored in 2 vec4s)
        let neighbor_state0 = states[target_idx * 8u];
        let neighbor_state1 = states[target_idx * 8u + 1u];

        // Accumulate weighted prediction
        for (var i = 0u; i < 4u; i = i + 1u) {
            pred.predicted_state[i] = pred.predicted_state[i] + neighbor_state0[i] * strength;
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            pred.predicted_state[i + 4u] = pred.predicted_state[i + 4u] + neighbor_state1[i] * strength;
        }

        total_weight = total_weight + strength;
    }

    // Normalize by total weight
    if total_weight > 0.001 {
        for (var i = 0u; i < 8u; i = i + 1u) {
            pred.predicted_state[i] = pred.predicted_state[i] / total_weight;
        }
        // Confidence based on connection strength (more connections = more confident)
        pred.confidence = min(total_weight / 3.0, 1.0);
    } else {
        // No connections = very low confidence (just guessing)
        pred.confidence = 0.05;
    }

    predictions[idx] = pred;
}
"#;

/// Evaluation phase: Compare predictions with actual states, apply rewards/penalties
/// Run AFTER the main tick to evaluate predictions
const PREDICTION_EVALUATE_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}

struct CellPrediction {
    predicted_state: array<f32, 8>,
    confidence: f32,
    last_error: f32,
    cumulative_score: f32,
    _pad: f32,
}

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;

// Prediction Law constants
const PREDICTION_REWARD_MAX: f32 = 0.02;   // Max energy gain per tick for good prediction
const PREDICTION_PENALTY_MAX: f32 = 0.01;  // Max energy loss per tick for bad prediction
const ACCURACY_GOOD_THRESHOLD: f32 = 0.7;  // Above this = good prediction
const ACCURACY_BAD_THRESHOLD: f32 = 0.3;   // Below this = bad prediction

struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read> states: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(2) var<storage, read_write> predictions: array<CellPrediction>;
@group(0) @binding(3) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(4) var<uniform> config: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }

    let cell_meta = metadata[idx];
    if (cell_meta.flags & (FLAG_SLEEPING | FLAG_DEAD)) != 0u { return; }

    var pred = predictions[idx];
    var cell_energy = energies[idx];

    // Get actual state (first 8 dims)
    let actual_state0 = states[idx * 8u];
    let actual_state1 = states[idx * 8u + 1u];

    // Calculate actual state magnitude - skip trivial predictions
    // "You can't claim to have predicted correctly if nothing happened"
    var actual_magnitude = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        actual_magnitude = actual_magnitude + actual_state0[i] * actual_state0[i];
        actual_magnitude = actual_magnitude + actual_state1[i] * actual_state1[i];
    }
    actual_magnitude = sqrt(actual_magnitude);

    // Skip cells with no meaningful activity - trivial predictions don't count
    if actual_magnitude < 0.1 {
        predictions[idx] = pred;
        return;
    }

    // Calculate prediction error (RMSE)
    var sum_sq_error = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let diff = pred.predicted_state[i] - actual_state0[i];
        sum_sq_error = sum_sq_error + diff * diff;
    }
    for (var i = 0u; i < 4u; i = i + 1u) {
        let diff = pred.predicted_state[i + 4u] - actual_state1[i];
        sum_sq_error = sum_sq_error + diff * diff;
    }
    pred.last_error = sqrt(sum_sq_error / 8.0);

    // Calculate accuracy (0.0 = very wrong, 1.0 = perfect)
    let accuracy = max(0.0, 1.0 - pred.last_error);

    // Apply reward/penalty based on accuracy and confidence
    var reward = 0.0;
    if accuracy > ACCURACY_GOOD_THRESHOLD {
        // Good prediction: reward proportional to confidence
        // "I was confident AND I was right" = big reward
        reward = accuracy * pred.confidence * PREDICTION_REWARD_MAX;
    } else if accuracy < ACCURACY_BAD_THRESHOLD {
        // Bad prediction: penalty worse if overconfident
        // "I was confident AND I was wrong" = big penalty (punish overconfidence)
        reward = -pred.last_error * pred.confidence * PREDICTION_PENALTY_MAX;
    } else {
        // Mediocre prediction: small adjustment
        reward = (accuracy - 0.5) * 0.005;
    }

    // Update cumulative score (exponential moving average)
    pred.cumulative_score = pred.cumulative_score * 0.99 + reward;

    // Apply energy change
    cell_energy.energy = clamp(cell_energy.energy + reward, 0.0, config.energy_cap);

    predictions[idx] = pred;
    energies[idx] = cell_energy;
}
"#;

// ============================================================================
// HEBBIAN SPATIAL ATTRACTION SHADERS
// "Cells that fire together, move together."
// ============================================================================

/// HEBBIAN_CENTROID_SHADER: Pass 1 - Calculate weighted centroid of active cells
/// Uses fixed-point i32 atomics since WGSL doesn't have atomicAdd for f32.
/// Scale factor: × 1000 for 0.001 precision.
const HEBBIAN_CENTROID_SHADER: &str = r#"
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
const HEBBIAN_ATTRACTION_SHADER: &str = r#"
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

// ============================================================================
// CLUSTER HYSTERESIS SHADERS
// "Stable clusters lock in, fading clusters release."
// ============================================================================

/// CLUSTER_STATS_SHADER: Pass 1 - Accumulate activity and count per cluster
/// Uses fixed-point u32 atomics for activity sum.
const CLUSTER_STATS_SHADER: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

// Per-cluster stats (256 clusters max)
struct ClusterStats {
    activity_sum: array<atomic<u32>, 256>,  // Fixed-point (×1000)
    count: array<atomic<u32>, 256>,
}

const FLAG_DEAD: u32 = 32u;
const FP_SCALE: f32 = 1000.0;
const MAX_CLUSTERS: u32 = 256u;

@group(0) @binding(0) var<storage, read> metadata: array<CellMetadata>;
@group(0) @binding(1) var<storage, read> energies: array<CellEnergy>;
@group(0) @binding(2) var<storage, read_write> cluster_stats: ClusterStats;
@group(0) @binding(3) var<uniform> cell_count: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= cell_count { return; }

    let cell_meta = metadata[idx];
    if (cell_meta.flags & FLAG_DEAD) != 0u { return; }
    if cell_meta.cluster_id == 0u { return; }
    if cell_meta.cluster_id >= MAX_CLUSTERS { return; }

    let activity = energies[idx].activity_level;
    let activity_fp = u32(activity * FP_SCALE);

    atomicAdd(&cluster_stats.activity_sum[cell_meta.cluster_id], activity_fp);
    atomicAdd(&cluster_stats.count[cell_meta.cluster_id], 1u);
}
"#;

/// CLUSTER_HYSTERESIS_SHADER: Pass 2 - Update hysteresis based on cluster activity
const CLUSTER_HYSTERESIS_SHADER: &str = r#"
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

// Per-cluster stats (256 clusters max)
struct ClusterStats {
    activity_sum: array<atomic<u32>, 256>,
    count: array<atomic<u32>, 256>,
}

const FLAG_DEAD: u32 = 32u;
const FP_SCALE: f32 = 1000.0;
const MAX_CLUSTERS: u32 = 256u;
const HIGH_ACTIVITY_THRESHOLD: f32 = 0.6;
const LOW_ACTIVITY_THRESHOLD: f32 = 0.2;
const HYSTERESIS_LOCK_RATE: f32 = 0.05;
const HYSTERESIS_RELEASE_RATE: f32 = 0.02;
const HYSTERESIS_NO_CLUSTER_DECAY: f32 = 0.1;

@group(0) @binding(0) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(1) var<storage, read> cluster_stats: ClusterStats;
@group(0) @binding(2) var<uniform> cell_count: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= cell_count { return; }

    var cell_meta = metadata[idx];
    if (cell_meta.flags & FLAG_DEAD) != 0u { return; }

    var hysteresis = cell_meta.hysteresis;

    if cell_meta.cluster_id > 0u && cell_meta.cluster_id < MAX_CLUSTERS {
        // Read cluster stats
        let count = max(atomicLoad(&cluster_stats.count[cell_meta.cluster_id]), 1u);
        let activity_sum_fp = atomicLoad(&cluster_stats.activity_sum[cell_meta.cluster_id]);
        let avg_activity = f32(activity_sum_fp) / (f32(count) * FP_SCALE);

        if avg_activity > HIGH_ACTIVITY_THRESHOLD {
            // Stable & Active: lock it in!
            hysteresis = min(hysteresis + HYSTERESIS_LOCK_RATE, 1.0);
        } else if avg_activity < LOW_ACTIVITY_THRESHOLD {
            // Fading out: release bond
            hysteresis = max(hysteresis - HYSTERESIS_RELEASE_RATE, 0.0);
        }
    } else {
        // No cluster: decay hysteresis faster
        hysteresis = max(hysteresis - HYSTERESIS_NO_CLUSTER_DECAY, 0.0);
    }

    cell_meta.hysteresis = hysteresis;
    metadata[idx] = cell_meta;
}
"#;

const SPATIAL_SIGNAL_TEMPLATE: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct CellPosition { position: array<f32, 16> }
struct SignalFragment { source_id_low: u32, source_id_high: u32, content: array<f32, 8>, position: array<f32, 8>, intensity: f32, _pad: array<f32, 3> }
struct GridRegion { count: u32, cell_indices: array<u32, 64>, _pad: array<u32, 3> }
struct SpatialConfig { grid_size: vec3<u32>, max_cells_per_region: u32, min_pos: vec3<f32>, region_size: f32, cell_count: u32, _pad1: u32, _pad2: u32, _pad3: u32 }
struct Config { energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32, cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32, signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32, tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32> }
struct CellConnections { targets: array<u32, 16>, strengths: array<f32, 16>, count: u32, _pad: array<u32, 3> }

const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
const NO_CONNECTION: u32 = 0xFFFFFFFFu;
const CONNECTION_SIGNAL_FACTOR: f32 = 0.5;
const HUB_INFLUENCE_FACTOR: f32 = 0.1;
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }

@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> positions: array<CellPosition>;
@group(0) @binding(2) var<storage, read_write> states: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(4) var<storage, read> signals: array<SignalFragment>;
@group(0) @binding(5) var<storage, read> grid: array<GridRegion>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<uniform> spatial_config: SpatialConfig;
@group(0) @binding(8) var<storage, read> connections: array<CellConnections>;
@group(0) @binding(9) var<storage, read> dna_pool: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> dna_indices: array<u32>;

fn position_to_grid(pos: vec3<f32>) -> vec3<i32> {
    let normalized = (pos - spatial_config.min_pos) / spatial_config.region_size;
    return vec3<i32>(i32(normalized.x), i32(normalized.y), i32(normalized.z));
}
fn grid_index(coord: vec3<i32>) -> u32 {
    let clamped = vec3<u32>(u32(clamp(coord.x, 0, i32(spatial_config.grid_size.x) - 1)), u32(clamp(coord.y, 0, i32(spatial_config.grid_size.y) - 1)), u32(clamp(coord.z, 0, i32(spatial_config.grid_size.z) - 1)));
    return clamped.x + clamped.y * spatial_config.grid_size.x + clamped.z * spatial_config.grid_size.x * spatial_config.grid_size.y;
}
fn calculate_resonance(signal_content: array<f32, 8>, cell_state_0: vec4<f32>, cell_state_1: vec4<f32>) -> f32 {
    var dot = 0.0; var norm_sig = 0.0; var norm_state = 0.0;
    for (var i = 0u; i < 4u; i++) { dot += signal_content[i] * cell_state_0[i]; norm_sig += signal_content[i] * signal_content[i]; norm_state += cell_state_0[i] * cell_state_0[i]; }
    for (var i = 0u; i < 4u; i++) { dot += signal_content[i+4u] * cell_state_1[i]; norm_sig += signal_content[i+4u] * signal_content[i+4u]; norm_state += cell_state_1[i] * cell_state_1[i]; }
    let denom = sqrt(norm_sig * norm_state);
    if denom > 0.001 { return (dot / denom + 1.0) * 0.5; }
    return 0.5;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let signal_idx = id.x;
    if signal_idx >= arrayLength(&signals) { return; }
    let signal = signals[signal_idx];
    if signal.intensity < 0.001 { return; }
    let signal_pos = vec3<f32>(signal.content[0], signal.content[1], signal.content[2]);
    let signal_grid = position_to_grid(signal_pos);
    let grid_radius = i32(ceil(config.signal_radius / spatial_config.region_size));
    for (var dx = -grid_radius; dx <= grid_radius; dx++) {
        for (var dy = -grid_radius; dy <= grid_radius; dy++) {
            for (var dz = -grid_radius; dz <= grid_radius; dz++) {
                let neighbor_coord = signal_grid + vec3<i32>(dx, dy, dz);
                if neighbor_coord.x < 0 || neighbor_coord.x >= i32(spatial_config.grid_size.x) || neighbor_coord.y < 0 || neighbor_coord.y >= i32(spatial_config.grid_size.y) || neighbor_coord.z < 0 || neighbor_coord.z >= i32(spatial_config.grid_size.z) { continue; }
                let region_idx = grid_index(neighbor_coord);
                let region = grid[region_idx];
                for (var i = 0u; i < min(region.count, spatial_config.max_cells_per_region); i++) {
                    let cell_idx = region.cell_indices[i];
                    if cell_idx >= config.cell_count { continue; }
                    var cell_energy = energies[cell_idx];
                    var cell_meta = metadata[cell_idx];
                    if (cell_meta.flags & FLAG_DEAD) != 0u { continue; }
                    let cell_pos = positions[cell_idx];
                    var dist_sq = 0.0;
                    for (var d = 0u; d < 8u; d++) { let diff = cell_pos.position[d] - signal.position[d]; dist_sq += diff * diff; }
                    let dist = sqrt(dist_sq);
                    if dist >= config.signal_radius { continue; }
                    let attenuation = 1.0 - (dist / config.signal_radius);
                    let intensity = signal.intensity * attenuation * config.reaction_amplification;
                    if (cell_meta.flags & FLAG_SLEEPING) != 0u && intensity > 0.1 { cell_meta.flags = cell_meta.flags & ~FLAG_SLEEPING; cell_energy.activity_level = 0.5; cell_energy.tension = 0.2; }
                    if (cell_meta.flags & FLAG_SLEEPING) == 0u {
                        let dna_idx = dna_indices[cell_idx];
                        let dna_base = dna_idx * 5u;
                        // DNA GENES
                        let gene_efficiency = dna_pool[dna_base+1u].x; // thresholds[4]
                        let gene_resonance = dna_pool[dna_base+1u].y;  // thresholds[5]
                        let gene_decay = dna_pool[dna_base+1u].z;      // thresholds[6]
                        let reflexivity_gain = dna_pool[dna_base+1u].w; // thresholds[7]

                        let attention_focus = dna_pool[dna_base+3u].z;
                        let semantic_filter = dna_pool[dna_base+3u].w;

                        // Calculate Traits from Genes (matching Rust logic)
                        let resonance_threshold = 0.05 + gene_resonance * 0.35; // [0.05, 0.4]
                        let efficiency = gene_efficiency; // [0.0, 1.0]

                        // [DYNAMIC_LOGIC]
                        let dynamic_intensity = intensity * attention_boost;
                        if (dynamic_intensity < semantic_threshold) { continue; }

                        var state0 = states[cell_idx * 8u]; var state1 = states[cell_idx * 8u + 1u];
                        for (var j = 0u; j < 4u; j++) { state0[j] += signal.content[j] * dynamic_intensity; }
                        for (var j = 0u; j < 4u; j++) { state1[j] += signal.content[j+4u] * dynamic_intensity; }
                        states[cell_idx * 8u] = state0; states[cell_idx * 8u + 1u] = state1;

                        let resonance = calculate_resonance(signal.content, state0, state1);

                        // LAW: Picky Eaters vs Trash Eaters
                        if resonance > resonance_threshold {
                            // Scale understanding from [Threshold, 1.0] -> [0.0, 1.0]
                            let understanding = (resonance - resonance_threshold) / (1.0 - resonance_threshold);

                            let energy_gain = config.signal_energy_base
                                * dynamic_intensity
                                * understanding
                                * efficiency
                                * (1.0 + resonance * config.signal_resonance_factor);

                            cell_energy.energy = min(cell_energy.energy + energy_gain, config.energy_cap);
                        }
                        cell_energy.activity_level += dynamic_intensity;
                    }
                    energies[cell_idx] = cell_energy; metadata[cell_idx] = cell_meta;
                }
            }
        }
    }
}
"#;

pub const CLUSTER_SYNTHESIS_TEMPLATE: &str = r#"
struct CellMetadata {
    flags: u32,
    cluster_id: u32,
    hysteresis: f32,
    _pad: u32,
}

struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }

@group(0) @binding(0) var<storage, read> energies: array<CellEnergy>;
@group(0) @binding(3) var<storage, read> metadata: array<CellMetadata>;

struct ClusterData {
    center_tension: f32,
    total_activity: f32,
    cell_count: u32,
    _pad: u32,
}

@group(1) @binding(0) var<storage, read_write> clusters: array<ClusterData>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let meta = metadata[idx];
    if (meta.flags & 32u) != 0u { return; } // DEAD

    let cid = meta.cluster_id;
    if cid == 0u { return; }

    let energy = energies[idx];

    // Synthesis logic: accumulate into cluster center
    // Use atomicAdd if possible or separate reduction pass
    // For now, simple stable placeholder
    clusters[cid].total_activity += energy.activity_level;
}
"#;
