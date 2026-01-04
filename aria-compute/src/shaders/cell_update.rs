//! Cell update shader - Core metabolism and state updates

pub const CELL_UPDATE_TEMPLATE: &str = r#"
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
