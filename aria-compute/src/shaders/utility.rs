//! Utility shaders - Sleeping drain and other helpers

pub const SLEEPING_DRAIN_SHADER: &str = r#"
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
