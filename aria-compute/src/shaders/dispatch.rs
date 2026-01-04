//! Sparse dispatch shaders - Compaction and indirect dispatch

pub const COMPACT_SHADER: &str = r#"
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

pub const PREPARE_DISPATCH_SHADER: &str = r#"
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
