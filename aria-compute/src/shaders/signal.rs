//! Signal propagation shaders

pub const SIGNAL_TEMPLATE: &str = r#"
struct CellEnergy { energy: f32, tension: f32, activity_level: f32, _pad: f32 }
struct Config {
    energy_cap: f32, reaction_amplification: f32, state_cap: f32, signal_radius: f32,
    cost_rest: f32, cost_signal: f32, cost_move: f32, cost_divide: f32,
    signal_energy_base: f32, signal_resonance_factor: f32, energy_gain: f32,
    tick: u32, cell_count: u32, workgroup_size: u32, _pad: vec2<u32>
}
struct SignalFragment { source_id_low: u32, source_id_high: u32, content: array<f32, 8>, position: array<f32, 8>, intensity: f32, _pad: array<f32, 3> }
const FLAG_SLEEPING: u32 = 1u;
const FLAG_DEAD: u32 = 32u;
struct CellMetadata { flags: u32, cluster_id: u32, hysteresis: f32, _pad: u32 }
struct CellPosition { position: array<f32, 16> }

@group(0) @binding(0) var<storage, read_write> energies: array<CellEnergy>;
@group(0) @binding(1) var<storage, read> positions: array<CellPosition>;
@group(0) @binding(2) var<storage, read_write> states: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> metadata: array<CellMetadata>;
@group(0) @binding(4) var<storage, read> dna_pool: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> signals: array<SignalFragment>;
@group(0) @binding(6) var<uniform> config: Config;
@group(0) @binding(7) var<storage, read> dna_indices: array<u32>;

// Simple hash for stochastic noise (same signal != same result)
fn hash(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x & 0xFFFFu) / 65535.0;  // [0, 1]
}

fn calculate_resonance(signal_content: array<f32, 8>, state0: vec4<f32>, state1: vec4<f32>) -> f32 {
    var dot = 0.0; var norm_sig = 0.0; var norm_state = 0.0;
    for (var i = 0u; i < 4u; i++) { dot += signal_content[i] * state0[i]; norm_sig += signal_content[i] * signal_content[i]; norm_state += state0[i] * state0[i]; }
    for (var i = 0u; i < 4u; i++) { dot += signal_content[i+4u] * state1[i]; norm_sig += signal_content[i+4u] * signal_content[i+4u]; norm_state += state1[i] * state1[i]; }
    let denom = sqrt(norm_sig * norm_state);
    if denom > 0.001 { return (dot / denom + 1.0) * 0.5; }
    return 0.5;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= config.cell_count { return; }
    var cell_meta = metadata[idx];
    if (cell_meta.flags & FLAG_DEAD) != 0u { return; }
    var cell_energy = energies[idx];
    let cell_pos = positions[idx];

    let dna_idx = dna_indices[idx];
    let dna_base = dna_idx * 5u;
    let gene_efficiency = dna_pool[dna_base+1u].x;
    let gene_resonance = dna_pool[dna_base+1u].y;
    let reflexivity_gain = dna_pool[dna_base+1u].w;
    let attention_focus = dna_pool[dna_base+3u].z;
    let semantic_filter = dna_pool[dna_base+3u].w;

    let resonance_threshold = 0.05 + gene_resonance * 0.35;
    let efficiency = gene_efficiency;

    // Stochastic seed: unique per cell per tick
    let noise_seed = idx * 31337u + config.tick * 7919u;

    // [DYNAMIC_LOGIC]

    // === DYNAMIC SIGNAL RADIUS (Session 32 Part 11) ===
    // At low population, signals must reach further to find cells
    // This prevents death spirals when cells are spread thin in 8D space
    let density_factor = 10000.0 / max(f32(config.cell_count), 1000.0);
    let effective_radius = config.signal_radius * sqrt(density_factor);
    // At 50k cells: radius = 30 * sqrt(0.2) = 13.4 (tighter)
    // At 10k cells: radius = 30 * sqrt(1.0) = 30 (baseline)
    // At 1k cells:  radius = 30 * sqrt(10) = 95 (wider to find cells)
    // At 500 cells: radius = 30 * sqrt(20) = 134 (very wide)

    // === WAVE PROPAGATION ===
    // Signals propagate like waves: intensity decreases with distance
    // Cells near the signal source receive it first and strongest
    let signal_count = arrayLength(&signals);
    for (var s = 0u; s < signal_count; s++) {
        let signal = signals[s];
        if signal.intensity < 0.001 { continue; }

        // Calculate distance in semantic space (8D)
        var dist_sq = 0.0;
        for (var d = 0u; d < 8u; d++) {
            let diff = cell_pos.position[d] - signal.position[d];
            dist_sq += diff * diff;
        }
        let dist = sqrt(dist_sq);

        // Skip if outside dynamic wave radius
        if dist >= effective_radius { continue; }

        // Wave attenuation: stronger near source, weaker far away
        let attenuation = 1.0 - (dist / effective_radius);

        // Add stochastic noise (±10%) so same signal != same result
        let noise = (hash(noise_seed + s) - 0.5) * 0.2;  // [-0.1, 0.1]
        let noisy_attenuation = clamp(attenuation + noise, 0.0, 1.0);

        let intensity = signal.intensity * noisy_attenuation * config.reaction_amplification;

        // Wake up sleeping cells if wave is strong enough
        if (cell_meta.flags & FLAG_SLEEPING) != 0u && intensity > 0.1 {
            cell_meta.flags = cell_meta.flags & ~FLAG_SLEEPING;
            cell_energy.activity_level = 0.5;
            cell_energy.tension = 0.2;
        }

        // Only awake cells process signals
        if (cell_meta.flags & FLAG_SLEEPING) == 0u {
            // Update cell state from signal with noise
            var state0 = states[idx * 8u];
            var state1 = states[idx * 8u + 1u];
            for (var j = 0u; j < 4u; j++) {
                let content_noise = (hash(noise_seed + s * 8u + j) - 0.5) * 0.1;
                state0[j] += (signal.content[j] + content_noise) * intensity;
            }
            for (var j = 0u; j < 4u; j++) {
                let content_noise = (hash(noise_seed + s * 8u + 4u + j) - 0.5) * 0.1;
                state1[j] += (signal.content[j + 4u] + content_noise) * intensity;
            }
            states[idx * 8u] = state0;
            states[idx * 8u + 1u] = state1;

            // Calculate resonance for energy
            let resonance = calculate_resonance(signal.content, state0, state1);

            // LA VRAIE FAIM: Only resonance feeds
            // Population scaling: more cells = more energy per signal (sqrt scaling)
            // At 10k: scale=1.0, at 40k: scale=2.0, at 100k: scale=3.2
            let population_scale = sqrt(max(f32(config.cell_count), 10000.0) / 10000.0);

            if resonance > resonance_threshold {
                let understanding = (resonance - resonance_threshold) / (1.0 - resonance_threshold);
                let energy_gain = config.signal_energy_base
                    * signal.intensity
                    * noisy_attenuation
                    * understanding
                    * efficiency
                    * population_scale
                    * (1.0 + resonance * config.signal_resonance_factor);
                cell_energy.energy = min(cell_energy.energy + energy_gain, config.energy_cap);
            }

            // Build tension from wave (cells want to re-emit)
            cell_energy.tension += intensity * 0.1;
            cell_energy.activity_level += intensity;
        }
    }

    // High tension = cell wants to emit (propagate the wave)
    // reflexivity_gain is already read from DNA, use it directly
    if cell_energy.tension > 0.8 {
        cell_energy.activity_level += 0.5 * (0.5 + reflexivity_gain);
        cell_energy.energy -= config.cost_signal;
        cell_energy.tension -= 0.3;  // Release some tension after "emitting"
    }

    energies[idx] = cell_energy;
    metadata[idx] = cell_meta;
}
"#;

pub const SPATIAL_SIGNAL_TEMPLATE: &str = r#"
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

    // === DYNAMIC SIGNAL RADIUS (Session 32 Part 11) ===
    // At low population, signals must reach further to find cells
    let density_factor = 10000.0 / max(f32(config.cell_count), 1000.0);
    let effective_radius = config.signal_radius * sqrt(density_factor);

    let signal_pos = vec3<f32>(signal.content[0], signal.content[1], signal.content[2]);
    let signal_grid = position_to_grid(signal_pos);
    let grid_radius = i32(ceil(effective_radius / spatial_config.region_size));
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
                    if dist >= effective_radius { continue; }
                    let attenuation = 1.0 - (dist / effective_radius);
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

                        // LA VRAIE FAIM: Population scaling - more cells = more energy per signal
                        // sqrt scaling: 10k=1.0, 40k=2.0, 100k=3.2
                        let population_scale = sqrt(max(f32(config.cell_count), 10000.0) / 10000.0);

                        // LAW: Picky Eaters vs Trash Eaters
                        if resonance > resonance_threshold {
                            // Scale understanding from [Threshold, 1.0] -> [0.0, 1.0]
                            let understanding = (resonance - resonance_threshold) / (1.0 - resonance_threshold);

                            let energy_gain = config.signal_energy_base
                                * dynamic_intensity
                                * understanding
                                * efficiency
                                * population_scale
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
