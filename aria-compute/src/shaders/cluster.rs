//! Cluster shaders - Statistics, hysteresis, and synthesis

/// CLUSTER_STATS_SHADER: Pass 1 - Accumulate activity and count per cluster
/// Uses fixed-point u32 atomics for activity sum.
pub const CLUSTER_STATS_SHADER: &str = r#"
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
pub const CLUSTER_HYSTERESIS_SHADER: &str = r#"
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
