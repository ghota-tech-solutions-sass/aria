//! Prediction Law shaders - "Cells that predict correctly, survive"

/// Prediction phase: Each cell predicts its next state based on connections
/// Run BEFORE the main tick to generate predictions
pub const PREDICTION_GENERATE_SHADER: &str = r#"
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
pub const PREDICTION_EVALUATE_SHADER: &str = r#"
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
