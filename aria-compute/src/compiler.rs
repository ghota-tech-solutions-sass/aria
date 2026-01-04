//! # ARIA JIT Compiler
//!
//! Dynamic shader generation and compilation for ARIA's GPU backend.

use crate::shaders;

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

    // Shader template getters - delegate to shaders module
    pub fn get_spatial_signal_template(&self) -> &str { shaders::SPATIAL_SIGNAL_TEMPLATE }
    pub fn get_cell_update_template(&self) -> &str { shaders::CELL_UPDATE_TEMPLATE }
    pub fn get_signal_template(&self) -> &str { shaders::SIGNAL_TEMPLATE }
    pub fn get_compact_shader(&self) -> &str { shaders::COMPACT_SHADER }
    pub fn get_prepare_dispatch_shader(&self) -> &str { shaders::PREPARE_DISPATCH_SHADER }
    pub fn get_sleeping_drain_shader(&self) -> &str { shaders::SLEEPING_DRAIN_SHADER }
    pub fn get_hebbian_shader(&self) -> &str { shaders::HEBBIAN_SHADER }

    // Prediction Law shaders
    pub fn get_prediction_generate_shader(&self) -> &str { shaders::PREDICTION_GENERATE_SHADER }
    pub fn get_prediction_evaluate_shader(&self) -> &str { shaders::PREDICTION_EVALUATE_SHADER }

    // Hebbian Spatial Attraction shaders (GPU migration)
    pub fn get_hebbian_centroid_shader(&self) -> &str { shaders::HEBBIAN_CENTROID_SHADER }
    pub fn get_hebbian_attraction_shader(&self) -> &str { shaders::HEBBIAN_ATTRACTION_SHADER }

    // Cluster Hysteresis shaders (GPU migration)
    pub fn get_cluster_stats_shader(&self) -> &str { shaders::CLUSTER_STATS_SHADER }
    pub fn get_cluster_hysteresis_shader(&self) -> &str { shaders::CLUSTER_HYSTERESIS_SHADER }

    // GPU Lifecycle Slot System shaders
    pub fn get_death_shader(&self) -> &str { shaders::DEATH_SHADER }
    pub fn get_birth_shader(&self) -> &str { shaders::BIRTH_SHADER }
    pub fn get_reset_lifecycle_counters_shader(&self) -> &str { shaders::RESET_LIFECYCLE_COUNTERS_SHADER }
}
