//! GPU configuration types for SoA backend
//!
//! Contains GPU-compatible structs that match WGSL shader layouts.

use aria_core::config::AriaConfig;
use bytemuck::{Pod, Zeroable};

/// GPU-compatible config (matches shader struct)
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GpuConfig {
    pub energy_cap: f32,
    pub reaction_amplification: f32,
    pub state_cap: f32,
    pub signal_radius: f32,
    pub cost_rest: f32,
    pub cost_signal: f32,
    pub cost_move: f32,
    pub cost_divide: f32,
    pub signal_energy_base: f32,
    pub signal_resonance_factor: f32,
    pub energy_gain: f32,
    pub tick: u32,
    pub cell_count: u32,
    pub workgroup_size: u32,
    pub _pad: [u32; 2],
}

impl GpuConfig {
    pub fn from_config(config: &AriaConfig, tick: u64, cell_count: usize) -> Self {
        Self {
            energy_cap: config.metabolism.energy_cap,
            reaction_amplification: config.signals.reaction_amplification,
            state_cap: config.signals.state_cap,
            signal_radius: config.signals.signal_radius,
            cost_rest: config.metabolism.cost_rest,
            cost_signal: config.metabolism.cost_signal,
            cost_move: config.metabolism.cost_move,
            cost_divide: config.metabolism.cost_divide,
            signal_energy_base: config.metabolism.signal_energy_base,
            signal_resonance_factor: config.metabolism.signal_resonance_factor,
            energy_gain: config.metabolism.energy_gain,
            tick: tick as u32,
            cell_count: cell_count as u32,
            workgroup_size: 256,
            _pad: [0; 2],
        }
    }
}

/// Atomic counter for sparse dispatch
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct AtomicCounter {
    pub count: u32,
    pub _pad: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_size() {
        // GpuConfig must be 64 bytes (16 f32/u32 values)
        assert_eq!(std::mem::size_of::<GpuConfig>(), 64);
    }

    #[test]
    fn test_atomic_counter_size() {
        // AtomicCounter must be 16 bytes (4 u32 values)
        assert_eq!(std::mem::size_of::<AtomicCounter>(), 16);
    }
}
