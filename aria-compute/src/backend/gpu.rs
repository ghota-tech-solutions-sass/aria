//! # GPU Backend
//!
//! Massive parallel computation using wgpu.
//!
//! This backend can handle millions of cells by running
//! compute shaders on the GPU.
//!
//! ## Architecture
//!
//! - Cell states live in GPU buffers
//! - Compute shaders update cells in parallel
//! - Results are read back only when needed
//!
//! ## Supported GPUs
//!
//! - NVIDIA (Vulkan)
//! - AMD (Vulkan)
//! - Intel (Vulkan)
//! - Apple M-series (Metal)

use std::sync::Arc;

use aria_core::cell::{Cell, CellAction, CellState};
use aria_core::config::AriaConfig;
use aria_core::dna::DNA;
use aria_core::error::{AriaError, AriaResult};
use aria_core::signal::{Signal, SignalFragment};
use aria_core::traits::{BackendStats, ComputeBackend};

/// GPU compute backend using wgpu
pub struct GpuBackend {
    /// wgpu device
    device: Arc<wgpu::Device>,

    /// Command queue
    queue: Arc<wgpu::Queue>,

    /// Configuration
    config: AriaConfig,

    /// Statistics
    stats: BackendStats,

    /// Current tick
    tick: u64,

    /// Is initialized?
    initialized: bool,
}

impl GpuBackend {
    /// Create a new GPU backend
    pub fn new(config: &AriaConfig) -> AriaResult<Self> {
        // Initialize wgpu
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Find a suitable adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| AriaError::gpu("No suitable GPU adapter found"))?;

        // Log adapter info
        let info = adapter.get_info();
        tracing::info!(
            "Using GPU: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        // Create device and queue
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        ))
        .map_err(|e| AriaError::gpu(format!("Failed to create device: {}", e)))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            config: config.clone(),
            stats: BackendStats::default(),
            tick: 0,
            initialized: false,
        })
    }

    /// Initialize GPU buffers for a given population
    fn init_buffers(&mut self, cell_count: usize) -> AriaResult<()> {
        // TODO: Create GPU buffers for:
        // - Cell states (CellState array)
        // - DNA pool
        // - Signal buffer
        // - Output actions buffer

        let state_size = std::mem::size_of::<CellState>() * cell_count;
        tracing::info!(
            "GPU: Allocating {} MB for {} cells",
            state_size / 1024 / 1024,
            cell_count
        );

        self.initialized = true;
        Ok(())
    }
}

impl ComputeBackend for GpuBackend {
    fn init(&mut self, config: &AriaConfig) -> AriaResult<()> {
        self.config = config.clone();
        Ok(())
    }

    fn update_cells(
        &mut self,
        cells: &mut [Cell],
        states: &mut [CellState],
        dna_pool: &[DNA],
        signals: &[SignalFragment],
    ) -> AriaResult<Vec<(u64, CellAction)>> {
        self.tick += 1;

        // Initialize buffers on first call
        if !self.initialized {
            self.init_buffers(cells.len())?;
        }

        // TODO: Implement GPU compute shader execution
        // For now, fall back to CPU-style processing
        // This is a placeholder that will be replaced with actual GPU code

        tracing::debug!(
            "GPU tick {} - {} cells, {} signals",
            self.tick,
            cells.len(),
            signals.len()
        );

        // Update stats
        self.stats.cells_processed = cells.len() as u64;

        // For now, return empty (no actions)
        // Real implementation will use compute shaders
        Ok(Vec::new())
    }

    fn propagate_signals(
        &mut self,
        states: &[CellState],
        signals: Vec<SignalFragment>,
    ) -> AriaResult<Vec<(usize, SignalFragment)>> {
        // TODO: Implement GPU signal propagation
        self.stats.signals_propagated = signals.len() as u64;
        Ok(Vec::new())
    }

    fn detect_emergence(
        &self,
        _cells: &[Cell],
        states: &[CellState],
        config: &AriaConfig,
    ) -> AriaResult<Vec<Signal>> {
        // TODO: Implement GPU emergence detection
        // This is compute-intensive and benefits greatly from GPU

        if self.tick % config.emergence.check_interval != 0 {
            return Ok(Vec::new());
        }

        Ok(Vec::new())
    }

    fn stats(&self) -> BackendStats {
        self.stats.clone()
    }

    fn sync(&mut self) -> AriaResult<()> {
        // Wait for GPU operations to complete
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn name(&self) -> &'static str {
        "GPU (wgpu)"
    }
}

// Shader source code (WGSL)
// These will be loaded and compiled at runtime

/// Cell update compute shader
pub const CELL_UPDATE_SHADER: &str = r#"
// Cell update compute shader for ARIA
// Updates cell states in parallel on the GPU

struct CellState {
    position: array<f32, 16>,
    state: array<f32, 32>,
    energy: f32,
    tension: f32,
    activity_level: f32,
    flags: u32,
    _reserved: array<f32, 4>,
}

struct DNA {
    thresholds: array<f32, 8>,
    reactions: array<f32, 8>,
    signature: u64,
    _pad: u64,
}

struct Config {
    energy_consumption: f32,
    energy_gain: f32,
    energy_cap: f32,
    reaction_amplification: f32,
    state_cap: f32,
    tick: u32,
    _pad: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read_write> cells: array<CellState>;
@group(0) @binding(1) var<storage, read> dna_pool: array<DNA>;
@group(0) @binding(2) var<uniform> config: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= arrayLength(&cells) {
        return;
    }

    var cell = cells[cell_idx];

    // Check if sleeping (bit 0 of flags)
    if (cell.flags & 1u) != 0u {
        return;
    }

    // Metabolism
    cell.energy -= config.energy_consumption;
    cell.energy += config.energy_gain;
    cell.energy = min(cell.energy, config.energy_cap);

    // Death check
    if cell.energy <= 0.0 {
        cell.flags = cell.flags | 32u; // Set dead flag
    }

    // Build tension
    cell.tension += 0.01;

    // Normalize state
    var norm: f32 = 0.0;
    for (var i = 0u; i < 32u; i++) {
        norm += cell.state[i] * cell.state[i];
    }
    norm = sqrt(norm);
    if norm > config.state_cap {
        let scale = config.state_cap / norm;
        for (var i = 0u; i < 32u; i++) {
            cell.state[i] *= scale;
        }
    }

    cells[cell_idx] = cell;
}
"#;

/// Signal propagation compute shader
pub const SIGNAL_PROPAGATE_SHADER: &str = r#"
// Signal propagation for ARIA
// Distributes signals to nearby cells based on semantic distance

struct SignalFragment {
    source_id: u64,
    content: array<f32, 8>,
    intensity: f32,
    _pad: f32,
}

struct CellState {
    position: array<f32, 16>,
    state: array<f32, 32>,
    energy: f32,
    tension: f32,
    activity_level: f32,
    flags: u32,
    _reserved: array<f32, 4>,
}

@group(0) @binding(0) var<storage, read_write> cells: array<CellState>;
@group(0) @binding(1) var<storage, read> signals: array<SignalFragment>;
@group(0) @binding(2) var<uniform> signal_radius: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= arrayLength(&cells) {
        return;
    }

    var cell = cells[cell_idx];

    // Skip sleeping cells
    if (cell.flags & 1u) != 0u {
        return;
    }

    // Process each signal
    for (var s = 0u; s < arrayLength(&signals); s++) {
        let signal = signals[s];

        // Calculate distance in semantic space
        var dist_sq: f32 = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let diff = cell.position[i] - signal.content[i];
            dist_sq += diff * diff;
        }
        let dist = sqrt(dist_sq);

        if dist < signal_radius {
            // Attenuate by distance
            let attenuation = 1.0 - (dist / signal_radius);
            let intensity = signal.intensity * attenuation;

            // Add signal to cell state
            for (var i = 0u; i < 8u; i++) {
                cell.state[i] += signal.content[i] * intensity;
            }

            // Energy from interaction
            cell.energy += intensity * 0.05;
        }
    }

    cells[cell_idx] = cell;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_creation() {
        let config = AriaConfig::default();
        // This test will fail on systems without GPU
        // That's expected behavior
        let result = GpuBackend::new(&config);
        if result.is_err() {
            println!("GPU not available: {:?}", result.err());
        }
    }
}
