//! # ARIA Compute
//!
//! Compute backends for ARIA's living substrate.
//!
//! This crate provides two backends:
//! - **CPU**: Uses Rayon for parallel computation (development, small populations)
//! - **GPU**: Uses wgpu with SoA layout + Hebbian learning (optimized for 5M+ cells)
//!
//! ## Backend Selection
//!
//! Set `ARIA_BACKEND` environment variable:
//! - `cpu` - Force CPU backend
//! - `gpu` or `gpu_soa` - Force GPU backend
//! - `auto` (default) - Auto-select GPU if available

pub mod backend;
pub mod compiler;
pub mod spatial;
pub mod spatial_gpu;

pub use backend::{CpuBackend, GpuSoABackend};
pub use spatial::SpatialHash;
pub use spatial_gpu::{SpatialHashConfig, GRID_SIZE, MAX_CELLS_PER_REGION};

// Backwards compatibility alias
pub use backend::GpuSoABackend as GpuBackend;

use aria_core::config::{AriaConfig, ComputeBackendType};
use aria_core::error::AriaResult;
use aria_core::traits::ComputeBackend;

/// Create the appropriate compute backend based on configuration
pub fn create_backend(config: &AriaConfig) -> AriaResult<Box<dyn ComputeBackend>> {
    match config.compute.backend {
        ComputeBackendType::Auto => {
            // Always try GPU first (SoA backend handles all sizes)
            match GpuSoABackend::new(config) {
                Ok(gpu) => {
                    tracing::info!(
                        "🎮 Using GPU backend (SoA + Hebbian) for {} cells",
                        config.population.target_population
                    );
                    Ok(Box::new(gpu))
                }
                Err(e) => {
                    tracing::warn!("GPU not available ({}), falling back to CPU", e);
                    Ok(Box::new(CpuBackend::new(config)?))
                }
            }
        }
        ComputeBackendType::Cpu => {
            tracing::info!("Using CPU backend (Rayon)");
            Ok(Box::new(CpuBackend::new(config)?))
        }
        ComputeBackendType::Gpu | ComputeBackendType::GpuSoA => {
            tracing::info!("🎮 Using GPU backend (SoA + Hebbian)");
            Ok(Box::new(GpuSoABackend::new(config)?))
        }
    }
}

/// Check if GPU is available on this system
pub fn gpu_available() -> bool {
    if std::env::var("ARIA_FORCE_CPU").is_ok() {
        return false;
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    !adapters.is_empty()
}

/// Get information about available compute devices
pub fn device_info() -> Vec<DeviceInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

    adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            DeviceInfo {
                name: info.name,
                vendor: info.vendor.to_string(),
                device_type: format!("{:?}", info.device_type),
                backend: format!("{:?}", info.backend),
            }
        })
        .collect()
}

/// Information about a compute device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
}
