//! # Compute Backends
//!
//! Implementations of the ComputeBackend trait for CPU and GPU.
//!
//! ## Backend Variants
//!
//! - `CpuBackend`: Fallback for systems without GPU
//! - `GpuBackend`: Original GPU backend with AoS layout (legacy)
//! - `GpuSoABackend`: Optimized GPU backend with SoA layout for 5M+ cells

mod cpu;
mod gpu;
mod gpu_soa;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;
pub use gpu_soa::GpuSoABackend;
