//! # Compute Backends
//!
//! Implementations of the ComputeBackend trait for CPU and GPU.
//!
//! ## Backend Variants
//!
//! - `CpuBackend`: Fallback for systems without GPU
//! - `GpuSoABackend`: Optimized GPU backend with SoA layout, Hebbian learning, 5M+ cells

mod cpu;
mod gpu_soa;

pub use cpu::CpuBackend;
pub use gpu_soa::GpuSoABackend;
