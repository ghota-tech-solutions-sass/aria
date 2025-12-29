//! # Compute Backends
//!
//! Implementations of the ComputeBackend trait for CPU and GPU.

mod cpu;
mod gpu;

pub use cpu::CpuBackend;
pub use gpu::GpuBackend;
