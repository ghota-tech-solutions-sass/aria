//! ARIA Brain Configuration
//!
//! Runtime configuration with GPU detection and fallback

use std::env;

/// Compute backend for cell processing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeBackend {
    /// CPU with Rayon parallelism
    Cpu,
    /// GPU with wgpu (Vulkan/Metal/DX12)
    Gpu,
}

/// Brain configuration
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of cells in the substrate
    pub cell_count: usize,
    /// Compute backend (GPU or CPU)
    pub backend: ComputeBackend,
    /// WebSocket/HTTP port
    pub port: u16,
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cell_count: 10_000,
            backend: ComputeBackend::Cpu,
            port: 8765,
            tick_interval_ms: 2,
        }
    }
}

#[allow(dead_code)]
impl Config {
    /// Create config from environment variables and GPU detection
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Port from environment
        if let Ok(port) = env::var("ARIA_PORT") {
            if let Ok(p) = port.parse() {
                config.port = p;
            }
        }

        // Cell count from environment
        if let Ok(count) = env::var("ARIA_CELLS") {
            if let Ok(c) = count.parse() {
                config.cell_count = c;
            }
        }

        // GPU mode from environment or auto-detect
        let gpu_requested = env::var("ARIA_GPU").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false);

        if gpu_requested {
            if Self::detect_gpu() {
                config.backend = ComputeBackend::Gpu;
                tracing::info!("🎮 GPU mode enabled (wgpu detected)");
            } else {
                tracing::warn!("⚠️ GPU requested but not available, falling back to CPU");
                config.backend = ComputeBackend::Cpu;
            }
        } else {
            config.backend = ComputeBackend::Cpu;
            tracing::info!("🖥️ CPU mode (use ARIA_GPU=1 for GPU acceleration)");
        }

        config
    }

    /// Detect if GPU is available using wgpu
    fn detect_gpu() -> bool {
        aria_compute::gpu_available()
    }

    /// Get recommended cell count based on backend
    pub fn recommended_cells(&self) -> usize {
        match self.backend {
            ComputeBackend::Cpu => 50_000,   // 50k for CPU
            ComputeBackend::Gpu => 500_000,  // 500k for GPU!
        }
    }
}

/// Print startup banner with config info
pub fn print_banner(config: &Config) {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║           🧠 ARIA Brain - Living Substrate 🧠            ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Cells: {:>10}                                      ║", config.cell_count);
    println!("║  Backend: {:>8}                                      ║",
        match config.backend {
            ComputeBackend::Cpu => "CPU",
            ComputeBackend::Gpu => "GPU 🎮",
        }
    );
    println!("║  Port: {:>11}                                      ║", config.port);
    println!("║  Tick: {:>9}ms                                      ║", config.tick_interval_ms);
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
}
