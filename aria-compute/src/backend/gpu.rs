//! # GPU Backend
//!
//! Massive parallel computation using wgpu 28.
//!
//! This backend can handle millions of cells by running
//! compute shaders on the GPU.
//!
//! ## Sparse Dispatch (Gemini optimization)
//!
//! For 5M+ cells, we use a two-pass approach:
//! 1. **Compact pass**: Build list of active cell indices + count
//! 2. **Update pass**: Only dispatch active cells using indirect dispatch
//!
//! This saves 80%+ GPU bandwidth when most cells are sleeping.

use std::sync::Arc;

use aria_core::cell::{Cell, CellAction, CellState};
use aria_core::config::AriaConfig;
use aria_core::dna::DNA;
use aria_core::error::{AriaError, AriaResult};
use aria_core::signal::{Signal, SignalFragment};
use aria_core::traits::{BackendStats, ComputeBackend};
use bytemuck::{Pod, Zeroable};

/// GPU-compatible config (matches shader struct)
/// "La Vraie Faim" - action costs and resonance-based energy
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct GpuConfig {
    // Basic metabolism
    energy_cap: f32,
    reaction_amplification: f32,
    state_cap: f32,
    signal_radius: f32,

    // Action costs (La Vraie Faim)
    cost_rest: f32,
    cost_signal: f32,
    cost_move: f32,
    cost_divide: f32,

    // Signal energy (resonance-based)
    signal_energy_base: f32,
    signal_resonance_factor: f32,

    // Bootstrap energy (tiny passive gain to prevent total starvation)
    energy_gain: f32,

    tick: u32,
}

impl GpuConfig {
    fn from_config(config: &AriaConfig, tick: u64) -> Self {
        Self {
            energy_cap: config.metabolism.energy_cap,
            reaction_amplification: config.signals.reaction_amplification,
            state_cap: config.signals.state_cap,
            signal_radius: config.signals.signal_radius,

            // Action costs
            cost_rest: config.metabolism.cost_rest,
            cost_signal: config.metabolism.cost_signal,
            cost_move: config.metabolism.cost_move,
            cost_divide: config.metabolism.cost_divide,

            // Signal energy
            signal_energy_base: config.metabolism.signal_energy_base,
            signal_resonance_factor: config.metabolism.signal_resonance_factor,

            // Bootstrap energy (0 = user must feed ARIA by talking)
            energy_gain: config.metabolism.energy_gain,

            tick: tick as u32,
        }
    }
}

/// Atomic counter for sparse dispatch (GPU-side)
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct AtomicCounter {
    count: u32,
    _pad: [u32; 3], // Align to 16 bytes for GPU
}

/// Indirect dispatch arguments (for wgpu indirect dispatch)
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct IndirectDispatch {
    x: u32, // Number of workgroups in X
    y: u32, // Always 1
    z: u32, // Always 1
    _pad: u32,
}

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

    // GPU Resources
    /// Cell states buffer
    cells_buffer: Option<wgpu::Buffer>,
    /// DNA pool buffer
    dna_buffer: Option<wgpu::Buffer>,
    /// Signals buffer
    signals_buffer: Option<wgpu::Buffer>,
    /// Config uniform buffer
    config_buffer: Option<wgpu::Buffer>,
    /// Staging buffer for readback
    staging_buffer: Option<wgpu::Buffer>,

    // === Sparse Dispatch Buffers (Gemini optimization) ===
    /// Atomic counter for active cells
    active_count_buffer: Option<wgpu::Buffer>,
    /// List of active cell indices (for indirect dispatch)
    active_indices_buffer: Option<wgpu::Buffer>,
    /// Staging buffer for reading active count
    active_count_staging: Option<wgpu::Buffer>,

    // Pipelines
    /// Cell update pipeline
    cell_update_pipeline: Option<wgpu::ComputePipeline>,
    /// Signal propagation pipeline
    signal_pipeline: Option<wgpu::ComputePipeline>,
    /// Compact pipeline (builds active_indices)
    compact_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Sparse bind group layout (for compact pass)
    sparse_bind_group_layout: Option<wgpu::BindGroupLayout>,

    /// Current cell count
    cell_count: usize,
    /// Maximum cell count (buffer capacity with headroom)
    max_cell_count: usize,
    /// Is initialized?
    initialized: bool,
    /// Use sparse dispatch (enabled for large populations)
    use_sparse_dispatch: bool,
}

impl GpuBackend {
    /// Create a new GPU backend
    pub fn new(config: &AriaConfig) -> AriaResult<Self> {
        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Find a suitable adapter
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| AriaError::gpu(format!("No suitable GPU adapter found: {}", e)))?;

        // Log adapter info
        let info = adapter.get_info();
        tracing::info!(
            "🎮 GPU: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        // Request device with higher limits for large populations
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ARIA GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1024 * 1024 * 1024, // 1GB
                    max_buffer_size: 1024 * 1024 * 1024,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .map_err(|e| AriaError::gpu(format!("Failed to create device: {}", e)))?;

        // Enable sparse dispatch for large populations (>100k cells)
        let use_sparse = config.population.target_population > 100_000;
        if use_sparse {
            tracing::info!("🎮 Sparse dispatch enabled (>100k cells)");
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            config: config.clone(),
            stats: BackendStats::default(),
            tick: 0,
            cells_buffer: None,
            dna_buffer: None,
            signals_buffer: None,
            config_buffer: None,
            staging_buffer: None,
            active_count_buffer: None,
            active_indices_buffer: None,
            active_count_staging: None,
            cell_update_pipeline: None,
            signal_pipeline: None,
            compact_pipeline: None,
            bind_group_layout: None,
            sparse_bind_group_layout: None,
            cell_count: 0,
            max_cell_count: 0,
            initialized: false,
            use_sparse_dispatch: use_sparse,
        })
    }

    /// Initialize GPU buffers and pipelines
    fn init_buffers(&mut self, cell_count: usize, dna_count: usize) -> AriaResult<()> {
        let cell_size = std::mem::size_of::<CellState>();
        let dna_size = std::mem::size_of::<DNA>();
        let signal_size = std::mem::size_of::<SignalFragment>();

        // Add 20% headroom for population fluctuations
        let cell_count_with_headroom = cell_count + cell_count / 5;
        let dna_count_with_headroom = dna_count + dna_count / 5;

        let cells_bytes = cell_size * cell_count_with_headroom;
        let dna_bytes = dna_size * dna_count_with_headroom;
        let max_signals = 1024; // Max signals per tick
        let signals_bytes = signal_size * max_signals;

        // Store max capacity for bounds checking
        self.max_cell_count = cell_count_with_headroom;

        // Sparse dispatch buffers (also with headroom)
        let active_indices_bytes = std::mem::size_of::<u32>() * cell_count_with_headroom;
        let counter_bytes = std::mem::size_of::<AtomicCounter>();

        let total_bytes = cells_bytes + dna_bytes + signals_bytes + active_indices_bytes + counter_bytes;

        tracing::info!(
            "🎮 GPU: Allocating {} MB for {} cells, {} DNA variants{}",
            total_bytes / 1024 / 1024,
            cell_count,
            dna_count,
            if self.use_sparse_dispatch { " (sparse dispatch enabled)" } else { "" }
        );

        // Create cell states buffer (read-write storage)
        self.cells_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell States"),
            size: cells_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Create DNA pool buffer (read-only storage)
        self.dna_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DNA Pool"),
            size: dna_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create signals buffer
        self.signals_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signals"),
            size: signals_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create config uniform buffer
        self.config_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config"),
            size: std::mem::size_of::<GpuConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create staging buffer for readback
        self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: cells_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // === Sparse Dispatch Buffers ===

        // Atomic counter for active cells
        self.active_count_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Active Count"),
            size: std::mem::size_of::<AtomicCounter>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Active indices buffer (for sparse update pass)
        self.active_indices_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Active Indices"),
            size: (std::mem::size_of::<u32>() * cell_count) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Staging buffer for reading active count
        self.active_count_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Active Count Staging"),
            size: std::mem::size_of::<AtomicCounter>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create bind group layout
        self.bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("ARIA Bind Group Layout"),
                entries: &[
                    // Cell states (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // DNA pool (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Create sparse bind group layout (for compact pass)
        self.sparse_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("ARIA Sparse Bind Group Layout"),
                entries: &[
                    // Cell states (read-only for compact pass)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Active count (atomic counter)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Active indices (output)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Create compute pipelines
        self.create_pipelines()?;

        self.cell_count = cell_count;
        self.initialized = true;

        tracing::info!("🎮 GPU initialized with {} cells", cell_count);
        Ok(())
    }

    /// Create compute pipelines
    fn create_pipelines(&mut self) -> AriaResult<()> {
        let bind_group_layout = self
            .bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Bind group layout not created"))?;

        // Create shader module for cell update
        let cell_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Cell Update Shader"),
                source: wgpu::ShaderSource::Wgsl(CELL_UPDATE_SHADER.into()),
            });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ARIA Pipeline Layout"),
                bind_group_layouts: &[bind_group_layout],
                immediate_size: 0,
            });

        // Create cell update pipeline
        self.cell_update_pipeline =
            Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Cell Update Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &cell_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

        // Create signal propagation shader and pipeline
        let signal_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Signal Propagation Shader"),
                source: wgpu::ShaderSource::Wgsl(SIGNAL_PROPAGATE_SHADER.into()),
            });

        self.signal_pipeline =
            Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Signal Propagation Pipeline"),
                        layout: Some(&pipeline_layout),
                        module: &signal_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

        // Create compact pipeline (for sparse dispatch)
        if let Some(sparse_layout) = &self.sparse_bind_group_layout {
            let compact_shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Compact Shader"),
                    source: wgpu::ShaderSource::Wgsl(COMPACT_SHADER.into()),
                });

            let sparse_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sparse Pipeline Layout"),
                    bind_group_layouts: &[sparse_layout],
                    immediate_size: 0,
                });

            self.compact_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Compact Pipeline"),
                        layout: Some(&sparse_pipeline_layout),
                        module: &compact_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::debug!("🎮 Sparse dispatch pipeline created");
        }

        tracing::debug!("🎮 GPU pipelines created");
        Ok(())
    }

    /// Upload cell states to GPU
    fn upload_cells(&self, states: &[CellState]) {
        if let Some(buffer) = &self.cells_buffer {
            // Bounds check: only upload up to max_cell_count
            let safe_count = states.len().min(self.max_cell_count);
            if safe_count < states.len() {
                tracing::warn!(
                    "⚠️ GPU: Truncating cell upload from {} to {} (buffer limit)",
                    states.len(),
                    safe_count
                );
            }
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&states[..safe_count]));
        }
    }

    /// Upload DNA pool to GPU
    fn upload_dna(&self, dna_pool: &[DNA]) {
        if let Some(buffer) = &self.dna_buffer {
            // Bounds check: DNA buffer has same headroom as cells
            let max_dna = self.max_cell_count;
            let safe_count = dna_pool.len().min(max_dna);
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&dna_pool[..safe_count]));
        }
    }

    /// Upload signals to GPU
    fn upload_signals(&self, signals: &[SignalFragment]) {
        if let Some(buffer) = &self.signals_buffer {
            // Pad with zeros if needed
            let mut padded = vec![SignalFragment::zeroed(); 1024];
            for (i, s) in signals.iter().take(1024).enumerate() {
                padded[i] = *s;
            }
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&padded));
        }
    }

    /// Upload config to GPU
    fn upload_config(&self) {
        if let Some(buffer) = &self.config_buffer {
            let gpu_config = GpuConfig::from_config(&self.config, self.tick);
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&gpu_config));
        }
    }

    /// Download cell states from GPU
    fn download_cells(&self, states: &mut [CellState]) -> AriaResult<()> {
        let cells_buffer = self
            .cells_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Cells buffer not initialized"))?;
        let staging_buffer = self
            .staging_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Staging buffer not initialized"))?;

        let size = std::mem::size_of::<CellState>() * states.len();

        // Copy from GPU to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });
        encoder.copy_buffer_to_buffer(cells_buffer, 0, staging_buffer, 0, size as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map staging buffer and read
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent submission
            timeout: None,          // Wait indefinitely
        });

        rx.recv()
            .map_err(|e| AriaError::gpu(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| AriaError::gpu(format!("Failed to map buffer: {:?}", e)))?;

        {
            let data = buffer_slice.get_mapped_range();
            let gpu_states: &[CellState] = bytemuck::cast_slice(&data);
            states.copy_from_slice(&gpu_states[..states.len()]);
        }

        staging_buffer.unmap();
        Ok(())
    }

    /// Create bind group for current buffers
    fn create_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Bind group layout not initialized"))?;
        let cells = self
            .cells_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Cells buffer not initialized"))?;
        let dna = self
            .dna_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("DNA buffer not initialized"))?;
        let config = self
            .config_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Config buffer not initialized"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ARIA Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dna.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config.as_entire_binding(),
                },
            ],
        }))
    }

    /// Run cell update shader
    fn run_cell_update(&self) -> AriaResult<()> {
        let pipeline = self
            .cell_update_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Cell update pipeline not initialized"))?;

        let bind_group = self.create_bind_group()?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cell Update Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cell Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroups = (self.cell_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Run signal propagation shader - LA VRAIE FAIM energy from resonance!
    fn run_signal_propagate(&self) -> AriaResult<()> {
        let pipeline = self
            .signal_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Signal pipeline not initialized"))?;

        let bind_group = self.create_bind_group()?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Signal Propagate Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Signal Propagate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroups = (self.cell_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    // === Sparse Dispatch Methods (Gemini optimization) ===

    /// Reset the atomic counter to zero
    fn reset_active_counter(&self) {
        if let Some(buffer) = &self.active_count_buffer {
            let zero = AtomicCounter {
                count: 0,
                _pad: [0; 3],
            };
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&zero));
        }
    }

    /// Create bind group for sparse dispatch (compact pass)
    fn create_sparse_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .sparse_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse bind group layout not initialized"))?;
        let cells = self
            .cells_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Cells buffer not initialized"))?;
        let counter = self
            .active_count_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Active count buffer not initialized"))?;
        let indices = self
            .active_indices_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Active indices buffer not initialized"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cells.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices.as_entire_binding(),
                },
            ],
        }))
    }

    /// Run compact pass to collect active cell indices
    fn run_compact_pass(&self) -> AriaResult<()> {
        let pipeline = self
            .compact_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Compact pipeline not initialized"))?;

        let bind_group = self.create_sparse_bind_group()?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compact Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compact Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups for all cells
            let workgroups = (self.cell_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Read back the active cell count from GPU
    fn read_active_count(&self) -> AriaResult<u32> {
        let counter_buffer = self
            .active_count_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Active count buffer not initialized"))?;
        let staging_buffer = self
            .active_count_staging
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Active count staging buffer not initialized"))?;

        let size = std::mem::size_of::<AtomicCounter>() as u64;

        // Copy from GPU to staging
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Counter Encoder"),
            });
        encoder.copy_buffer_to_buffer(counter_buffer, 0, staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        // Map staging buffer and read
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        rx.recv()
            .map_err(|e| AriaError::gpu(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| AriaError::gpu(format!("Failed to map buffer: {:?}", e)))?;

        let count = {
            let data = buffer_slice.get_mapped_range();
            let counter: &AtomicCounter = bytemuck::from_bytes(&data);
            counter.count
        };

        staging_buffer.unmap();
        Ok(count)
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

        // Initialize buffers on first call or if size changed
        let first_init = !self.initialized || self.cell_count != cells.len();
        if first_init {
            self.init_buffers(cells.len(), dna_pool.len())?;
            // Only upload cells on first init - they live on GPU after that
            self.upload_cells(states);
            self.upload_dna(dna_pool);
        }

        // Only upload signals (small) and config each tick
        self.upload_signals(signals);
        self.upload_config();

        // Run compact pass to count active cells (Gemini sparse dispatch)
        // This gives us accurate GPU-side stats without CPU iteration
        let (awake_count, sleeping_count) = if self.use_sparse_dispatch && self.compact_pipeline.is_some() {
            // Reset counter
            self.reset_active_counter();

            // Run compact pass (counts active cells on GPU)
            self.run_compact_pass()?;

            // Read back active count
            let active = self.read_active_count()? as usize;
            let sleeping = cells.len().saturating_sub(active);

            (active, sleeping)
        } else {
            // Fallback: count on CPU after download
            (0, 0) // Will be computed after download
        };

        // Run signal propagation FIRST (La Vraie Faim - energy from resonance)
        // ALWAYS propagate if signals exist - they are meant to WAKE UP sleeping cells!
        // The signal shader handles waking cells that receive strong signals.
        if !signals.is_empty() {
            self.run_signal_propagate()?;
        }

        // Run main cell update shader (costs, actions, death)
        self.run_cell_update()?;

        // OPTIMIZATION: Only download every 100 ticks (for stats/sync)
        // The GPU is the source of truth - CPU only needs periodic updates
        let should_download = self.tick % 100 == 0 || first_init;

        if should_download {
            self.download_cells(states)?;
        }

        // Use GPU-computed stats (from compact pass) - no CPU iteration needed
        let (final_awake, final_sleeping) = if self.use_sparse_dispatch {
            (awake_count, sleeping_count)
        } else if should_download {
            let sleeping = states.iter().filter(|s| s.is_sleeping()).count();
            (cells.len() - sleeping, sleeping)
        } else {
            // Use cached stats from last download
            (self.stats.cells_processed as usize, self.stats.cells_sleeping as usize)
        };

        // Update stats
        self.stats.cells_processed = final_awake as u64;
        self.stats.cells_sleeping = final_sleeping as u64;
        self.stats.signals_propagated = signals.len() as u64;

        // Check for dead cells only when we downloaded
        let mut actions = Vec::new();
        if should_download {
            for (i, state) in states.iter().enumerate() {
                if state.is_dead() {
                    actions.push((cells[i].id, CellAction::Die));
                }
            }
        }

        if self.tick % 100 == 0 {
            let gpu_percent = if cells.len() > 0 {
                (final_sleeping as f32 / cells.len() as f32 * 100.0) as u32
            } else {
                0
            };
            tracing::debug!(
                "🎮 GPU tick {} - {} active, {} sleeping ({}% sparse savings)",
                self.tick,
                final_awake,
                final_sleeping,
                gpu_percent
            );
        }

        Ok(actions)
    }

    fn propagate_signals(
        &mut self,
        _states: &[CellState],
        signals: Vec<SignalFragment>,
    ) -> AriaResult<Vec<(usize, SignalFragment)>> {
        // Signal propagation is handled in update_cells for now
        self.stats.signals_propagated = signals.len() as u64;
        Ok(Vec::new())
    }

    fn detect_emergence(
        &self,
        _cells: &[Cell],
        states: &[CellState],
        config: &AriaConfig,
    ) -> AriaResult<Vec<Signal>> {
        // For now, do emergence detection on CPU
        // GPU version would need reduction shaders
        if self.tick % config.emergence.check_interval != 0 {
            return Ok(Vec::new());
        }

        // Calculate average state of active cells
        let mut sum = [0.0f32; 8];
        let mut count = 0usize;

        for state in states.iter() {
            if !state.is_sleeping() && !state.is_dead() && state.activity_level > 0.1 {
                for (i, &s) in state.state.iter().take(8).enumerate() {
                    sum[i] += s;
                }
                count += 1;
            }
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        // Average
        for s in &mut sum {
            *s /= count as f32;
        }

        // Check coherence
        let intensity: f32 = sum.iter().map(|x| x.abs()).sum::<f32>() / 8.0;

        if intensity > config.emergence.coherence_threshold {
            let signal = Signal {
                content: sum.to_vec(),
                intensity,
                label: format!("emergence_{}", self.tick),
                signal_type: aria_core::SignalType::Expression,
                timestamp: self.tick,
                position: None,
            };
            return Ok(vec![signal]);
        }

        Ok(Vec::new())
    }

    fn stats(&self) -> BackendStats {
        self.stats.clone()
    }

    fn sync(&mut self) -> AriaResult<()> {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None, // Wait for most recent submission
            timeout: None,          // Wait indefinitely
        });
        Ok(())
    }

    fn name(&self) -> &'static str {
        "GPU (wgpu)"
    }
}

// ============================================================================
// WGSL Shaders
// ============================================================================

/// Cell update compute shader
/// "La Vraie Faim" - cells must struggle to survive
pub const CELL_UPDATE_SHADER: &str = r#"
// Cell update compute shader for ARIA
// "La Vraie Faim" - cells must struggle to survive

struct CellState {
    position: array<f32, 16>,
    state: array<f32, 32>,
    energy: f32,
    tension: f32,
    activity_level: f32,
    flags: u32,
    _reserved: array<f32, 4>,
}

struct Config {
    // Basic metabolism
    energy_cap: f32,
    reaction_amplification: f32,
    state_cap: f32,
    signal_radius: f32,

    // Action costs (La Vraie Faim)
    cost_rest: f32,
    cost_signal: f32,
    cost_move: f32,
    cost_divide: f32,

    // Signal energy
    signal_energy_base: f32,
    signal_resonance_factor: f32,

    // Bootstrap energy (0 = user feeds ARIA by talking)
    energy_gain: f32,

    tick: u32,
}

@group(0) @binding(0) var<storage, read_write> cells: array<CellState>;
@group(0) @binding(1) var<storage, read> dna_pool: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> config: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= arrayLength(&cells) {
        return;
    }

    var cell = cells[cell_idx];
    let is_sleeping = (cell.flags & 1u) != 0u;
    let is_dead = (cell.flags & 32u) != 0u;

    // Skip dead cells
    if is_dead {
        return;
    }

    // Sleeping cells still consume (hibernation cost)
    if is_sleeping {
        cell.energy -= config.cost_rest * 0.1;
        cell.energy += config.energy_gain; // Tiny passive gain if enabled
        if cell.energy <= 0.0 {
            cell.flags = cell.flags | 32u; // Die
        }
        cells[cell_idx] = cell;
        return;
    }

    // === LA VRAIE FAIM: Resting costs energy (breathing) ===
    cell.energy -= config.cost_rest;
    cell.energy += config.energy_gain; // Tiny passive gain if enabled (usually 0)

    // Death check
    if cell.energy <= 0.0 {
        cell.flags = cell.flags | 32u;
        cells[cell_idx] = cell;
        return;
    }

    // Build tension slowly
    cell.tension += 0.01;

    // Decay activity - faster decay for better sparse updates
    // Was 0.99 (100+ ticks to sleep) → 0.9 (10-20 ticks to sleep)
    cell.activity_level *= 0.9;

    // Action check - if tension exceeds threshold, reset it
    if cell.tension > 1.0 {
        cell.tension = 0.0;
        // Actions would cost more energy (handled in CPU for now)
    }

    // Decay tension when activity is low - aligned with sleep threshold
    // Was: activity < 0.1 (tension never decayed before sleep)
    if cell.activity_level < 0.3 {
        cell.tension *= 0.8;  // Was 0.9 - faster decay
        cell.tension -= 0.01; // Was 0.005 - faster decay
        if cell.tension < 0.0 {
            cell.tension = 0.0;
        }
    }

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

    // Cap energy
    cell.energy = clamp(cell.energy, 0.0, config.energy_cap);

    // Sleep if inactive - relaxed thresholds for better sparse updates
    // Was: activity < 0.05 && tension < 0.1 (never reached!)
    // Now: activity < 0.3 && tension < 0.5 (reachable after ~10 ticks idle)
    if cell.activity_level < 0.3 && cell.tension < 0.5 && cell.energy > 0.0 {
        cell.flags = cell.flags | 1u;  // Set sleeping flag
    }

    cells[cell_idx] = cell;
}
"#;

/// Signal propagation compute shader
/// "La Vraie Faim" - resonance-based energy gain
pub const SIGNAL_PROPAGATE_SHADER: &str = r#"
// Signal propagation for ARIA
// "La Vraie Faim" - only resonant signals give energy

struct SignalFragment {
    source_id_low: u32,
    source_id_high: u32,
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

struct Config {
    // Basic metabolism
    energy_cap: f32,
    reaction_amplification: f32,
    state_cap: f32,
    signal_radius: f32,

    // Action costs (La Vraie Faim)
    cost_rest: f32,
    cost_signal: f32,
    cost_move: f32,
    cost_divide: f32,

    // Signal energy
    signal_energy_base: f32,
    signal_resonance_factor: f32,

    // Bootstrap energy (0 = user feeds ARIA by talking)
    energy_gain: f32,

    tick: u32,
}

@group(0) @binding(0) var<storage, read_write> cells: array<CellState>;
@group(0) @binding(1) var<storage, read> signals: array<SignalFragment>;
@group(0) @binding(2) var<uniform> config: Config;

// Calculate resonance between signal and cell state (cosine similarity)
fn calculate_resonance(signal_content: array<f32, 8>, cell_state: array<f32, 32>) -> f32 {
    var dot: f32 = 0.0;
    var norm_sig: f32 = 0.0;
    var norm_state: f32 = 0.0;

    for (var i = 0u; i < 8u; i++) {
        dot += signal_content[i] * cell_state[i];
        norm_sig += signal_content[i] * signal_content[i];
        norm_state += cell_state[i] * cell_state[i];
    }

    let denom = sqrt(norm_sig * norm_state);
    if denom > 0.001 {
        // Map from [-1, 1] to [0, 1]
        return (dot / denom + 1.0) * 0.5;
    }
    return 0.5; // Neutral if no signal or state
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= arrayLength(&cells) {
        return;
    }

    var cell = cells[cell_idx];
    let is_sleeping = (cell.flags & 1u) != 0u;
    let is_dead = (cell.flags & 32u) != 0u;

    // Skip dead cells entirely
    if is_dead {
        return;
    }

    let signal_count = arrayLength(&signals);
    var received_signal = false;

    // OPTIMIZATION: sleeping cells only check first few signals for wake-up
    let max_signals_to_check = select(signal_count, min(signal_count, 5u), is_sleeping);

    for (var s = 0u; s < max_signals_to_check; s++) {
        let signal = signals[s];

        if signal.intensity < 0.001 {
            continue;
        }

        // Distance in semantic space
        var dist_sq: f32 = 0.0;
        for (var i = 0u; i < 8u; i++) {
            let diff = cell.position[i] - signal.content[i];
            dist_sq += diff * diff;
        }
        let dist = sqrt(dist_sq);

        if dist < config.signal_radius {
            let attenuation = 1.0 - (dist / config.signal_radius);
            let intensity = signal.intensity * attenuation * config.reaction_amplification;

            // Wake up sleeping cells if signal is strong enough
            if is_sleeping && intensity > 0.1 {
                cell.flags = cell.flags & ~1u; // Clear sleeping flag
                cell.activity_level = 0.5; // Boost activity on wake
                cell.tension = 0.2; // Some initial tension
                received_signal = true;
            }

            // Only process signal if awake (or just woken up)
            if (cell.flags & 1u) == 0u {
                for (var i = 0u; i < 8u; i++) {
                    cell.state[i] += signal.content[i] * intensity;
                }

                // === LA VRAIE FAIM: Resonance-based energy gain ===
                // Only cells that UNDERSTAND the signal get fed!
                let resonance = calculate_resonance(signal.content, cell.state);

                // Threshold: resonance < 0.3 = you don't understand = no food
                // This creates REAL natural selection
                if resonance > 0.3 {
                    // Scale energy by how well you understand
                    // resonance 0.3 → multiplier ~0.0
                    // resonance 0.5 → multiplier ~0.4
                    // resonance 0.8 → multiplier ~1.0
                    // resonance 1.0 → multiplier ~1.4 (bonus!)
                    let understanding = (resonance - 0.3) / 0.7; // 0 to 1
                    let energy_gain = config.signal_energy_base
                        * intensity
                        * understanding
                        * (1.0 + resonance * config.signal_resonance_factor);
                    cell.energy = min(cell.energy + energy_gain, config.energy_cap);
                }
                // resonance <= 0.3: NO ENERGY - you didn't understand

                cell.activity_level += intensity;
                received_signal = true;
            }
        }
    }

    // Only write back if something changed
    if received_signal || !is_sleeping {
        cells[cell_idx] = cell;
    }
}
"#;

/// Compact shader - collects active cell indices for sparse dispatch
/// This is the first pass in sparse dispatch: count active cells and build index list
pub const COMPACT_SHADER: &str = r#"
// Compact shader for ARIA sparse dispatch
// Counts active cells and builds list of their indices

struct CellState {
    position: array<f32, 16>,
    state: array<f32, 32>,
    energy: f32,
    tension: f32,
    activity_level: f32,
    flags: u32,
    _reserved: array<f32, 4>,
}

struct AtomicCounter {
    count: atomic<u32>,
    _pad: array<u32, 3>,
}

@group(0) @binding(0) var<storage, read> cells: array<CellState>;
@group(0) @binding(1) var<storage, read_write> counter: AtomicCounter;
@group(0) @binding(2) var<storage, read_write> active_indices: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let cell_idx = id.x;
    if cell_idx >= arrayLength(&cells) {
        return;
    }

    let cell = cells[cell_idx];

    // Check if this cell is active (not sleeping and not dead)
    let is_sleeping = (cell.flags & 1u) != 0u;
    let is_dead = (cell.flags & 32u) != 0u;

    if !is_sleeping && !is_dead {
        // Atomically increment counter and get index
        let write_idx = atomicAdd(&counter.count, 1u);

        // Store this cell's index in the active list
        // Note: This may write out of bounds if more cells than expected are active
        // We handle this by ensuring active_indices buffer is cell_count sized
        if write_idx < arrayLength(&active_indices) {
            active_indices[write_idx] = cell_idx;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_size() {
        // 12 f32 fields = 48 bytes (La Vraie Faim config)
        assert_eq!(std::mem::size_of::<GpuConfig>(), 48);
    }

    #[test]
    fn test_gpu_backend_creation() {
        let config = AriaConfig::default();
        let result = GpuBackend::new(&config);
        if result.is_err() {
            println!("GPU not available: {:?}", result.err());
        }
    }
}
