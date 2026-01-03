//! # GPU Backend with SoA (Structure of Arrays)
//!
//! Optimized GPU backend for 5M+ cells using separate buffers.
//!
//! ## Key Differences from gpu.rs
//!
//! 1. **Separate Buffers**: Energy, Position, State, Flags are in different GPU buffers
//! 2. **Indirect Dispatch**: GPU computes workgroup count, no CPU roundtrip
//! 3. **Hysteresis Sleep**: Schmitt trigger prevents oscillation
//! 4. **Spatial Hashing**: O(1) neighbor lookup instead of O(N²)
//!
//! ## Performance Gains
//!
//! - +40% FPS from better memory coalescing
//! - No CPU-GPU sync for dispatch count
//! - Better cache utilization
//! - 9000x reduction in signal propagation calculations

use std::sync::Arc;

use aria_core::cell::{Cell, CellAction, CellState};
use aria_core::config::AriaConfig;
use aria_core::dna::DNA;
use aria_core::error::{AriaError, AriaResult};
use aria_core::signal::{Signal, SignalFragment};
use aria_core::soa::{
    CellConnections, CellEnergy, CellMetadata, CellInternalState, CellPosition, IndirectDispatchArgs,
    CellPrediction,
};
use aria_core::traits::{BackendStats, ComputeBackend};
use bytemuck::{Pod, Zeroable};

use crate::spatial_gpu::{self, SpatialHashConfig, GRID_SIZE, TOTAL_REGIONS};

/// GPU-compatible config (matches shader struct)
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct GpuConfig {
    energy_cap: f32,
    reaction_amplification: f32,
    state_cap: f32,
    signal_radius: f32,
    cost_rest: f32,
    cost_signal: f32,
    cost_move: f32,
    cost_divide: f32,
    signal_energy_base: f32,
    signal_resonance_factor: f32,
    energy_gain: f32,
    tick: u32,
    cell_count: u32,
    workgroup_size: u32,
    _pad: [u32; 2],
}

impl GpuConfig {
    fn from_config(config: &AriaConfig, tick: u64, cell_count: usize) -> Self {
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
struct AtomicCounter {
    count: u32,
    _pad: [u32; 3],
}

/// GPU compute backend with SoA layout
pub struct GpuSoABackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: AriaConfig,
    stats: BackendStats,
    tick: u64,

    // SoA Buffers
    energy_buffer: Option<wgpu::Buffer>,
    position_buffer: Option<wgpu::Buffer>,
    state_buffer: Option<wgpu::Buffer>,
    metadata_buffer: Option<wgpu::Buffer>,

    // Other buffers
    dna_buffer: Option<wgpu::Buffer>,
    dna_indices_buffer: Option<wgpu::Buffer>,
    signals_buffer: Option<wgpu::Buffer>,
    config_buffer: Option<wgpu::Buffer>,

    // Sparse dispatch buffers (merged into one to save storage bindings)
    sparse_dispatch_buffer: Option<wgpu::Buffer>,
    indirect_buffer: Option<wgpu::Buffer>,

    // Spatial hash buffers
    grid_buffer: Option<wgpu::Buffer>,
    spatial_config_buffer: Option<wgpu::Buffer>,

    // Hebbian learning buffer
    connection_buffer: Option<wgpu::Buffer>,

    // Prediction Law buffer
    prediction_buffer: Option<wgpu::Buffer>,

    // Hebbian Spatial Attraction buffer (fixed-point centroid accumulation)
    centroid_buffer: Option<wgpu::Buffer>,

    // Cluster Hysteresis buffer (256 clusters × (activity_sum + count))
    cluster_stats_buffer: Option<wgpu::Buffer>,

    // Staging buffers for readback
    energy_staging: Option<wgpu::Buffer>,
    metadata_staging: Option<wgpu::Buffer>,
    counter_staging: Option<wgpu::Buffer>,

    // Pipelines
    cell_update_pipeline: Option<wgpu::ComputePipeline>,
    cell_update_sparse_pipeline: Option<wgpu::ComputePipeline>,
    signal_pipeline: Option<wgpu::ComputePipeline>,
    signal_with_hash_pipeline: Option<wgpu::ComputePipeline>,
    compact_pipeline: Option<wgpu::ComputePipeline>,
    prepare_dispatch_pipeline: Option<wgpu::ComputePipeline>,
    clear_grid_pipeline: Option<wgpu::ComputePipeline>,
    build_grid_pipeline: Option<wgpu::ComputePipeline>,
    hebbian_pipeline: Option<wgpu::ComputePipeline>,
    sleeping_drain_pipeline: Option<wgpu::ComputePipeline>,

    // Prediction Law pipelines
    prediction_generate_pipeline: Option<wgpu::ComputePipeline>,
    prediction_evaluate_pipeline: Option<wgpu::ComputePipeline>,

    // Hebbian Spatial Attraction pipelines
    hebbian_centroid_pipeline: Option<wgpu::ComputePipeline>,
    hebbian_attraction_pipeline: Option<wgpu::ComputePipeline>,

    // Cluster Hysteresis pipelines
    cluster_stats_pipeline: Option<wgpu::ComputePipeline>,
    cluster_hysteresis_pipeline: Option<wgpu::ComputePipeline>,

    // Bind group layouts
    main_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sparse_cell_bind_group_layout: Option<wgpu::BindGroupLayout>,
    signal_with_hash_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sparse_bind_group_layout: Option<wgpu::BindGroupLayout>,
    grid_bind_group_layout: Option<wgpu::BindGroupLayout>,
    hebbian_bind_group_layout: Option<wgpu::BindGroupLayout>,
    sleeping_drain_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Prediction Law bind group layouts
    prediction_generate_bind_group_layout: Option<wgpu::BindGroupLayout>,
    prediction_evaluate_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Hebbian Spatial Attraction bind group layouts
    hebbian_centroid_bind_group_layout: Option<wgpu::BindGroupLayout>,
    hebbian_attraction_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Cluster Hysteresis bind group layouts
    cluster_stats_bind_group_layout: Option<wgpu::BindGroupLayout>,
    cluster_hysteresis_bind_group_layout: Option<wgpu::BindGroupLayout>,

    // Sparse dispatch state
    use_indirect_dispatch: bool,
    last_active_count: u32,

    cell_count: usize,
    max_cell_count: usize,
    max_buffer_size: usize, // GPU's actual buffer limit (queried at init)
    initialized: bool,
    use_spatial_hash: bool,
    compiler: crate::compiler::ShaderCompiler,
}

impl GpuSoABackend {
    /// Create a new GPU SoA backend
    pub fn new(config: &AriaConfig) -> AriaResult<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| AriaError::gpu(format!("No suitable GPU adapter found: {}", e)))?;

        let info = adapter.get_info();
        tracing::info!(
            "🎮 GPU SoA: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        // Query the adapter's actual limits
        let adapter_limits = adapter.limits();
        // Cap at 1GB to avoid u32 overflow issues with max_storage_buffer_binding_size
        let gpu_max_buffer = (adapter_limits.max_buffer_size as usize).min(1024 * 1024 * 1024);
        tracing::info!("🎮 GPU max buffer size: {} MB", gpu_max_buffer / 1024 / 1024);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ARIA GPU SoA"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: gpu_max_buffer as u32,
                    max_buffer_size: gpu_max_buffer as u64,
                    // Request actual hardware limits for storage buffers (fixes crash on limited hardware)
                    max_storage_buffers_per_shader_stage: adapter_limits.max_storage_buffers_per_shader_stage,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .map_err(|e| AriaError::gpu(format!("Failed to create device: {}", e)))?;

        // Enable spatial hash for large populations (>100k cells)
        let use_spatial_hash = config.population.target_population > 100_000;
        if use_spatial_hash {
            tracing::info!("🎮 Spatial hashing enabled for {}+ cells", config.population.target_population);
        }

        // Enable indirect dispatch for larger populations
        let use_indirect_dispatch = config.population.target_population > 50_000;
        if use_indirect_dispatch {
            tracing::info!("🎮 Indirect dispatch enabled for sparse cell updates");
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            config: config.clone(),
            stats: BackendStats::default(),
            tick: 0,
            energy_buffer: None,
            position_buffer: None,
            state_buffer: None,
            metadata_buffer: None,
            dna_buffer: None,
            signals_buffer: None,
            config_buffer: None,
            sparse_dispatch_buffer: None,
            indirect_buffer: None,
            grid_buffer: None,
            dna_indices_buffer: None,
            spatial_config_buffer: None,
            energy_staging: None,
            metadata_staging: None,
            counter_staging: None,
            cell_update_pipeline: None,
            cell_update_sparse_pipeline: None,
            signal_pipeline: None,
            signal_with_hash_pipeline: None,
            compact_pipeline: None,
            prepare_dispatch_pipeline: None,
            clear_grid_pipeline: None,
            build_grid_pipeline: None,
            hebbian_pipeline: None,
            sleeping_drain_pipeline: None,
            prediction_generate_pipeline: None,
            prediction_evaluate_pipeline: None,
            hebbian_centroid_pipeline: None,
            hebbian_attraction_pipeline: None,
            cluster_stats_pipeline: None,
            cluster_hysteresis_pipeline: None,
            main_bind_group_layout: None,
            sparse_cell_bind_group_layout: None,
            signal_with_hash_bind_group_layout: None,
            sparse_bind_group_layout: None,
            grid_bind_group_layout: None,
            hebbian_bind_group_layout: None,
            sleeping_drain_bind_group_layout: None,
            prediction_generate_bind_group_layout: None,
            prediction_evaluate_bind_group_layout: None,
            hebbian_centroid_bind_group_layout: None,
            hebbian_attraction_bind_group_layout: None,
            cluster_stats_bind_group_layout: None,
            cluster_hysteresis_bind_group_layout: None,
            connection_buffer: None,
            prediction_buffer: None,
            centroid_buffer: None,
            cluster_stats_buffer: None,
            use_indirect_dispatch,
            last_active_count: 0,
            cell_count: 0,
            max_cell_count: 0,
            max_buffer_size: gpu_max_buffer,
            initialized: false,
            use_spatial_hash,
            compiler: crate::compiler::ShaderCompiler::new(),
        })
    }

    /// Initialize GPU buffers with SoA layout
    fn init_buffers(&mut self, cell_count: usize, dna_count: usize) -> AriaResult<()> {
        // Use GPU's actual max buffer size (queried at init)
        // CellConnections is the LARGEST buffer at 144 bytes/cell
        // (16 targets × u32 + 16 strengths × f32 + count + padding)
        let connections_size = std::mem::size_of::<CellConnections>(); // 144 bytes
        let max_cells_in_buffer = self.max_buffer_size / connections_size;

        // Calculate headroom: aim for 100% but cap at GPU limit
        let desired_headroom = cell_count * 2;
        let cell_count_with_headroom = desired_headroom.min(max_cells_in_buffer);
        let dna_count_with_headroom = (dna_count * 2).min(max_cells_in_buffer);

        if cell_count_with_headroom < desired_headroom {
            tracing::warn!("⚠️ GPU buffer limit: headroom reduced from {}M to {}M cells",
                desired_headroom / 1_000_000, cell_count_with_headroom / 1_000_000);
        }

        self.max_cell_count = cell_count_with_headroom;

        // Calculate buffer sizes
        let energy_bytes = std::mem::size_of::<CellEnergy>() * cell_count_with_headroom;
        let position_bytes = std::mem::size_of::<CellPosition>() * cell_count_with_headroom;
        let state_bytes = std::mem::size_of::<CellInternalState>() * cell_count_with_headroom;
        let metadata_bytes = std::mem::size_of::<CellMetadata>() * cell_count_with_headroom;
        let dna_bytes = std::mem::size_of::<DNA>() * dna_count_with_headroom;
        let signals_bytes = std::mem::size_of::<SignalFragment>() * 1024;
        let indices_bytes = std::mem::size_of::<u32>() * cell_count_with_headroom;
        let counter_bytes = 16; // Atomic counter + padding
        let indirect_bytes = std::mem::size_of::<IndirectDispatchArgs>();
        let dna_indices_bytes = std::mem::size_of::<u32>() * cell_count_with_headroom;

        let sparse_bytes = counter_bytes + indices_bytes;

        let total_bytes = energy_bytes
            + position_bytes
            + state_bytes
            + metadata_bytes
            + dna_bytes
            + signals_bytes
            + sparse_bytes
            + indirect_bytes
            + dna_indices_bytes;

        tracing::info!(
            "🎮 GPU SoA: Allocating {} MB for {} cells (SoA layout)",
            total_bytes / 1024 / 1024,
            cell_count
        );

        // Create SoA buffers
        self.energy_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energy Buffer"),
            size: energy_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.position_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Position Buffer"),
            size: position_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.state_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("State Buffer"),
            size: state_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.metadata_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Metadata Buffer"),
            size: metadata_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.dna_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DNA Pool"),
            size: dna_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.signals_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Signals"),
            size: signals_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.config_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config"),
            size: std::mem::size_of::<GpuConfig>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.dna_indices_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DNA Indices"),
            size: dna_indices_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Sparse dispatch buffer (contains BOTH counter and indices)
        self.sparse_dispatch_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Dispatch Buffer"),
            size: sparse_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Indirect dispatch buffer - GPU writes workgroup count here
        self.indirect_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Dispatch"),
            size: indirect_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Staging buffers for readback
        self.energy_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energy Staging"),
            size: energy_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.metadata_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Metadata Staging"),
            size: metadata_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.counter_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Counter Staging"),
            size: counter_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Spatial hash buffers (only if enabled)
        if self.use_spatial_hash {
            let grid_bytes = spatial_gpu::grid_buffer_size();
            let spatial_config_bytes = std::mem::size_of::<SpatialHashConfig>();

            tracing::info!(
                "🎮 Spatial hash: Allocating {} MB for {}³ grid",
                grid_bytes / 1024 / 1024,
                GRID_SIZE
            );

            self.grid_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Spatial Grid"),
                size: grid_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.spatial_config_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Spatial Config"),
                size: spatial_config_bytes as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // Upload initial spatial config
            let spatial_config = SpatialHashConfig::default_aria(cell_count);
            self.queue.write_buffer(
                self.spatial_config_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&spatial_config),
            );
        }

        // Hebbian connection buffer
        // 144 bytes per cell (16 targets + 16 strengths + count + padding)
        let connection_bytes = std::mem::size_of::<CellConnections>() * cell_count_with_headroom;
        tracing::info!(
            "🧠 Hebbian: Allocating {} MB for {} cell connections",
            connection_bytes / 1024 / 1024,
            cell_count
        );

        self.connection_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hebbian Connections"),
            size: connection_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Initialize connections to zero
        let empty_connections = vec![CellConnections::default(); cell_count_with_headroom];
        self.queue.write_buffer(
            self.connection_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&empty_connections),
        );

        // Prediction Law buffer
        // 48 bytes per cell (predicted_state[8] + confidence + last_error + cumulative_score + _pad)
        let prediction_bytes = std::mem::size_of::<CellPrediction>() * cell_count_with_headroom;
        tracing::info!(
            "🔮 Prediction Law: Allocating {} MB for {} cell predictions",
            prediction_bytes / 1024 / 1024,
            cell_count
        );

        self.prediction_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Predictions"),
            size: prediction_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Initialize predictions to default
        let empty_predictions = vec![CellPrediction::default(); cell_count_with_headroom];
        self.queue.write_buffer(
            self.prediction_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&empty_predictions),
        );

        // Centroid buffer for Hebbian spatial attraction
        // Structure: 16 × i32 (weighted_pos) + u32 (total_mass) + u32 (count) + 2 × u32 (padding)
        // Total: 80 bytes
        let centroid_bytes = 80;
        self.centroid_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hebbian Centroid"),
            size: centroid_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Cluster stats buffer for cluster hysteresis
        // Structure: 256 × u32 (activity_sum) + 256 × u32 (count)
        // Total: 2048 bytes
        let cluster_stats_bytes = 256 * 4 * 2;
        self.cluster_stats_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cluster Stats"),
            size: cluster_stats_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.create_bind_group_layouts()?;
        self.create_pipelines()?;

        self.cell_count = cell_count;
        self.initialized = true;

        tracing::info!("🎮 GPU SoA initialized with {} cells", cell_count);
        Ok(())
    }

    fn create_bind_group_layouts(&mut self) -> AriaResult<()> {
        // Main bind group layout for SoA buffers
        self.main_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("SoA Main Bind Group Layout"),
                entries: &[
                    // Energy (read-write)
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
                    // Position (read-only)
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
                    // State (read-write)
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
                    // Flags (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // DNA (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Signals (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 7: DNA Indices (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Sparse cell update bind group layout (main + active_indices + counter)
        self.sparse_cell_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Sparse Cell Bind Group Layout"),
                entries: &[
                    // 0: Energy (read-write)
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
                    // 1: Position (read-only)
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
                    // 2: State (read-write)
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
                    // 3: Flags (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: DNA (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5: Signals (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 7: Sparse Dispatch (read-only counter + indices)
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 8: DNA Indices (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        // Signal with spatial hash bind group layout (8 bindings)
        if self.use_spatial_hash {
            self.signal_with_hash_bind_group_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Signal With Hash Bind Group Layout"),
                    entries: &[
                        // 0: energies (read-write)
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
                        // 1: positions (read-only)
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
                        // 2: states (read-write)
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
                        // 3: flags (read-write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 4: signals (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 5: grid (read-only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 6: config (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 7: spatial_config (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 8: connections (read-only) - Hebbian signal propagation
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 9: dna_pool (read-only) - Selective Attention (Axe 3)
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // 10: dna_indices (read-only) - Selective Attention (Axe 3)
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            ));
        }

        // Sparse bind group layout (Flags + Sparse Dispatch + Indirect + Config)
        self.sparse_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("SoA Sparse Bind Group Layout"),
                entries: &[
                    // 0: Flags (read-only for compact)
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
                    // 1: Sparse Dispatch (counter + indices)
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
                    // 2: Indirect dispatch (output)
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
                    // 3: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        // Grid bind group layout (for spatial hashing)
        if self.use_spatial_hash {
            self.grid_bind_group_layout = Some(self.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Grid Bind Group Layout"),
                    entries: &[
                        // Positions (read-only)
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
                        // Grid (read-write)
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
                        // Spatial config (uniform)
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
        }

        // Hebbian bind group layout (for "fire together, wire together")
        // Uses: energies, metadata, connections, config
        self.hebbian_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Hebbian Bind Group Layout"),
                entries: &[
                    // 0: Energies (read) - to check activity_level
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
                    // 1: Metadata (read) - to check sleeping flag
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
                    // 2: Connections (read-write) - Hebbian learning happens here
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
                    // 3: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        // Sleeping drain bind group layout (simple: energies, flags, config)
        self.sleeping_drain_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Sleeping Drain Bind Group Layout"),
                entries: &[
                    // 0: Energies (read-write)
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
                    // 1: Flags (read-write)
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
                    // 2: Config (uniform)
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

        // Prediction Generate bind group layout
        // Used to generate predictions based on connections
        self.prediction_generate_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Prediction Generate Bind Group Layout"),
                entries: &[
                    // 0: States (read)
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
                    // 1: Metadata (read)
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
                    // 2: Connections (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: Predictions (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        // Prediction Evaluate bind group layout
        // Used to evaluate predictions and apply energy rewards/penalties
        self.prediction_evaluate_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Prediction Evaluate Bind Group Layout"),
                entries: &[
                    // 0: States (read)
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
                    // 1: Metadata (read)
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
                    // 2: Predictions (read-write)
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
                    // 3: Energies (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        // Hebbian Centroid bind group layout
        // Bindings: energies (read), positions (read), metadata (read), centroid (read-write), cell_count (uniform)
        self.hebbian_centroid_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Hebbian Centroid Bind Group Layout"),
                entries: &[
                    // 0: Energies (read)
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
                    // 1: Positions (read)
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
                    // 2: Metadata (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: Centroid (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: Cell count (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        // Hebbian Attraction bind group layout
        // Bindings: positions (read-write), energies (read), metadata (read), centroid (read), config (uniform)
        self.hebbian_attraction_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Hebbian Attraction Bind Group Layout"),
                entries: &[
                    // 0: Positions (read-write)
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
                    // 1: Energies (read)
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
                    // 2: Metadata (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 3: Centroid (read)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 4: Config (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        // Cluster Stats bind group layout
        // Bindings: metadata (read), energies (read), cluster_stats (read-write), cell_count (uniform)
        self.cluster_stats_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Cluster Stats Bind Group Layout"),
                entries: &[
                    // 0: Metadata (read)
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
                    // 1: Energies (read)
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
                    // 2: Cluster Stats (read-write)
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
                    // 3: Cell count (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        // Cluster Hysteresis bind group layout
        // Bindings: metadata (read-write), cluster_stats (read), cell_count (uniform)
        self.cluster_hysteresis_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Cluster Hysteresis Bind Group Layout"),
                entries: &[
                    // 0: Metadata (read-write)
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
                    // 1: Cluster Stats (read)
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
                    // 2: Cell count (uniform)
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

        Ok(())
    }

    fn create_pipelines(&mut self) -> AriaResult<()> {
        // Initial compilation of dynamic pipelines (using default checksum 0)
        self.recompile_dynamic_pipelines(0)?;

        let sparse_layout = self
            .sparse_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse bind group layout not created"))?;

        // Compact shader
        let compact_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SoA Compact Shader"),
            source: wgpu::ShaderSource::Wgsl(self.compiler.get_compact_shader().into()),
        });

        let sparse_pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SoA Sparse Pipeline Layout"),
                bind_group_layouts: &[sparse_layout],
                immediate_size: 0,
            });

        self.compact_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("SoA Compact Pipeline"),
                    layout: Some(&sparse_pipeline_layout),
                    module: &compact_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        // Prepare indirect dispatch shader
        let prepare_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Prepare Dispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(self.compiler.get_prepare_dispatch_shader().into()),
        });

        self.prepare_dispatch_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Prepare Dispatch Pipeline"),
                    layout: Some(&sparse_pipeline_layout),
                    module: &prepare_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        // Spatial hash pipelines (only if enabled)
        if self.use_spatial_hash {
            let grid_layout = self
                .grid_bind_group_layout
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Grid bind group layout not created"))?;

            let grid_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Grid Pipeline Layout"),
                    bind_group_layouts: &[grid_layout],
                    immediate_size: 0,
                });

            // Clear grid shader
            let clear_shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Clear Grid Shader"),
                    source: wgpu::ShaderSource::Wgsl(spatial_gpu::CLEAR_GRID_SHADER.into()),
                });

            self.clear_grid_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Clear Grid Pipeline"),
                        layout: Some(&grid_pipeline_layout),
                        module: &clear_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            // Build grid shader
            let build_shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Build Grid Shader"),
                    source: wgpu::ShaderSource::Wgsl(spatial_gpu::BUILD_GRID_SHADER.into()),
                });

            self.build_grid_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Build Grid Pipeline"),
                        layout: Some(&grid_pipeline_layout),
                        module: &build_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            // Signal with spatial hash pipeline
            let signal_hash_layout = self
                .signal_with_hash_bind_group_layout
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Signal with hash bind group layout not created"))?;

            let signal_hash_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Signal With Hash Pipeline Layout"),
                    bind_group_layouts: &[signal_hash_layout],
                    immediate_size: 0,
                });

            let signal_hash_shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Signal With Hash Shader"),
                    source: wgpu::ShaderSource::Wgsl(spatial_gpu::SIGNAL_WITH_SPATIAL_HASH_SHADER.into()),
                });

            self.signal_with_hash_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Signal With Hash Pipeline"),
                        layout: Some(&signal_hash_pipeline_layout),
                        module: &signal_hash_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::debug!("🎮 Spatial hash pipelines created");

            // Hebbian learning pipeline (requires spatial hash)
            let hebbian_layout = self
                .hebbian_bind_group_layout
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Hebbian bind group layout not created"))?;

            let hebbian_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Hebbian Pipeline Layout"),
                    bind_group_layouts: &[hebbian_layout],
                    immediate_size: 0,
                });

            let hebbian_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hebbian Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_hebbian_shader().into()),
            });

            self.hebbian_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Hebbian Pipeline"),
                        layout: Some(&hebbian_pipeline_layout),
                        module: &hebbian_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::info!("🧠 Hebbian learning pipeline created");
        }

        // Sleeping drain pipeline (runs periodically to drain sleeping cells)
        if let Some(sleeping_layout) = self.sleeping_drain_bind_group_layout.as_ref() {
            let sleeping_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sleeping Drain Pipeline Layout"),
                    bind_group_layouts: &[sleeping_layout],
                    immediate_size: 0,
                });

            let sleeping_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sleeping Drain Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_sleeping_drain_shader().into()),
            });

            self.sleeping_drain_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Sleeping Drain Pipeline"),
                        layout: Some(&sleeping_pipeline_layout),
                        module: &sleeping_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::info!("💤 Sleeping drain pipeline created");
        }

        // Prediction Law pipelines (always created)
        if let Some(pred_gen_layout) = self.prediction_generate_bind_group_layout.as_ref() {
            let pred_gen_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Prediction Generate Pipeline Layout"),
                    bind_group_layouts: &[pred_gen_layout],
                    immediate_size: 0,
                });

            let pred_gen_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Prediction Generate Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_prediction_generate_shader().into()),
            });

            self.prediction_generate_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Prediction Generate Pipeline"),
                        layout: Some(&pred_gen_pipeline_layout),
                        module: &pred_gen_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );
        }

        if let Some(pred_eval_layout) = self.prediction_evaluate_bind_group_layout.as_ref() {
            let pred_eval_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Prediction Evaluate Pipeline Layout"),
                    bind_group_layouts: &[pred_eval_layout],
                    immediate_size: 0,
                });

            let pred_eval_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Prediction Evaluate Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_prediction_evaluate_shader().into()),
            });

            self.prediction_evaluate_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Prediction Evaluate Pipeline"),
                        layout: Some(&pred_eval_pipeline_layout),
                        module: &pred_eval_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::info!("🔮 Prediction Law pipelines created");
        }

        // Create Hebbian Spatial Attraction pipelines
        if let Some(centroid_layout) = self.hebbian_centroid_bind_group_layout.as_ref() {
            let centroid_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Hebbian Centroid Pipeline Layout"),
                    bind_group_layouts: &[centroid_layout],
                    immediate_size: 0,
                });

            let centroid_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hebbian Centroid Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_hebbian_centroid_shader().into()),
            });

            self.hebbian_centroid_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Hebbian Centroid Pipeline"),
                        layout: Some(&centroid_pipeline_layout),
                        module: &centroid_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );
        }

        if let Some(attraction_layout) = self.hebbian_attraction_bind_group_layout.as_ref() {
            let attraction_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Hebbian Attraction Pipeline Layout"),
                    bind_group_layouts: &[attraction_layout],
                    immediate_size: 0,
                });

            let attraction_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hebbian Attraction Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_hebbian_attraction_shader().into()),
            });

            self.hebbian_attraction_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Hebbian Attraction Pipeline"),
                        layout: Some(&attraction_pipeline_layout),
                        module: &attraction_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::info!("🧲 Hebbian Spatial Attraction pipelines created");
        }

        // Create Cluster Hysteresis pipelines
        if let Some(stats_layout) = self.cluster_stats_bind_group_layout.as_ref() {
            let stats_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cluster Stats Pipeline Layout"),
                    bind_group_layouts: &[stats_layout],
                    immediate_size: 0,
                });

            let stats_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Cluster Stats Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_cluster_stats_shader().into()),
            });

            self.cluster_stats_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Cluster Stats Pipeline"),
                        layout: Some(&stats_pipeline_layout),
                        module: &stats_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );
        }

        if let Some(hysteresis_layout) = self.cluster_hysteresis_bind_group_layout.as_ref() {
            let hysteresis_pipeline_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cluster Hysteresis Pipeline Layout"),
                    bind_group_layouts: &[hysteresis_layout],
                    immediate_size: 0,
                });

            let hysteresis_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Cluster Hysteresis Shader"),
                source: wgpu::ShaderSource::Wgsl(self.compiler.get_cluster_hysteresis_shader().into()),
            });

            self.cluster_hysteresis_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Cluster Hysteresis Pipeline"),
                        layout: Some(&hysteresis_pipeline_layout),
                        module: &hysteresis_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );

            tracing::info!("📊 Cluster Hysteresis pipelines created");
        }

        tracing::debug!("🎮 GPU SoA pipelines created");
        Ok(())
    }

    /// Upload cell data to SoA buffers
    fn upload_cells(&self, states: &[CellState]) {
        let count = states.len().min(self.max_cell_count);

        // Convert to SoA
        let energies: Vec<CellEnergy> = states[..count]
            .iter()
            .map(|s| CellEnergy {
                energy: s.energy,
                tension: s.tension,
                activity_level: s.activity_level,
                _pad: 0.0,
            })
            .collect();

        let positions: Vec<CellPosition> = states[..count]
            .iter()
            .map(|s| CellPosition {
                position: s.position,
            })
            .collect();

        let internal_states: Vec<CellInternalState> = states[..count]
            .iter()
            .map(|s| CellInternalState { state: s.state })
            .collect();

        let metadata: Vec<CellMetadata> = states[..count]
            .iter()
            .map(|s| CellMetadata {
                flags: s.flags,
                cluster_id: s.cluster_id,
                hysteresis: s.hysteresis,
                _pad: 0,
            })
            .collect();

        if let Some(buf) = &self.energy_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&energies));
        }
        if let Some(buf) = &self.position_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&positions));
        }
        if let Some(buf) = &self.state_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&internal_states));
        }
        if let Some(buf) = &self.metadata_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&metadata));
        }
    }

    /// Upload only NEW cells (from old_count to current len) - Session 32
    /// This avoids O(n) upload when only a few cells were added
    fn upload_new_cells(&self, states: &[CellState], old_count: usize) {
        let new_count = states.len().min(self.max_cell_count);
        if old_count >= new_count {
            return; // Nothing new to upload
        }

        let new_states = &states[old_count..new_count];
        let offset_bytes = |size: usize| (old_count * size) as u64;

        // Convert new cells to SoA
        let energies: Vec<CellEnergy> = new_states
            .iter()
            .map(|s| CellEnergy {
                energy: s.energy,
                tension: s.tension,
                activity_level: s.activity_level,
                _pad: 0.0,
            })
            .collect();

        let positions: Vec<CellPosition> = new_states
            .iter()
            .map(|s| CellPosition { position: s.position })
            .collect();

        let internal_states: Vec<CellInternalState> = new_states
            .iter()
            .map(|s| CellInternalState { state: s.state })
            .collect();

        let metadata: Vec<CellMetadata> = new_states
            .iter()
            .map(|s| CellMetadata {
                flags: s.flags,
                cluster_id: s.cluster_id,
                hysteresis: s.hysteresis,
                _pad: 0,
            })
            .collect();

        if let Some(buf) = &self.energy_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes(std::mem::size_of::<CellEnergy>()),
                bytemuck::cast_slice(&energies),
            );
        }
        if let Some(buf) = &self.position_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes(std::mem::size_of::<CellPosition>()),
                bytemuck::cast_slice(&positions),
            );
        }
        if let Some(buf) = &self.state_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes(std::mem::size_of::<CellInternalState>()),
                bytemuck::cast_slice(&internal_states),
            );
        }
        if let Some(buf) = &self.metadata_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes(std::mem::size_of::<CellMetadata>()),
                bytemuck::cast_slice(&metadata),
            );
        }
    }

    /// Upload only NEW DNA entries - Session 32
    fn upload_new_dna(&self, dna_pool: &[DNA], old_count: usize) {
        if let Some(buffer) = &self.dna_buffer {
            let new_count = dna_pool.len().min(self.max_cell_count);
            if old_count >= new_count {
                return;
            }
            let offset = (old_count * std::mem::size_of::<DNA>()) as u64;
            self.queue.write_buffer(
                buffer,
                offset,
                bytemuck::cast_slice(&dna_pool[old_count..new_count]),
            );
        }
    }

    fn upload_dna(&self, dna_pool: &[DNA]) {
        if let Some(buffer) = &self.dna_buffer {
            let safe_count = dna_pool.len().min(self.max_cell_count);
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&dna_pool[..safe_count]));
        }
    }

    fn upload_signals(&self, signals: &[SignalFragment]) {
        if let Some(buffer) = &self.signals_buffer {
            let mut padded = vec![SignalFragment::zeroed(); 1024];
            for (i, s) in signals.iter().take(1024).enumerate() {
                padded[i] = *s;
            }
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&padded));
        }
    }

    fn upload_config(&self) {
        if let Some(buffer) = &self.config_buffer {
            let gpu_config = GpuConfig::from_config(&self.config, self.tick, self.cell_count);
            self.queue
                .write_buffer(buffer, 0, bytemuck::bytes_of(&gpu_config));
        }
    }

    /// Create bind group for signal propagation with spatial hash
    fn create_signal_with_hash_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .signal_with_hash_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Signal with hash bind group layout not initialized"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Signal With Hash Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.energy_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.position_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.state_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.metadata_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.signals_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.grid_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.config_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.spatial_config_buffer.as_ref().unwrap().as_entire_binding(),
                },
                // 8: connections for Hebbian signal propagation
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.connection_buffer.as_ref().unwrap().as_entire_binding(),
                },
                // 9: dna_pool for Selective Attention
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.dna_buffer.as_ref().unwrap().as_entire_binding(),
                },
                // 10: dna_indices for Selective Attention
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.dna_indices_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for sparse cell update (includes active_indices and counter)
    fn create_sparse_cell_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .sparse_cell_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse cell bind group layout not initialized"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sparse Cell Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.energy_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.position_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.state_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.metadata_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.dna_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.signals_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.config_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.sparse_dispatch_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.dna_indices_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for sparse compact/prepare passes
    fn create_sparse_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .sparse_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse bind group layout not initialized"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SoA Sparse Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.metadata_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sparse_dispatch_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.indirect_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.config_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for spatial hash operations
    fn create_grid_bind_group(&self) -> Option<wgpu::BindGroup> {
        if !self.use_spatial_hash {
            return None;
        }

        let layout = self.grid_bind_group_layout.as_ref()?;
        let position_buffer = self.position_buffer.as_ref()?;
        let grid_buffer = self.grid_buffer.as_ref()?;
        let spatial_config_buffer = self.spatial_config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grid Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spatial_config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for Hebbian learning
    /// "Fire together, wire together" - co-active cells strengthen connections
    fn create_hebbian_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.hebbian_bind_group_layout.as_ref()?;
        let energy_buffer = self.energy_buffer.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let connection_buffer = self.connection_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hebbian Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: energy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: connection_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for sleeping drain pass
    fn create_sleeping_drain_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.sleeping_drain_bind_group_layout.as_ref()?;
        let energy_buffer = self.energy_buffer.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sleeping Drain Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: energy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for prediction evaluate pass
    fn create_prediction_evaluate_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.prediction_evaluate_bind_group_layout.as_ref()?;
        let state_buffer = self.state_buffer.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let prediction_buffer = self.prediction_buffer.as_ref()?;
        let energy_buffer = self.energy_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Prediction Evaluate Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prediction_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: energy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for Hebbian centroid accumulation pass
    fn create_hebbian_centroid_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.hebbian_centroid_bind_group_layout.as_ref()?;
        let energy_buffer = self.energy_buffer.as_ref()?;
        let position_buffer = self.position_buffer.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let centroid_buffer = self.centroid_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hebbian Centroid Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: energy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: centroid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for Hebbian attraction pass
    fn create_hebbian_attraction_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.hebbian_attraction_bind_group_layout.as_ref()?;
        let position_buffer = self.position_buffer.as_ref()?;
        let energy_buffer = self.energy_buffer.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let centroid_buffer = self.centroid_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hebbian Attraction Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: energy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: centroid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for cluster stats accumulation pass
    fn create_cluster_stats_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.cluster_stats_bind_group_layout.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let energy_buffer = self.energy_buffer.as_ref()?;
        let cluster_stats_buffer = self.cluster_stats_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cluster Stats Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: energy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cluster_stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Create bind group for cluster hysteresis pass
    fn create_cluster_hysteresis_bind_group(&self) -> Option<wgpu::BindGroup> {
        let layout = self.cluster_hysteresis_bind_group_layout.as_ref()?;
        let metadata_buffer = self.metadata_buffer.as_ref()?;
        let cluster_stats_buffer = self.cluster_stats_buffer.as_ref()?;
        let config_buffer = self.config_buffer.as_ref()?;

        Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cluster Hysteresis Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cluster_stats_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        }))
    }

    /// Build spatial hash grid (clear + build passes)
    /// Call this once per tick before signal propagation
    fn build_spatial_grid(&self) -> AriaResult<()> {
        if !self.use_spatial_hash {
            return Ok(());
        }

        let grid_bind_group = self
            .create_grid_bind_group()
            .ok_or_else(|| AriaError::gpu("Failed to create grid bind group"))?;

        let clear_pipeline = self
            .clear_grid_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Clear grid pipeline not initialized"))?;

        let build_pipeline = self
            .build_grid_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Build grid pipeline not initialized"))?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Build Spatial Grid"),
            });

        // Pass 1: Clear grid
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(clear_pipeline);
            pass.set_bind_group(0, &grid_bind_group, &[]);
            // Dispatch for all grid regions (64^3 / 256 workgroups)
            let workgroups = (TOTAL_REGIONS as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Build grid (assign cells to regions)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(build_pipeline);
            pass.set_bind_group(0, &grid_bind_group, &[]);
            let workgroups = (self.cell_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    fn reset_counter(&self) {
        if let Some(buffer) = &self.sparse_dispatch_buffer {
            let zero: [u32; 4] = [0; 4]; // Atomic count + padding
            self.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&zero));
        }
    }

    fn read_active_count(&self) -> AriaResult<u32> {
        let sparse_buffer = self
            .sparse_dispatch_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse dispatch buffer not initialized"))?;
        let staging_buffer = self
            .counter_staging
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Counter staging not initialized"))?;

        let size = 16; // Read only the counter part (16 bytes with padding)

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Counter"),
            });
        encoder.copy_buffer_to_buffer(sparse_buffer, 0, staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

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

    fn create_main_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .main_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Main bind group layout not initialized"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SoA Main Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .energy_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .position_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.state_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.metadata_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.dna_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.signals_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.config_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.dna_indices_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }))
    }
}

impl ComputeBackend for GpuSoABackend {
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

        // Initialize on first call or when exceeding buffer capacity (Session 32)
        // Previously: reallocated on ANY size change (very expensive)
        // Now: only reallocate when cells.len() > max_cell_count
        let needs_realloc = !self.initialized || cells.len() > self.max_cell_count;
        let old_count = self.cell_count;
        let new_count = cells.len();

        if needs_realloc {
            self.init_buffers(new_count, dna_pool.len())?;
            self.upload_cells(states);
            self.upload_dna(dna_pool);
        } else if new_count > old_count {
            // Session 32: Only upload the NEW cells, not all 1M
            // This reduces upload from O(n) to O(births) = ~500 cells
            self.cell_count = new_count;
            self.upload_new_cells(states, old_count);
            self.upload_new_dna(dna_pool, old_count);
        } else if new_count != old_count {
            // Population decreased (deaths) - just update count
            // GPU will handle dead cells via flags
            self.cell_count = new_count;
        }

        self.upload_signals(signals);
        self.upload_config();

        // Reset counter for sparse dispatch
        self.reset_counter();

        // Run compact pass (count active cells on GPU)
        let compact_pipeline = self
            .compact_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Compact pipeline not initialized"))?;
        let sparse_bind_group = self.create_sparse_bind_group()?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compact Pass"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compact"),
                timestamp_writes: None,
            });
            pass.set_pipeline(compact_pipeline);
            pass.set_bind_group(0, &sparse_bind_group, &[]);
            let workgroups = (self.cell_count as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // If using indirect dispatch, also run prepare_dispatch to compute workgroup count
        if self.use_indirect_dispatch {
            let prepare_pipeline = self
                .prepare_dispatch_pipeline
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Prepare dispatch pipeline not initialized"))?;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Prepare Dispatch"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(prepare_pipeline);
                pass.set_bind_group(0, &sparse_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        // Only read active count periodically to avoid GPU→CPU sync every tick
        let should_read_stats = self.tick % 100 == 0 || needs_realloc;
        let (active_count, sleeping_count) = if should_read_stats {
            let count = self.read_active_count()? as usize;
            self.last_active_count = count as u32;
            (count, self.cell_count.saturating_sub(count))
        } else {
            // Use cached value for stats
            let count = self.last_active_count as usize;
            (count, self.cell_count.saturating_sub(count))
        };

        // Build spatial hash grid (if enabled)
        // This prepares the grid for O(1) neighbor lookup in signal propagation
        if self.use_spatial_hash && !signals.is_empty() {
            self.build_spatial_grid()?;
        }

        // Run signal propagation
        if !signals.is_empty() {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Signal Pass"),
                });

            if self.use_spatial_hash {
                // Use spatial hash for O(1) neighbor lookup
                // Dispatched per-signal (each signal finds nearby cells via grid)
                let signal_hash_bind_group = self.create_signal_with_hash_bind_group()?;
                let signal_hash_pipeline = self
                    .signal_with_hash_pipeline
                    .as_ref()
                    .ok_or_else(|| AriaError::gpu("Signal with hash pipeline not initialized"))?;

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Signal With Hash"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(signal_hash_pipeline);
                    pass.set_bind_group(0, &signal_hash_bind_group, &[]);
                    // Dispatch per signal (workgroup_size=1 in shader)
                    let signal_count = signals.len().min(1024) as u32;
                    pass.dispatch_workgroups(signal_count, 1, 1);
                }
            } else {
                // Legacy O(N²) signal propagation
                let main_bind_group = self.create_main_bind_group()?;
                let signal_pipeline = self
                    .signal_pipeline
                    .as_ref()
                    .ok_or_else(|| AriaError::gpu("Signal pipeline not initialized"))?;

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Signal"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(signal_pipeline);
                    pass.set_bind_group(0, &main_bind_group, &[]);
                    let workgroups = (self.cell_count as u32 + 255) / 256;
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }

            self.queue.submit(Some(encoder.finish()));
        }

        // Run cell update
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cell Update Pass"),
            });

        if self.use_indirect_dispatch {
            // Use indirect dispatch - GPU decides workgroup count based on active cells
            let sparse_cell_bind_group = self.create_sparse_cell_bind_group()?;
            let sparse_cell_pipeline = self
                .cell_update_sparse_pipeline
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Sparse cell update pipeline not initialized"))?;
            let indirect_buffer = self
                .indirect_buffer
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Indirect buffer not initialized"))?;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Sparse Cell Update"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(sparse_cell_pipeline);
                pass.set_bind_group(0, &sparse_cell_bind_group, &[]);
                // GPU decides workgroup count - no CPU roundtrip!
                pass.dispatch_workgroups_indirect(indirect_buffer, 0);
            }
        } else {
            // Legacy: dispatch all cells
            let main_bind_group = self.create_main_bind_group()?;
            let cell_pipeline = self
                .cell_update_pipeline
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Cell update pipeline not initialized"))?;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Cell Update"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(cell_pipeline);
                pass.set_bind_group(0, &main_bind_group, &[]);
                let workgroups = (self.cell_count as u32 + 255) / 256;
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        // Run Hebbian learning (requires spatial hash for neighbor lookup)
        // "Cells that fire together, wire together"
        if self.use_spatial_hash {
            if let Some(hebbian_bind_group) = self.create_hebbian_bind_group() {
                if let Some(hebbian_pipeline) = self.hebbian_pipeline.as_ref() {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Hebbian Pass"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Hebbian Learning"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(hebbian_pipeline);
                        pass.set_bind_group(0, &hebbian_bind_group, &[]);
                        let workgroups = (self.cell_count as u32 + 255) / 256;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));
                }
            }
        }

        // SLEEPING DRAIN: Run every 100 ticks to drain energy from sleeping cells
        // Sparse dispatch skips sleeping cells, so we need this periodic pass
        if self.tick % 100 == 0 && self.use_indirect_dispatch {
            if let Some(drain_bind_group) = self.create_sleeping_drain_bind_group() {
                if let Some(drain_pipeline) = self.sleeping_drain_pipeline.as_ref() {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Sleeping Drain Pass"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Sleeping Drain"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(drain_pipeline);
                        pass.set_bind_group(0, &drain_bind_group, &[]);
                        let workgroups = (self.cell_count as u32 + 255) / 256;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));
                }
            }
        }

        // PREDICTION EVALUATE: Run every tick to evaluate predictions and apply energy rewards/penalties
        // "Surprise costs energy" - cells that predict correctly gain energy
        if let Some(pred_bind_group) = self.create_prediction_evaluate_bind_group() {
            if let Some(pred_pipeline) = self.prediction_evaluate_pipeline.as_ref() {
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Prediction Evaluate Pass"),
                    });

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Prediction Evaluate"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pred_pipeline);
                    pass.set_bind_group(0, &pred_bind_group, &[]);
                    let workgroups = (self.cell_count as u32 + 255) / 256;
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }

                self.queue.submit(Some(encoder.finish()));
            }
        }

        // HEBBIAN SPATIAL ATTRACTION: Run every 5 ticks
        // "Cells that fire together, move together" - active cells attract to centroid
        if self.tick % 5 == 0 {
            // Pass 1: Clear centroid buffer and accumulate weighted positions
            if let Some(centroid_buffer) = self.centroid_buffer.as_ref() {
                // Clear centroid buffer (80 bytes of zeros)
                self.queue.write_buffer(centroid_buffer, 0, &[0u8; 80]);
            }

            if let Some(centroid_bind_group) = self.create_hebbian_centroid_bind_group() {
                if let Some(centroid_pipeline) = self.hebbian_centroid_pipeline.as_ref() {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Hebbian Centroid Pass"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Hebbian Centroid Accumulation"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(centroid_pipeline);
                        pass.set_bind_group(0, &centroid_bind_group, &[]);
                        let workgroups = (self.cell_count as u32 + 255) / 256;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));
                }
            }

            // Pass 2: Apply attraction force towards centroid
            if let Some(attraction_bind_group) = self.create_hebbian_attraction_bind_group() {
                if let Some(attraction_pipeline) = self.hebbian_attraction_pipeline.as_ref() {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Hebbian Attraction Pass"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Hebbian Attraction"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(attraction_pipeline);
                        pass.set_bind_group(0, &attraction_bind_group, &[]);
                        let workgroups = (self.cell_count as u32 + 255) / 256;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));
                }
            }
        }

        // CLUSTER HYSTERESIS: Run every 50 ticks
        // "Stable clusters lock in, fading clusters release."
        if self.tick % 50 == 0 {
            // Pass 1: Clear cluster stats buffer and accumulate stats
            if let Some(cluster_stats_buffer) = self.cluster_stats_buffer.as_ref() {
                // Clear cluster stats buffer (2048 bytes of zeros)
                self.queue.write_buffer(cluster_stats_buffer, 0, &[0u8; 2048]);
            }

            if let Some(stats_bind_group) = self.create_cluster_stats_bind_group() {
                if let Some(stats_pipeline) = self.cluster_stats_pipeline.as_ref() {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Cluster Stats Pass"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Cluster Stats Accumulation"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(stats_pipeline);
                        pass.set_bind_group(0, &stats_bind_group, &[]);
                        let workgroups = (self.cell_count as u32 + 255) / 256;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));
                }
            }

            // Pass 2: Update hysteresis based on cluster activity
            if let Some(hysteresis_bind_group) = self.create_cluster_hysteresis_bind_group() {
                if let Some(hysteresis_pipeline) = self.cluster_hysteresis_pipeline.as_ref() {
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Cluster Hysteresis Pass"),
                        });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Cluster Hysteresis Update"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(hysteresis_pipeline);
                        pass.set_bind_group(0, &hysteresis_bind_group, &[]);
                        let workgroups = (self.cell_count as u32 + 255) / 256;
                        pass.dispatch_workgroups(workgroups, 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));
                }
            }
        }

        // Update stats
        self.stats.cells_processed = active_count as u64;
        self.stats.cells_sleeping = sleeping_count as u64;
        self.stats.signals_propagated = signals.len() as u64;

        // Periodic download for CPU sync
        // Session 31: Reduced from 100 to 1000 ticks for better performance
        let should_download = self.tick % 1000 == 0 || needs_realloc;
        let mut actions = Vec::new();

        if should_download {
            // Download energy and flags (minimal data needed)
            // Full state download only when needed
            self.download_to_states(states)?;

            for (i, state) in states.iter().enumerate() {
                if state.is_dead() {
                    actions.push((cells[i].id, CellAction::Die));
                }
            }
        }

        if self.tick % 1000 == 0 {
            let gpu_percent = if self.cell_count > 0 {
                (sleeping_count as f32 / self.cell_count as f32 * 100.0) as u32
            } else {
                0
            };
            tracing::debug!(
                "🎮 GPU SoA tick {} - {} active, {} sleeping ({}% sparse savings)",
                self.tick,
                active_count,
                sleeping_count,
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
        self.stats.signals_propagated = signals.len() as u64;
        Ok(Vec::new())
    }

    fn detect_emergence(
        &self,
        _cells: &[Cell],
        states: &[CellState],
        config: &AriaConfig,
    ) -> AriaResult<Vec<Signal>> {
        if self.tick % config.emergence.check_interval != 0 {
            return Ok(Vec::new());
        }

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

        for s in &mut sum {
            *s /= count as f32;
        }

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
            submission_index: None,
            timeout: None,
        });
        Ok(())
    }

    fn recompile(&mut self, structural_checksum: u64) -> AriaResult<()> {
        self.recompile_dynamic_pipelines(structural_checksum)
    }

    fn name(&self) -> &'static str {
        "GPU SoA (wgpu)"
    }
}

impl GpuSoABackend {
    /// Download GPU data back to CellState slice
    fn download_to_states(&self, states: &mut [CellState]) -> AriaResult<()> {
        let energy_buffer = self
            .energy_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Energy buffer not initialized"))?;
        let metadata_buffer = self
            .metadata_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Metadata buffer not initialized"))?;
        let energy_staging = self
            .energy_staging
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Energy staging not initialized"))?;
        let metadata_staging = self
            .metadata_staging
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Metadata staging not initialized"))?;

        let energy_size = std::mem::size_of::<CellEnergy>() * states.len();
        let metadata_size = std::mem::size_of::<CellMetadata>() * states.len();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download States"),
            });
        encoder.copy_buffer_to_buffer(energy_buffer, 0, energy_staging, 0, energy_size as u64);
        encoder.copy_buffer_to_buffer(metadata_buffer, 0, metadata_staging, 0, metadata_size as u64);
        self.queue.submit(Some(encoder.finish()));

        // Map energy
        let energy_slice = energy_staging.slice(..);
        let (tx1, rx1) = std::sync::mpsc::channel();
        energy_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx1.send(r);
        });

        // Map metadata
        let metadata_slice = metadata_staging.slice(..);
        let (tx2, rx2) = std::sync::mpsc::channel();
        metadata_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx2.send(r);
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        rx1.recv()
            .map_err(|e| AriaError::gpu(format!("Energy map failed: {}", e)))?
            .map_err(|e| AriaError::gpu(format!("Energy map error: {:?}", e)))?;

        rx2.recv()
            .map_err(|e| AriaError::gpu(format!("Metadata map failed: {}", e)))?
            .map_err(|e| AriaError::gpu(format!("Metadata map error: {:?}", e)))?;

        // Copy data
        {
            let energy_data = energy_slice.get_mapped_range();
            let energies: &[CellEnergy] = bytemuck::cast_slice(&energy_data);
            for (i, state) in states.iter_mut().enumerate() {
                if i < energies.len() {
                    state.energy = energies[i].energy;
                    state.tension = energies[i].tension;
                    state.activity_level = energies[i].activity_level;
                }
            }
        }

        {
            let metadata_data = metadata_slice.get_mapped_range();
            let metadata: &[CellMetadata] = bytemuck::cast_slice(&metadata_data);
            for (i, state) in states.iter_mut().enumerate() {
                if i < metadata.len() {
                    state.flags = metadata[i].flags;
                    state.cluster_id = metadata[i].cluster_id;
                    state.hysteresis = metadata[i].hysteresis;
                }
            }
        }

        energy_staging.unmap();
        metadata_staging.unmap();
        Ok(())
    }

    /// Recompile dynamic pipelines with new logic from DNA
    pub fn recompile_dynamic_pipelines(&mut self, structural_checksum: u64) -> AriaResult<()> {
        let dna_logic = self.compiler.generate_dna_logic(structural_checksum);

        let cell_update_source = self.compiler.generate_shader(self.compiler.get_cell_update_template(), &dna_logic);
        let signal_source = self.compiler.generate_shader(self.compiler.get_signal_template(), &dna_logic);

        let main_layout = self
            .main_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Main bind group layout not created"))?;

        let main_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamic Main Pipeline Layout"),
            bind_group_layouts: &[main_layout],
            immediate_size: 0,
        });

        // Cell Update Pipeline
        let cell_shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Cell Update Shader"),
            source: wgpu::ShaderSource::Wgsl(cell_update_source.into()),
        });

        self.cell_update_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Cell Update Pipeline"),
            layout: Some(&main_pipeline_layout),
            module: &cell_shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        // Signal Pipeline
        let signal_shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Signal Shader"),
            source: wgpu::ShaderSource::Wgsl(signal_source.into()),
        });

        self.signal_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Signal Pipeline"),
            layout: Some(&main_pipeline_layout),
            module: &signal_shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        // Also update sparse version (uses same logic but different layout)
        let sparse_cell_layout = self
            .sparse_cell_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse cell bind group layout not created"))?;

        let sparse_cell_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Dynamic Sparse Cell Pipeline Layout"),
            bind_group_layouts: &[sparse_cell_layout],
            immediate_size: 0,
        });

        self.cell_update_sparse_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Sparse Cell Update Pipeline"),
            layout: Some(&sparse_cell_pipeline_layout),
            module: &cell_shader_module, // Reuse same module
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        // Signal With Hash Pipeline (Spatial Propagation)
        if self.use_spatial_hash {
            let signal_hash_source = self.compiler.generate_shader(self.compiler.get_spatial_signal_template(), &dna_logic);
            let signal_hash_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dynamic Signal With Hash Shader"),
                source: wgpu::ShaderSource::Wgsl(signal_hash_source.into()),
            });

            let signal_hash_layout = self.signal_with_hash_bind_group_layout.as_ref().unwrap();
            let signal_hash_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dynamic Signal Hash Pipeline Layout"),
                bind_group_layouts: &[signal_hash_layout],
                immediate_size: 0,
            });

            self.signal_with_hash_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dynamic Signal With Hash Pipeline"),
                layout: Some(&signal_hash_pipeline_layout),
                module: &signal_hash_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }));
            tracing::info!("🎮 Dynamic Spatial Signal pipeline recompiled");
        }

        tracing::info!("🧬 GPU: Dynamic pipelines recompiled (checksum: {}, logic size: {})", structural_checksum, dna_logic.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_size() {
        assert_eq!(std::mem::size_of::<GpuConfig>(), 64);
    }
}
