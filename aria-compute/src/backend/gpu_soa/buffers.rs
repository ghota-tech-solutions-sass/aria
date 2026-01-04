//! Buffer initialization and data upload for GPU SoA backend
//!
//! Handles creation, resize, and data transfer for all GPU buffers.

use aria_core::cell::CellState;
use aria_core::dna::DNA;
use aria_core::error::AriaResult;
use aria_core::signal::SignalFragment;
use aria_core::soa::{
    CellConnections, CellEnergy, CellInternalState, CellMetadata, CellPosition, CellPrediction,
    IndirectDispatchArgs,
};
use aria_core::LifecycleCounters;

use crate::spatial_gpu::{self, SpatialHashConfig, GRID_SIZE, TOTAL_REGIONS};

use super::config::GpuConfig;
use super::GpuSoABackend;

impl GpuSoABackend {
    /// Initialize GPU buffers with SoA layout
    pub(super) fn init_buffers(&mut self, cell_count: usize, dna_count: usize) -> AriaResult<()> {
        // Use GPU's actual max buffer size (queried at init)
        // CellConnections is the LARGEST buffer at 144 bytes/cell
        let connections_size = std::mem::size_of::<CellConnections>();
        let max_cells_in_buffer = self.max_buffer_size / connections_size;

        // Calculate headroom based on population size
        let headroom_factor = if cell_count > 3_000_000 {
            1.25
        } else if cell_count > 1_000_000 {
            1.5
        } else {
            2.0
        };
        let desired_headroom = (cell_count as f64 * headroom_factor) as usize;
        let cell_count_with_headroom = desired_headroom.min(max_cells_in_buffer);
        let dna_count_with_headroom =
            ((dna_count as f64 * headroom_factor) as usize).min(max_cells_in_buffer);

        if cell_count_with_headroom < desired_headroom {
            tracing::warn!(
                "⚠️ GPU buffer limit: headroom reduced from {}M to {}M cells",
                desired_headroom / 1_000_000,
                cell_count_with_headroom / 1_000_000
            );
        }
        tracing::info!(
            "📊 Headroom: {:.0}% ({} → {} cells)",
            (headroom_factor - 1.0) * 100.0,
            cell_count,
            cell_count_with_headroom
        );

        self.max_cell_count = cell_count_with_headroom;

        // Calculate buffer sizes
        let energy_bytes = std::mem::size_of::<CellEnergy>() * cell_count_with_headroom;
        let position_bytes = std::mem::size_of::<CellPosition>() * cell_count_with_headroom;
        let state_bytes = std::mem::size_of::<CellInternalState>() * cell_count_with_headroom;
        let metadata_bytes = std::mem::size_of::<CellMetadata>() * cell_count_with_headroom;
        let dna_bytes = std::mem::size_of::<DNA>() * dna_count_with_headroom;
        let signals_bytes = std::mem::size_of::<SignalFragment>() * 1024;
        let indices_bytes = std::mem::size_of::<u32>() * cell_count_with_headroom;
        let counter_bytes = 16;
        let indirect_bytes = std::mem::size_of::<IndirectDispatchArgs>();
        let dna_indices_bytes = std::mem::size_of::<u32>() * cell_count_with_headroom;

        let sparse_bytes = counter_bytes + indices_bytes;

        // Buffer size estimation for VRAM logging
        let connection_bytes_est =
            std::mem::size_of::<CellConnections>() * cell_count_with_headroom;
        let prediction_bytes_est = std::mem::size_of::<CellPrediction>() * cell_count_with_headroom;
        let grid_bytes_est = TOTAL_REGIONS * 4 * 21;

        let total_bytes = energy_bytes
            + position_bytes
            + state_bytes
            + metadata_bytes
            + dna_bytes
            + signals_bytes
            + sparse_bytes
            + indirect_bytes
            + dna_indices_bytes
            + connection_bytes_est
            + prediction_bytes_est
            + grid_bytes_est;

        let total_mb = total_bytes / 1024 / 1024;
        tracing::info!(
            "🎮 GPU SoA: Allocating ~{} MB VRAM for {} cells (capacity: {})",
            total_mb,
            cell_count,
            cell_count_with_headroom
        );
        if total_mb > 6000 {
            tracing::warn!(
                "⚠️ High VRAM usage ({}MB) - risk of 'Device lost' on 8GB GPUs",
                total_mb
            );
        }

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

        // Sparse dispatch buffer
        self.sparse_dispatch_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sparse Dispatch Buffer"),
            size: sparse_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Indirect dispatch buffer
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

        // Spatial hash buffers
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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            self.spatial_config_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Spatial Config"),
                size: spatial_config_bytes as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            let spatial_config = SpatialHashConfig::default_aria(cell_count);
            self.queue.write_buffer(
                self.spatial_config_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&spatial_config),
            );
        }

        // Hebbian connection buffer
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

        let empty_connections = vec![CellConnections::default(); cell_count_with_headroom];
        self.queue.write_buffer(
            self.connection_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&empty_connections),
        );

        // Prediction Law buffer
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

        let empty_predictions = vec![CellPrediction::default(); cell_count_with_headroom];
        self.queue.write_buffer(
            self.prediction_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&empty_predictions),
        );

        // Centroid buffer for Hebbian spatial attraction
        let centroid_bytes = 80;
        self.centroid_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hebbian Centroid"),
            size: centroid_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Cluster stats buffer
        let cluster_stats_bytes = 256 * 4 * 2;
        self.cluster_stats_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cluster Stats"),
            size: cluster_stats_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // GPU Lifecycle buffers
        let free_list_bytes = std::mem::size_of::<u32>() * cell_count_with_headroom;
        tracing::info!(
            "🔄 Lifecycle: Allocating {} MB for free_list ({} slots)",
            free_list_bytes / 1024 / 1024,
            cell_count_with_headroom
        );

        self.free_list_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free List"),
            size: free_list_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let initial_free_slots: Vec<u32> =
            (cell_count as u32..cell_count_with_headroom as u32).collect();
        if !initial_free_slots.is_empty() {
            self.queue.write_buffer(
                self.free_list_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&initial_free_slots),
            );
        }

        let lifecycle_counters_bytes = std::mem::size_of::<LifecycleCounters>();
        self.lifecycle_counters_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lifecycle Counters"),
            size: lifecycle_counters_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let initial_counters = LifecycleCounters::new(
            cell_count_with_headroom,
            cell_count,
            self.config.metabolism.reproduction_threshold,
            self.config.metabolism.child_energy,
        );
        self.queue.write_buffer(
            self.lifecycle_counters_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&initial_counters),
        );

        self.lifecycle_counters_staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lifecycle Counters Staging"),
            size: lifecycle_counters_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.create_bind_group_layouts()?;
        self.create_pipelines()?;

        self.cell_count = cell_count;
        self.initialized = true;

        tracing::info!("🎮 GPU SoA initialized with {} cells", cell_count);
        Ok(())
    }

    /// Upload cell data to SoA buffers
    pub(super) fn upload_cells(&self, states: &[CellState]) {
        let count = states.len().min(self.max_cell_count);
        if count < states.len() {
            tracing::warn!(
                "⚠️ upload_cells truncated: {} → {} cells (max_cell_count: {})",
                states.len(),
                count,
                self.max_cell_count
            );
        }

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

    /// Upload only NEW cells (optimization for births)
    pub(super) fn upload_new_cells(&self, states: &[CellState], old_count: usize) {
        if old_count >= states.len() {
            return;
        }

        // Safety check: don't write beyond buffer capacity
        let safe_end = states.len().min(self.max_cell_count);
        if old_count >= safe_end {
            tracing::warn!(
                "⚠️ upload_new_cells skipped: old_count {} >= safe_end {} (max_cell_count: {})",
                old_count,
                safe_end,
                self.max_cell_count
            );
            return;
        }

        let new_cells = &states[old_count..safe_end];
        let offset_bytes_energy = old_count * std::mem::size_of::<CellEnergy>();
        let offset_bytes_position = old_count * std::mem::size_of::<CellPosition>();
        let offset_bytes_state = old_count * std::mem::size_of::<CellInternalState>();
        let offset_bytes_metadata = old_count * std::mem::size_of::<CellMetadata>();

        let energies: Vec<CellEnergy> = new_cells
            .iter()
            .map(|s| CellEnergy {
                energy: s.energy,
                tension: s.tension,
                activity_level: s.activity_level,
                _pad: 0.0,
            })
            .collect();

        let positions: Vec<CellPosition> = new_cells
            .iter()
            .map(|s| CellPosition {
                position: s.position,
            })
            .collect();

        let internal_states: Vec<CellInternalState> = new_cells
            .iter()
            .map(|s| CellInternalState { state: s.state })
            .collect();

        let metadata: Vec<CellMetadata> = new_cells
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
                offset_bytes_energy as u64,
                bytemuck::cast_slice(&energies),
            );
        }
        if let Some(buf) = &self.position_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes_position as u64,
                bytemuck::cast_slice(&positions),
            );
        }
        if let Some(buf) = &self.state_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes_state as u64,
                bytemuck::cast_slice(&internal_states),
            );
        }
        if let Some(buf) = &self.metadata_buffer {
            self.queue.write_buffer(
                buf,
                offset_bytes_metadata as u64,
                bytemuck::cast_slice(&metadata),
            );
        }
    }

    /// Upload only NEW DNA entries
    pub(super) fn upload_new_dna(&self, dna_pool: &[DNA], old_count: usize) {
        if old_count >= dna_pool.len() {
            return;
        }

        // Safety check: don't write beyond buffer capacity
        let safe_end = dna_pool.len().min(self.max_cell_count);
        if old_count >= safe_end {
            tracing::warn!(
                "⚠️ upload_new_dna skipped: old_count {} >= safe_end {} (max_cell_count: {})",
                old_count,
                safe_end,
                self.max_cell_count
            );
            return;
        }

        let new_dna = &dna_pool[old_count..safe_end];
        let offset_bytes = old_count * std::mem::size_of::<DNA>();

        // Final safety: verify we won't overflow
        let end_bytes = offset_bytes + new_dna.len() * std::mem::size_of::<DNA>();
        let buffer_size = self.max_cell_count * std::mem::size_of::<DNA>();
        if end_bytes > buffer_size {
            tracing::error!(
                "🚨 DNA OVERFLOW PREVENTED: {}..{} > buffer size {}",
                offset_bytes,
                end_bytes,
                buffer_size
            );
            return;
        }

        if let Some(buf) = &self.dna_buffer {
            self.queue
                .write_buffer(buf, offset_bytes as u64, bytemuck::cast_slice(new_dna));
        }
    }

    /// Upload DNA pool
    pub(super) fn upload_dna(&self, dna_pool: &[DNA]) {
        if let Some(buf) = &self.dna_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(dna_pool));
        }
    }

    /// Upload signals
    pub(super) fn upload_signals(&self, signals: &[SignalFragment]) {
        if signals.is_empty() {
            return;
        }
        let to_upload = &signals[..signals.len().min(1024)];
        if let Some(buf) = &self.signals_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(to_upload));
        }
    }

    /// Upload config
    pub(super) fn upload_config(&self) {
        let gpu_config = GpuConfig::from_config(&self.config, self.tick, self.cell_count);
        if let Some(buf) = &self.config_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&gpu_config));
        }
    }
}
