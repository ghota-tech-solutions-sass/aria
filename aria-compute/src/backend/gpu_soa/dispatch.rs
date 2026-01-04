//! Dispatch logic and GPU-CPU synchronization for SoA backend
//!
//! Contains spatial grid building, counter management, and data readback.

use aria_core::cell::CellState;
use aria_core::error::{AriaError, AriaResult};
use aria_core::soa::{CellEnergy, CellMetadata};

use super::config::AtomicCounter;
use super::GpuSoABackend;

impl GpuSoABackend {
    /// Build spatial hash grid for O(1) neighbor lookup
    pub(super) fn build_spatial_grid(&self) -> AriaResult<()> {
        let clear_pipeline = self
            .clear_grid_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Clear grid pipeline not initialized"))?;
        let build_pipeline = self
            .build_grid_pipeline
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Build grid pipeline not initialized"))?;
        let grid_bind_group = self
            .create_grid_bind_group()
            .ok_or_else(|| AriaError::gpu("Grid bind group creation failed"))?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Spatial Grid Pass"),
            });

        // Clear grid
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(clear_pipeline);
            pass.set_bind_group(0, &grid_bind_group, &[]);
            let grid_workgroups = (crate::spatial_gpu::TOTAL_REGIONS as u32 + 255) / 256;
            pass.dispatch_workgroups(grid_workgroups, 1, 1);
        }

        // Build grid
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(build_pipeline);
            pass.set_bind_group(0, &grid_bind_group, &[]);
            let cell_workgroups = (self.cell_count as u32 + 255) / 256;
            pass.dispatch_workgroups(cell_workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Reset sparse dispatch counter
    pub(super) fn reset_counter(&self) {
        let counter = AtomicCounter {
            count: 0,
            _pad: [0; 3],
        };
        if let Some(buf) = &self.sparse_dispatch_buffer {
            self.queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&counter));
        }
    }

    /// Read active cell count from GPU
    pub(super) fn read_active_count(&self) -> AriaResult<u32> {
        let sparse_buffer = self
            .sparse_dispatch_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse dispatch buffer not initialized"))?;
        let counter_staging = self
            .counter_staging
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Counter staging not initialized"))?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Counter"),
            });
        encoder.copy_buffer_to_buffer(sparse_buffer, 0, counter_staging, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        let slice = counter_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        // Session 35: Add timeout to prevent infinite blocking
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(50)),
        });

        // Non-blocking receive with fallback to last known count
        match rx.try_recv() {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range();
                let counter: &AtomicCounter = bytemuck::from_bytes(&data[..16]);
                let count = counter.count;
                drop(data);
                counter_staging.unmap();
                Ok(count)
            }
            Ok(Err(e)) => Err(AriaError::gpu(format!("Counter map error: {:?}", e))),
            Err(_) => {
                // GPU not ready yet - return last known count
                Ok(self.last_active_count)
            }
        }
    }

    /// Read lifecycle counters from GPU
    pub fn read_lifecycle_counters(&self) -> AriaResult<(u32, u32, u32, u32)> {
        let counters_buffer = self
            .lifecycle_counters_buffer
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Lifecycle counters buffer not initialized"))?;
        let counters_staging = self
            .lifecycle_counters_staging
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Lifecycle counters staging not initialized"))?;

        let size = std::mem::size_of::<aria_core::LifecycleCounters>() as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Read Lifecycle Counters"),
            });
        encoder.copy_buffer_to_buffer(counters_buffer, 0, counters_staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = counters_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        // Session 35: Add timeout to prevent infinite blocking
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(50)),
        });

        // Non-blocking receive with fallback
        match rx.try_recv() {
            Ok(Ok(())) => {
                let data = slice.get_mapped_range();
                let counters: &aria_core::LifecycleCounters = bytemuck::from_bytes(&data);
                let result = (
                    counters.alive_count,
                    counters.free_count,
                    counters.births_this_tick,
                    counters.deaths_this_tick,
                );
                drop(data);
                counters_staging.unmap();
                Ok(result)
            }
            Ok(Err(e)) => Err(AriaError::gpu(format!("Lifecycle counters map error: {:?}", e))),
            Err(_) => {
                // GPU not ready yet - return zeros (non-critical stats)
                Ok((0, 0, 0, 0))
            }
        }
    }

    /// Download GPU data back to CellState slice
    pub(super) fn download_to_states(&self, states: &mut [CellState]) -> AriaResult<()> {
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
        encoder.copy_buffer_to_buffer(
            metadata_buffer,
            0,
            metadata_staging,
            0,
            metadata_size as u64,
        );
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

        // Session 35: Add timeout to prevent long blocking
        // For 120k cells, this can take several seconds - limit to 100ms
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_millis(100)),
        });

        // Check if both mappings completed
        let energy_ready = rx1.try_recv();
        let metadata_ready = rx2.try_recv();

        match (energy_ready, metadata_ready) {
            (Ok(Ok(())), Ok(Ok(()))) => {
                // Both ready - copy data
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
            (Ok(Err(e)), _) => Err(AriaError::gpu(format!("Energy map error: {:?}", e))),
            (_, Ok(Err(e))) => Err(AriaError::gpu(format!("Metadata map error: {:?}", e))),
            _ => {
                // GPU not ready yet - skip this download, try again next interval
                tracing::debug!("⏳ GPU download timeout - will retry");
                Ok(())
            }
        }
    }
}
