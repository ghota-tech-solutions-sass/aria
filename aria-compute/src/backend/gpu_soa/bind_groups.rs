//! Bind group creation for GPU SoA backend
//!
//! Creates bind group instances that connect buffers to pipelines.

use aria_core::error::{AriaError, AriaResult};

use super::GpuSoABackend;

impl GpuSoABackend {
    pub(super) fn create_signal_with_hash_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .signal_with_hash_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Signal with hash bind group layout not created"))?;

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
                    resource: self
                        .spatial_config_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self
                        .connection_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.dna_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self
                        .dna_indices_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        }))
    }

    pub(super) fn create_sparse_cell_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .sparse_cell_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse cell bind group layout not created"))?;

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
                    resource: self
                        .sparse_dispatch_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self
                        .dna_indices_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        }))
    }

    pub(super) fn create_sparse_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .sparse_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse bind group layout not created"))?;

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
                    resource: self
                        .sparse_dispatch_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
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

    pub(super) fn create_grid_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_hebbian_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_sleeping_drain_bind_group(&self) -> Option<wgpu::BindGroup> {
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


    pub(super) fn create_prediction_evaluate_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_hebbian_centroid_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_hebbian_attraction_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_cluster_stats_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_cluster_hysteresis_bind_group(&self) -> Option<wgpu::BindGroup> {
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

    pub(super) fn create_main_bind_group(&self) -> AriaResult<wgpu::BindGroup> {
        let layout = self
            .main_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Main bind group layout not created"))?;

        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SoA Main Bind Group"),
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
                    resource: self
                        .dna_indices_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                },
            ],
        }))
    }
}
