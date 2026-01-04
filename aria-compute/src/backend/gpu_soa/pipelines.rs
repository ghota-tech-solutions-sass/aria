//! Pipeline creation for GPU SoA backend
//!
//! Creates all compute pipelines used for cell updates, signal propagation,
//! Hebbian learning, prediction, and lifecycle management.

use aria_core::error::{AriaError, AriaResult};

use crate::spatial_gpu;

use super::GpuSoABackend;

impl GpuSoABackend {
    pub(super) fn create_pipelines(&mut self) -> AriaResult<()> {
        // Initial compilation of dynamic pipelines
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

        let sparse_pipeline_layout =
            self.device
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

        // Spatial hash pipelines
        if self.use_spatial_hash {
            let grid_layout = self
                .grid_bind_group_layout
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Grid bind group layout not created"))?;

            let grid_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Grid Pipeline Layout"),
                        bind_group_layouts: &[grid_layout],
                        immediate_size: 0,
                    });

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

            let signal_hash_layout = self
                .signal_with_hash_bind_group_layout
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Signal with hash bind group layout not created"))?;

            let signal_hash_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Signal With Hash Pipeline Layout"),
                        bind_group_layouts: &[signal_hash_layout],
                        immediate_size: 0,
                    });

            let signal_hash_shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Signal With Hash Shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        spatial_gpu::SIGNAL_WITH_SPATIAL_HASH_SHADER.into(),
                    ),
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

            // Hebbian learning pipeline
            let hebbian_layout = self
                .hebbian_bind_group_layout
                .as_ref()
                .ok_or_else(|| AriaError::gpu("Hebbian bind group layout not created"))?;

            let hebbian_pipeline_layout =
                self.device
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

        // Sleeping drain pipeline
        if let Some(sleeping_layout) = self.sleeping_drain_bind_group_layout.as_ref() {
            let sleeping_pipeline_layout =
                self.device
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

        // Prediction Law pipelines
        if let Some(pred_gen_layout) = self.prediction_generate_bind_group_layout.as_ref() {
            let pred_gen_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Prediction Generate Pipeline Layout"),
                        bind_group_layouts: &[pred_gen_layout],
                        immediate_size: 0,
                    });

            let pred_gen_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Prediction Generate Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    self.compiler.get_prediction_generate_shader().into(),
                ),
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
            let pred_eval_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Prediction Evaluate Pipeline Layout"),
                        bind_group_layouts: &[pred_eval_layout],
                        immediate_size: 0,
                    });

            let pred_eval_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Prediction Evaluate Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    self.compiler.get_prediction_evaluate_shader().into(),
                ),
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

        // Hebbian Spatial Attraction pipelines
        if let Some(centroid_layout) = self.hebbian_centroid_bind_group_layout.as_ref() {
            let centroid_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Hebbian Centroid Pipeline Layout"),
                        bind_group_layouts: &[centroid_layout],
                        immediate_size: 0,
                    });

            let centroid_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Hebbian Centroid Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    self.compiler.get_hebbian_centroid_shader().into(),
                ),
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
            let attraction_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Hebbian Attraction Pipeline Layout"),
                        bind_group_layouts: &[attraction_layout],
                        immediate_size: 0,
                    });

            let attraction_shader =
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Hebbian Attraction Shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            self.compiler.get_hebbian_attraction_shader().into(),
                        ),
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

        // Cluster Hysteresis pipelines
        if let Some(stats_layout) = self.cluster_stats_bind_group_layout.as_ref() {
            let stats_pipeline_layout =
                self.device
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
            let hysteresis_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Cluster Hysteresis Pipeline Layout"),
                        bind_group_layouts: &[hysteresis_layout],
                        immediate_size: 0,
                    });

            let hysteresis_shader =
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Cluster Hysteresis Shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            self.compiler.get_cluster_hysteresis_shader().into(),
                        ),
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

        // GPU Lifecycle pipelines
        let lifecycle_layout = self
            .lifecycle_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Lifecycle bind group layout not created"))?;

        let lifecycle_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Lifecycle Pipeline Layout"),
                    bind_group_layouts: &[lifecycle_layout],
                    immediate_size: 0,
                });

        let death_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Death Shader"),
            source: wgpu::ShaderSource::Wgsl(self.compiler.get_death_shader().into()),
        });

        self.death_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Death Pipeline"),
                    layout: Some(&lifecycle_pipeline_layout),
                    module: &death_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        let birth_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Birth Shader"),
            source: wgpu::ShaderSource::Wgsl(self.compiler.get_birth_shader().into()),
        });

        self.birth_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Birth Pipeline"),
                    layout: Some(&lifecycle_pipeline_layout),
                    module: &birth_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        let reset_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reset Lifecycle Counters Shader"),
            source: wgpu::ShaderSource::Wgsl(
                self.compiler.get_reset_lifecycle_counters_shader().into(),
            ),
        });

        let reset_counters_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Reset Counters Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let reset_counters_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Reset Counters Pipeline Layout"),
                    bind_group_layouts: &[&reset_counters_layout],
                    immediate_size: 0,
                });

        self.reset_lifecycle_counters_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Reset Lifecycle Counters Pipeline"),
                    layout: Some(&reset_counters_pipeline_layout),
                    module: &reset_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        tracing::info!("🔄 GPU Lifecycle pipelines created (death, birth, reset)");
        tracing::debug!("🎮 GPU SoA pipelines created");
        Ok(())
    }

    /// Recompile dynamic pipelines with new logic from DNA
    pub fn recompile_dynamic_pipelines(&mut self, structural_checksum: u64) -> AriaResult<()> {
        let dna_logic = self.compiler.generate_dna_logic(structural_checksum);

        let cell_update_source = self
            .compiler
            .generate_shader(self.compiler.get_cell_update_template(), &dna_logic);
        let signal_source = self
            .compiler
            .generate_shader(self.compiler.get_signal_template(), &dna_logic);

        let main_layout = self
            .main_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Main bind group layout not created"))?;

        let main_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Dynamic Main Pipeline Layout"),
                    bind_group_layouts: &[main_layout],
                    immediate_size: 0,
                });

        let cell_shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Cell Update Shader"),
            source: wgpu::ShaderSource::Wgsl(cell_update_source.into()),
        });

        self.cell_update_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dynamic Cell Update Pipeline"),
                    layout: Some(&main_pipeline_layout),
                    module: &cell_shader_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        let signal_shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Signal Shader"),
            source: wgpu::ShaderSource::Wgsl(signal_source.into()),
        });

        self.signal_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dynamic Signal Pipeline"),
                    layout: Some(&main_pipeline_layout),
                    module: &signal_shader_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        let sparse_cell_layout = self
            .sparse_cell_bind_group_layout
            .as_ref()
            .ok_or_else(|| AriaError::gpu("Sparse cell bind group layout not created"))?;

        let sparse_cell_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Dynamic Sparse Cell Pipeline Layout"),
                    bind_group_layouts: &[sparse_cell_layout],
                    immediate_size: 0,
                });

        self.cell_update_sparse_pipeline = Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dynamic Sparse Cell Update Pipeline"),
                    layout: Some(&sparse_cell_pipeline_layout),
                    module: &cell_shader_module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );

        if self.use_spatial_hash {
            let signal_hash_source = self.compiler.generate_shader(
                self.compiler.get_spatial_signal_template(),
                &dna_logic,
            );
            let signal_hash_shader =
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Dynamic Signal With Hash Shader"),
                        source: wgpu::ShaderSource::Wgsl(signal_hash_source.into()),
                    });

            let signal_hash_layout = self.signal_with_hash_bind_group_layout.as_ref().unwrap();
            let signal_hash_pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Dynamic Signal Hash Pipeline Layout"),
                        bind_group_layouts: &[signal_hash_layout],
                        immediate_size: 0,
                    });

            self.signal_with_hash_pipeline = Some(
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Dynamic Signal With Hash Pipeline"),
                        layout: Some(&signal_hash_pipeline_layout),
                        module: &signal_hash_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
            );
            tracing::info!("🎮 Dynamic Spatial Signal pipeline recompiled");
        }

        tracing::info!(
            "🧬 GPU: Dynamic pipelines recompiled (checksum: {}, logic size: {})",
            structural_checksum,
            dna_logic.len()
        );
        Ok(())
    }
}
