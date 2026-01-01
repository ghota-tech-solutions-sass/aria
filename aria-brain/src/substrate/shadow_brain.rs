use aria_core::{Cell, CellState, DNA, AriaConfig, POSITION_DIMS};
use aria_core::traits::ComputeBackend;
use aria_core::error::AriaResult;
use aria_compute::create_backend;

/// A lightweight substrate clone for testing experimental code
pub struct ShadowBrain {
    backend: Box<dyn ComputeBackend>,
    cells: Vec<Cell>,
    states: Vec<CellState>,
    dna_pool: Vec<DNA>,
}

impl ShadowBrain {
    /// Create a new Shadow Brain with a small population
    pub fn new(main_config: &AriaConfig) -> AriaResult<Self> {
        let mut shadow_config = main_config.clone();
        shadow_config.population.target_population = 1024; // Small sample

        let backend = create_backend(&shadow_config)?;

        // Initialize with default DNA and cells
        let dna = DNA::default();
        let dna_pool = vec![dna; 1];
        let mut cells = Vec::with_capacity(1024);
        let mut states = Vec::with_capacity(1024);

        for i in 0..1024 {
            let cell = Cell::new(i as u64, 0);
            cells.push(cell);
            let mut state = CellState::new();
            state.position = [0.0; POSITION_DIMS]; // Start at origin
            state.energy = 1.0; // Ensure full energy for tests
            states.push(state);
        }

        Ok(Self {
            backend,
            cells,
            states,
            dna_pool,
        })
    }

    /// Evaluate a new structural checksum
    /// Returns a fitness score (higher is better)
    pub fn evaluate_candidate(&mut self, checksum: u64) -> AriaResult<f32> {
        // 1. Recompile backend with candidate checksum
        self.backend.recompile(checksum)?;

        // 2. Reset states to baseline
        for state in &mut self.states {
            state.energy = 1.0;
            state.tension = 0.0;
            state.activity_level = 0.5;
        }

        // 3. Run simulation for N ticks
        let test_ticks = 100;
        let mut total_stability = 0.0;

        for _ in 0..test_ticks {
            let signals = vec![];
            let _actions = self.backend.update_cells(
                &mut self.cells,
                &mut self.states,
                &self.dna_pool,
                &signals
            )?;

            // Measure average stability (avoid extreme energy drain or explosion)
            let avg_energy: f32 = self.states.iter().map(|s| s.energy).sum::<f32>() / self.states.len() as f32;
            total_stability += 1.0 - (avg_energy - 1.0).abs();
        }

        let score = (total_stability / test_ticks as f32).clamp(0.0, 1.0);

        tracing::info!("👻 SHADOW BRAIN: Candidate {} score: {:.4}", checksum, score);

        Ok(score)
    }
}
