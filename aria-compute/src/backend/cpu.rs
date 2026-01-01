//! # CPU Backend
//!
//! Parallel computation using Rayon.
//!
//! This backend is ideal for:
//! - Development and debugging
//! - Small populations (< 100k cells)
//! - Systems without GPU support
//!
//! ## Features
//!
//! - Full sparse update support
//! - Spatial hashing for O(1) neighbor lookup
//! - Uses all CPU cores via Rayon

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use aria_core::activity::SleepConfig;
use aria_core::cell::{Cell, CellAction, CellState};
use aria_core::config::AriaConfig;
use aria_core::dna::DNA;
use aria_core::error::AriaResult;
use aria_core::signal::{Signal, SignalFragment, SignalType};
use aria_core::traits::{BackendStats, ComputeBackend};
use aria_core::{POSITION_DIMS, SIGNAL_DIMS, STATE_DIMS};

use crate::spatial::SpatialHash;

/// CPU compute backend using Rayon
pub struct CpuBackend {
    /// Configuration
    config: AriaConfig,

    /// Spatial hash for neighbor lookup
    spatial: SpatialHash,

    /// Sleep configuration
    sleep_config: SleepConfig,

    /// Statistics
    stats: BackendStats,

    /// Tick counter
    tick: AtomicU64,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new(config: &AriaConfig) -> AriaResult<Self> {
        let spatial = SpatialHash::new(config.compute.grid_resolution as usize);

        Ok(Self {
            config: config.clone(),
            spatial,
            sleep_config: config.activity.clone(),
            stats: BackendStats::default(),
            tick: AtomicU64::new(0),
        })
    }

    /// Process a single cell for one tick
    ///
    /// "La Vraie Faim" - Cells must struggle to survive.
    fn process_cell(
        cell: &mut Cell,
        state: &mut CellState,
        dna: &DNA,
        signals: &[SignalFragment],
        config: &AriaConfig,
        sleep_config: &SleepConfig,
    ) -> Option<CellAction> {
        // Skip sleeping cells unless they receive a signal
        if cell.activity.sleeping {
            // Sleeping cells consume significant energy (hibernation still costs!)
            // "La Vraie Faim" - even sleep isn't free
            state.energy -= config.metabolism.cost_rest * 0.5;

            // Check if any signal should wake this cell
            let max_stimulus: f32 = signals
                .iter()
                .map(|s| s.intensity)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            if cell.activity.should_wake(max_stimulus, sleep_config) {
                cell.activity.wake();
                state.set_sleeping(false); // Sync CellState.flags with Cell.activity
            } else {
                // Stay asleep - but check for death even while sleeping
                if state.energy <= 0.0 {
                    state.set_dead();
                    return Some(CellAction::Die);
                }
                cell.activity.update(0.0, 0.0, sleep_config);
                return None;
            }
        }

        // Track energy before
        let energy_before = state.energy;

        // Age
        cell.age += 1;

        // === LA VRAIE FAIM: Process signals with RESONANCE ===
        for signal in signals {
            // Calculate resonance: how similar is this signal to cell's state?
            let resonance = Self::calculate_resonance(&signal.content, &state.state);

            // Reflexivity: How much does ARIA listen to her own thoughts?
            let reflexivity_gain = if signal.source_id == u64::MAX {
                dna.reflexivity_gain()
            } else {
                1.0
            };

            for (i, s) in signal.content.iter().enumerate() {
                if i < SIGNAL_DIMS {
                    let reaction = dna.reactions[i];
                    // Amplified reaction with reflexivity multiplier
                    state.state[i] += s * signal.intensity * reaction * config.signals.reaction_amplification * reflexivity_gain;
                }
            }

            // Echo to higher dimensions
            for (i, s) in signal.content.iter().enumerate() {
                if i < SIGNAL_DIMS {
                    state.state[i + 8] += s * signal.intensity * 5.0;
                    state.state[i + 16] += s * signal.intensity * 2.5;
                }
            }

            // === LA VRAIE FAIM: ENERGY FROM RESONANCE ===
            // Cells that resonate with the signal get fed!
            // resonance < 0.1 = pure noise = no food
            // resonance > 0.1 = some understanding = food scaled by resonance
            if resonance > 0.1 {
                // Scale energy by how well you understand (0.1->0 to 1.0->1)
                let understanding = (resonance - 0.1) / 0.9;
                let energy_gain = config.metabolism.signal_energy_base
                    * signal.intensity
                    * understanding
                    * (1.0 + resonance * config.metabolism.signal_resonance_factor);
                state.energy += energy_gain;
            }
            // resonance <= 0.1: NO ENERGY - pure noise
        }

        // Normalize state
        state.normalize_state(config.signals.state_cap);

        // Build tension
        state.tension += 0.01 + signals.len() as f32 * 0.05;

        // === BUG FIX: Deduct cost_rest BEFORE delta calculation ===
        // So the sleep system "sees" the energy loss from breathing
        state.energy -= config.metabolism.cost_rest;

        // Calculate energy delta for activity tracking (now includes cost_rest)
        let energy_delta = state.energy - energy_before;
        let state_magnitude: f32 = state.state.iter().map(|x| x.abs()).sum();
        cell.activity.update(energy_delta, state_magnitude, sleep_config);

        // === BUG FIX: Decay tension when activity is low (like GPU shader) ===
        // This allows cells to reach the sleep threshold
        if cell.activity.last_energy_delta.abs() < sleep_config.energy_delta_threshold {
            state.tension *= 0.8;  // Match GPU: faster decay
            state.tension -= 0.01;
            if state.tension < 0.0 {
                state.tension = 0.0;
            }
        }

        // Check for sleep
        if cell.activity.should_sleep(sleep_config) {
            cell.activity.sleep();
            state.set_sleeping(true);
        }

        // Death check BEFORE action
        if state.energy <= 0.0 {
            state.set_dead();
            return Some(CellAction::Die);
        }

        // Action threshold
        let action_threshold = dna.action_threshold();

        if state.tension > action_threshold {
            state.tension = 0.0;
            let action = Self::choose_action(state, dna, config);

            // === LA VRAIE FAIM: Actions cost energy ===
            let action_cost = match &action {
                CellAction::Signal(_) => config.metabolism.cost_signal,
                CellAction::Divide => config.metabolism.cost_divide,
                CellAction::Move(_) => config.metabolism.cost_move,
                CellAction::Rest => config.metabolism.cost_rest,
                CellAction::Die => 0.0,
                CellAction::Connect(_) => config.metabolism.cost_signal * 0.5,
            };
            state.energy -= action_cost;

            // Cap energy
            state.energy = state.energy.clamp(0.0, config.metabolism.energy_cap);

            // Check death after action cost
            if state.energy <= 0.0 {
                state.set_dead();
                return Some(CellAction::Die);
            }

            return Some(action);
        }

        // cost_rest already deducted above (before delta calculation)
        // Just cap energy
        state.energy = state.energy.clamp(0.0, config.metabolism.energy_cap);

        // Final death check
        if state.energy <= 0.0 {
            state.set_dead();
            return Some(CellAction::Die);
        }

        Some(CellAction::Rest)
    }

    /// Calculate resonance between a signal and cell state
    ///
    /// Returns 0.0 for noise, 1.0 for perfect match.
    /// Cells that receive signals matching their internal state gain more energy.
    fn calculate_resonance(signal: &[f32; SIGNAL_DIMS], state: &[f32; STATE_DIMS]) -> f32 {
        // Cosine similarity between signal and first SIGNAL_DIMS of state
        let mut dot = 0.0f32;
        let mut norm_sig = 0.0f32;
        let mut norm_state = 0.0f32;

        for i in 0..SIGNAL_DIMS {
            dot += signal[i] * state[i];
            norm_sig += signal[i] * signal[i];
            norm_state += state[i] * state[i];
        }

        let denom = (norm_sig * norm_state).sqrt();
        if denom > 0.001 {
            // Map from [-1, 1] to [0, 1]
            (dot / denom + 1.0) * 0.5
        } else {
            0.5 // Neutral if no signal or state
        }
    }

    /// Choose an action based on state and DNA
    fn choose_action(state: &CellState, dna: &DNA, config: &AriaConfig) -> CellAction {
        let activation: f32 = state.state[0..4].iter().sum();
        let valence: f32 = state.state[4..8].iter().sum();

        // Divide if energetic enough
        if activation > dna.thresholds[1] && state.energy > config.metabolism.reproduction_threshold {
            return CellAction::Divide;
        }

        // Emit signal
        if activation.abs() > dna.thresholds[3] {
            let signal: [f32; SIGNAL_DIMS] = std::array::from_fn(|i| {
                state.state[i] * dna.reactions[i]
            });
            return CellAction::Signal(signal);
        }

        // Move
        let direction: [f32; POSITION_DIMS] = std::array::from_fn(|i| {
            let state_contribution = state.state[i % STATE_DIMS];
            let dna_contribution = dna.thresholds[i % 4];
            state_contribution * dna_contribution * 0.1 * valence.signum()
        });
        CellAction::Move(direction)
    }
}

impl ComputeBackend for CpuBackend {
    fn init(&mut self, config: &AriaConfig) -> AriaResult<()> {
        self.config = config.clone();
        self.sleep_config = config.activity.clone();
        Ok(())
    }

    fn update_cells(
        &mut self,
        cells: &mut [Cell],
        states: &mut [CellState],
        dna_pool: &[DNA],
        signals: &[SignalFragment],
    ) -> AriaResult<Vec<(u64, CellAction)>> {
        let start = Instant::now();
        let tick = self.tick.fetch_add(1, Ordering::SeqCst);

        // Update spatial hash
        if tick % 10 == 0 {
            self.spatial.clear();
            for (i, state) in states.iter().enumerate() {
                self.spatial.insert(i, &state.position);
            }
        }

        // Count sleeping cells
        let sleeping_count = cells.iter().filter(|c| c.activity.sleeping).count();

        // Process cells in parallel
        let config = &self.config;
        let sleep_config = &self.sleep_config;

        let actions: Vec<(u64, CellAction)> = cells
            .par_iter_mut()
            .zip(states.par_iter_mut())
            .filter_map(|(cell, state)| {
                // Get DNA for this cell
                let dna = dna_pool.get(cell.dna_index as usize)?;

                // Get signals near this cell (simplified - all signals for now)
                // TODO: Use spatial hash for better locality
                let nearby_signals = signals;

                let action = Self::process_cell(
                    cell,
                    state,
                    dna,
                    nearby_signals,
                    config,
                    sleep_config,
                )?;

                if action != CellAction::Rest {
                    Some((cell.id, action))
                } else {
                    None
                }
            })
            .collect();

        // Update stats
        self.stats.cells_processed = (cells.len() - sleeping_count) as u64;
        self.stats.cells_sleeping = sleeping_count as u64;
        self.stats.compute_time_us = start.elapsed().as_micros() as u64;

        Ok(actions)
    }

    fn propagate_signals(
        &mut self,
        states: &[CellState],
        signals: Vec<SignalFragment>,
    ) -> AriaResult<Vec<(usize, SignalFragment)>> {
        if signals.is_empty() {
            return Ok(Vec::new());
        }

        let signal_radius = self.config.signals.signal_radius;

        // For each signal, find cells within radius
        let distributions: Vec<(usize, SignalFragment)> = signals
            .par_iter()
            .flat_map(|signal| {
                // Find cells that should receive this signal
                states
                    .iter()
                    .enumerate()
                    .filter_map(|(i, state)| {
                        // Skip sleeping cells
                        if state.is_sleeping() {
                            return None;
                        }

                        // Calculate distance (simplified - should use signal position)
                        let dist: f32 = state.position
                            .iter()
                            .zip(signal.content.iter().take(POSITION_DIMS))
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt();

                        if dist < signal_radius {
                            // Attenuate by distance
                            let attenuation = 1.0 - (dist / signal_radius);
                            let mut attenuated = *signal;
                            attenuated.intensity *= attenuation;
                            Some((i, attenuated))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        self.stats.signals_propagated = distributions.len() as u64;

        Ok(distributions)
    }

    fn detect_emergence(
        &self,
        _cells: &[Cell],
        states: &[CellState],
        config: &AriaConfig,
    ) -> AriaResult<Vec<Signal>> {
        let tick = self.tick.load(Ordering::SeqCst);

        // Only check at intervals
        if tick % config.emergence.check_interval != 0 {
            return Ok(Vec::new());
        }

        // Find active cells
        let active: Vec<(usize, &CellState)> = states
            .iter()
            .enumerate()
            .filter(|(_i, s)| {
                !s.is_sleeping()
                    && !s.is_dead()
                    && s.activity_level > config.emergence.activation_threshold
            })
            .collect();

        if active.len() < config.emergence.min_active_cells {
            return Ok(Vec::new());
        }

        // Calculate average state
        let mut avg_state = [0.0f32; STATE_DIMS];
        for (_, state) in &active {
            for (i, v) in state.state.iter().enumerate() {
                avg_state[i] += v;
            }
        }
        for v in &mut avg_state {
            *v /= active.len() as f32;
        }

        // Calculate coherence (inverse of variance)
        let variance: f32 = active
            .iter()
            .flat_map(|(_, s)| {
                s.state
                    .iter()
                    .zip(avg_state.iter())
                    .map(|(v, avg)| (v - avg).powi(2))
            })
            .sum::<f32>()
            / (active.len() * STATE_DIMS) as f32;

        let coherence = 1.0 / (1.0 + variance);

        if coherence < config.emergence.coherence_threshold {
            return Ok(Vec::new());
        }

        // Calculate intensity
        let intensity: f32 = avg_state.iter().map(|x| x.abs()).sum::<f32>() / STATE_DIMS as f32;

        if intensity < config.emergence.expression_threshold {
            return Ok(Vec::new());
        }

        // Create emergence signal
        let signal = Signal {
            content: avg_state.to_vec(),
            intensity,
            label: format!("emergence@{}", tick),
            signal_type: SignalType::Expression,
            timestamp: tick,
            position: None,
        };

        Ok(vec![signal])
    }

    fn stats(&self) -> BackendStats {
        self.stats.clone()
    }

    fn sync(&mut self) -> AriaResult<()> {
        // No-op for CPU backend
        Ok(())
    }

    fn name(&self) -> &'static str {
        "CPU (Rayon)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let config = AriaConfig::default();
        let backend = CpuBackend::new(&config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_sparse_updates() {
        let config = AriaConfig::default();
        let mut backend = CpuBackend::new(&config).unwrap();

        // Create some cells
        let mut cells: Vec<Cell> = (0..100).map(|i| Cell::new(i, 0)).collect();
        let mut states: Vec<CellState> = (0..100).map(|_| CellState::new()).collect();
        let dna_pool = vec![DNA::random()];

        // Run several ticks
        for _ in 0..200 {
            let _ = backend.update_cells(&mut cells, &mut states, &dna_pool, &[]);
        }

        // Some cells should be sleeping now
        let sleeping = cells.iter().filter(|c| c.activity.sleeping).count();
        assert!(sleeping > 0, "Expected some cells to be sleeping");
    }
}
