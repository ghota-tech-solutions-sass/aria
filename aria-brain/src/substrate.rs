//! Substrate - The universe where cells live
//!
//! The substrate is not a grid. It's a topological space where
//! distances are semantic, not geometric.

use crate::cell::{Cell, CellAction, SignalFragment, Emotion};
use crate::signal::Signal;
use crate::memory::LongTermMemory;

use dashmap::DashMap;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// The living substrate
pub struct Substrate {
    /// All living cells
    cells: DashMap<u64, Cell>,

    /// Next cell ID
    next_id: AtomicU64,

    /// Current tick
    tick: AtomicU64,

    /// Attractors in semantic space
    attractors: RwLock<Vec<Attractor>>,

    /// Recent signals for pattern detection
    signal_buffer: RwLock<Vec<Signal>>,

    /// Long-term memory reference
    memory: Arc<RwLock<LongTermMemory>>,

    /// Global energy available
    global_energy: AtomicU64,
}

/// An attractor in semantic space
#[derive(Clone, Debug)]
pub struct Attractor {
    pub position: [f32; 16],
    pub strength: f32,
    pub label: String,
    pub created_at: u64,
}

/// Statistics about the substrate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubstrateStats {
    pub tick: u64,
    pub alive_cells: usize,
    pub total_energy: f32,
    pub entropy: f32,
    pub active_clusters: usize,
    pub dominant_emotion: String,
    pub signals_per_second: f32,
    pub oldest_cell_age: u64,
    pub average_connections: f32,
}

impl Substrate {
    /// Create a new substrate with initial cells
    pub fn new(initial_cells: usize, memory: Arc<RwLock<LongTermMemory>>) -> Self {
        let cells = DashMap::new();

        // Create primordial cells
        for i in 0..initial_cells {
            cells.insert(i as u64, Cell::new(i as u64));
        }

        // Check if we have elite DNA to seed from memory
        {
            let mem = memory.read();
            if !mem.elite_dna.is_empty() {
                tracing::info!("Seeding {} cells with elite DNA from memory", mem.elite_dna.len().min(100));
                for (i, elite) in mem.elite_dna.iter().take(100).enumerate() {
                    if let Some(mut cell) = cells.get_mut(&(i as u64)) {
                        cell.dna = elite.dna.clone();
                    }
                }
            }
        }

        Self {
            cells,
            next_id: AtomicU64::new(initial_cells as u64),
            tick: AtomicU64::new(0),
            attractors: RwLock::new(Vec::new()),
            signal_buffer: RwLock::new(Vec::new()),
            memory,
            global_energy: AtomicU64::new(10000),
        }
    }

    /// Inject an external signal (perception)
    pub fn inject_signal(&self, signal: Signal) {
        // Transform signal to fragments for cells
        let fragment = SignalFragment {
            source_id: 0, // External source
            content: signal.to_vector(),
            intensity: signal.intensity,
        };

        // Get semantic position of the signal
        let target_position = signal.semantic_position();

        // Distribute to ALL cells (external signals are broadcast)
        // Intensity decreases with semantic distance
        self.cells.iter_mut().for_each(|mut entry| {
            let cell = entry.value_mut();
            let distance = semantic_distance(&cell.position, &target_position);

            // All cells receive external signals, but with distance-based attenuation
            let mut attenuated_fragment = fragment.clone();
            attenuated_fragment.intensity = fragment.intensity / (1.0 + distance * 0.5);

            cell.receive(attenuated_fragment);

            // External signals also give energy boost (attention/arousal)
            cell.energy = (cell.energy + 0.01 * fragment.intensity).min(1.5);
        });

        // Create a temporary attractor
        {
            let mut attractors = self.attractors.write();
            attractors.push(Attractor {
                position: target_position,
                strength: signal.intensity,
                label: signal.label.clone(),
                created_at: self.tick.load(Ordering::Relaxed),
            });
        }

        // Store in signal buffer
        {
            let mut buffer = self.signal_buffer.write();
            buffer.push(signal);
            if buffer.len() > 1000 {
                buffer.remove(0);
            }
        }
    }

    /// One tick of life
    pub fn tick(&self) -> Vec<Signal> {
        let current_tick = self.tick.fetch_add(1, Ordering::SeqCst);

        let mut new_cells: Vec<Cell> = Vec::new();
        let mut dead_cells: Vec<u64> = Vec::new();
        let mut connection_requests: Vec<(u64, u64)> = Vec::new();
        let mut emitted_signals: Vec<([f32; 8], [f32; 16], f32)> = Vec::new();

        // Phase 1: Each cell lives (parallel)
        self.cells.iter_mut().for_each(|mut entry| {
            let cell = entry.value_mut();
            let action = cell.live();

            match action {
                CellAction::Die => {
                    // Will be removed later
                }
                CellAction::Divide => {
                    // Mark for division
                }
                CellAction::Connect => {
                    // Will create connection later
                }
                CellAction::Signal(content) => {
                    // Collect emitted signals
                }
                CellAction::Move(direction) => {
                    // Apply movement
                    for (i, d) in direction.iter().enumerate() {
                        cell.position[i] = (cell.position[i] + d).clamp(-10.0, 10.0);
                    }
                }
                CellAction::Rest => {}
            }
        });

        // Phase 2: Process actions (sequential for consistency)
        let actions: Vec<(u64, CellAction)> = self.cells.iter()
            .map(|entry| {
                let cell = entry.value();
                let action = if cell.energy <= 0.0 {
                    CellAction::Die
                } else if cell.tension == 0.0 && cell.energy > 0.6 && cell.age % 100 == 0 {
                    CellAction::Divide
                } else {
                    CellAction::Rest
                };
                (*entry.key(), action)
            })
            .collect();

        for (cell_id, action) in actions {
            match action {
                CellAction::Die => {
                    dead_cells.push(cell_id);
                }
                CellAction::Divide => {
                    if let Some(parent) = self.cells.get(&cell_id) {
                        if parent.energy > 0.6 {
                            let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                            let child = Cell::from_parent(new_id, &parent);
                            new_cells.push(child);
                        }
                    }
                }
                _ => {}
            }
        }

        // Remove dead cells
        for id in dead_cells {
            self.cells.remove(&id);
        }

        // Add new cells
        for cell in new_cells {
            self.cells.insert(cell.id, cell);
        }

        // Phase 3: Propagate signals between cells
        self.propagate_internal_signals();

        // Phase 4: Detect emergent patterns
        let emergent = self.detect_emergence(current_tick);

        // Phase 5: Decay attractors
        {
            let mut attractors = self.attractors.write();
            attractors.retain_mut(|a| {
                a.strength *= 0.99;
                a.strength > 0.01
            });
        }

        // Phase 6: Apply attractor influence
        self.apply_attractors();

        // Phase 7: Maintain population (natural selection) - run frequently
        if current_tick % 10 == 0 {
            self.natural_selection();
        }

        emergent
    }

    fn propagate_internal_signals(&self) {
        // Collect signals from active cells
        let signals: Vec<(u64, [f32; 16], [f32; 8], f32)> = self.cells.iter()
            .filter_map(|entry| {
                let cell = entry.value();
                let activation: f32 = cell.state.iter().map(|x| x.abs()).sum();
                if activation > 0.5 {
                    let content: [f32; 8] = std::array::from_fn(|i| cell.state[i]);
                    Some((cell.id, cell.position, content, activation))
                } else {
                    None
                }
            })
            .collect();

        // Distribute to nearby cells
        for (source_id, source_pos, content, intensity) in signals {
            self.cells.iter_mut().for_each(|mut entry| {
                let cell = entry.value_mut();
                if cell.id != source_id {
                    let distance = semantic_distance(&cell.position, &source_pos);
                    if distance < 2.0 {
                        cell.receive(SignalFragment {
                            source_id,
                            content,
                            intensity: intensity / (1.0 + distance),
                        });
                    }
                }
            });
        }
    }

    fn detect_emergence(&self, current_tick: u64) -> Vec<Signal> {
        // Only check for emergence every 50 ticks (throttle output)
        if current_tick % 50 != 0 {
            return Vec::new();
        }

        // Find cells with any meaningful activation (lowered threshold)
        let active_cells: Vec<_> = self.cells.iter()
            .filter(|entry| {
                let cell = entry.value();
                cell.state.iter().map(|x| x.abs()).sum::<f32>() > 0.1
            })
            .take(1000) // Limit for performance
            .collect();

        // Need at least a few active cells
        if active_cells.len() < 3 {
            return Vec::new();
        }

        // Calculate average state
        let mut average_state = [0.0f32; 8];
        for entry in &active_cells {
            for (i, s) in entry.value().state[0..8].iter().enumerate() {
                average_state[i] += s;
            }
        }
        let n = active_cells.len() as f32;
        for a in &mut average_state {
            *a /= n;
        }

        // Check coherence (lowered threshold for baby ARIA)
        let coherence = self.calculate_cluster_coherence(&active_cells);

        if coherence > 0.1 {
            // This is an emergent thought!
            let mut signal = Signal::from_vector(average_state, format!("emergence@{}", current_tick));
            signal.intensity = coherence;

            // Learn this pattern
            {
                let mut memory = self.memory.write();
                memory.learn_pattern(
                    vec![average_state],
                    average_state,
                    coherence
                );
            }

            vec![signal]
        } else {
            Vec::new()
        }
    }

    fn calculate_cluster_coherence(&self, cells: &[dashmap::mapref::multiple::RefMulti<u64, Cell>]) -> f32 {
        if cells.len() < 2 {
            return 0.0;
        }

        // Calculate variance of positions
        let mut mean_pos = [0.0f32; 16];
        for entry in cells {
            for (i, p) in entry.value().position.iter().enumerate() {
                mean_pos[i] += p;
            }
        }
        let n = cells.len() as f32;
        for p in &mut mean_pos {
            *p /= n;
        }

        let variance: f32 = cells.iter()
            .map(|entry| semantic_distance(&entry.value().position, &mean_pos).powi(2))
            .sum::<f32>() / n;

        // Low variance = high coherence
        (1.0 / (1.0 + variance)).min(1.0)
    }

    fn apply_attractors(&self) {
        let attractors = self.attractors.read().clone();

        if attractors.is_empty() {
            return;
        }

        self.cells.iter_mut().for_each(|mut entry| {
            let cell = entry.value_mut();

            for attractor in &attractors {
                let distance = semantic_distance(&cell.position, &attractor.position);
                if distance > 0.1 && distance < 5.0 {
                    // Move toward attractor
                    let pull = attractor.strength / (distance * distance);
                    for (i, (p, a)) in cell.position.iter_mut().zip(attractor.position.iter()).enumerate() {
                        *p += (a - *p) * pull * 0.01 * cell.dna.connectivity[i % 4];
                    }
                }
            }
        });
    }

    fn natural_selection(&self) {
        let mut rng = rand::thread_rng();

        // Remove cells with no energy
        self.cells.retain(|_, cell| cell.energy > 0.0);

        let current_pop = self.cells.len();
        let target_pop = 10_000;

        if current_pop < target_pop {
            // If population is very low, create new primordial cells
            if current_pop < 100 {
                let cells_to_create = (target_pop / 10).min(1000);
                for _ in 0..cells_to_create {
                    let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                    self.cells.insert(new_id, Cell::new(new_id));
                }
                return;
            }

            // Reproduce the best performers (lowered threshold)
            let best_cells: Vec<_> = self.cells.iter()
                .filter(|e| e.value().energy > 0.3)
                .take(100)
                .map(|e| e.value().clone())
                .collect();

            for cell in best_cells {
                if self.cells.len() >= target_pop {
                    break;
                }

                let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                let child = Cell::from_parent(new_id, &cell);
                self.cells.insert(new_id, child);

                // Save elite DNA periodically
                if rng.gen::<f32>() < 0.01 {
                    let mut memory = self.memory.write();
                    memory.preserve_elite(
                        cell.dna.clone(),
                        cell.energy,
                        cell.generation,
                        "survivor"
                    );
                }
            }
        } else if current_pop > target_pop + 1000 {
            // Cull the weakest
            let weak_ids: Vec<u64> = self.cells.iter()
                .filter(|e| e.value().energy < 0.3)
                .take(current_pop - target_pop)
                .map(|e| *e.key())
                .collect();

            for id in weak_ids {
                self.cells.remove(&id);
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> SubstrateStats {
        let alive = self.cells.len();

        let total_energy: f32 = self.cells.iter()
            .map(|e| e.value().energy)
            .sum();

        let entropy = self.calculate_entropy();

        let oldest = self.cells.iter()
            .map(|e| e.value().age)
            .max()
            .unwrap_or(0);

        let total_connections: usize = self.cells.iter()
            .map(|e| e.value().connections.len())
            .sum();

        let avg_connections = if alive > 0 {
            total_connections as f32 / alive as f32
        } else {
            0.0
        };

        // Count clusters (simplified: cells with high activity)
        let active_clusters = self.cells.iter()
            .filter(|e| e.value().state.iter().map(|x| x.abs()).sum::<f32>() > 1.0)
            .count() / 10;

        // Dominant emotion
        let emotions: Vec<Emotion> = self.cells.iter()
            .take(100)
            .map(|e| e.value().emotion())
            .collect();

        let dominant_emotion = self.most_common_emotion(&emotions);

        SubstrateStats {
            tick: self.tick.load(Ordering::Relaxed),
            alive_cells: alive,
            total_energy,
            entropy,
            active_clusters: active_clusters.max(1),
            dominant_emotion: dominant_emotion.to_string(),
            signals_per_second: self.signal_buffer.read().len() as f32 / 10.0,
            oldest_cell_age: oldest,
            average_connections: avg_connections,
        }
    }

    fn calculate_entropy(&self) -> f32 {
        let states: Vec<f32> = self.cells.iter()
            .take(1000)
            .flat_map(|e| e.value().state.to_vec())
            .collect();

        if states.is_empty() {
            return 0.0;
        }

        let mean: f32 = states.iter().sum::<f32>() / states.len() as f32;
        let variance: f32 = states.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / states.len() as f32;

        variance.sqrt()
    }

    fn most_common_emotion(&self, emotions: &[Emotion]) -> Emotion {
        use std::collections::HashMap;
        let mut counts: HashMap<Emotion, usize> = HashMap::new();
        for e in emotions {
            *counts.entry(*e).or_insert(0) += 1;
        }
        counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(emotion, _)| emotion)
            .unwrap_or(Emotion::Calm)
    }
}

/// Calculate semantic distance between two positions
fn semantic_distance(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_creation() {
        let memory = Arc::new(RwLock::new(LongTermMemory::new()));
        let substrate = Substrate::new(100, memory);
        assert_eq!(substrate.stats().alive_cells, 100);
    }

    #[test]
    fn test_signal_injection() {
        let memory = Arc::new(RwLock::new(LongTermMemory::new()));
        let substrate = Substrate::new(100, memory);

        let signal = Signal::from_text("Hello");
        substrate.inject_signal(signal);

        // Attractors should be created
        assert!(!substrate.attractors.read().is_empty());
    }

    #[test]
    fn test_tick() {
        let memory = Arc::new(RwLock::new(LongTermMemory::new()));
        let substrate = Substrate::new(100, memory);

        let signals = substrate.tick();
        assert_eq!(substrate.stats().tick, 1);
    }
}
