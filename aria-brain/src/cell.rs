//! Cell - The fundamental unit of life in ARIA
//!
//! A cell is not a neuron. It's a living entity with:
//! - DNA that defines its behavior
//! - Energy that it needs to survive
//! - Tension that drives it to act
//! - Connections to other cells

use serde::{Deserialize, Serialize};
use rand::Rng;

/// Computational DNA - defines the "character" of a cell
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DNA {
    /// Genes controlling thresholds (when to act)
    pub thresholds: [f32; 8],
    /// Genes controlling reactions (how to respond)
    pub reactions: [f32; 8],
    /// Genes controlling connectivity preferences
    pub connectivity: [f32; 4],
    /// Unique signature
    pub signature: u64,
}

impl DNA {
    /// Create random DNA
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            thresholds: std::array::from_fn(|_| rng.gen()),
            reactions: std::array::from_fn(|_| rng.gen()),
            connectivity: std::array::from_fn(|_| rng.gen()),
            signature: rng.gen(),
        }
    }

    /// Mutate the DNA with a given rate
    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();

        for t in &mut self.thresholds {
            if rng.gen::<f32>() < rate {
                *t = (*t + rng.gen::<f32>() * 0.2 - 0.1).clamp(0.0, 1.0);
            }
        }

        for r in &mut self.reactions {
            if rng.gen::<f32>() < rate {
                *r = (*r + rng.gen::<f32>() * 0.2 - 0.1).clamp(0.0, 1.0);
            }
        }

        for c in &mut self.connectivity {
            if rng.gen::<f32>() < rate {
                *c = (*c + rng.gen::<f32>() * 0.2 - 0.1).clamp(0.0, 1.0);
            }
        }
    }

    /// Sexual reproduction with another DNA (crossover + mutation)
    pub fn reproduce_with(&self, other: &DNA) -> DNA {
        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(0..8);

        let mut child = self.clone();

        // Crossover
        for i in crossover_point..8 {
            child.thresholds[i] = other.thresholds[i];
            child.reactions[i] = other.reactions[i];
        }

        // New signature
        child.signature = rng.gen();

        // Slight mutation
        child.mutate(0.05);

        child
    }

    /// Calculate similarity with another DNA (0.0 = different, 1.0 = identical)
    pub fn similarity(&self, other: &DNA) -> f32 {
        let threshold_sim: f32 = self.thresholds.iter()
            .zip(other.thresholds.iter())
            .map(|(a, b)| 1.0 - (a - b).abs())
            .sum::<f32>() / 8.0;

        let reaction_sim: f32 = self.reactions.iter()
            .zip(other.reactions.iter())
            .map(|(a, b)| 1.0 - (a - b).abs())
            .sum::<f32>() / 8.0;

        (threshold_sim + reaction_sim) / 2.0
    }
}

/// A living cell
#[derive(Clone)]
pub struct Cell {
    /// Unique identifier
    pub id: u64,
    /// Genetic code
    pub dna: DNA,

    /// Position in semantic space (not geometric!)
    pub position: [f32; 16],

    /// Internal state - activation vector
    pub state: [f32; 32],

    /// Vital energy - without energy, the cell dies
    pub energy: f32,

    /// Tension - desire to act
    pub tension: f32,

    /// Age in ticks
    pub age: u64,

    /// Generation (how many ancestors)
    pub generation: u64,

    /// Connections to other cells
    pub connections: Vec<Connection>,

    /// Incoming signal buffer
    pub inbox: Vec<SignalFragment>,
}

/// A connection to another cell
#[derive(Clone, Debug)]
pub struct Connection {
    pub target_id: u64,
    pub weight: f32,
    pub plasticity: f32, // How much this connection can change
    pub last_used: u64,
}

/// A fragment of a signal received from another cell
#[derive(Clone, Debug)]
pub struct SignalFragment {
    pub source_id: u64,
    pub content: [f32; 8],
    pub intensity: f32,
}

/// What a cell can do
#[derive(Clone, Debug)]
pub enum CellAction {
    /// Do nothing
    Rest,
    /// Die (no more energy)
    Die,
    /// Reproduce (divide)
    Divide,
    /// Create a connection
    Connect,
    /// Emit a signal
    Signal([f32; 8]),
    /// Move in semantic space
    Move([f32; 16]),
}

impl Cell {
    /// Create a new cell with random DNA
    pub fn new(id: u64) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            id,
            dna: DNA::random(),
            position: std::array::from_fn(|_| rng.gen::<f32>() * 2.0 - 1.0),
            state: [0.0; 32],
            energy: 1.0,
            tension: 0.0,
            age: 0,
            generation: 0,
            connections: Vec::new(),
            inbox: Vec::new(),
        }
    }

    /// Create a child cell from a parent
    pub fn from_parent(id: u64, parent: &Cell) -> Self {
        let mut rng = rand::thread_rng();
        let mut child = Self::new(id);

        child.dna = parent.dna.clone();
        child.dna.mutate(0.1);
        child.generation = parent.generation + 1;

        // Slight position variation from parent
        for (i, p) in parent.position.iter().enumerate() {
            child.position[i] = p + rng.gen::<f32>() * 0.2 - 0.1;
        }

        child.energy = 0.5; // Child starts with half energy

        child
    }

    /// The heart of life: one tick of existence
    pub fn live(&mut self) -> CellAction {
        self.age += 1;

        // 1. Consume energy to live (metabolism)
        self.energy -= 0.001;

        // 2. Process received signals
        self.process_inbox();

        // 3. Tension builds up over time and with stimuli
        self.tension += 0.01 + self.inbox.len() as f32 * 0.05;

        // 4. Decay unused connections
        self.decay_connections();

        // 5. Decide what to do
        if self.energy <= 0.0 {
            return CellAction::Die;
        }

        // Action threshold depends on DNA
        let action_threshold = self.dna.thresholds[0] * 2.0 + 0.5;

        if self.tension > action_threshold {
            self.tension = 0.0; // Release tension
            return self.choose_action();
        }

        CellAction::Rest
    }

    fn process_inbox(&mut self) {
        for signal in self.inbox.drain(..) {
            // Integrate signal into internal state
            for (i, s) in signal.content.iter().enumerate() {
                if i < 8 {
                    let reaction = self.dna.reactions[i];
                    self.state[i] += s * signal.intensity * reaction;
                }
            }

            // Gain a bit of energy from interactions
            self.energy += signal.intensity * 0.01;

            // Strengthen connection if it exists
            for conn in &mut self.connections {
                if conn.target_id == signal.source_id {
                    conn.weight = (conn.weight + 0.1 * conn.plasticity).min(1.0);
                    conn.last_used = self.age;
                    break;
                }
            }
        }

        // Normalize state to prevent explosion
        let norm: f32 = self.state.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1.0 {
            for s in &mut self.state {
                *s /= norm;
            }
        }
    }

    fn decay_connections(&mut self) {
        // Connections that aren't used weaken over time
        for conn in &mut self.connections {
            if self.age - conn.last_used > 1000 {
                conn.weight *= 0.99;
            }
        }

        // Remove very weak connections
        self.connections.retain(|c| c.weight > 0.01);
    }

    fn choose_action(&self) -> CellAction {
        // The action depends on state and DNA
        let activation: f32 = self.state[0..4].iter().sum();
        let valence: f32 = self.state[4..8].iter().sum(); // Positive or negative

        // Enough energy and motivated: reproduce
        if activation > self.dna.thresholds[1] && self.energy > 0.6 {
            return CellAction::Divide;
        }

        // Want to connect
        if activation > self.dna.thresholds[2] && self.connections.len() < 10 {
            return CellAction::Connect;
        }

        // Emit a signal
        if activation.abs() > self.dna.thresholds[3] {
            let signal = std::array::from_fn(|i| {
                self.state[i] * self.dna.reactions[i]
            });
            return CellAction::Signal(signal);
        }

        // Move in semantic space
        let direction: [f32; 16] = std::array::from_fn(|i| {
            let state_contribution = self.state[i % 32];
            let dna_contribution = self.dna.connectivity[i % 4];
            state_contribution * dna_contribution * 0.1 * valence.signum()
        });
        CellAction::Move(direction)
    }

    /// Calculate semantic distance to a position
    pub fn distance_to(&self, other: &[f32; 16]) -> f32 {
        self.position.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Receive a signal fragment
    pub fn receive(&mut self, fragment: SignalFragment) {
        self.inbox.push(fragment);
    }

    /// Get the cell's current "emotion" based on state
    pub fn emotion(&self) -> Emotion {
        let activation: f32 = self.state[0..4].iter().sum();
        let valence: f32 = self.state[4..8].iter().sum();

        if activation.abs() < 0.2 {
            Emotion::Calm
        } else if activation > 0.5 && valence > 0.0 {
            Emotion::Excited
        } else if activation > 0.5 && valence < 0.0 {
            Emotion::Frustrated
        } else if valence > 0.3 {
            Emotion::Content
        } else if valence < -0.3 {
            Emotion::Distressed
        } else {
            Emotion::Curious
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Emotion {
    Calm,
    Curious,
    Excited,
    Content,
    Frustrated,
    Distressed,
}

impl std::fmt::Display for Emotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Emotion::Calm => write!(f, "calm"),
            Emotion::Curious => write!(f, "curious"),
            Emotion::Excited => write!(f, "excited"),
            Emotion::Content => write!(f, "content"),
            Emotion::Frustrated => write!(f, "frustrated"),
            Emotion::Distressed => write!(f, "distressed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_creation() {
        let cell = Cell::new(1);
        assert_eq!(cell.id, 1);
        assert_eq!(cell.energy, 1.0);
        assert_eq!(cell.age, 0);
    }

    #[test]
    fn test_cell_lives() {
        let mut cell = Cell::new(1);
        let action = cell.live();
        assert_eq!(cell.age, 1);
        assert!(cell.energy < 1.0);
    }

    #[test]
    fn test_dna_mutation() {
        let mut dna = DNA::random();
        let original = dna.clone();
        dna.mutate(1.0); // 100% mutation rate
        // At least some values should be different
        assert!(dna.thresholds != original.thresholds || dna.reactions != original.reactions);
    }

    #[test]
    fn test_dna_reproduction() {
        let parent1 = DNA::random();
        let parent2 = DNA::random();
        let child = parent1.reproduce_with(&parent2);
        // Child should be different from both parents
        assert_ne!(child.signature, parent1.signature);
        assert_ne!(child.signature, parent2.signature);
    }
}
