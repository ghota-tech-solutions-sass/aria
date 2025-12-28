# ARIA Architecture

This document describes the technical architecture of ARIA.

## Overview

ARIA consists of two main components that communicate over WebSocket:

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│                 │◄─────────────────────────►│                 │
│   aria-body     │    Signals (JSON)         │   aria-brain    │
│   (Interface)   │                           │   (Substrate)   │
│                 │                           │                 │
└─────────────────┘                           └─────────────────┘
```

## aria-brain

The brain is the computational core where the living substrate exists.

### Components

#### Cell (`cell.rs`)

The fundamental unit of computation. Each cell is a simple agent with:

```rust
struct Cell {
    id: u64,           // Unique identifier
    dna: DNA,          // Genetic code
    position: [f32; 16], // Position in semantic space
    state: [f32; 32],  // Internal activation
    energy: f32,       // Vital energy (0.0 = death)
    tension: f32,      // Desire to act
    age: u64,          // Ticks lived
    connections: Vec<Connection>,
    inbox: Vec<SignalFragment>,
}
```

**DNA Structure:**
- `thresholds[8]`: Control when actions are triggered
- `reactions[8]`: Control response intensity
- `connectivity[4]`: Control connection preferences

**Cell Actions:**
- `Rest`: Do nothing
- `Die`: Cell death (no energy)
- `Divide`: Reproduce (create child cell)
- `Connect`: Create synapse to another cell
- `Signal`: Emit information to nearby cells
- `Move`: Change position in semantic space

#### Substrate (`substrate.rs`)

The universe where cells live. Not a grid - a topological space where distance is semantic.

```rust
struct Substrate {
    cells: DashMap<u64, Cell>,  // All living cells
    attractors: Vec<Attractor>, // Points of interest
    signal_buffer: Vec<Signal>, // Recent external signals
    memory: LongTermMemory,     // Persistent storage
}
```

**Tick Cycle:**
1. Cells receive signals from inbox
2. Each cell lives one tick (parallel)
3. Process cell actions (divide, connect, etc.)
4. Propagate internal signals
5. Detect emergent patterns
6. Apply attractor influence
7. Natural selection

#### Signal (`signal.rs`)

The quantum of information that travels through the system.

```rust
struct Signal {
    content: Vec<f32>,     // Vector representation
    intensity: f32,        // Strength
    label: String,         // Human-readable label
    signal_type: SignalType,
}
```

**Signal Types:**
- `Perception`: External input (from humans)
- `Expression`: Emergent output (from ARIA)
- `Internal`: Between cells

**Text Encoding:**
- Characters 0-19: Character codes normalized
- Position 20: Length feature
- Position 21: Uppercase ratio
- Position 22: Whitespace ratio
- Position 23: Numeric ratio
- Positions 24-27: Punctuation features
- Positions 28-31: Emotional markers

#### Memory (`memory/mod.rs`)

Persistent storage that survives restarts.

```rust
struct LongTermMemory {
    elite_dna: Vec<EliteDNA>,      // Best genetic codes
    learned_patterns: Vec<Pattern>, // Recurring sequences
    associations: Vec<Association>, // Stimulus-response pairs
    memories: Vec<Memory>,          // Emotional moments
    vocabulary: HashMap<String, WordMeaning>,
}
```

**Memory Types:**

1. **Elite DNA**: Preserved from high-performing cells
   - Fitness score
   - Generation (lineage depth)
   - Specialization label

2. **Patterns**: Learned sequences
   - Sequence of vectors
   - Typical response
   - Valence (emotional charge)
   - Frequency count

3. **Associations**: Hebbian learning
   - Stimulus vector
   - Response vector
   - Strength (reinforcement count)

4. **Memories**: Significant moments
   - Trigger (what caused it)
   - Internal state snapshot
   - Emotional intensity
   - Outcome (positive/negative)

### Data Flow

```
External Input
      │
      ▼
┌─────────────┐
│   Signal    │ ◄── from_text()
└─────────────┘
      │
      ▼
┌─────────────┐
│ inject_signal│ ◄── Creates attractor
└─────────────┘      Distributes fragments
      │
      ▼
┌─────────────┐
│   Cells     │ ◄── Receive in inbox
│   process   │     Update state
└─────────────┘     Take actions
      │
      ▼
┌─────────────┐
│  Emergence  │ ◄── Detect synchronized clusters
│  Detection  │     Generate expression
└─────────────┘
      │
      ▼
┌─────────────┐
│   Signal    │ ◄── to_expression()
└─────────────┘
      │
      ▼
External Output
```

## aria-body

The interface for human interaction.

### Modes

1. **Simple Mode**: Text-only chat interface
2. **Visual Mode**: Full TUI with statistics and graphs

### Components

#### Visualizer (`visualizer.rs`)

Real-time TUI using Ratatui.

Displays:
- Energy gauge
- Entropy gauge
- Activity sparkline
- Dominant emotion
- Population graph
- Conversation history
- Input field

## Communication Protocol

### WebSocket Messages

All messages are JSON-encoded.

**Signal Message:**
```json
{
    "content": [0.5, 0.3, ...],
    "intensity": 0.7,
    "label": "hello",
    "signal_type": "Perception",
    "timestamp": 12345
}
```

**Stats Request:**
```json
{
    "type": "stats"
}
```

**Stats Response:**
```json
{
    "tick": 50000,
    "alive_cells": 10234,
    "total_energy": 8543.2,
    "entropy": 0.234,
    "active_clusters": 12,
    "dominant_emotion": "curious",
    "signals_per_second": 45.2,
    "oldest_cell_age": 12000,
    "average_connections": 3.4
}
```

## Semantic Space

Cells exist in a 16-dimensional semantic space. Distance in this space represents conceptual similarity, not physical location.

**Properties:**
- Cells close together share signals more easily
- Attractors pull cells toward concepts
- Movement is influenced by DNA connectivity genes

## Evolution Mechanics

### Natural Selection

Every 100 ticks:
1. Remove cells with zero energy
2. If population < target: reproduce best cells
3. If population > target + buffer: cull weakest

### Reproduction

When a cell divides:
1. Parent energy is halved
2. Child inherits DNA with mutations
3. Child spawns near parent in semantic space
4. Child starts at generation = parent + 1

### Mutation

DNA mutation rate: 10% per gene per reproduction

Mutation amount: ±10% of current value

## Emergence Detection

Emergent signals are generated when:
1. At least 5 cells have activation > 1.0
2. Cluster coherence > 0.5 (low position variance)
3. Average state is computed
4. Signal is created from average

This represents a "thought" - many cells arriving at similar conclusions simultaneously.

## Performance Considerations

### Parallelization

- Cell ticks: Parallel via Rayon
- Signal propagation: Parallel iteration
- Stats calculation: Parallel aggregation

### Memory Management

- DashMap for concurrent cell access
- Circular buffers for histories
- Periodic pruning of weak patterns/memories

### Optimization Opportunities

- GPU acceleration for:
  - Distance calculations
  - Signal propagation
  - State updates
- SIMD for vector operations
- Connection pooling for WebSocket
