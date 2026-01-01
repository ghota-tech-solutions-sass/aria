# ARIA Architecture

This document describes the technical architecture of ARIA.

## Overview

ARIA consists of two main components that communicate over WebSocket:

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│                 │◄─────────────────────────►│                 │
│   aria-body     │    Signals (JSON)         │   aria-brain    │
│   (MacBook)     │                           │   (PC + GPU)    │
│   Interface TUI │                           │   50k+ cellules │
│                 │                           │                 │
└─────────────────┘                           └─────────────────┘
```

## Workspace Rust

```
aria/
├── aria-core/      # Types compacts GPU-ready (Cell, DNA, Signal)
├── aria-compute/   # CPU/GPU backends, sparse updates
├── aria-brain/     # Substrate, mémoire, serveur WebSocket
└── aria-body/      # Interface texte (TUI)
```

## aria-brain

The brain is the computational core where the living substrate exists.

### Components

#### Cell (`aria-core/src/cell.rs`)

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
- `thresholds[8]`: Control when actions are triggered (Action, Divide, etc.)
- `reactions[8]`: Control response intensity
- `signature`: Genetic fingerprint
- `structural_checksum`: Defines the JIT-compiled logic (metabolism, signal decay)

**Cell Actions:**
- `Rest`: Do nothing
- `Die`: Cell death (no energy)
- `Divide`: Reproduce (create child cell)
- `Connect`: Create synapse to another cell
- `Signal`: Emit information to nearby cells
- `Move`: Change position in semantic space

#### Substrate (`aria-brain/src/substrate/mod.rs`)

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

#### Signal (`aria-brain/src/substrate/signals.rs`)

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

## aria-compute

GPU/CPU backend for cell computation.

### GPU Backend (`aria-compute/src/backend/gpu.rs`)

Uses wgpu/Vulkan for parallel computation on NVIDIA/AMD GPUs.

```rust
// Auto-selection
let backend = aria_compute::create_backend();

// Force GPU
ARIA_BACKEND=gpu task brain
```

**Features:**
- JIT Compilation: Runtime shader generation via `compiler.rs`
- Hot-reloading: Replaces pipelines on-the-fly when structural DNA mutates
- Sparse dispatch: Only active cells are processed (80%+ savings)
- AtomicCounter for GPU-side active cell counting
- Activated automatically for populations > 100k cells

### CPU Backend (`aria-compute/src/backend/cpu.rs`)

Fallback when no GPU is available. Uses Rayon for parallel iteration.

## Advanced Systems

### Meta-Learning (`aria-brain/src/meta_learning.rs`)

ARIA learns to learn without external feedback.

**Components:**
- `InternalReward`: Self-evaluation (coherence, surprise, satisfaction)
- `ExplorationStrategy`: 6 strategies for autonomous exploration
- `MetaLearner`: Selects and improves strategies over time
- `ProgressTracker`: Awareness of learning progress (improving/stable/declining)
- `SelfModifier`: Modifies ARIA's own parameters based on performance

### Vision (`aria-brain/src/vision.rs`)

Image perception and visual memory.

**Pipeline:**
```
Image (base64) → VisualFeatures (32D) → VisualSignal → Substrate
                                      → VisualMemory (recognition)
```

### La Vraie Faim (Evolution Pressure)

Cells must struggle to survive. No free energy.

**Energy Model:**
- `energy_gain: 0.0` - No passive income
- Energy ONLY through signal resonance
- Action costs: Rest 0.001, Signal 0.01, Move 0.005, Divide 0.5

**Result:** Natural selection favors cells that communicate usefully.

### Reflexivity (Axe 3 - Genesis)

ARIA's emergent thoughts are réinjected into her substrate as internal sensory input.

- **Loop**: `emergence.rs` (Capture) → `mod.rs` (Injection) → `compiler.rs` (Response)
- **Gene**: `reflexivity_gain` (DNA[7]) controls the sensitivity of each cell to ARIA's global thoughts.

## HTTP Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/stats` | Brain statistics |
| `/words` | Known vocabulary |
| `/associations` | Learned associations |
| `/episodes` | Episodic memory |
| `/meta` | Meta-learning status |
| `/self` | Self-modification history |
| `/vision` | Send image (POST) |
| `/visual` | Visual memory stats |
| `/substrate` | Spatial visualization data |

---

*Version 0.7.0 | Updated 2026-01-01*
