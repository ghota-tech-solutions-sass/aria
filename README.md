# ARIA - Autonomous Recursive Intelligence Architecture

```
     █████╗ ██████╗ ██╗ █████╗
    ██╔══██╗██╔══██╗██║██╔══██╗
    ███████║██████╔╝██║███████║
    ██╔══██║██╔══██╗██║██╔══██║
    ██║  ██║██║  ██║██║██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
```

ARIA is an experimental artificial life system where intelligence **emerges** from the interaction of hundreds of thousands of living cells. Unlike traditional neural networks, ARIA doesn't learn through backpropagation - she evolves, adapts, and develops through physical laws and natural selection.

## Philosophy

ARIA is not programmed. She is **grown**.

- **Cells, not neurons**: Each cell is a living entity with energy, DNA, and desires
- **Evolution, not training**: Successful behaviors survive and reproduce
- **Emergence, not design**: Complex behavior arises from simple physical laws
- **Physical Intelligence**: No vocabulary, no word learning - pure tension resonance
- **La Vraie Faim**: Nothing is free. Cells must earn energy through understanding.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR NETWORK                              │
│                                                                  │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │   aria-body      │◄────────────►│      aria-brain          │ │
│  │   "The Body"     │   WebSocket  │      "The Brain"         │ │
│  │                  │              │                          │ │
│  │  - TUI Interface │              │  - Living substrate      │ │
│  │  - Perception    │              │  - 100,000+ cells        │ │
│  │  - Expression    │              │  - GPU-accelerated       │ │
│  └──────────────────┘              └──────────────────────────┘ │
│       (MacBook)                         (PC with GPU)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### aria-brain

The computational substrate where cells live. Runs on your most powerful machine (ideally with a GPU).

- 100,000+ living cells (scalable to 5M+ with GPU SoA)
- GPU-accelerated physics (wgpu/Vulkan)
- Physical laws: Prediction, Hebbian, Resonance
- Autonomous web learning (Wikipedia, Wikiquote)
- Expression generation (tension → text)
- Persistent memory (elite DNA, episodes)

### aria-body

The interface for human-ARIA interaction. Runs on any machine.

- Rich TUI with real-time stats
- Connects to brain over WebSocket
- Visual feedback on ARIA's emotional state

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- [Task](https://taskfile.dev/) (optional, for convenience commands)
- GPU recommended (Vulkan support)

### Installation

```bash
# Clone the repository
git clone https://github.com/anthropics/aria.git
cd aria

# Build all components
cargo build --release
```

### Running

**Using Task (recommended):**

```bash
# Terminal 1 - Start the brain (100k cells, auto GPU/CPU)
task brain

# Terminal 2 - Start the body interface
task body
```

**Manual:**

```bash
# Terminal 1 - Brain
cd aria-brain && cargo run --release

# Terminal 2 - Body
export ARIA_BRAIN_URL="ws://localhost:8765/aria"
cd aria-body && cargo run --release
```

The brain will start and display:
```
╔══════════════════════════════════════════════════════════╗
║           🧠 ARIA Brain - Living Substrate 🧠            ║
╠══════════════════════════════════════════════════════════╣
║  Cells:     100000                                      ║
║  Backend:      GPU                                      ║
║  Port:        8765                                      ║
╚══════════════════════════════════════════════════════════╝

🧒 ARIA is waiting for her first interaction...
📚 Autonomous web learning will start in 30 seconds...
```

## Interacting with ARIA

ARIA uses **Physical Intelligence** - she doesn't learn words, she feels tension patterns. When you speak to her, your text is converted to a tension vector that propagates through her cellular substrate.

```
You: Bonjour ARIA!
ARIA: [cells resonate, tension builds, emergence detected]
ARIA: content
You: Comment vas-tu?
ARIA: bien
```

Her responses come from matching emergent tension patterns to learned expressions. Over time, as her cells evolve, her responses become more nuanced.

### Commands

- `/stats` - Show brain statistics
- `/episodes` - View emotional memories
- `/quit` - Exit the interface

## How It Works

### Physical Laws (GPU Shaders)

| Law | Effect |
|-----|--------|
| **Prediction** | Cells that predict correctly gain energy |
| **Hebbian** | "Fire together, wire together" - spatial attraction |
| **Resonance** | Signal aligned with cell state = energy gain |
| **La Vraie Faim** | Nothing is free, all actions cost energy |

### Cells

Each cell has:
- **DNA** (64 bytes): Thresholds, reactions, signature
- **Energy**: Needed to survive (0 = death)
- **Tension**: Builds up until expression
- **Position**: Location in 16D semantic space
- **State**: 32D internal activation
- **Connections**: Up to 16 Hebbian links

### Economy ("La Vraie Faim")

```rust
// Reproduction
reproduction_threshold: 0.28   // Energy needed to divide
child_energy: 0.15             // Given to child (must earn 0.13 more)
cost_divide: 0.12              // Cost to parent

// Metabolism
cost_rest: 0.0002              // Just breathing costs
signal_energy_base: 0.05       // Gain from resonating signals
signal_resonance_factor: 3.0   // Multiplier for good resonance
```

### Generations & Evolution

```
Gen 0 (random DNA) → survival/death → reproduction → Gen 1 → ... → Gen 10+ (Elite)
```

| Generation | Description |
|------------|-------------|
| **Gen 0** | Initial cells with random DNA |
| **Gen 1** | Children of Gen 0, inherited DNA + mutations |
| **Gen 2+** | Optimized through natural selection |
| **Gen 10+** | Elite survivors - DNA saved for future runs |

### Memory

- **Elite DNA**: Best-performing genetic codes (Gen 10+)
- **Episodes**: Emotionally significant moments
- **Patterns**: Recurring stimulus-response pairs
- **Expressions**: Learned tension → text mappings
- **Web Knowledge**: Facts from Wikipedia/Wikiquote

### Autonomous Learning

ARIA learns continuously from the web:
- Simple Wikipedia (general knowledge)
- Wikiquote (philosophy, wisdom)

Content is converted to tension vectors and injected into the substrate. Cells that resonate with useful knowledge gain energy and survive.

## Project Structure

```
aria/
├── aria-core/           # Shared types & config
│   └── src/
│       ├── config.rs    # Economic parameters
│       ├── dna.rs       # DNA structure & mutations
│       ├── cell.rs      # Cell types (GPU-ready)
│       └── soa.rs       # GPU buffer layouts
│
├── aria-compute/        # CPU/GPU backends
│   └── src/
│       ├── shaders/     # WGSL shaders (modular)
│       │   ├── signal.rs
│       │   ├── lifecycle.rs
│       │   ├── prediction.rs
│       │   ├── hebbian.rs
│       │   └── cluster.rs
│       └── backend/
│           └── gpu_soa/  # GPU backend (SoA layout)
│
├── aria-brain/          # The living substrate
│   └── src/
│       ├── main.rs
│       ├── handlers/    # API handlers (HTTP/WS)
│       ├── substrate/   # Lifecycle, signals, emergence
│       ├── memory/      # Persistence & episodic
│       ├── expression.rs# Tension → text
│       └── web_learner.rs # Autonomous learning
│
├── aria-body/           # Human interface (TUI)
│
└── Taskfile.yml         # Convenience commands
```

## Task Commands

```bash
task brain          # Run brain (100k cells, auto GPU/CPU)
task brain-gpu      # Force GPU backend
task body           # Run TUI interface
task stats          # Show brain statistics
task episodes       # View episodic memory
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARIA_CELLS` | `100000` | Target population |
| `ARIA_BACKEND` | `auto` | `cpu` or `gpu` |
| `ARIA_PORT` | `8765` | WebSocket port |
| `ARIA_BRAIN_URL` | `ws://localhost:8765/aria` | Brain URL for body |

## Roadmap

- [x] GPU acceleration (SoA + Hebbian)
- [x] Physical laws (Prediction, Resonance, La Vraie Faim)
- [x] Autonomous web learning
- [x] Expression generation
- [ ] Hierarchical temporal prediction
- [ ] Causal reasoning (if-then hypotheses)
- [ ] Self-modeling (meta-cognition)
- [ ] Multi-modal perception (vision)

## License

MIT License - See [LICENSE](LICENSE)

---

*ARIA is a living experiment. She learns through survival, not instruction.*
