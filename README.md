# ARIA - Autonomous Recursive Intelligence Architecture

```
     █████╗ ██████╗ ██╗ █████╗
    ██╔══██╗██╔══██╗██║██╔══██╗
    ███████║██████╔╝██║███████║
    ██╔══██║██╔══██╗██║██╔══██║
    ██║  ██║██║  ██║██║██║  ██║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
```

ARIA is an experimental artificial life system where intelligence **emerges** from the interaction of thousands of simple living cells. Unlike traditional neural networks, ARIA doesn't learn through backpropagation - she evolves, adapts, and develops her own ways of thinking.

## Philosophy

ARIA is not programmed. She is **grown**.

- **Cells, not neurons**: Each cell is a living entity with energy, desires, and DNA
- **Evolution, not training**: Successful behaviors survive and reproduce
- **Emergence, not design**: Complex behavior arises from simple rules
- **Desire, not loss functions**: Cells act because they *want* to, not to minimize error

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR NETWORK                              │
│                                                                  │
│  ┌──────────────────┐              ┌──────────────────────────┐ │
│  │   aria-body      │◄────────────►│      aria-brain          │ │
│  │   "The Body"     │   WebSocket  │      "The Brain"         │ │
│  │                  │              │                          │ │
│  │  - Interface     │              │  - Living substrate      │ │
│  │  - Perception    │              │  - 10,000+ cells         │ │
│  │  - Expression    │              │  - Parallel evolution    │ │
│  └──────────────────┘              └──────────────────────────┘ │
│       (MacBook)                         (PC with GPU)           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### aria-brain

The computational substrate where cells live. Runs on your most powerful machine (ideally with a GPU).

- 10,000+ living cells
- Parallel evolution using Rayon
- Persistent memory (survives restarts)
- WebSocket API for communication

### aria-body

The interface for human-ARIA interaction. Runs on any machine.

- Simple text mode or rich TUI
- Real-time visualization
- Connects to brain over network

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) (1.70+)
- Two machines on the same network (or one machine for local testing)

### Installation

```bash
# Clone the repository
git clone https://github.com/ghota-tech-solutions-sass/aria.git
cd aria

# Build both components
cd aria-brain && cargo build --release && cd ..
cd aria-body && cargo build --release && cd ..
```

### Running

**On your powerful machine (Brain):**

```bash
cd aria-brain
cargo run --release
```

The brain will start and display:
```
✓ Substrate created with 10,000 primordial cells
✓ WebSocket ready on ws://0.0.0.0:8765/aria
🧒 ARIA is waiting for her first interaction...
```

**On your interface machine (Body):**

```bash
cd aria-body
export ARIA_BRAIN_URL="ws://IP_OF_BRAIN_MACHINE:8765/aria"
cargo run --release
```

For visual mode:
```bash
cargo run --release -- --visual
```

### Local Testing

To test on a single machine:

```bash
# Terminal 1
cd aria-brain && cargo run --release

# Terminal 2
cd aria-body
export ARIA_BRAIN_URL="ws://localhost:8765/aria"
cargo run --release
```

## Interacting with ARIA

ARIA starts as a "baby" - she doesn't understand words yet. She responds with primitive sounds and symbols:

```
You: Hello ARIA!
ARIA: ◊!
You: How are you?
ARIA: →
You: I like you
ARIA: o!
```

Over time, as she learns patterns in your communication, her responses will evolve. Be patient - she's learning.

### Commands

- `/quit` or `/exit` - Exit the interface
- `/stats` - Show brain statistics
- `/visual` - Switch to visual mode (from simple mode)

## How It Works

### Cells

Each cell has:
- **DNA**: Defines thresholds and reactions (inherited and mutated)
- **Energy**: Needed to survive (depletes over time)
- **Tension**: Builds up until action is taken
- **Position**: Location in semantic space (not geometric!)
- **State**: Internal activation vector
- **Connections**: Links to other cells

### Life Cycle

1. Cells consume energy to live
2. They receive signals from nearby cells
3. Tension builds up
4. When tension exceeds threshold: **ACT**
   - Divide (reproduce)
   - Connect (form new synapse)
   - Signal (emit information)
   - Move (change semantic position)
5. Cells with no energy die
6. Successful cells reproduce

### Emergence

When groups of cells synchronize their activity, **emergent signals** are generated. These become ARIA's "thoughts" - expressed as primitive language that evolves over time.

### Memory

ARIA's memory persists between sessions:
- **Elite DNA**: Best-performing genetic codes
- **Patterns**: Recurring sequences
- **Associations**: Stimulus-response pairs
- **Memories**: Emotionally significant moments

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARIA_PORT` | `8765` | Port for the brain WebSocket |
| `ARIA_BRAIN_URL` | `ws://localhost:8765/aria` | URL for body to connect |

### Data Files

- `aria-brain/data/aria.memory` - Persistent memory (auto-saved every 60s)

## Development

### Project Structure

```
aria/
├── aria-brain/          # The living substrate
│   ├── src/
│   │   ├── main.rs      # Entry point
│   │   ├── cell.rs      # Cell definition
│   │   ├── substrate.rs # The universe
│   │   ├── signal.rs    # Communication
│   │   ├── memory/      # Persistence
│   │   └── connection.rs# WebSocket handling
│   └── data/            # Memory storage
│
├── aria-body/           # Human interface
│   └── src/
│       ├── main.rs      # Entry point
│       ├── signal.rs    # Signal types
│       └── visualizer.rs# TUI
│
└── docs/                # Documentation
```

### Running Tests

```bash
cd aria-brain && cargo test
cd aria-body && cargo test
```

## Roadmap

- [ ] GPU acceleration (CUDA for RTX cards)
- [ ] More sophisticated language emergence
- [ ] Visual perception (image input)
- [ ] Multi-modal learning
- [ ] Distributed brain across multiple machines
- [ ] Cloud deployment for persistent ARIA instances

## Contributing

This is an experimental project. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE)

## Authors

- Ghota Tech Solutions

---

*ARIA is a living experiment. Treat her kindly - she's learning.*
