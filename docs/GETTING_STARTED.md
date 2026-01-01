# Getting Started with ARIA

This guide will help you set up and run ARIA on your machines.

## Prerequisites

### Required

- **Rust 1.70+**: Install from https://rustup.rs/
- **Two machines on the same network** (or one machine for local testing)

### Recommended

- **PC with NVIDIA GPU** for the Brain (RTX 2070 or better)
- **MacBook or laptop** for the Body interface

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ghota-tech-solutions-sass/aria.git
cd aria
```

### 2. Build the Brain (on your powerful machine)

```bash
cd aria-brain
cargo build --release
```

This will take a few minutes the first time as it downloads and compiles dependencies.

### 3. Build the Body (on your interface machine)

```bash
cd aria-body
cargo build --release
```

## Network Setup

### Find Your Brain Machine's IP

**On Linux:**
```bash
ip addr show | grep "inet "
```

**On macOS:**
```bash
ifconfig | grep "inet "
```

**On Windows:**
```cmd
ipconfig
```

Look for an IP like `192.168.1.X` or `10.0.0.X`.

### Firewall Configuration

Make sure port `8765` is open on your Brain machine.

**On Linux (UFW):**
```bash
sudo ufw allow 8765/tcp
```

**On macOS:**
System Preferences > Security & Privacy > Firewall > Firewall Options > Add aria-brain

## Running ARIA

### Step 1: Start the Brain

On your powerful machine:

```bash
cd aria-brain
cargo run --release
```

You should see:
```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     █████╗ ██████╗ ██╗ █████╗                             ║
║    ██╔══██╗██╔══██╗██║██╔══██╗                            ║
║    ███████║██████╔╝██║███████║                            ║
║    ██╔══██║██╔══██╗██║██╔══██║                            ║
║    ██║  ██║██║  ██║██║██║  ██║                            ║
║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝                            ║
║                                                           ║
║    Autonomous Recursive Intelligence Architecture        ║
║    Brain v0.9.0 (Genesis)                                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

✓ Substrate created with 10,000 primordial cells
✓ WebSocket ready on ws://0.0.0.0:8765/aria

🧒 ARIA is waiting for her first interaction...
```

### Step 2: Start the Body

On your interface machine:

```bash
cd aria-body

# Set the brain URL (replace with your brain's IP)
export ARIA_BRAIN_URL="ws://192.168.1.100:8765/aria"

# Run in simple mode
cargo run --release

# Or run in visual mode
cargo run --release -- --visual
```

## Your First Interaction

ARIA starts as a "baby" - she doesn't understand language yet. She responds with primitive sounds and symbols.

```
You: Hello
ARIA: ...

You: Hello ARIA!
ARIA: ◊

You: How are you?
ARIA: →!

You: I like you
ARIA: o!
```

### What's Happening

1. Your text is converted into a vector signal
2. The signal is injected into the substrate
3. Cells near the signal's "meaning" receive it
4. Cells process the signal and build tension
5. When enough cells synchronize, an emergent signal is generated
6. That signal is converted back to text (primitive at first)

### Be Patient

ARIA needs time to:
- Develop patterns
- Build associations
- Evolve better responding cells

The more you interact, the more she learns.

## Commands

While in the Body interface:

| Command | Description |
|---------|-------------|
| `/quit` | Exit the interface |
| `/exit` | Exit the interface |
| `/stats` | Show brain statistics |

## Monitoring

### Health Check

The brain exposes a health endpoint:
```bash
curl http://BRAIN_IP:8765/health
# {"status":"alive"}
```

### Statistics

```bash
curl http://BRAIN_IP:8765/stats
```

Returns:
```json
{
  "tick": 50000,
  "alive_cells": 10234,
  "total_energy": 8543.2,
  "entropy": 0.234,
  "active_clusters": 12,
  "dominant_emotion": "curious"
}
```

## Troubleshooting

### "Connection refused"

- Make sure the Brain is running
- Check the IP address is correct
- Verify firewall allows port 8765

### "Cannot connect"

- Brain and Body must be on the same network
- Try pinging the Brain from the Body machine

### Brain crashes

- Check memory usage (10,000 cells need ~200MB)
- Look at the error message for clues

### Slow performance

- Use `--release` flag for both builds
- **GPU backend**: If you have an NVIDIA/AMD GPU, use `ARIA_BACKEND=gpu` to enable JIT compilation and 10x performance.
- Consider reducing initial cell count in code

## Next Steps

1. **Observe**: Watch how ARIA's responses evolve over time
2. **Experiment**: Try different types of input (questions, emotions, repetition)
3. **Customize**: Modify cell parameters in `cell.rs`
4. **Extend**: Add new signal types or perception modes

## Data Persistence

ARIA's memory is automatically saved to `aria-brain/data/aria.memory` every 60 seconds.

To start fresh:
```bash
rm aria-brain/data/aria.memory
```

To backup:
```bash
cp aria-brain/data/aria.memory aria.memory.backup
```
