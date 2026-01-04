#!/usr/bin/env python3
"""
ARIA Autonomous Trainer - Runs ARIA overnight for continuous learning.

This script:
1. Sends stimulating signals to ARIA
2. Triggers web learning
3. Monitors evolution and learning progress
4. Logs all activity for morning review

Usage:
    python scripts/autonomous_trainer.py

Press Ctrl+C to stop gracefully.
"""

import asyncio
import json
import random
import hashlib
import time
import websockets
from datetime import datetime
import requests
import signal
import sys

# Configuration
BRAIN_HOST = "localhost"
BRAIN_PORT = 8765
WS_URL = f"ws://{BRAIN_HOST}:{BRAIN_PORT}/aria"
HTTP_URL = f"http://{BRAIN_HOST}:{BRAIN_PORT}"

# Training parameters
STIMULATION_INTERVAL = 5  # seconds
STATS_INTERVAL = 60  # seconds
WEB_LEARN_INTERVAL = 300  # seconds (5 minutes)
DIVERSITY_INTERVAL = 30  # seconds - try different patterns

# Stimulation corpus - things to teach ARIA
STIMULI = [
    # Emotions
    "content", "triste", "curieux", "calme", "excite", "fatigue",
    "heureux", "interesse", "ennuye", "surpris", "confus", "confiant",
    # Actions
    "apprendre", "explorer", "penser", "observer", "comprendre", "chercher",
    "decouvrir", "sentir", "voir", "ecouter", "essayer", "grandir",
    # Concepts
    "vie", "intelligence", "memoire", "evolution", "conscience", "temps",
    "espace", "energie", "connexion", "pattern", "emergence", "adaptation",
    # Greetings
    "bonjour", "salut", "coucou", "hey", "hello", "bonsoir",
    # Questions
    "pourquoi", "comment", "quoi", "qui", "quand", "ou",
    # Phrases courtes
    "je suis", "je pense", "je ressens", "je vois", "je comprends",
    "ca va", "merci", "oui", "non", "peut-etre", "bien sur",
]

# Pattern variations - different emotional intensities
PATTERNS = [
    {"arousal": 0.8, "valence": 0.6, "prefix": "excitement"},  # Excited positive
    {"arousal": 0.2, "valence": 0.7, "prefix": "calm"},        # Calm positive
    {"arousal": 0.6, "valence": 0.0, "prefix": "neutral"},     # Neutral active
    {"arousal": 0.9, "valence": -0.3, "prefix": "anxiety"},    # Anxious
    {"arousal": 0.3, "valence": -0.5, "prefix": "melancholy"}, # Sad calm
    {"arousal": 0.7, "valence": 0.4, "prefix": "curiosity"},   # Curious
]

# Stats tracking
class TrainingStats:
    def __init__(self):
        self.signals_sent = 0
        self.emergences_received = 0
        self.web_learns = 0
        self.start_time = time.time()
        self.last_population = 0
        self.last_generation = 0
        self.expressions_learned = 0

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
        # Also log to file for morning review
        with open("data/trainer_log.txt", "a") as f:
            f.write(f"[{timestamp}] {msg}\n")

stats = TrainingStats()

def text_to_tension(text):
    """Convert text to 8D tension vector using semantic hashing."""
    h = hashlib.md5(text.encode()).digest()
    tension = []
    for i in range(8):
        # Map to [-1, 1]
        val = ((h[i] / 255.0) * 2.0 - 1.0)
        tension.append(val)

    # Add some randomness for stochasticity
    for i in range(len(tension)):
        tension[i] += random.uniform(-0.1, 0.1)
        tension[i] = max(-1.0, min(1.0, tension[i]))

    return tension

def create_signal(text, pattern=None):
    """Create a Signal JSON for ARIA."""
    tension = text_to_tension(text)

    # Apply pattern modulation if specified
    if pattern:
        tension[0] = pattern["arousal"]
        tension[1] = pattern["valence"]

    # Pad to 32 dimensions
    content = tension + [0.0] * 24

    magnitude = sum(t*t for t in tension) ** 0.5
    intensity = 0.3 + 0.5 * min(magnitude / 2.0, 1.0)

    return {
        "content": content,
        "intensity": intensity,
        "label": text,
        "signal_type": "Perception"
    }

async def get_brain_stats():
    """Fetch current brain stats via HTTP."""
    try:
        resp = requests.get(f"{HTTP_URL}/stats", timeout=5)
        return resp.json()
    except:
        return None

async def get_expression_stats():
    """Fetch expression generator stats."""
    try:
        resp = requests.get(f"{HTTP_URL}/express", timeout=5)
        return resp.json()
    except:
        return None

async def get_learn_stats():
    """Fetch web learner stats."""
    try:
        resp = requests.get(f"{HTTP_URL}/learn", timeout=5)
        return resp.json()
    except:
        return None

async def stimulate_aria(ws):
    """Send a stimulating signal to ARIA."""
    # Pick a random stimulus
    text = random.choice(STIMULI)

    # Sometimes use a pattern variation
    if random.random() < 0.3:
        pattern = random.choice(PATTERNS)
        signal = create_signal(f"{pattern['prefix']}:{text}", pattern)
    else:
        signal = create_signal(text)

    await ws.send(json.dumps(signal))
    stats.signals_sent += 1

    if stats.signals_sent % 10 == 0:
        stats.log(f"📤 Sent {stats.signals_sent} signals (last: '{text}')")

async def log_stats():
    """Log current training statistics."""
    brain = await get_brain_stats()
    expr = await get_expression_stats()
    learn = await get_learn_stats()

    if brain:
        pop = brain.get("alive_cells", 0)
        gen = brain.get("max_generation", 0)
        energy = brain.get("total_energy", 0)

        pop_change = pop - stats.last_population if stats.last_population else 0
        gen_change = gen - stats.last_generation if stats.last_generation else 0

        stats.last_population = pop
        stats.last_generation = gen

        stats.log(f"🧠 Brain: {pop} cells ({pop_change:+d}), Gen {gen} ({gen_change:+d}), Energy: {energy:.0f}")

    if expr:
        learned = expr.get("total_learned", 0)
        total = expr.get("total_expressions", 0)
        stats.log(f"💬 Expressions: {total} total, {learned} learned")

    if learn:
        web_learned = learn.get("total_learned", 0)
        stats.log(f"📚 Web knowledge: {web_learned} items")

    elapsed = time.time() - stats.start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    stats.log(f"⏱️ Training time: {hours}h {minutes}m, {stats.signals_sent} signals sent")

async def receive_emergences(ws):
    """Receive and log emergences from ARIA."""
    try:
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
            data = json.loads(msg)
            label = data.get("label", "")
            intensity = data.get("intensity", 0)
            stats.emergences_received += 1

            # Only log interesting emergences
            if "says:" in label or intensity > 0.5:
                stats.log(f"🌟 EMERGENCE: {label} (intensity: {intensity:.2f})")
    except asyncio.TimeoutError:
        pass
    except Exception as e:
        pass

async def training_loop():
    """Main training loop."""
    stats.log("🚀 ARIA Autonomous Trainer starting...")
    stats.log(f"📡 Connecting to {WS_URL}")

    last_stim_time = 0
    last_stats_time = 0
    last_diversity_time = 0
    current_pattern_idx = 0

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                stats.log("✅ Connected to ARIA brain!")

                while True:
                    current_time = time.time()

                    # Stimulate periodically
                    if current_time - last_stim_time >= STIMULATION_INTERVAL:
                        await stimulate_aria(ws)
                        last_stim_time = current_time

                    # Diversify patterns periodically
                    if current_time - last_diversity_time >= DIVERSITY_INTERVAL:
                        current_pattern_idx = (current_pattern_idx + 1) % len(PATTERNS)
                        pattern = PATTERNS[current_pattern_idx]

                        # Send a burst of related signals
                        for text in random.sample(STIMULI, 3):
                            signal = create_signal(text, pattern)
                            await ws.send(json.dumps(signal))
                            stats.signals_sent += 1

                        last_diversity_time = current_time

                    # Log stats periodically
                    if current_time - last_stats_time >= STATS_INTERVAL:
                        await log_stats()
                        last_stats_time = current_time

                    # Receive emergences
                    await receive_emergences(ws)

                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.1)

        except Exception as e:
            stats.log(f"❌ Connection lost: {e}")
            stats.log("🔄 Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    stats.log("🛑 Trainer stopping...")
    stats.log(f"📊 Final stats: {stats.signals_sent} signals, {stats.emergences_received} emergences")
    sys.exit(0)

if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs("data", exist_ok=True)

    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Run training loop
    asyncio.run(training_loop())
