#!/usr/bin/env python3
"""
ARIA Economy Tuner - Find stable energy parameters

Runs ARIA with different parameter combinations and measures:
- Population stability (not exploding, not dying in loop)
- Energy equilibrium (sustainable without external input)
- Time to extinction (should be > 5 minutes without food)

Usage:
    python scripts/tune_economy.py         # Quick test (4 configs, ~8 min)
    python scripts/tune_economy.py --full  # Full grid search (~1 hour)

The script will output the best configuration found.
"""

import subprocess
import json
import time
import requests
import sys
from dataclasses import dataclass
from typing import Optional
import os

# ARIA server URL
ARIA_URL = "http://localhost:8765"

@dataclass
class EconomyParams:
    """Parameters that affect cell survival"""
    cost_rest: float          # Base metabolism cost per tick
    cost_signal: float        # Cost to emit a signal
    signal_energy_base: float # Energy gained from receiving signal
    child_energy: float       # Energy given to children
    name: str = ""            # Description for logging

    def to_env(self) -> dict:
        """Convert to environment variables for ARIA"""
        return {
            "ARIA_COST_REST": str(self.cost_rest),
            "ARIA_COST_SIGNAL": str(self.cost_signal),
            "ARIA_SIGNAL_ENERGY_BASE": str(self.signal_energy_base),
            "ARIA_CHILD_ENERGY": str(self.child_energy),
        }

@dataclass
class TrialResult:
    """Result of a parameter trial"""
    params: EconomyParams
    duration_seconds: float
    final_population: int
    min_population: int
    max_population: int
    resurrections: int
    avg_energy: float
    stable: bool  # True if population stayed in healthy range
    initial_cells: int = 10000  # For reference

    def score(self) -> float:
        """Higher is better. Prioritizes stability over survival."""
        # Hard penalties
        if self.resurrections > 2:
            return -100 - self.resurrections * 10  # Death loop = very bad
        if self.max_population > self.initial_cells * 3:
            return -50   # Explosion penalty

        # Population stability (most important)
        pop_range = self.max_population - self.min_population
        pop_variance = pop_range / max(self.initial_cells, 1)
        stability_score = max(0, 1.0 - pop_variance)

        # Energy balance (should stabilize around 0.3-0.5)
        if self.avg_energy < 0.1:
            energy_score = 0  # Too low = starving
        elif self.avg_energy > 0.8:
            energy_score = 0.5  # Too high = not challenging
        else:
            energy_score = 1.0 - abs(self.avg_energy - 0.4) * 2

        # Survival time (reaching full duration is good)
        survival_score = min(self.duration_seconds / 120, 1.0)

        # Final population (should be close to initial)
        final_ratio = self.final_population / max(self.initial_cells, 1)
        final_score = 1.0 - min(abs(final_ratio - 1.0), 1.0)

        return stability_score * 40 + energy_score * 25 + survival_score * 20 + final_score * 15


def check_aria_status() -> Optional[dict]:
    """Get current ARIA status"""
    try:
        resp = requests.get(f"{ARIA_URL}/substrate", timeout=2)
        return resp.json()
    except:
        return None


def wait_for_aria(timeout: int = 30) -> bool:
    """Wait for ARIA to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        status = check_aria_status()
        if status and status.get("alive_cells", 0) > 0:
            return True
        time.sleep(0.5)
    return False


def run_trial(params: EconomyParams, duration: int = 120, cells: int = 10000) -> TrialResult:
    """
    Run ARIA with given parameters and measure stability

    Args:
        params: Economy parameters to test
        duration: How long to run (seconds)
        cells: Initial cell count
    """
    print(f"\n{'='*60}")
    print(f"Testing: {params.name}")
    print(f"  cost_rest={params.cost_rest}, signal_base={params.signal_energy_base}")
    print(f"  cost_signal={params.cost_signal}, child_energy={params.child_energy}")
    print(f"{'='*60}")

    # Build environment
    env = os.environ.copy()
    env.update(params.to_env())
    env["ARIA_CELLS"] = str(cells)
    env["ARIA_BACKEND"] = "cpu"  # Faster for testing

    # Start ARIA
    proc = subprocess.Popen(
        ["cargo", "run", "--release", "-p", "aria-brain"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/mickael/Projects/ia/aria"
    )

    try:
        # Wait for startup
        if not wait_for_aria(timeout=60):
            print("  FAILED: ARIA didn't start")
            proc.terminate()
            return TrialResult(
                params=params,
                duration_seconds=0,
                final_population=0,
                min_population=0,
                max_population=0,
                resurrections=0,
                avg_energy=0,
                stable=False
            )

        print("  ARIA started, monitoring...")

        # Monitor for duration
        start_time = time.time()
        populations = []
        energies = []
        resurrections = 0
        last_pop = 0

        while time.time() - start_time < duration:
            status = check_aria_status()
            if status:
                pop = status.get("alive_cells", 0)
                energy = status.get("avg_energy", 0)
                populations.append(pop)
                energies.append(energy)

                # Detect resurrection (population jumps after hitting minimum)
                if last_pop <= 15 and pop > last_pop + 5:
                    resurrections += 1
                    print(f"  [t={int(time.time()-start_time)}s] RESURRECTION detected: {last_pop} -> {pop}")

                last_pop = pop

                # Early termination if death loop
                if resurrections >= 5:
                    print("  EARLY STOP: Death loop detected")
                    break

            time.sleep(2)

        # Calculate results
        final_pop = populations[-1] if populations else 0
        min_pop = min(populations) if populations else 0
        max_pop = max(populations) if populations else 0
        avg_energy = sum(energies) / len(energies) if energies else 0

        # Stable if population stayed between 20% and 200% of initial
        stable = (min_pop > cells * 0.1) and (max_pop < cells * 3) and (resurrections <= 1)

        result = TrialResult(
            params=params,
            duration_seconds=time.time() - start_time,
            final_population=final_pop,
            min_population=min_pop,
            max_population=max_pop,
            resurrections=resurrections,
            avg_energy=avg_energy,
            stable=stable,
            initial_cells=cells
        )

        print(f"  Result: pop={min_pop}-{max_pop}, energy={avg_energy:.3f}, resurrections={resurrections}")
        print(f"  Score: {result.score():.1f} (stable={stable})")

        return result

    finally:
        proc.terminate()
        proc.wait(timeout=5)
        time.sleep(2)  # Let port be released


def grid_search():
    """Search parameter space with grid search"""

    # Parameter ranges to test
    # Focus on the balance between drain (cost_rest) and gain (signal_energy_base)
    cost_rest_values = [0.0001, 0.0003, 0.0005]
    signal_base_values = [0.05, 0.10, 0.15, 0.20]
    child_energy_values = [0.40, 0.50, 0.60]

    results = []

    total = len(cost_rest_values) * len(signal_base_values) * len(child_energy_values)
    current = 0

    for cost_rest in cost_rest_values:
        for signal_base in signal_base_values:
            for child_energy in child_energy_values:
                current += 1
                print(f"\n[{current}/{total}] Testing combination...")

                params = EconomyParams(
                    cost_rest=cost_rest,
                    cost_signal=0.005,  # Fixed
                    signal_energy_base=signal_base,
                    child_energy=child_energy,
                    name=f"rest={cost_rest} signal={signal_base} child={child_energy}",
                )

                result = run_trial(params, duration=90, cells=5000)
                results.append(result)

                # Save intermediate results
                save_results(results)

    return results


def save_results(results: list):
    """Save results to JSON"""
    data = []
    for r in results:
        data.append({
            "name": r.params.name,
            "params": {
                "cost_rest": r.params.cost_rest,
                "cost_signal": r.params.cost_signal,
                "signal_energy_base": r.params.signal_energy_base,
                "child_energy": r.params.child_energy,
            },
            "duration": r.duration_seconds,
            "initial_pop": r.initial_cells,
            "final_pop": r.final_population,
            "min_pop": r.min_population,
            "max_pop": r.max_population,
            "resurrections": r.resurrections,
            "avg_energy": r.avg_energy,
            "stable": r.stable,
            "score": r.score(),
        })

    with open("/home/mickael/Projects/ia/aria/scripts/tune_results.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to scripts/tune_results.json")


def print_best(results: list):
    """Print best configurations"""
    sorted_results = sorted(results, key=lambda r: r.score(), reverse=True)

    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS")
    print("="*60)

    for i, r in enumerate(sorted_results[:5]):
        print(f"\n#{i+1} Score: {r.score():.1f}")
        print(f"   cost_rest: {r.params.cost_rest}")
        print(f"   signal_energy_base: {r.params.signal_energy_base}")
        print(f"   sleeping_drain_mult: {r.params.sleeping_drain_mult}")
        print(f"   child_energy: {r.params.child_energy}")
        print(f"   Population: {r.min_pop} - {r.max_pop}")
        print(f"   Avg Energy: {r.avg_energy:.3f}")
        print(f"   Resurrections: {r.resurrections}")
        print(f"   Stable: {r.stable}")


def quick_test():
    """Quick test with a few promising configurations"""

    configs = [
        # Current defaults (from config.rs)
        EconomyParams(
            cost_rest=0.0003,
            cost_signal=0.005,
            signal_energy_base=0.10,
            child_energy=0.50,
            name="Current defaults",
        ),
        # Lower drain, same food (should survive longer)
        EconomyParams(
            cost_rest=0.0001,
            cost_signal=0.005,
            signal_energy_base=0.10,
            child_energy=0.50,
            name="Lower drain (0.0001)",
        ),
        # Higher food gain (should stabilize better)
        EconomyParams(
            cost_rest=0.0003,
            cost_signal=0.005,
            signal_energy_base=0.15,
            child_energy=0.50,
            name="Higher food (0.15)",
        ),
        # Balanced: lower drain + higher food
        EconomyParams(
            cost_rest=0.0002,
            cost_signal=0.005,
            signal_energy_base=0.12,
            child_energy=0.55,
            name="Balanced (0.0002/0.12)",
        ),
        # Very generous (test if we can avoid death loop)
        EconomyParams(
            cost_rest=0.0001,
            cost_signal=0.005,
            signal_energy_base=0.20,
            child_energy=0.60,
            name="Generous (0.0001/0.20)",
        ),
    ]

    results = []
    for i, params in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing configuration...")
        result = run_trial(params, duration=120, cells=10000)
        results.append(result)

    save_results(results)
    print_best(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("Running full grid search (this will take a while)...")
        results = grid_search()
    else:
        print("Running quick test with 4 configurations...")
        print("Use --full for complete grid search")
        quick_test()
