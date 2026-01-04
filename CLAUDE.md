# ARIA - Claude Context File

> Contexte pour reprendre ARIA. **Ne pas supprimer.**

## Identité

**ARIA** = Autonomous Recursive Intelligence Architecture
IA expérimentale où l'intelligence **émerge** de cellules vivantes. Pas un LLM.

## Architecture

```
aria-body (MacBook) ◄──WebSocket──► aria-brain (PC + RTX 2070)
```

| Crate | Rôle |
|-------|------|
| `aria-core` | Types compacts GPU-ready, DNA, config |
| `aria-compute` | CPU/GPU backends, shaders WGSL, spatial hash |
| `aria-brain` | Substrate, mémoire, WebSocket server |
| `aria-body` | Interface TUI |

## Commandes essentielles

```bash
task brain              # 50k cellules, auto GPU/CPU
task brain-gpu          # Forcer GPU
task body               # Interface TUI
task stats              # Stats cerveau
task episodes           # Mémoire épisodique
./scripts/run_overnight.sh  # Training autonome 24h
```

## Économie actuelle (Session 35)

```rust
// Reproduction
reproduction_threshold: 0.18   // Seuil pour se reproduire
child_energy: 0.15             // Énergie de départ des enfants
cost_divide: 0.12              // Coût parent

// Métabolisme
cost_rest: 0.0003              // Drain passif par tick
signal_energy_base: 0.20       // Session 35: augmenté 0.08→0.20 + fix efficiency [0.5-1.5]
signal_resonance_factor: 3.0   // Multiplicateur résonance
signal_radius: 30.0            // Portée en 8D

// Population scale (signal.rs)
// sqrt(10k / cell_count) → dilution ressources
// 100k = 0.316x, 200k = 0.22x énergie par signal
// Effective signal = 0.15 × 0.22 = 0.034 à 200k cells

// Timing
EMISSION_COOLDOWN_TICKS: 5000  // ~5 sec entre émissions
TPS: ~1000                     // Rate limité dans main.rs
```

## Générations & ADN Élite

| Gen | Description |
|-----|-------------|
| 0 | ADN aléatoire |
| 1 | Enfants Gen0, ADN hérité + mutations |
| 2+ | Optimisé par sélection |
| **10+** | **Élite** - ADN sauvegardé dans `aria.memory` |

**Élite** = cellules ayant survécu 10+ cycles. Leur ADN est préservé pour accélérer l'évolution future.

## Lois physiques (GPU shaders)

| Loi | Effet |
|-----|-------|
| **Prédiction** | Cellules qui prédisent bien gagnent énergie |
| **Hebb** | "Fire together, move together" - attraction spatiale |
| **Résonance** | Signal aligné avec état = énergie |
| **La Vraie Faim** | Rien n'est gratuit, actions coûtent |

## Fichiers clés

| Fichier | Contenu |
|---------|---------|
| `aria-core/src/config.rs` | Tous les paramètres économiques |
| `aria-core/src/dna.rs` | Structure ADN, mutations |
| `aria-compute/src/shaders/` | Shaders WGSL modulaires |
| `aria-compute/src/backend/gpu_soa/` | Backend GPU SoA (6 modules) |
| `aria-brain/src/handlers/` | API handlers HTTP/WebSocket |
| `aria-brain/src/substrate/` | Logique substrate (lifecycle, signals, emergence) |
| `aria-brain/src/expression.rs` | Génération parole sans LLM |
| `aria-brain/src/memory/persistence.rs` | Types sérialisables (DNA, patterns, etc.) |

## Contexte personnel

- **Moka** : Bengal de Mickael
- **Obrigada** : Abyssin

---
*Version 0.9.8 | Session 35 | 2026-01-04*

## Historique des sessions (résumé)

| Session | Changement principal |
|---------|---------------------|
| 12 | GPU backend wgpu/Vulkan |
| 14-16 | Méta-apprentissage, auto-modification |
| 17 | Sparse dispatch GPU, neuroplasticité |
| 18 | "La Vraie Faim" - économie stricte |
| 20 | Architecture SoA pour 5M+ cellules |
| 23 | JIT compilation DNA→WGSL |
| 25-29 | Lois physiques (Prédiction, Hebb, Compression) |
| 30 | Fix évolution multi-générationnelle |
| 31 | Suppression vocabulaire, intelligence physique |
| 32 | Full GPU migration, TPS rate limiting |
| 33 | Web learner, expression generator |
| 34 | Refactoring modulaire + fix économie (population_scale inversé, buffer 70%) + fix web learner (scraper lib) |
| 35 | Anti-freeze: ring buffer signals, GPU timeouts, 90% cells sleeping at start, DNA bloat detection, sampling O(1) |

## Problèmes courants résolus

| Problème | Solution |
|----------|----------|
| Gen 0 forever | `reproduction_threshold` trop haut, `cost_divide` tuait parent |
| Émissions spam | TPS non limité (ajouter sleep 1ms dans main.rs) |
| GPU freeze | Réallocation constante → headroom 100% |
| 0 épisodes | Normal - envoyer "Bravo!" après émergence |
| Population collapse | Sleeping drain trop fort (2.0 → 1.0) |
| **Population explosion** | `population_scale` était inversé ! Plus de cellules = plus d'énergie → fix: dilution `sqrt(10k/count)` |
| **GPU buffer overflow** | Réalloc proactive à 70% capacité + hard limit check |
| **Web learner 0 items** | Parser HTML manuel cassé (`<header>` matchait `<head>`) → remplacé par lib `scraper` |
| **REALLOC spam** | DNA pool croissait indéfiniment (190k+ DNA pour 105k cells) → check seulement cells.len() à 80% |
| **Cells can't reproduce** | `reproduction_threshold=0.28` mais `max_energy=0.24` → abaissé à 0.22, augmenté `signal_energy_base` 0.03→0.04 |
| **Freeze before mass die-off** | Signaux s'accumulaient pendant GPU work → Ring buffer (256 cap) + drain max 64/tick |
| **Startup freeze (120k awake)** | Toutes cells awake au démarrage → 90% start sleeping dans `CellState::new()` |
| **DNA pool 854k entries** | DNA croît sans limite → `compact_dead_cells()` aussi si DNA > 2× cells |
| **GPU poll blocking** | `device.poll()` sans timeout → ajout timeout 50-100ms + fallback |
| **Énergie trop diluée** | À 200k cells, `population_scale=0.22` → `signal_energy_base` 0.08→0.20 |
| **Efficiency crush** | `efficiency` DNA [0-1] multipliait l'énergie → changé à [0.5-1.5] (bonus pas pénalité) |
