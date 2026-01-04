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

## Économie actuelle (Session 32)

```rust
// Reproduction
reproduction_threshold: 0.28   // Énergie nécessaire
child_energy: 0.24             // Donné à l'enfant
cost_divide: 0.12              // Coût parent

// Métabolisme
cost_rest: 0.0002              // Respirer
signal_energy_base: 0.60       // Gain par signal
signal_resonance_factor: 3.0   // Multiplicateur résonance
signal_radius: 30.0            // Portée en 8D

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
*Version 0.9.7 | 2026-01-04*

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
| 34 | Refactoring: shaders/, gpu_soa/, handlers/ modulaires |

## Problèmes courants résolus

| Problème | Solution |
|----------|----------|
| Gen 0 forever | `reproduction_threshold` trop haut, `cost_divide` tuait parent |
| Émissions spam | TPS non limité (ajouter sleep 1ms dans main.rs) |
| GPU freeze | Réallocation constante → headroom 100% |
| 0 épisodes | Normal - envoyer "Bravo!" après émergence |
| Population collapse | Sleeping drain trop fort (2.0 → 1.0) |
