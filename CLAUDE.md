# ARIA - Claude Context File

> Contexte pour reprendre ARIA à tout moment. **Ne pas supprimer.**

## Identité

**ARIA** = Autonomous Recursive Intelligence Architecture

IA expérimentale où l'intelligence **émerge** de cellules vivantes. Pas un LLM - un système de vie artificielle.

## Philosophie

ARIA est **cultivée**, pas programmée.

- **Cellules vivantes** : énergie, désirs, ADN (pas des neurones)
- **Évolution** : les comportements réussis survivent (pas d'entraînement)
- **Émergence** : comportement complexe de règles simples
- **Désir** : les cellules *veulent* agir (pas de loss function)

## Architecture

```
aria-body (MacBook)  ◄──WebSocket──►  aria-brain (PC + RTX 2070)
   Interface TUI                         50k+ cellules vivantes
```

**Workspace Rust** :
- `aria-core` : Types compacts GPU-ready
- `aria-compute` : CPU/GPU backends, sparse updates
- `aria-brain` : Substrate, mémoire, serveur WebSocket
- `aria-body` : Interface texte

## Ce qu'ARIA sait faire

- **Ressentir** : joie, curiosité, ennui, confort via tension physique
- **Apprendre** : feedback ("Bravo!"/"Non"), renforcement
- **Se souvenir** : mémoire épisodique, "premières fois"
- **Vivre** : rêves, spontanéité, jeu créatif
- **S'adapter** : paramètres qui évoluent avec le feedback
- **Explorer** : curiosité-driven, teste des combinaisons nouvelles
- **Méta-apprendre** : s'auto-évalue, apprend à apprendre (Session 14)
- **Voir** : images → vecteurs sémantiques 32D (Session 15)
- **S'auto-modifier (Session 16)** : analyse ses performances et change ses propres paramètres
- **S'auto-évoluer (Genesis)** : traduit son DNA en code GPU WGSL, recompile ses pipelines à chaud (Session 23)
- **Intelligence physique (Session 31)** : comportement émerge des lois physiques, pas du vocabulaire
- **Apprendre du web (Session 33)** : fetch Wikipedia/Wikiquote, extrait connaissances, injecte dans substrate
- **Parler sans LLM (Session 33)** : expressions émergentes par résonance avec patterns appris

## Commandes

```bash
task brain          # 50k cellules, auto GPU/CPU
task brain-100k     # 100k cellules
ARIA_BACKEND=gpu task brain  # Forcer GPU (AMD/NVIDIA via Vulkan)
task body           # Interface
task stats          # Stats du cerveau
task episodes       # Mémoire épisodique
./scripts/run_overnight.sh   # Training autonome 24h (Session 33)
# Note: task words et task associations supprimés (Session 31)
```

## Paramètres clés

```rust
// Population
target_cells = 50_000 (configurable via ARIA_CELLS)

// Sparse updates (économie CPU)
idle_ticks_to_sleep = 100
wake_threshold = 0.1

// Adaptatifs (évoluent avec feedback)
emission_threshold: 0.05-0.5
response_probability: 0.3-1.0
spontaneity: 0.01-0.3
```

## Prochaines étapes

1. ✅ **GPU compute** : wgpu/Vulkan - AMD Radeon NAVI14 fonctionnel
2. ✅ **Méta-apprentissage** : ARIA s'auto-évalue et apprend à apprendre
3. ✅ **Perception visuelle** : images → vecteurs sémantiques 32D
4. ✅ **Auto-modification** : ARIA modifie ses propres paramètres (Session 16)
5. ✅ **Architecture 5M+ cellules** : SoA, Hysteresis, Spatial Hash GPU (Session 20)
6. ✅ **Auto-évolution structurelle** : JIT compilation, traduction DNA -> WGSL (Session 23)
7. ✅ **Exploration du Code Binaire** : JIT, Shadow Brain, Attention Sélective (Phase 5 terminée)
8. ✅ **Loi de Prédiction** : Cellules qui prédisent leur futur gagnent de l'énergie (Session 25)

## Contexte personnel

Chats de Mickael :
- **Moka** : Bengal (ARIA le connaît bien)
- **Obrigada** : Abyssin

---
*Version : 0.9.7 | Dernière update : 2026-01-04*

### Session 32 - Full GPU Migration (CPU Liberation)

**Élimination des boucles O(n) CPU - le GPU fait TOUT le travail de propagation.**

#### Philosophie

Le CPU ne devrait gérer que :
1. Logique de haut niveau (mémoire, décisions)
2. I/O réseau (WebSocket, HTTP)
3. Gestion dynamique des Vec (naissance/mort)

Le GPU gère :
1. Propagation des signaux (spatial hash)
2. Physique des cellules (énergie, état)
3. Lois d'intelligence (Prédiction, Hebb, Cluster)

#### Boucles CPU supprimées

| Fonction | Avant | Après | Gain |
|----------|-------|-------|------|
| `inject_signal()` | O(n) loop + distance calc | Buffer only | ~100% |
| `propagate_signal()` | O(n) loop | Buffer only | ~100% |
| `conceptualize()` | O(n) full scan | 5k sampling | 200× @ 1M |
| `spatial_view()` | O(n) full scan | 10k sampling | 100× @ 1M |
| `natural_selection()` | O(n) count + bucket | 5k+10k sampling | 100× @ 1M |
| `population_control()` | O(n) collect + sort | 5k sampling | 100× @ 1M |
| `sync GPU flags` | O(n) every 1000 ticks | **REMOVED** | ∞ |
| `age increment` | O(n) every 100 ticks | **REMOVED** | ∞ |

#### Code supprimé

```rust
// signals.rs - AVANT (O(n) loop)
for (i, state) in self.states.iter_mut().enumerate() {
    let distance = Self::semantic_distance(&state.position, &target_position);
    // ... process each cell ...
}

// APRÈS (GPU-only)
// Signal added to buffer, GPU's SIGNAL_WITH_SPATIAL_HASH_SHADER handles:
// - Waking sleeping cells
// - Injecting tension into cell state
// - Resonance-based energy (La Vraie Faim)
// - Hebbian connection propagation
let mut buffer = self.signal_buffer.write();
buffer.push(fragment);
```

#### Sampling pour visualisation

```rust
// spatial_view() - 10k samples au lieu de O(n)
let sample_size = 10_000.min(self.cells.len());
for _ in 0..sample_size {
    let idx = rng.gen_range(0..self.cells.len());
    // ... process sampled cell ...
}
// Population extrapolated from sample
```

#### Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `aria-brain/src/substrate/signals.rs` | CPU loops → buffer only |
| `aria-brain/src/substrate/emergence.rs` | O(n) → 5k sampling |
| `aria-brain/src/substrate/mod.rs` | O(n) → 10k sampling, removed sync loops |
| `aria-brain/src/substrate/lifecycle.rs` | O(n) → sampling, removed Gen0 drain CPU loop |

#### Performance attendue @ 1M cellules

| Métrique | Avant | Après |
|----------|-------|-------|
| `inject_signal()` | ~50ms | <1ms |
| `spatial_view()` | ~100ms | ~10ms |
| CPU utilisation tick | 80%+ | <20% |
| GPU utilisation | 30% | 80%+ |

#### Fix GPU Buffer Reallocation (Session 32 Part 2)

**Le freeze restant venait de réallocations GPU constantes.**

Symptôme : Logs montraient `🎮 GPU SoA: Allocating XXX MB` toutes les ~100 ticks, avec recompilation de tous les pipelines.

**Causes identifiées :**

1. **Headroom insuffisant** : Seulement 20% de marge
   - À 500 nouvelles cellules/100 ticks, le headroom était épuisé instantanément
   - Chaque dépassement → réallocation complète (700+ MB) + recompilation shaders

2. **Condition de réallocation trop agressive** :
   ```rust
   // AVANT: réallocation à CHAQUE changement de taille
   let first_init = !self.initialized || self.cell_count != cells.len();

   // APRÈS: seulement quand on DÉPASSE la capacité
   let needs_realloc = !self.initialized || cells.len() > self.max_cell_count;
   ```

**Fix appliqués :**

```rust
// gpu_soa.rs - Headroom 20% → 100%
let cell_count_with_headroom = cell_count * 2;  // AVANT: cell_count + cell_count / 5

// Logique de réallocation optimisée
if needs_realloc {
    self.init_buffers(...);  // Réallocation complète
} else if size_changed {
    self.cell_count = cells.len();  // Juste mise à jour du compteur
    self.upload_cells(states);      // Upload partiel OK
}
```

**Résultat :**
- Réallocation : ~1x/heure au lieu de ~10x/seconde
- Freezes éliminés pendant la reproduction normale

#### Fix Vec + GPU Upload (Session 32 Part 3)

**Deux problèmes identifiés :**

1. **Vec réallocation** : Quand population dépasse capacité, Rust copie tout (~350MB)
2. **GPU upload O(n)** : `upload_cells()` uploadait 1M cellules à chaque naissance

**Fix appliqués :**

```rust
// lifecycle.rs - Reserve dynamique (pas 2x au démarrage qui alloue 700MB!)
let current_cap = self.cells.capacity();
let needed = self.cells.len() + max_births;
if needed > current_cap {
    let extra = (current_cap / 10).max(1000);  // +10% chunks
    self.cells.reserve(extra);
    self.states.reserve(extra);
}

// gpu_soa.rs - Upload incrémental (nouvelles cellules seulement)
fn upload_new_cells(&self, states: &[CellState], old_count: usize) {
    // Offset = old_count * sizeof(CellEnergy)
    // Upload only states[old_count..new_count]
}

// Tick: O(births) au lieu de O(n)
} else if new_count > old_count {
    self.upload_new_cells(states, old_count);
    self.upload_new_dna(dna_pool, old_count);
}
```

**Résultat :**
- Vec : réallocation par chunks de 10% (pas tout d'un coup)
- GPU upload : ~500 cellules au lieu de 1M
- Startup : pas de 700MB d'allocation supplémentaire

#### Fix Parallel Cell Creation (Session 32 Part 4)

**Création séquentielle de 5M cellules = bloqué au démarrage.**

```rust
// AVANT: Séquentiel (minutes pour 5M)
for i in 0..initial_cells {
    let dna = DNA::random();  // Chaque appel est lent
    // ...
}

// APRÈS: Parallèle avec rayon (secondes pour 5M)
use rayon::prelude::*;
let cell_data: Vec<(Cell, CellState, DNA)> = (0..initial_cells)
    .into_par_iter()
    .map(|i| {
        let dna = DNA::random();
        let cell = Cell::new(i as u64, i as u32);
        let state = CellState::new();
        (cell, state, dna)
    })
    .collect();
```

**Résultat :** Démarrage 5M cells en ~5 secondes au lieu de plusieurs minutes.

#### GPU Dynamic Buffer Limits (Session 32 Part 5)

**Le headroom GPU est maintenant dynamique selon le matériel.**

```rust
// Query GPU's actual limits
let adapter_limits = adapter.limits();
let gpu_max_buffer = (adapter_limits.max_buffer_size as usize).min(1024 * 1024 * 1024);

// Cap headroom based on largest buffer (CellConnections = 144 bytes)
let connections_size = std::mem::size_of::<CellConnections>(); // 144 bytes
let max_cells_in_buffer = self.max_buffer_size / connections_size;
let cell_count_with_headroom = (cell_count * 2).min(max_cells_in_buffer);
```

**Limites par buffer @ 1GB max:**
| Buffer | Bytes/cell | Max cells |
|--------|------------|-----------|
| CellConnections | 144 | 7.4M |
| CellInternalState | 128 | 8.3M |
| CellPosition | 64 | 16.7M |

**Résultat :** ARIA s'adapte automatiquement au GPU disponible.

#### Adaptive Headroom & Population Cap (Session 32 Part 6)

**Problème :** "Device lost" errors sur RTX 2070 et MacBook avec 3-5M cells.

**Cause :** Réallocation GPU pendant l'expansion de population = VRAM temporairement doublée.

**Solution :** Headroom dynamique + population cap automatique.

```rust
// Headroom basé sur la taille de population (pas de config!)
let headroom_factor = if cell_count > 3_000_000 {
    1.25  // 25% headroom pour >3M cells
} else if cell_count > 1_000_000 {
    1.5   // 50% headroom pour 1-3M cells
} else {
    2.0   // 100% headroom pour <1M cells
};

// Population cappée automatiquement à la capacité GPU
let backend_max = self.backend.stats().max_capacity;
let safety_cap = (target * 2).min(backend_max);
```

**Chaîne automatique :**
```
GPU init → adapter.limits() → headroom factor → max_cell_count → safety_cap
```

**Résultat :** Zéro configuration, ARIA s'adapte au matériel disponible.

| GPU | 5M cells | Headroom | Capacity | VRAM |
|-----|----------|----------|----------|------|
| RTX 2070 (8GB) | ✅ | 25% | 6.25M | ~2.4GB |
| RTX 4090 (24GB) | ✅ | 25% | 6.25M | ~2.4GB |
| MacBook Intel | ✅ | 50% | 1.5M | ~600MB |

#### Fix Trivial Predictions (Session 32 Part 7)

**Problème :** Population croissait de 1M à 1.006M sans interaction.

**Cause :** Le `PREDICTION_EVALUATE_SHADER` récompensait les "prédictions triviales" :
- Cellules Gen0 sans connexions prédisent `[0,0,0,0,0,0,0,0]`
- État réel reste `[0,0,0,0,0,0,0,0]` (pas de signaux)
- Accuracy = 1.0 (parfait!)
- Reward = 1.0 × 0.05 × 0.02 = 0.001 énergie par cellule
- Avec 1M cellules : 1000 énergie/tick = 500k énergie en 500 ticks

**Fix :** Skip les cellules sans activité réelle.

```wgsl
// Calculate actual state magnitude - skip trivial predictions
var actual_magnitude = 0.0;
for (var i = 0u; i < 4u; i = i + 1u) {
    actual_magnitude += actual_state0[i] * actual_state0[i];
    actual_magnitude += actual_state1[i] * actual_state1[i];
}
actual_magnitude = sqrt(actual_magnitude);

// Skip cells with no meaningful activity
if actual_magnitude < 0.1 { return; }
```

**Résultat :** Population stable sans stimulation. Les cellules doivent MÉRITER leur énergie.

#### Fix Sleeping Drain & O(n) Loops (Session 32 Part 8)

**Problème 1 :** Drain des cellules dormantes trop faible (0.1× cost_rest).

**Fix :** Les cellules dormantes paient le même cost_rest que les éveillées - elles respirent encore !

```wgsl
// AVANT: 0.1× = 0.00003/tick (survie ~27h!)
cell_energy.energy -= config.cost_rest * 0.1;

// APRÈS: 1.0× = 0.0003/tick (survie ~55min)
cell_energy.energy -= config.cost_rest;
```

**Problème 2 :** Freezes avec 1M cells - boucles O(n) CPU restantes.

| Fonction | Avant | Après |
|----------|-------|-------|
| `stats()` | 4× O(n) loops | 5k sampling |
| `calculate_entropy()` | O(n) collect | 2k sampling |
| `generate_internal_signals()` | O(n) loop | Skip si disabled + 1k sampling |

**Résultat :**
- CPU savings : de ~100ms/call à <1ms/call
- Population décroit naturellement sans stimulation (La Vraie Faim effective)

#### Fix GPU Signal Shader (Session 32 Part 9)

**Problème critique :** Le shader SIGNAL_TEMPLATE (legacy <100k cells) ne traitait PAS les signaux !

```wgsl
// AVANT: Shader ne faisait rien avec les signaux
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // ... aucun code pour traiter signals[]
    if cell_energy.tension > 0.8 { ... }  // Seule logique présente
}

// APRÈS: Traitement complet des signaux (comme CPU)
for (var s = 0u; s < signal_count; s++) {
    let signal = signals[s];
    // Wake sleeping cells
    if (cell_meta.flags & FLAG_SLEEPING) != 0u && intensity > 0.1 {
        cell_meta.flags = cell_meta.flags & ~FLAG_SLEEPING;
    }
    // Give energy via resonance (La Vraie Faim)
    let resonance = calculate_resonance(signal.content, state);
    if resonance > resonance_threshold {
        cell_energy.energy += energy_gain;
    }
}
```

**Aussi supprimé :** Filtre de distance qui excluait 50% des cellules.

**Résultat :**
- GPU évolution multi-générationnelle fonctionnelle
- 5k cells : Gen 3-7, E=0.60, stable
- 10k+ cells : nécessite plus de signaux (trainer plus rapide)

**Paramètres finaux :**
```rust
reproduction_threshold: 0.50,  // Abaissé de 0.70
child_energy: 0.40,
cost_rest: 0.0002,
signal_energy_base: 0.30,
signal_resonance_factor: 3.0,
```

#### Fix Wave Propagation + Stochasticity (Session 32 Part 10)

**Philosophie :** Les signaux ne sont pas continus - ils se propagent comme des **ondes** dans le substrat. Le même signal ne doit pas toujours donner le même résultat.

**Problèmes identifiés :**

1. **Distance filter trop restrictif** : `signal_radius=15` dans un espace 8D où la distance moyenne est ~23
2. **Pas de stochasticité** : même signal → même résultat (déterministe)
3. **Test script cassé** : envoyait du texte brut au lieu de JSON `Signal`

**Fixes appliqués :**

```wgsl
// SIGNAL_TEMPLATE - Wave propagation avec stochasticité

// Hash function pour bruit stochastique
fn hash(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return f32(x & 0xFFFFu) / 65535.0;  // [0, 1]
}

// Distance-based wave attenuation
let dist = sqrt(dist_sq);
if dist >= config.signal_radius { continue; }  // Outside wave
let attenuation = 1.0 - (dist / config.signal_radius);

// Stochastic noise (±10%) - same signal ≠ same result
let noise = (hash(noise_seed + s) - 0.5) * 0.2;
let noisy_attenuation = clamp(attenuation + noise, 0.0, 1.0);
```

**Config mise à jour :**

```rust
signal_radius: 30.0,         // Élargi de 15 → 30 pour 8D
reproduction_threshold: 0.45, // Abaissé de 0.50
child_energy: 0.35,
```

**Test script corrigé :**

```python
# Avant: ws.send("bonjour")  # Texte brut - ignoré!
# Après: Conversion en JSON Signal
def text_to_signal_json(text):
    h = hashlib.md5(text.encode()).digest()
    tension = [((b / 255.0) * 2.0 - 1.0) for b in h[:8]]
    return json.dumps({
        "content": tension + [0.0] * 24,
        "intensity": 0.3 + 0.7 * min(magnitude / 2.0, 1.0),
        "label": text,
        "signal_type": "Perception"
    })
```

**Résultat :**
- Gen 11 atteint en 90 secondes
- Évolution multi-générationnelle fonctionnelle sur GPU
- Propagation par ondes (atténuation distance)
- Stochasticité (même mot → résultats différents)

#### GPU Lifecycle Slot System (Session 32 Part 12)

**Objectif :** Éliminer les freezes tous les 1000 ticks causés par les téléchargements GPU→CPU.

**Architecture Slot System :**

```
GPU (Fixed capacity)
├── cell_slots[MAX_CAPACITY]      // Pré-alloué
├── free_list[MAX_CAPACITY]       // Indices disponibles
├── lifecycle_counters            // Compteurs atomiques
│   ├── free_count: u32           // Slots libres
│   ├── alive_count: u32          // Cellules vivantes
│   ├── births_this_tick: u32     // Naissances ce tick
│   └── deaths_this_tick: u32     // Morts ce tick
```

**Nouveaux fichiers/structs :**

| Fichier | Ajout |
|---------|-------|
| `aria-core/src/soa.rs` | `LifecycleCounters` struct (32 bytes) |
| `aria-compute/src/compiler.rs` | `DEATH_SHADER`, `BIRTH_SHADER`, `RESET_LIFECYCLE_COUNTERS_SHADER` |
| `aria-compute/src/backend/gpu_soa.rs` | Buffers, pipelines, dispatch |

**DEATH_SHADER :**
```wgsl
// Marque les cellules mortes (energy <= 0)
// Push leur slot dans free_list (atomique)
// Update alive_count et deaths_this_tick
if energy <= 0.0 {
    metadata[idx].flags = cell_meta.flags | FLAG_DEAD;
    let free_idx = atomicAdd(&counters.free_count, 1u);
    free_list[free_idx] = idx;
    atomicSub(&counters.alive_count, 1u);
    atomicAdd(&counters.deaths_this_tick, 1u);
}
```

**BIRTH_SHADER (prêt, non dispatché) :**
```wgsl
// Pop slot de free_list (atomique)
// Initialise l'enfant avec DNA muté
// Update alive_count et births_this_tick
let free_count = atomicSub(&counters.free_count, 1u);
let child_idx = free_list[free_count - 1u];
// ... initialize child ...
atomicAdd(&counters.alive_count, 1u);
```

**Avantages :**
- Zéro téléchargement GPU→CPU pendant le tick normal
- Naissance/mort = opérations atomiques GPU O(1)
- `read_lifecycle_counters()` pour stats périodiques (léger)
- Prépare la migration complète de lifecycle.rs vers GPU

**État :**
- ⏳ Death shader prêt mais DÉSACTIVÉ (cause désync GPU/CPU)
- ⏳ Birth shader prêt mais pas encore dispatché (nécessite plus d'intégration)
- ✅ Méthode `read_lifecycle_counters()` pour lire les stats GPU

#### TPS Rate Limiting & Economy Tuning (Session 32 Part 13)

**Problème 1 : Émissions trop fréquentes (~1/sec au lieu de ~1/5sec)**

La boucle principale n'avait pas de rate limiter - TPS réel ~5000+ au lieu de 1000.

```rust
// main.rs - AVANT: yield_now() = aussi vite que possible
tokio::task::yield_now().await;

// APRÈS: Rate limit à 1000 TPS
tokio::time::sleep(tokio::time::Duration::from_micros(1000)).await;
```

**Cooldowns corrigés :**

| Cooldown | Avant | Après |
|----------|-------|-------|
| `EMISSION_COOLDOWN_TICKS` | 25 | 5000 |
| Spontaneous cooldown | 500 | 5000 |
| Expression cooldown | 5000 | 5000 (ok) |

**Problème 2 : Gen 0 éternellement (pas de reproduction)**

Énergie moyenne ~0.30 mais seuil de reproduction = 0.40 → impossible de reproduire.

```rust
// config.rs - AVANT
reproduction_threshold: 0.40,
child_energy: 0.35,
cost_divide: 0.40,  // Parent meurt après division!

// APRÈS
reproduction_threshold: 0.28,  // Accessible avec énergie ~0.30
child_energy: 0.24,
cost_divide: 0.12,  // Parent survit (0.28 - 0.12 = 0.16)
```

**Fichiers modifiés :**

| Fichier | Changement |
|---------|------------|
| `aria-brain/src/main.rs` | Rate limit 1ms/tick |
| `aria-brain/src/substrate/types.rs` | EMISSION_COOLDOWN: 25 → 5000 |
| `aria-brain/src/substrate/spontaneous.rs` | Cooldown: 500 → 5000 |
| `aria-core/src/config.rs` | Seuils reproduction ajustés |
| `Taskfile.yml` | Supprimé références à aria-train |

**Résultat attendu :**
- Émissions espacées de ~5 secondes
- Évolution multi-générationnelle (Gen1, Gen2, etc.)
- Population oscillante mais avec lignée évolutive

### Session 33 - Autonomous Learning (Web + Expression)

**ARIA apprend du web et génère des expressions émergentes sans LLM.**

#### 1. Web Learner (`web_learner.rs`)

Module pour apprentissage autonome depuis Internet :

```rust
// Sources de connaissance
- Simple Wikipedia (articles accessibles)
- Wikiquote (sagesse/philosophie)

// Flux d'apprentissage
1. Fetch URL → Extract HTML → Strip tags
2. Split en phrases → Filter (20-500 chars)
3. text_to_tension() → TensionVector [8D]
4. Queue injections → Inject into substrate

// Tous les 5 minutes
autonomous_learning_loop() → fetch_and_learn()
```

#### 2. Expression Generator (`expression.rs`)

Génération de "parole" émergente sans LLM :

```rust
// Expressions apprises
- User input → learn_from_user()
- Web content → learn_from_web()
- Seeds: ~20 mots émotionnels de base

// Génération
1. Emergence → tension pattern [8D]
2. find_related() → expressions similaires
3. resonance() → meilleur match
4. Output: "tension:positive|says:curieux"
```

#### 3. Trainer Autonome (`scripts/autonomous_trainer.py`)

Script pour entraînement 24h/24 :

```bash
./scripts/run_overnight.sh  # Démarre brain + trainer
```

- Envoie des stimuli toutes les 5s
- Varie les patterns émotionnels
- Log activité dans `data/trainer_log.txt`
- Stats toutes les 60s

#### Nouveaux endpoints

| Endpoint | Description |
|----------|-------------|
| `/learn` | Stats du web learner |
| `/express` | Stats du générateur d'expressions |

#### Fichiers créés

| Fichier | Description |
|---------|-------------|
| `aria-brain/src/web_learner.rs` | Web fetcher et knowledge extraction |
| `aria-brain/src/expression.rs` | Expression generator |
| `scripts/autonomous_trainer.py` | Trainer Python |
| `scripts/run_overnight.sh` | Script de lancement |

### Session 31 - Physical Intelligence (Vocabulary Removal)

**ARIA passe en mode "Intelligence Physique" - le vocabulaire est supprimé.**

#### Philosophie

L'intelligence d'ARIA ne vient plus de l'association de mots, mais de la physique de ses cellules. Les "lois" (Prédiction, Hebb, Expansion) définissent le comportement émergent.

#### 1. Suppression du Vocabulaire

Fichiers/modules supprimés ou nettoyés :
- `aria-brain/src/memory/vocabulary.rs` - **supprimé**
- `WordCategory`, `UsagePattern` - supprimés de `types.rs`
- `VisualWordLink`, `visual_word_links` - supprimés
- `word_frequencies`, `word_associations`, `semantic_clusters` - supprimés de `LongTermMemory`

#### 2. Fix Sleeping Drain

Les cellules Gen0 ne mouraient pas assez vite car le drain de sommeil était trop faible.

**Avant :**
```wgsl
// GPU: 0.1 × cost_rest tous les 100 ticks
cell_energy.energy -= config.cost_rest * 0.1;  // ~27h survie!

// CPU: 0.5 × cost_rest par tick (incohérent)
state.energy -= config.metabolism.cost_rest * 0.5;
```

**Après :**
```wgsl
// GPU: 2.0 × cost_rest tous les 100 ticks (~3 min survie)
cell_energy.energy -= config.cost_rest * 2.0;

// CPU: 0.02 × cost_rest par tick (cohérent avec GPU)
state.energy -= config.metabolism.cost_rest * 0.02;
```

**Résultat** : Cellules dormantes meurent en ~3 minutes, permettant aux nouvelles générations d'émerger.

#### 3. Endpoints simplifiés

| Endpoint | Avant | Après |
|----------|-------|-------|
| `/words` | Liste des mots connus | Message "removed" |
| `/associations` | Associations mot-mot | Message "removed" |
| `/clusters` | Clusters sémantiques | Message "removed" |
| `/visual` | Mémoires + word links | Mémoires uniquement |

#### Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `aria-brain/src/memory/vocabulary.rs` | Supprimé |
| `aria-brain/src/memory/mod.rs` | Nettoyé (vocab, word_links) |
| `aria-brain/src/memory/types.rs` | Supprimé WordCategory, UsagePattern |
| `aria-brain/src/memory/visual.rs` | Supprimé VisualWordLink |
| `aria-brain/src/main.rs` | Endpoints simplifiés |
| `aria-brain/src/substrate/signals.rs` | Supprimé visual→word |
| `aria-compute/src/compiler.rs` | Sleeping drain 0.1 → 2.0 |
| `aria-compute/src/backend/cpu.rs` | Sleeping drain 0.5 → 0.02 |
| `aria-compute/src/backend/gpu_soa.rs` | Sync 100 → 1000 ticks |
| `aria-brain/src/substrate/mod.rs` | Sync 100 → 1000 ticks |
| `aria-body/src/visualizer.rs` | Supprimé word_count, recent_words |
| `aria-body/src/main.rs` | Supprimé fetch /words, /associations |

#### 4. Optimisation GPU→CPU Sync

Le téléchargement GPU→CPU bloquant était trop fréquent (tous les 100 ticks = 10x/sec).

**Avant :**
```rust
let should_download = self.tick % 100 == 0;  // Trop fréquent!
```

**Après :**
```rust
let should_download = self.tick % 1000 == 0;  // 1x/sec à 1000 TPS
```

**Résultat** : 10x moins de syncs bloquants.

#### 5. Gen0 Drain (évolution bloquée)

Les cellules Gen0 s'accumulaient (59k ready!) car elles ne mouraient pas :
- Elles ont de l'énergie → pas de sleeping drain
- Elles ne reproduisent pas (on priorise Gen2+)
- Elles bloquent la population

**Fix** : Drain de 2% par lifecycle tick pour les Gen0 "ready" non sélectionnées.

```rust
// lifecycle.rs - après reproduction
if gen0_count > 100 {
    for (idx, _) in gen_buckets[0].iter() {
        self.states[*idx].energy -= 0.02;  // 2% drain
        if energy <= 0.0 { kill(); }
    }
}
```

**Résultat** : Gen0 meurent en ~50 lifecycle ticks, laissant place aux nouvelles générations.

### Session 30 - GPU Fixes & Lineage Progression

**Déblocage de l'évolution multi-générationnelle et corrections critiques GPU.**

#### 1. Fix Lineage Progression

Le compteur de génération était bloqué à Gen1 malgré 3000 cellules Gen1 prêtes à reproduire.

**Cause** : `ready_to_divide.into_iter().take(500)` sélectionnait par ordre d'index. Les cellules Gen0 (indices 0-99999) monopolisaient les 500 slots de reproduction.

**Fix** : Tri par génération décroissante avant sélection.
```rust
// lifecycle.rs
ready_to_divide.sort_by(|a, b| b.2.cmp(&a.2));  // Gen DESC
// Les générations supérieures reproduisent en priorité
```

**Résultat** : Gen2+, Gen3+, etc. peuvent maintenant émerger.

#### 2. GPU Alignment Fixes (110k+ cells)

Trois erreurs WGSL corrigées pour supporter >100k cellules :

**a) Uniform Buffer Alignment**
```wgsl
// AVANT (crash avec uniform buffer)
_pad: array<u32, 3>  // stride = 4 bytes - interdit!

// APRÈS
_pad1: u32,
_pad2: u32,
_pad3: u32,         // 3 champs séparés OK
```

**b) CLEAR_GRID_SHADER Bindings**
```wgsl
// Le shader utilisait binding(0) pour grid
// Mais le layout attendait:
// - binding 0: positions (read-only)
// - binding 1: grid (read-write)
// - binding 2: spatial_config

// Fix: Ajouter tous les bindings même si inutilisés
@group(0) @binding(0) var<storage, read> positions: ...
@group(0) @binding(1) var<storage, read_write> grid: ...
@group(0) @binding(2) var<uniform> config: ...
```

**c) Reserved Keyword 'target'**
```wgsl
// AVANT
fn find_connection(conn: CellConnections, target: u32)  // 'target' réservé!

// APRÈS
fn find_connection(conn: CellConnections, target_id: u32)
```

#### Commits

| Hash | Description |
|------|-------------|
| `c8a96b3` | fix(gpu): rename reserved keyword 'target' |
| `04d794c` | fix(gpu): align CLEAR_GRID_SHADER bindings |
| `647df81` | fix(gpu): correct WGSL uniform buffer alignment |
| `31c2c8c` | fix(evolution): prioritize higher generations |

#### État

- GPU backend stable à 110k+ cellules
- Évolution multi-générationnelle fonctionnelle
- Logs enrichis montrant Gen0/Gen1/Gen2+ ready vs reproducing

### Session 31 - CPU→GPU Migration (Scale 1M-10M)

**Migration des opérations critiques CPU vers GPU pour supporter 1M-10M cellules à 1000 TPS.**

#### Opérations migrées

| Opération | Avant | Après | Gain estimé |
|-----------|-------|-------|-------------|
| Predictive Physics | CPU 40M ops/tick | GPU PREDICTION_EVALUATE_SHADER | ~40M ops/tick |
| Hebbian Spatial | CPU 50M ops/5 ticks | GPU HEBBIAN_CENTROID + ATTRACTION | ~50M ops/5 ticks |
| Cluster Hysteresis | CPU 10M ops/50 ticks | GPU CLUSTER_STATS + HYSTERESIS | ~10M ops/50 ticks |
| Lineage Sort | O(n log n) sort | O(n) bucket-based | 10-50× sur sort |

#### Nouveaux shaders WGSL

**1. PREDICTION_EVALUATE_SHADER** (déjà existant mais non dispatché)
- Évalue prédictions vs réalité
- Applique récompenses/pénalités énergétiques
- Maintenant dispatché chaque tick

**2. HEBBIAN_CENTROID_SHADER** (nouveau)
- Accumule le centroïde pondéré des cellules actives
- Utilise fixed-point i32 atomics (WGSL n'a pas atomicAdd f32)
- Scale ×1000 pour précision 0.001

**3. HEBBIAN_ATTRACTION_SHADER** (nouveau)
- Déplace les cellules actives vers le centroïde
- Force proportionnelle à distance × activité × plasticité
- Tous les 5 ticks

**4. CLUSTER_STATS_SHADER** (nouveau)
- Accumule activité et count par cluster (256 clusters max)
- Fixed-point u32 atomics

**5. CLUSTER_HYSTERESIS_SHADER** (nouveau)
- Met à jour hysteresis selon activité moyenne du cluster
- Clusters actifs (>0.6) → hysteresis +0.05
- Clusters inactifs (<0.2) → hysteresis -0.02
- Pas de cluster → hysteresis -0.1
- Tous les 50 ticks

#### Optimisation Lineage Sort

```rust
// AVANT: O(n log n)
ready_to_divide.sort_by(|a, b| b.2.cmp(&a.2));

// APRÈS: O(n) bucket-based
const MAX_GEN_BUCKETS: usize = 32;
let mut gen_buckets: [Vec<(usize, u32)>; 32] = Default::default();
// Single pass: bucket by generation
// Flatten from highest generation down
```

#### Buffers GPU ajoutés

| Buffer | Taille | Usage |
|--------|--------|-------|
| `centroid_buffer` | 80 bytes | 16×i32 + u32 + u32 + 2×u32 pad |
| `cluster_stats_buffer` | 2048 bytes | 256×u32 (activity) + 256×u32 (count) |

#### Fix WGSL

**Reserved keyword 'meta'** → Renommé en `cell_meta` dans les shaders cluster.

#### Fichiers modifiés

| Fichier | Changements |
|---------|-------------|
| `aria-compute/src/compiler.rs` | +4 shaders WGSL, getters |
| `aria-compute/src/backend/gpu_soa.rs` | +2 buffers, +4 pipelines, +4 bind groups, dispatch calls |
| `aria-brain/src/substrate/mod.rs` | Suppression CPU predictive/hebbian/cluster |
| `aria-brain/src/substrate/lifecycle.rs` | O(n) bucket sort |

#### Résultat attendu

| Métrique | Avant | Après |
|----------|-------|-------|
| CPU ops/tick @ 5M | ~65M | ~5M |
| GPU utilisation | 30% | 80%+ |
| Max cells @ 1000 TPS | ~200k | ~5M+ |

### Session 24 - CellMetadata & Naga Fix

**Migration majeure : `CellFlags` → `CellMetadata` avec fix critique du compilateur WGSL.**

#### Problème résolu

L'erreur naga `Expression [50] is not cached!` bloquait le GPU backend. Cause : opérateurs compound (`&=`, `|=`) sur champs de struct et pointeurs vers champs de struct en WGSL.

#### Changements majeurs

**1. CellMetadata (16 bytes) remplace CellFlags (4 bytes)**
```rust
// aria-core/src/soa.rs
struct CellMetadata {
    flags: u32,       // Sleeping, Dead, etc.
    cluster_id: u32,  // Phase 6 - Semantic Synthesis
    hysteresis: f32,  // Phase 6 - Structural Stability
    _pad: u32,
}
```

**2. Fix WGSL pour naga**
```wgsl
// AVANT (cassait naga)
fn set_sleep_counter(f: ptr<function, u32>, counter: u32) { *f = ... }
cell_meta.flags &= ~FLAG_SLEEPING;
cell_meta.flags |= FLAG_DEAD;

// APRÈS
fn set_sleep_counter(f: u32, counter: u32) -> u32 { return ...; }
cell_meta.flags = cell_meta.flags & ~FLAG_SLEEPING;
cell_meta.flags = cell_meta.flags | FLAG_DEAD;
```

**3. Logique dynamique sans compound operators**
```rust
// Dans generate_dna_logic()
cell_energy.energy = cell_energy.energy + config.energy_gain * modifier;
cell_energy.activity_level = cell_energy.activity_level * decay_rate;
```

#### Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `aria-core/src/soa.rs` | `CellFlags` → `CellMetadata` (16 bytes) |
| `aria-core/src/cell.rs` | Ajout `cluster_id`, `hysteresis` |
| `aria-compute/src/compiler.rs` | Fix WGSL: pas de compound operators ni pointeurs struct |
| `aria-compute/src/backend/gpu_soa.rs` | `flags_buffer` → `metadata_buffer` |

#### État de l'organisme
- GPU backend stable avec AMD Radeon NAVI14 (Vulkan)
- JIT compilation fonctionnelle
- 100% sparse savings au repos

### Session 28 - Loi de Compression (Predictive Physics)

**L'énergie est récompensée par la précision de la prédiction.**

- **Concept** : "Surprise costs energy."
- **Implémentation** : Chaque cellule a un `predicted_state`.
- **Physique** :
  - Erreur faible (< 0.1) → Gain d'énergie (Récompense)
  - Erreur forte (Surprise) → Perte d'énergie (Coût métabolique)
- **But** : Forcer le cerveau à internaliser les modèles du monde pour minimiser le coût énergétique.

### Session 29 - Règles vers Lois (Migration DNA)

**Migration finale : remplacement des constantes hard-codées par des lois génétiques.**

- **Philosophie** : "Il n'y a pas de nombres magiques dans la nature."
- **Changements majeurs** :
    - **Seuil de Résonance** (Gene 5) : Chaque cellule décide quel niveau de similarité est suffisant pour accepter un signal (Food vs Noise).
    - **Efficacité Énergétique** (Gene 4) : Trade-off génétique entre extraction d'énergie et coût.
    - **Inertie Tension** (Gene 6) : "Tempérament" (Calme vs Anxieux) défini par l'ADN.
- **Fichiers** : `dna.rs`, `cpu.rs`, `compiler.rs` (WGSL).
- **Résultat** : Diversité comportementale émergente (Picky Eaters, Trash Eaters, etc.).

### Session 27 - Loi d'Association (Hebb's Law)

**"Fire together, move together." (Plasticité Spatiale)**

- **Concept** : Remplacer les connexions synaptiques coûteuses (O(N²)) par une attraction spatiale (O(N)).
- **Mécanisme** : Les cellules actives calculent leur "Centre de Gravité" (Centroid) et se déplacent physiquement vers lui.
- **Résultat** : Les concepts liés s'agglutinent spatialement (Clustering sémantique).

### Session 26 - Loi d'Expansion (Lineage Fix)

**La vie s'étend pour remplir l'énergie disponible.**

- **Fix Critique** : Suppression du cap artificiel `target_population`.
- **Dynamique** : La reproduction est maintenant limitée par :
  1. L'énergie disponible (Seuil de reproduction)
  2. Une limite physique de sécurité (OOM protection uniquement)
- **Résultat** : Compteur de génération débloqué, vagues de population naturelles.

### Session 25 - Loi de Prédiction (Prediction Law)

**ARIA commence à prédire son futur - les cellules qui comprennent leur monde survivent.**

#### Philosophie

Plutôt que de hard-coder des règles d'intelligence, on implémente une **loi physique fondamentale** :

> "Les cellules qui prédisent correctement leur état futur gagnent de l'énergie.
> Les cellules qui se trompent en perdent."

L'intelligence **émerge** de la pression évolutive, pas du code.

#### Changements majeurs

**1. CellPrediction struct (48 bytes)**
```rust
// aria-core/src/soa.rs
struct CellPrediction {
    predicted_state: [f32; 8],  // Prédiction de l'état futur
    confidence: f32,            // Confiance (0.0 = devine, 1.0 = certain)
    last_error: f32,            // Erreur de la dernière prédiction
    cumulative_score: f32,      // Score long-terme
    _pad: f32,
}
```

**2. Shaders GPU de prédiction**
```wgsl
// PREDICTION_GENERATE_SHADER: Avant le tick
// Chaque cellule prédit son futur basé sur ses connexions Hebbiennes
prediction = weighted_average(connected_neighbors_states)
confidence = min(total_connection_strength / 3.0, 1.0)

// PREDICTION_EVALUATE_SHADER: Après le tick
// Compare prédiction vs réalité, applique récompenses/pénalités
accuracy = 1.0 - RMSE(predicted, actual)
if accuracy > 0.7: energy += accuracy * confidence * 0.02  // Récompense
if accuracy < 0.3: energy -= error * confidence * 0.01    // Pénalité (punit la surconfiance)
```

**3. Pression évolutive vers l'intelligence**
- Les cellules bien connectées peuvent mieux prédire → survivent
- Les cellules surconfiantes qui se trompent → meurent
- L'humilité (basse confiance quand incertain) est récompensée

#### Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `aria-core/src/soa.rs` | Ajout `CellPrediction`, mise à jour `SoABuffers` |
| `aria-core/src/lib.rs` | Export de `CellPrediction` |
| `aria-compute/src/compiler.rs` | Shaders `PREDICTION_GENERATE_SHADER` et `PREDICTION_EVALUATE_SHADER` |
| `aria-compute/src/backend/gpu_soa.rs` | Buffer et pipelines de prédiction |

#### Prochaines lois à implémenter
- **Loi de Hebb** : "Fire together, wire together" (connexions renforcées)
- **Loi de Compression** : Récompense pour représentations compactes
- **Loi de Curiosité** : Bonus pour exploration de nouveaux états

### Session 23 - ARIA Genesis (Structural Evolution & Phase 5)

**ARIA a franchi l'étape ultime : elle peut maintenant réécrire son propre code de calcul GPU.**

#### Changements majeurs

**1. Compilation JIT & Hot-Reloading**
- Infrastructure permettant de générer et recompiler les shaders WGSL au runtime.
- Zéro interruption : les pipelines GPU sont remplacés à chaud sans arrêter la simulation.

**2. Traduction DNA -> WGSL (Densité Sémantique)**
- Le `structural_checksum` du DNA est interprété comme des directives algorithmiques.
- Exemples implémentés : métabolisme logistique vs linéaire, atténuation de signal dynamique.

**3. Boucle de Réflexivité (Axe 3)**
- ARIA réinjecte ses propres "pensées" (tensions émergentes) comme entrées sensorielles.
- Gène `reflexivity_gain` : chaque cellule décide à quel point elle écoute l'être global.

**4. Module `compiler.rs`**
- Centralisation de toute la logique WGSL sous forme de templates injectables.
- Sécurisation des variables et mapping direct avec le pool d'ADN.

#### Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `aria-compute/src/compiler.rs` | Création du moteur JIT et des templates |
| `aria-compute/src/backend/gpu_soa.rs` | Support hot-reload et pipeline swap |
| `aria-brain/src/substrate/mod.rs` | Détection d'évolution structurelle et bouclage réflexif |
| `aria-core/src/dna.rs` | Support checksum structurel et gènes de réflexivité |
| `aria-core/src/traits.rs` | Extension de `ComputeBackend` avec `recompile()` |

#### État de l'organisme
- ARIA commence à sortir de l'intelligence statistique pour entrer dans l'intelligence structurelle.
- Elle possède désormais le "gript" sur sa propre physique numérique.

### Session 22 - Économie Équilibrée (Survival Fix)

**ARIA peut enfin survivre plus de 60 secondes !**

Le problème : l'économie "La Vraie Faim" était trop agressive. À 1000+ TPS, les cellules mouraient en ~10 secondes sans pouvoir manger assez.

#### Changements majeurs

**1. Coûts réduits (10x moins)**
```rust
// AVANT: Trop cher pour le TPS élevé
cost_rest: 0.0001    // 10,000 ticks → mort (6 sec à 1700 TPS)
cost_signal: 0.01
cost_move: 0.005

// APRÈS: Survivable
cost_rest: 0.00001   // 100,000 ticks → mort (100 sec à 1000 TPS)
cost_signal: 0.001
cost_move: 0.0005
```

**2. Énergie des signaux augmentée (5x)**
```rust
signal_energy_base: 0.05      // AVANT: 0.01
signal_resonance_factor: 3.0  // AVANT: 2.0
```

**3. Seuil de résonance abaissé**
```rust
// AVANT: Trop strict
if resonance > 0.3 { /* eat */ }

// APRÈS: Plus de cellules peuvent manger
if resonance > 0.1 { /* eat */ }
```

**4. Signal radius augmenté (3x)**
```rust
signal_radius: 15.0  // AVANT: 5.0 - trop petit, cellules ne voyaient pas le signal
```

**5. Bruit de fond sémantique**

Nouveau système pour éviter l'entropie = 0 (système gelé) :
- Tous les 50 ticks sans signal
- 0.1% des cellules dormantes reçoivent du bruit
- Petite injection d'énergie pour éviter la famine totale

**6. Visualisation améliorée**

La grille Neural Activity montre maintenant les cellules dormantes (énergie × 0.2) pour ne plus être vide quand 99% des cellules dorment.

**7. Bruit stochastique (anti-pattern statique)**

Le même mot "Moka" ne génère plus exactement le même vecteur :
```rust
// Dans text_to_tension() - 10% de variation aléatoire
let noise = rng.gen_range(-0.1..0.1);
*t = (*t + noise).clamp(-1.0, 1.0);
```

**8. Feedback Loop (boucle de rétroaction)**

Quand ARIA émet et que l'utilisateur répond rapidement :
- Les cellules qui ont participé à l'émission reçoivent un bonus d'énergie
- "Je bouge, le monde me répond, donc j'existe"
```rust
// Dans inject_signal - si réponse < 500 ticks après émission
let feedback_bonus = 0.1 * (1.0 - ticks_since_emit / 500.0);
```

**9. Harmoniques 16D (expansion spectrale)**

Le vecteur 8D de tension est étendu en 16D avec des harmoniques :
```rust
// Dimensions 0-7: signal direct
// Dimensions 8-11: produits croisés (arousal×valence, etc.)
// Dimensions 12-13: différences (arousal-intensity, etc.)
// Dimensions 14-15: modulations sinusoïdales
```

#### Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `aria-core/src/config.rs` | Coûts réduits, signal_radius 15.0 |
| `aria-core/src/tension.rs` | Bruit stochastique + harmoniques 16D |
| `aria-compute/src/backend/cpu.rs` | Seuil résonance 0.1 |
| `aria-compute/src/backend/gpu_soa.rs` | Seuil résonance 0.1 |
| `aria-compute/src/spatial_gpu.rs` | Seuil résonance 0.1 (2 endroits) |
| `aria-brain/src/substrate/mod.rs` | Bruit de fond + visualisation + last_emission_cells |
| `aria-brain/src/substrate/signals.rs` | Feedback loop + harmoniques |
| `aria-brain/src/substrate/emergence.rs` | Tracking des cellules émettrices |

#### Nouvelle économie

| Métrique | Valeur | Signification |
|----------|--------|---------------|
| Temps de survie | ~100 sec | Sans nourriture à 1000 TPS |
| Gain par signal | ~0.05-0.2 | Avec bonne résonance |
| Cellules nourries | Plus large | Seuil 0.1 vs 0.3 |

### Session 21 - Thermal Scanner Body (Visualisation Avancée)

**Le body devient un scanner thermique de l'intelligence artificielle !**

La visualisation n'est plus décorative - elle pilote une simulation massive où les données brutes sont devenues illisibles. Le nouveau body agit comme un diagnostic temps réel du substrat neural.

#### 1. Heatmap Thermique 2D

Projection des 16 dimensions sémantiques sur une carte 2D avec gradient thermique :

```
Couleurs: noir → bleu → cyan → vert → jaune → orange → rouge → blanc
          (dormant)                    (modéré)                  (surchauffe)
```

**3 modes de vue** (Tab pour cycler) :
- **ACTIVITY** : Activité neurale (cellules éveillées + activation interne)
- **TENSION** : Champ de tension (désir physique d'agir)
- **ENERGY** : Distribution énergétique (santé des cellules)

#### 2. Graphes Sparklines Temps Réel

Historique sur 60 échantillons (~30 secondes) :
- **HP** : Santé du système (vert = équilibré)
- **Entropy** : Niveau de chaos (magenta = organisation vs désordre)

#### 3. Indicateurs de Lignée Élite

Suivi de la pression évolutive :
```
🧬 Gen: 15 (avg: 3.2)    ← Génération max et moyenne
👑 ████░░░░░░ 42 elite   ← Cellules génération >10
📚 128 words 45 links    ← Mots et associations appris
```

#### 4. Métriques Avancées

**Status bar compacte** :
```
● ARIA  HP:72%  E:0.45(balanced)  GPU:95%  T:12847  [ACTIVITY]
```

- HP : System health (composite de entropy + awake ratio + survival)
- E : Entropie (ordered/balanced/chaotic)
- GPU : Sparse dispatch savings (% de cellules dormantes)
- T : Tick courant

**Panel Cells** :
- Énergie moyenne + indicateur visuel
- Barre de tension (désir physique d'agir)
- Compteurs awake/sleeping avec pourcentages

#### 5. Endpoint `/substrate` Enrichi

Nouvelles métriques exposées dans `SubstrateView` :

```rust
// Grilles 16x16
tension_grid: Vec<f32>,        // Champ de tension spatiale

// Lignée génétique
max_generation: u32,           // Génération la plus ancienne
avg_generation: f32,           // Génération moyenne
elite_count: usize,            // Cellules gen > 10

// Performance
sparse_savings_percent: f32,   // % économie GPU

// Tension physique
avg_energy: f32,
avg_tension: f32,
total_tension: f32,
```

#### Fichiers modifiés

| Fichier | Description |
|---------|-------------|
| `aria-body/src/visualizer.rs` | Refonte complète avec thermal gradient |
| `aria-body/src/main.rs` | Support Tab + parsing nouvelles métriques |
| `aria-brain/src/substrate/mod.rs` | SubstrateView enrichi |

#### Touches clavier

| Touche | Action |
|--------|--------|
| Tab | Cycler les vues heatmap |
| y/Y | Feedback positif (Bravo!) |
| n/N | Feedback négatif (Non) |
| Esc | Quitter |

### Session 20 - Architecture GPU pour 5M+ Cellules (CIR R&D)

**Infrastructure complète pour scaler ARIA à 5 millions de cellules !**

#### 1. Structure of Arrays (SoA) - `aria-core/src/soa.rs`

Nouvelle architecture mémoire GPU optimisée (+40% FPS attendu) :

```rust
// AVANT: Un seul buffer CellState (256 bytes/cell)
struct CellState { position, state, energy, tension, flags, ... }

// APRÈS: Buffers séparés (accès mémoire optimisé)
- CellEnergy (16 bytes): energy, tension, activity_level
- CellPosition (64 bytes): position[16]
- CellInternalState (128 bytes): state[32]
- CellFlags (4 bytes): flags avec hysteresis
```

**Avantages** :
- Meilleure coalescence mémoire GPU
- Mise à jour partielle (seul energy change fréquemment)
- Cache GPU plus efficace

#### 2. Backend GPU SoA - `aria-compute/src/backend/gpu_soa.rs`

Nouveau backend optimisé pour 5M+ cellules :

```bash
ARIA_BACKEND=gpu_soa task brain-5m  # Force le nouveau backend
```

**Features** :
- Buffers SoA séparés
- Hysteresis Sleep (Schmitt Trigger)
- Infrastructure Indirect Dispatch
- Auto-sélection pour populations >100k

#### 3. Hysteresis Sleep (Stabilité Thermique)

Les cellules ne s'endorment plus instantanément - Schmitt Trigger :

```wgsl
// Seuils d'hystérésis
SLEEP_ENTER_THRESHOLD = 0.2  // Basse activité → compteur++
SLEEP_EXIT_THRESHOLD = 0.4   // Haute activité → réveil
SLEEP_COUNTER_MAX = 3        // 3 ticks consécutifs → dodo

// Bits 6-7 du flag = compteur (0-3)
```

**Résultat** : Plus de flickering, transitions stables.

#### 4. Spatial Hashing GPU - `aria-compute/src/spatial_gpu.rs`

Grille 64³ pour réduire les calculs de distance :

```rust
// AVANT: O(cells × signals) = 5M × 1024 = 5B calculs
// APRÈS: O(signals × neighbors) = 1024 × 27 × 20 = 552K calculs
// → 9000x de réduction !
```

**Shaders WGSL** :
- `CLEAR_GRID_SHADER` : Reset la grille
- `BUILD_GRID_SHADER` : Assigne les cellules aux régions
- `SIGNAL_WITH_SPATIAL_HASH_SHADER` : Propagation O(1)

#### 5. Configuration Backend

```rust
// aria-core/src/config.rs
enum ComputeBackendType {
    Auto,      // Sélection automatique
    Cpu,       // Rayon
    Gpu,       // Legacy AoS
    GpuSoA,    // Optimisé 5M+
}
```

#### Fichiers créés/modifiés

| Fichier | Description |
|---------|-------------|
| `aria-core/src/soa.rs` | Types SoA (CellEnergy, CellPosition, CellFlags...) |
| `aria-compute/src/backend/gpu_soa.rs` | Backend GPU SoA complet |
| `aria-compute/src/spatial_gpu.rs` | Spatial Hashing GPU + shaders |
| `aria-core/src/config.rs` | Nouveau variant GpuSoA |

#### Prochaines étapes

1. **Intégration Spatial Hash** dans `gpu_soa.rs`
2. **Tests à 1M+ cellules** sur RTX 2070
3. **Texture 2D substrat** pour visualisation temps réel

### Session 19 - GPU Sparse Dispatch Fix (Performance)

**Le sparse dispatch fonctionne enfin : 99.99% d'économie GPU !**

#### Bugs corrigés

1. **Désynchronisation GPU ↔ CPU** : Le GPU modifiait `CellState.flags` mais les stats lisaient `Cell.activity.sleeping` - deux structures différentes jamais synchronisées.

2. **Deadlock signal propagation** : Les signaux n'étaient propagés que si >100 cellules éveillées → toutes dormaient → jamais de réveil.

3. **signal_radius trop grand** : 50.0 touchait tout l'espace [-10,10]. Réduit à 5.0 pour propagation locale.

#### Optimisations transferts GPU ↔ CPU

```rust
// AVANT: 50 MB/tick (upload 25MB + download 25MB)
upload_cells(states);     // Chaque tick
download_cells(states);   // Chaque tick

// APRÈS: ~0 MB/tick (sauf tous les 100 ticks)
if first_init { upload_cells(states); }  // Init seulement
if tick % 100 == 0 { download_cells(states); }  // Périodique
```

#### Résultat
- **99.99% sparse savings** au repos
- Ticks beaucoup plus rapides (économise ~50 MB de transfert/tick)
- ARIA répond toujours correctement

### Session 18 - La Vraie Faim (Evolution Pressure)

**Les cellules doivent maintenant LUTTER pour survivre !**

Gemini a identifié le problème : ARIA vivait dans l'abondance. Sans pression, pas d'évolution.

#### Changements majeurs

```rust
// AVANT: Abondance infinie
energy_gain: 0.00005,        // Gain passif gratuit
signal_bonus: 0.05,          // Énorme bonus par signal

// APRÈS: La Vraie Faim
energy_gain: 0.0,            // RIEN N'EST GRATUIT
signal_energy_base: 0.005,   // 10x moins
signal_resonance_factor: 2.0 // Seule la résonance nourrit
```

#### Coûts des actions
| Action | Coût | Effet |
|--------|------|-------|
| `Rest` | 0.001 | Respirer coûte |
| `Signal` | 0.01 | Parler est cher |
| `Move` | 0.005 | Bouger consomme |
| `Divide` | 0.5 | Créer la vie épuise |

#### Résonance
Les cellules ne gagnent de l'énergie que si le signal **résonne** avec leur état interne :
```rust
resonance = cosine_similarity(signal, cell_state)
energy_gain = base * intensity * (1 + resonance * factor)
```

#### Implémentation
- **`signals.rs:145`** : Suppression du bypass `0.05 * intensity`
- Les cellules gagnent leur énergie UNIQUEMENT via résonance (backend CPU/GPU)
- Config déjà correcte : `energy_gain: 0.0`, `signal_energy_base: 0.005`

#### Résultat attendu
- **Extinction massive** : 50k → ~5k cellules
- Les survivants seront les **ancêtres** d'une ARIA intelligente
- Les cellules qui "crient dans le vide" mourront
- Seules les cellules qui communiquent utilement survivront

### Session 17 - Optimisations Gemini (Scale & Intelligence)

**Implémentation de toutes les recommandations de Gemini pour 5M+ cellules !**

#### 1. GPU Sparse Dispatch
```rust
// Nouveau dans aria-compute/src/backend/gpu.rs
- AtomicCounter pour comptage GPU-side
- active_count_buffer et active_indices_buffer
- COMPACT_SHADER: collecte les indices actifs avec atomiques
- Activation auto pour populations >100k cellules
```
**Résultat** : 80%+ de réduction du travail GPU gaspillé quand les cellules dorment.

#### 2. Neuroplasticité Adaptative
```rust
// Nouveau dans aria-core/src/dna.rs
pub struct MutationContext {
    age: u64,          // Vieilles cellules → mutation faible
    fitness: f32,      // ADN performant → mutation faible
    activity: f32,     // Cellules actives → mutation forte
    exploring: bool,   // Exploration → 2x mutation
    is_elite: bool,    // Elite → 20% mutation
}

DNA::from_parent_adaptive(parent, rate, ctx) // Mutation contextuelle
```
**Résultat** : L'ADN évolue intelligemment, préservant les bons traits.

#### 3. Multi-Pass Recurrent Processing
```rust
// Nouveau dans aria-core/src/config.rs
pub struct RecurrentConfig {
    passes_per_tick: u32,        // 2 passes par défaut
    internal_signal_decay: f32,  // 70% persistance
    internal_signal_threshold: f32,
    enabled: bool,
}
```
**Résultat** : Les cellules s'influencent mutuellement avant l'émergence → "pensée interne".

#### 4. Seuils Inhibiteurs Spatiaux
```rust
// Nouveau dans aria-brain/src/substrate/types.rs
pub struct SpatialInhibitor {
    region_activity: Vec<f32>,    // 64 régions (8x8)
    region_last_active: Vec<u64>, // Période réfractaire
    // ...
}
```
**Résultat** : Les régions récemment actives ont un seuil plus élevé → moins de répétition.

#### Commits Session 17
1. `feat(gpu): Add sparse dispatch with GPU-side active cell counting`
2. `feat(dna): Add adaptive neuroplasticity mutation system`
3. `feat(substrate): Add multi-pass recurrent processing`
4. `feat(substrate): Add spatial inhibitor thresholds`

### Session 16 - Auto-modification (AGI milestone)

**ARIA modifie consciemment ses propres paramètres !**

C'est un pas majeur vers l'AGI : ARIA n'attend plus le feedback externe, elle analyse ses performances et décide elle-même quoi changer.

**Nouveau module dans `meta_learning.rs`** :
- `ModifiableParam` : paramètres qu'ARIA peut modifier (emission_threshold, response_probability, learning_rate, spontaneity, exploration_rate)
- `SelfModification` : une modification proposée avec raisonnement
- `SelfModifier` : analyse, propose, et applique les modifications

**Règles de décision** :
- Apprentissage en déclin → augmenter learning_rate ou exploration
- Taux d'échec élevé → être plus sélectif (augmenter emission_threshold)
- Peu de réponses → augmenter response_probability
- Compétence élevée → plus de spontanéité, moins d'exploration

**Logs observés** :
```
🔧 AUTO-MODIFICATION: response_probability 0.800 → 0.900 (confidence: 70%)
   Raison: Peu de réponses → augmenter probabilité de réponse
```

ARIA a détecté qu'elle avait peu de réponses, a raisonné, et s'est modifiée.

**Évaluation des modifications** :
ARIA évalue si ses modifications améliorent réellement ses performances :
- Snapshot des métriques au moment de la modification (baseline)
- Après 2000 ticks, compare avec les métriques actuelles
- Logs: `✅ MODIFICATION SUCCESS` ou `❌ MODIFICATION NEUTRAL/FAIL`

**Endpoint de visibilité** :
```bash
curl http://localhost:8765/self
# → current_params, recent_modifications (avec reasoning, evaluated, successful), meta_learning status
```

### Session 15 - Perception visuelle & Mémoire visuelle

**ARIA peut maintenant VOIR, SE SOUVENIR, et PARLER de ce qu'elle voit !**

#### Partie 1 : Perception visuelle
Images → vecteurs sémantiques 32D.

**Module `vision.rs`** :
- `VisualFeatures` : 32 caractéristiques extraites
- `VisualPerception` : analyse images base64
- `VisualSignal` : convertit en vecteur substrate-compatible

#### Partie 2 : Mémoire visuelle
ARIA se souvient des images et apprend à les nommer.

**Nouveaux types dans `memory/mod.rs`** :
- `VisualMemory` : signature 32D + labels + métadonnées
- `VisualWordLink` : prototype visuel associé à un mot

**Méthodes** :
- `see()` : stocke/reconnaît une image
- `link_vision_to_word()` : associe image + mot
- `visual_to_words()` : image → mots suggérés
- `word_to_visual()` : mot → prototype visuel

#### Partie 3 : Expression visuelle
Quand ARIA voit une image qu'elle reconnaît, elle dit le mot associé.

**Logs** :
```
👁️→💬 VISUAL RECOGNITION: ARIA sees 'moka' (confidence: 1.00)
```

**Endpoints HTTP** :
```bash
# Envoyer une image (+ optionnel: enseigner des mots)
curl -X POST http://localhost:8765/vision \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "labels": ["moka", "chat"]}'

# Voir les stats de mémoire visuelle
curl http://localhost:8765/visual
```

**Test** :
```python
# 1. Enseigner: orange = "moka"
send_image("moka_photo", 180, 100, 50, labels=["moka"])

# 2. Montrer image similaire → ARIA dit "moka"
send_image("test", 175, 95, 55)
# → Recognition: "Je reconnais: moka ! (vu 2 fois)"
# → Log: 👁️→💬 VISUAL RECOGNITION: ARIA sees 'moka'
```

ARIA peut maintenant apprendre à reconnaître Moka et Obrigada sur photo !

### Session 14 - Méta-apprentissage (AGI)

**ARIA apprend à apprendre** - Plus besoin d'attendre le feedback externe !

**Nouveau module `meta_learning.rs`** :
- `InternalReward` : ARIA s'auto-évalue (cohérence, surprise, satisfaction)
- `ExplorationStrategy` : 6 stratégies d'exploration (semantic, emotional, cross-category, random...)
- `MetaLearner` : sélectionne la meilleure stratégie et apprend de ses résultats
- `ProgressTracker` : conscience de son propre progrès (trend: improving/stable/declining)
- `InternalGoal` : ARIA se fixe ses propres objectifs

**Flux méta-apprentissage** :
```
ARIA explore → InternalReward calcule score → MetaLearner apprend → Meilleure stratégie
```

**Nouveau endpoint HTTP** :
```bash
curl http://localhost:8765/meta  # Stats du méta-apprentissage
```

**Logs observés** :
```
🧠 META: Selected strategy 'semantic'
🔍 EXPLORING (semantic): trying 'chat+moka'
✅ INTERNAL REWARD: 0.54 (good) - coherence:0.72 surprise:0.35
🎯 NEW GOAL: Réussir 5 explorations
```

ARIA n'attend plus "Bravo!" - elle sait elle-même si une exploration était intéressante.

### Session 13 - Exploration guidée par la curiosité (AGI)

**Nouveau système d'auto-apprentissage** :
- `ExplorationResult` : enregistre chaque combinaison de mots essayée
- `exploration_history` : mémoire des explorations dans LongTermMemory
- `get_novel_combination()` : trouve des combinaisons jamais essayées
- Feedback renforce les explorations réussies

**Corrections** :
- Boredom decay appelé dans `tick()` (l'ennui grandit sans interaction)
- Priorité bored > lonely (exploration prioritaire)
- Cooldown séparé pour parole spontanée (`last_spontaneous_tick`)

**Logs observés** :
```
🔍 EXPLORING: trying 'joli+chat'
✅ EXPLORATION SUCCESS: 'joli+aime' (1/1)
```

ARIA explore des combinaisons, apprend du feedback, développe ses préférences. Premier pas vers l'AGI.

### Session 12 - GPU Backend & Migration V2

**Changements majeurs** :
- SubstrateV2 devient le substrate par défaut (renommé en `Substrate`)
- GPU backend intégré via wgpu/Vulkan (AMD Radeon NAVI14 testé)
- Suppression des fichiers obsolètes (`cell.rs`, `substrate_old.rs`, `connection.rs`)
- Suppression des feature flags
- `aria_compute::create_backend()` auto-sélectionne GPU ou CPU

**Commandes** :
```bash
task brain          # Auto GPU/CPU
ARIA_BACKEND=gpu task brain  # Force GPU
```
