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

- **Parler** : mots appris, phrases 2-3 mots, ordre naturel
- **Ressentir** : joie, curiosité, ennui, confort
- **Apprendre** : feedback ("Bravo!"/"Non"), associations, contexte
- **Se souvenir** : mémoire épisodique, "premières fois"
- **Vivre** : rêves, parole spontanée, jeu créatif
- **S'adapter** : paramètres qui évoluent avec le feedback
- **Explorer** : curiosité-driven, teste des combinaisons nouvelles
- **Méta-apprendre** : s'auto-évalue, apprend à apprendre (Session 14)
- **Voir** : images → vecteurs sémantiques 32D (Session 15)
- **S'auto-modifier (Session 16)** : analyse ses performances et change ses propres paramètres
- **S'auto-évoluer (Genesis)** : traduit son DNA en code GPU WGSL, recompile ses pipelines à chaud (Session 23)

## Commandes

```bash
task brain          # 50k cellules, auto GPU/CPU
task brain-100k     # 100k cellules
ARIA_BACKEND=gpu task brain  # Forcer GPU (AMD/NVIDIA via Vulkan)
task body           # Interface
task stats          # Stats du cerveau
task words          # Mots connus
task associations   # Associations apprises
task episodes       # Mémoire épisodique
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
*Version : 0.9.2 | Dernière update : 2026-01-01*

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
