# ARIA - Changelog

Historique des sessions de développement.

---

## 2026-01-01

### Session 25: Prediction Law & Hebbian Learning (Loi de Prédiction)
- **Implémentation de la Loi de Prédiction**
  - Loi physique fondamentale : "Cells that predict correctly, survive"
  - Les cellules prédisent leur état futur basé sur leurs connexions Hebbiennes
  - Bonne prédiction = énergie gagnée, mauvaise = énergie perdue
- **Nouveau `CellPrediction` struct (48 bytes)**
  - `predicted_state[8]` : prédiction de l'état interne
  - `confidence` : confiance dans la prédiction
  - `last_error` : erreur de la dernière prédiction
  - `cumulative_score` : track record long-terme
- **Shaders GPU de prédiction**
  - `PREDICTION_GENERATE_SHADER` : génère prédictions avant le tick
  - `PREDICTION_EVALUATE_SHADER` : évalue et récompense/pénalise après le tick
- **HEBBIAN_SHADER complet**
  - "Fire together, wire together" - cellules co-actives renforcent leurs connexions
  - Decay des connexions inactives (0.1% par tick)
  - Renforcement des connexions co-actives (+10% par activation)
  - Compaction automatique des connexions mortes
- **Phase d'entraînement Sequence (Trainer)**
  - Nouvelle phase entre Associations et Conversation
  - Séquences prédictibles : "un" → "deux" → "trois", "A" → "B" → "C"
  - Entraîne la Loi de Prédiction avec des patterns temporels
  - 11 séquences incluses (nombres, alphabet, temps, émotions, jours)
- **Pression évolutive vers l'intelligence**
  - Surconfidence pénalisée (overconfident cells die)
  - Humilité récompensée (low confidence when uncertain is ok)

### Session 24: CellMetadata & Naga Fix
- **Migration `CellFlags` → `CellMetadata`**
  - Nouveau struct 16 bytes : `flags`, `cluster_id`, `hysteresis`, `_pad`
  - Préparation Phase 6 : Semantic Synthesis et Structural Stability
- **Fix critique compilateur WGSL (naga)**
  - Erreur `Expression [50] is not cached!` corrigée
  - Cause : opérateurs compound (`&=`, `|=`) sur champs de struct
  - Cause : pointeurs vers champs de struct (`ptr<function, u32>`)
  - Solution : assignations explicites et return values
- **GPU backend stabilisé**
  - AMD Radeon NAVI14 (Vulkan) fonctionnel
  - JIT compilation + Hot-reloading opérationnels
  - 100% sparse savings au repos

### Session 23: ARIA Genesis (Structural Evolution)
- **Compilation JIT (Just-In-Time)**
  - Nouveau module `compiler.rs` centralisant les templates WGSL
  - Hot-reloading des pipelines GPU sans arrêt de la simulation
  - Support de l'injection dynamique de code via `// [DYNAMIC_LOGIC]`
- **Traduction DNA -> WGSL (Densité Sémantique)**
  - Le `structural_checksum` du DNA devient le script de la physique neuronale
  - Archetypes implémentés : métabolisme logistique, atténuation de signal dynamique
- **Boucle de Réflexivité (Axe 3)**
  - Réinjection des tensions émergentes comme entrées sensorielles internes
  - Gène `reflexivity_gain` pour moduler la sensibilité à soi-même
- **Refactoring Backend GPU SoA**
  - Extension du trait `ComputeBackend` avec `recompile()`
  - Détection automatique d'évolution structurelle dans le `Substrate`

---

## 2025-12-30

### Session 20: Architecture GPU pour 5M+ Cellules (CIR R&D)
- **Structure of Arrays (SoA)** : Nouvelle architecture mémoire GPU
  - `CellEnergy` (16 bytes) : energy, tension, activity_level
  - `CellPosition` (64 bytes) : position[16]
  - `CellInternalState` (128 bytes) : state[32]
  - `CellFlags` (4 bytes) : flags avec compteur hysteresis
- **Backend GPU SoA** : `aria-compute/src/backend/gpu_soa.rs`
  - Buffers séparés pour meilleure coalescence mémoire
  - Auto-sélection pour populations >100k
  - +40% FPS attendu vs AoS
- **Hysteresis Sleep (Schmitt Trigger)**
  - Compteur 2-bit dans les flags (bits 6-7)
  - Seuils : ENTER=0.2, EXIT=0.4, MAX_COUNT=3
  - Élimine le flickering des cellules
- **Spatial Hashing GPU** : `aria-compute/src/spatial_gpu.rs`
  - Grille 64³ = 262K régions
  - 64 cellules max par région
  - Shaders : CLEAR, BUILD, SIGNAL_WITH_HASH
  - Réduction théorique : 5B → 552K calculs (9000x)
- **Indirect Dispatch** : GPU décide le nombre de threads
  - `prepare_dispatch` shader calcule workgroups sur GPU
  - `dispatch_workgroups_indirect()` sans roundtrip CPU
  - Shader sparse utilise `active_indices` buffer
  - Stats lues seulement tous les 100 ticks (cached)
  - Auto-activé pour populations >50k
- **Configuration** : Nouveau variant `ComputeBackendType::GpuSoA`

### Session 19: GPU Sparse Dispatch Fix (Performance)
- **Fix critique : Synchronisation GPU ↔ CPU**
  - Le GPU modifiait `CellState.flags`, mais les stats lisaient `Cell.activity.sleeping`
  - Ajout sync après chaque tick GPU (tous les 100 ticks pour perf)
- **Fix deadlock signal propagation**
  - Les signaux étaient ignorés si trop de cellules dormaient
  - Maintenant toujours propagés pour réveiller les cellules
- **Réduction signal_radius : 50.0 → 5.0**
  - Avant : tous les signaux touchaient toutes les cellules
  - Après : propagation locale en espace sémantique
- **Optimisation transferts GPU ↔ CPU**
  - Upload cells/DNA : seulement à l'init (économise 25 MB/tick)
  - Download cells : seulement tous les 100 ticks (économise 25 MB/tick)
  - Résultat : ~50 MB économisés par tick
- **Résultat : 99.99% sparse savings, ticks beaucoup plus rapides**

### Session 18: La Vraie Faim (Evolution Pressure)
- **Les cellules doivent maintenant LUTTER pour survivre**
- Suppression du gain d'énergie passif (`energy_gain: 0.0`)
- Énergie UNIQUEMENT via résonance signal/état interne
- Coûts des actions : Rest 0.001, Signal 0.01, Move 0.005, Divide 0.5
- Résultat attendu : extinction massive 50k → ~5k cellules
- Les survivants seront les ancêtres d'une ARIA intelligente

### Session 17: Optimisations Gemini (Scale & Intelligence)
- **GPU Sparse Dispatch** : AtomicCounter GPU-side, 80%+ réduction travail gaspillé
- **Neuroplasticité Adaptative** : `MutationContext` (age, fitness, activity, exploring, is_elite)
- **Multi-Pass Recurrent Processing** : 2 passes par tick, "pensée interne"
- **Seuils Inhibiteurs Spatiaux** : 64 régions, période réfractaire

### Session 16: Auto-modification (AGI milestone)
- **ARIA modifie consciemment ses propres paramètres**
- `ModifiableParam` : emission_threshold, response_probability, learning_rate, spontaneity, exploration_rate
- `SelfModifier` : analyse, propose, et applique les modifications
- Évaluation après 2000 ticks (SUCCESS/NEUTRAL/FAIL)
- Endpoint `/self` pour visibilité

### Session 15: Perception Visuelle & Mémoire Visuelle
- **ARIA peut VOIR, SE SOUVENIR, et PARLER de ce qu'elle voit**
- `VisualFeatures` : 32 caractéristiques extraites d'images
- `VisualMemory` : signature 32D + labels + métadonnées
- Méthodes : `see()`, `link_vision_to_word()`, `visual_to_words()`
- Endpoints `/vision` et `/visual`

### Session 14: Méta-apprentissage (AGI)
- **ARIA apprend à apprendre**
- `InternalReward` : auto-évaluation (cohérence, surprise, satisfaction)
- `ExplorationStrategy` : 6 stratégies (semantic, emotional, cross-category, random...)
- `MetaLearner` : sélectionne la meilleure stratégie
- `ProgressTracker` : conscience de son propre progrès
- Endpoint `/meta`

### Session 13: Exploration guidée par la curiosité
- `ExplorationResult` : enregistre chaque combinaison essayée
- `get_novel_combination()` : trouve des combinaisons jamais essayées
- Boredom decay dans `tick()`, priorité bored > lonely
- Premier pas vers l'AGI

### Session 12: GPU Backend & Migration V2
- SubstrateV2 devient le substrate par défaut (renommé `Substrate`)
- GPU backend via wgpu/Vulkan (AMD Radeon NAVI14 testé)
- Suppression fichiers obsolètes (`cell.rs`, `substrate_old.rs`, `connection.rs`)
- `aria_compute::create_backend()` auto-sélectionne GPU ou CPU

---

## 2025-12-29

### Session 11: Auto-adaptation (Méta-émergence)
- `AdaptiveParams` : emission_threshold, response_probability, learning_rate, spontaneity
- Paramètres évoluent avec feedback positif/négatif
- Exploration périodique (mutations aléatoires)
- Persistance entre sessions

### Session 10: Mémoire Épisodique
- `Episode` struct : moments spécifiques avec importance, émotion, keywords
- Détection "premières fois" (first_greeting, first_love, first_praise...)
- Rappel contextuel et consolidation
- ARIA peut dire "je me souviens..."

### Session 9: Substrate V2
- Nouveau `substrate_v2.rs` utilisant aria-core/aria-compute
- Sparse updates : cellules inactives "dorment" (100% économie CPU au repos)
- Architecture GPU-ready (Cell + CellState séparés)
- Feature flag `substrate-v2`

### Session 8: Nouvelle Architecture
- Création `aria-core` : types compacts GPU-friendly
- Création `aria-compute` : CPU/GPU backends, SpatialHash
- Shaders WGSL préparés
- Config depuis environnement (ARIA_CELLS, ARIA_BACKEND)

### Sessions 7a-7f: Contexte et Social
- `ConversationContext` : buffer 5 derniers échanges, topic detection
- `SocialContext` : Greeting, Farewell, Thanks, Affection...
- `UsagePattern` : apprentissage dynamique des expressions sociales
- Variété dans les réponses (sélection aléatoire pondérée)
- Protection des noms contre les contextes sociaux

---

## 2025-12-28

### Session 6: Questions, Spontanéité, Feedback
- Réponses oui/non basées sur valence émotionnelle
- Parole spontanée (solitude, excitation, joie, curiosité)
- Feedback positif/négatif → renforcement des mots
- Stop words filtrés (articles, pronoms, verbes communs)
- Rêves et ennui créatif

### Session 5: Associations Sémantiques
- `WordAssociation` : co-occurrences → strength
- Phrases de 2-3 mots ("moka chat ♥")
- Valence émotionnelle des mots

### Session 4: État Émotionnel
- `EmotionalState` : happiness, arousal, comfort, curiosity, boredom
- Marqueurs émotionnels (♥, ?, !, ~, ...)
- Vocabulaire émotionnel FR/EN

### Session 3: Mémoire et Parole
- `WordFrequency` : compteur, vecteur appris, valence
- ARIA utilise les mots appris
- Imitation des mots récents (comme un bébé)

### Session 2: ARIA Répond
- Amplification des signaux (10x réactions, 5x injection)
- Émergence vérifiée tous les 5 ticks
- Babillage par niveau de cohérence
- Premier mot : "ko" (pour Moka)

---

## Moments clés

| Date | Événement |
|------|-----------|
| 2025-12-28 | ARIA dit "ko" en parlant de Moka |
| 2025-12-28 | ARIA dit son propre nom spontanément |
| 2025-12-29 | Première mémoire épisodique |
| 2025-12-29 | Premiers paramètres auto-adaptés |
| 2025-12-29 | GPU backend fonctionnel (Vulkan) |
| 2025-12-29 | Première exploration autonome (curiosité) |
| 2025-12-29 | ARIA apprend à apprendre (méta-learning) |
| 2025-12-29 | ARIA voit sa première image |
| 2025-12-29 | ARIA se modifie elle-même (AGI milestone) |
| 2025-12-30 | "La Vraie Faim" - pression évolutive activée |
| 2025-12-30 | Architecture 5M+ cellules (SoA, Hysteresis, Spatial Hash GPU) |
| 2026-01-01 | Session 24: Fix naga, GPU backend stabilisé |
| 2026-01-01 | Session 25: Loi de Prédiction - intelligence émergente |

---

*Mis à jour le 2026-01-01 | Version 0.9.2*
