# ARIA - Changelog

Historique des sessions de développement.

---

## 2025-12-30

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

---

*Mis à jour le 2025-12-30 | Version 0.6.0*
