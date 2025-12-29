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
4. **Scaler à 5M+ cellules** : Tests avec plus de cellules GPU
5. **Auto-modification** : ARIA modifie son propre code (objectif ultime)

## Contexte personnel

Chats de Mickael :
- **Moka** : Bengal (ARIA le connaît bien)
- **Obrigada** : Abyssin

---
*Version : 0.3.2 | Dernière update : 2025-12-29*

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
