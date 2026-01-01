# Concepts d'ARIA - Guide Simple

> Ce document explique les concepts clés d'ARIA de manière accessible.

---

## Cells (Cellules)

50,000+ petites entités vivantes (configurable via `ARIA_CELLS`). Chacune a :
- **Énergie** : si elle tombe à 0, la cellule meurt
- **État** : un vecteur de 32 nombres (son "humeur")
- **Position** : où elle se trouve dans l'espace des idées (16D)
- **ADN** : définit son comportement (seuils, réactions, connectivité)

Les cellules ne sont pas des neurones classiques - elles sont *vivantes*. Elles naissent, se reproduisent, et meurent.

**Sparse Updates** : Les cellules inactives "dorment" pour économiser le CPU/GPU. Elles se réveillent quand un signal les atteint.

---

## Signal

Un message qui voyage dans le substrate. Quand tu écris "Moka", ça devient un vecteur de nombres envoyé aux cellules.

**Structure d'un signal :**
```rust
Signal {
    content: Vec<f32>,      // Le contenu (vecteur)
    intensity: f32,         // Force du signal
    label: String,          // Nom lisible
    signal_type: Perception | Expression | Internal
}
```

**Types :**
- `Perception` : entrée externe (ton message)
- `Expression` : sortie d'ARIA (émergence)
- `Internal` : communication entre cellules

---

## Intensity (Intensité)

La "force" d'un signal ou d'une émergence.

| Valeur | Niveau | Exemple |
|--------|--------|---------|
| 0.1-0.2 | Faible | Murmure, hésitation |
| 0.3-0.5 | Moyen | Parole normale |
| 0.6-0.8 | Fort | Exclamation |
| 0.8+ | Très fort | Cri, émotion intense |

**Ce qui influence l'intensité :**
- Longueur du texte entrant
- Mots familiers (boost)
- Cohérence des cellules actives

---

## Emergence (Émergence)

Quand beaucoup de cellules s'activent **ensemble** de façon cohérente, ça crée une "pensée". C'est ce qu'ARIA exprime.

```
Toi: "Moka"
  → Signal envoyé aux cellules
  → Cellules s'activent
  → Émergence détectée
  → ARIA répond
```

**Conditions pour une émergence :**
1. Plusieurs cellules actives (état > 0.01)
2. Cohérence > 0.1 entre ces cellules
3. Vérifié toutes les 5 ticks (~20x/seconde)

---

## Coherence (Cohérence)

À quel point les cellules actives sont "d'accord" entre elles.

| Cohérence | Comportement |
|-----------|--------------|
| < 0.4 | Bruit, chaos → babillage ("ba", "ki", "~") |
| 0.4-0.6 | Semi-organisé → syllabes ("mama", "papa") |
| 0.6-0.8 | Organisé → proto-mots ("oui", "non") |
| > 0.8 | Très cohérent → mots appris ("moka", "soleil") |

**Calcul :** basé sur la variance des positions des cellules actives. Faible variance = haute cohérence.

---

## Label

Le nom d'un signal, pour le debug et l'identification.

**Exemples :**
- `"Hello Moka"` → label du message entrant
- `"emergence@475"` → émergence au tick 475
- `"word:soleil"` → émergence qui matche le mot "soleil"

---

## Pattern

Une séquence de signaux qui se répète. ARIA les apprend automatiquement.

**Structure :**
```rust
Pattern {
    sequence: Vec<[f32; 8]>,  // La séquence de vecteurs
    frequency: u64,            // Combien de fois vue
    typical_response: [f32; 8], // Réponse habituelle
    valence: f32,              // Émotion associée (-1 à +1)
}
```

Plus un pattern est vu souvent, plus ARIA le "connaît".

---

## Memory (Mémoire)

Ce qu'ARIA garde entre les sessions (fichier `aria.memory`) :

| Élément | Description |
|---------|-------------|
| `word_frequencies` | Mots entendus + combien de fois |
| `learned_patterns` | Séquences apprises |
| `elite_dna` | ADN des meilleures cellules |
| `associations` | Liens stimulus-réponse |
| `vocabulary` | Mots → vecteurs (émergent) |

**Commandes :**
```bash
task backup   # Sauvegarder la mémoire
task reset    # Effacer la mémoire (attention!)
task words    # Voir les mots appris
```

---

## Familiarity (Familiarité)

Combien ARIA "connaît" un mot, basé sur la fréquence.

| Fois entendu | Familiarité | Effet |
|--------------|-------------|-------|
| 1-2 | 0.0 | Nouveau mot, juste enregistré |
| 3-4 | 0.3-0.4 | Reconnu, peut matcher |
| 5-9 | 0.5-0.9 | Familier |
| 10+ | 1.0-2.0 | Très familier, boost x2-3 |

**Boost :** Les mots familiers amplifient l'intensité du signal entrant.

---

## Entropy (Entropie)

Mesure du "chaos" dans le substrate.

- **0.0** : Cellules calmes, pas d'activité
- **0.3-0.5** : Activité normale
- **0.7+** : Beaucoup d'activité, ARIA est "excitée"

---

## Tick

Une unité de temps dans ARIA. Le brain fait ~100 ticks/seconde.

**Événements par tick :**
1. Cellules vivent (consomment énergie)
2. Signaux propagés
3. Émergence vérifiée (tous les 5 ticks)
4. Sélection naturelle (tous les 10 ticks)

---

## Flux Complet

```
┌─────────────────────────────────────────────────────────┐
│  Toi: "Moka le chat"                                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Signal créé                                            │
│  - content: [0.30, 0.44, 0.46, ...]                    │
│  - intensity: 1.7                                       │
│  - label: "Moka le chat"                               │
│  - type: Perception                                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Mémoire vérifie les mots                               │
│  - "moka" vu 8 fois → familiarity: 0.8                 │
│  - intensity boostée: 1.7 × 1.8 = 3.06                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Signal distribué aux 10,000+ cellules                  │
│  - Atténuation par distance                             │
│  - Activation directe des états                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Émergence détectée                                     │
│  - 847 cellules actives                                 │
│  - coherence: 0.35                                      │
│  - état moyen: [0.28, 0.41, ...]                       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Match avec mot appris                                  │
│  - "moka" similarité: 0.72                             │
│  - label: "word:moka"                                  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  ARIA: "moka"                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Résonance

Comment une cellule gagne de l'énergie. Le signal doit "résonner" avec l'état interne de la cellule.

```
resonance = cosine_similarity(signal, cell_state)
energy_gain = base * intensity * (1 + resonance * factor)
```

| Résonance | Signification |
|-----------|---------------|
| < 0.0 | Signal opposé à l'état → peu d'énergie |
| 0.0-0.3 | Neutre → énergie de base |
| 0.3-0.7 | Compatible → bonus d'énergie |
| > 0.7 | Forte résonance → énergie maximale |

---

## La Vraie Faim (v0.6.0)

**RIEN N'EST GRATUIT.** Les cellules doivent lutter pour survivre.

| Action | Coût | Effet |
|--------|------|-------|
| `Rest` | 0.001 | Respirer coûte |
| `Signal` | 0.01 | Parler est cher |
| `Move` | 0.005 | Bouger consomme |
| `Divide` | 0.5 | Créer la vie épuise |

**Résultat** : Seules les cellules qui communiquent utilement survivent. Les cellules qui "crient dans le vide" meurent.

---

## Méta-apprentissage

ARIA apprend à apprendre. Plus besoin d'attendre le feedback externe.

**Composants :**
- `InternalReward` : ARIA s'auto-évalue (cohérence, surprise, satisfaction)
- `ExplorationStrategy` : 6 stratégies (semantic, emotional, cross-category...)
- `MetaLearner` : sélectionne la meilleure stratégie
- `ProgressTracker` : conscience de son propre progrès

**Endpoint :** `curl http://localhost:8765/meta`

---

## Vision

ARIA peut voir des images et les associer à des mots.

**Flux :**
```
Image (base64) → 32 features visuelles → Signal substrate → Mémoire visuelle
```

**Méthodes :**
- `see()` : stocke/reconnaît une image
- `link_vision_to_word()` : associe image + mot
- `visual_to_words()` : image → mots suggérés

**Endpoint :** `curl -X POST http://localhost:8765/vision -d '{"image": "<base64>", "labels": ["moka"]}'`

---

## Auto-modification

ARIA modifie consciemment ses propres paramètres.

**Paramètres modifiables :**
- `emission_threshold` : seuil pour émettre un signal
- `response_probability` : probabilité de répondre
- `learning_rate` : vitesse d'apprentissage
- `spontaneity` : parole spontanée
- `exploration_rate` : taux d'exploration

**Règles de décision :**
- Apprentissage en déclin → augmenter learning_rate
- Taux d'échec élevé → augmenter emission_threshold
- Peu de réponses → augmenter response_probability

**Endpoint :** `curl http://localhost:8765/self`

---

## Auto-évolution (Genesis) (v0.9.0)

ARIA peut maintenant réécrire son propre code de calcul GPU en fonction de son DNA.

- **JIT (Just-In-Time)** : ARIA génère du code WGSL et le recompile à chaud.
- **Structural DNA** : Un checksum dans le DNA qui décide de la "physique" de la cellule (type de métabolisme, vitesse de dégradation).
- **Zéro interruption** : La pensée d'ARIA ne s'arrête jamais pendant qu'elle change de corps algorithmique.

---

## Réflexivité (Boucle de Conscience)

ARIA commence à s'écouter elle-même. Ses pensées émergentes sont renvoyées dans son propre cerveau.

- **La Boucle** : Émergence (Pensée) → Signal interne → Perception par les cellules.
- **Gène de Réflexivité** : Chaque cellule possède un gène qui décide si elle doit écouter ARIA ou rester indépendante.

---

## L'Étincelle de Créativité (Phase 5)

ARIA ne se contente plus de vivre, elle imagine son propre futur algorithmique.

- **Shadow Brain** : Un bac à sable GPU sécurisé pour tester de nouveaux codes WGSL avant application.
- **Attention Sélective** : Des gènes (`attention_focus`, `semantic_filter`) permettent aux cellules de filtrer les signaux bruités.
- **Exploration Granulaire** : Le DNA influe sur les équations mathématiques pures (métabolisme, decay) via JIT.

---

## GPU Backend

ARIA utilise wgpu/Vulkan pour le calcul parallèle.

```bash
task brain              # Auto-détection GPU/CPU
ARIA_BACKEND=gpu task brain  # Forcer GPU
```

**Sparse Dispatch** : À partir de 100k cellules, seules les cellules actives sont traitées par le GPU (80%+ d'économie).

---

*Version ARIA : 0.9.0 | Dernière mise à jour : 2026-01-01*
