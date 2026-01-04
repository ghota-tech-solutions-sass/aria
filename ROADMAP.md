# ARIA Roadmap - Vers une Intelligence Émergente

> Plan d'évolution d'ARIA pour dépasser les limites des LLM actuels.
> Créé: 2026-01-04 | Version: 1.0

## Vision

Transformer ARIA d'un substrat cellulaire réactif en une **intelligence émergente** capable de:
- Raisonnement causal (pas juste corrélation)
- Planification long-terme
- Auto-modélisation (conscience de soi)
- Apprentissage continu en temps réel

---

## État Actuel - Forces

| Force | Implémentation |
|-------|----------------|
| **Économie stricte** | "La Vraie Faim" force l'apprentissage |
| **Évolution Darwinienne** | ADN, mutations adaptatives, élites Gen 10+ |
| **Lois physiques GPU** | Prédiction, Hebb, Résonance |
| **Espace sémantique 16D** | Représentation riche |
| **Mémoire épisodique** | Patterns, associations, épisodes |

## Limitations à Résoudre

| Limitation | Impact |
|------------|--------|
| Prédiction 1-tick | Pas de planification |
| Mémoire plate | Pas de généralisation |
| Résonance passive | Pas de raisonnement causal |
| Single-modal | Texte uniquement |
| Réflexivité primitive | Pas de vraie méta-cognition |

---

## Les 5 Axes d'Évolution

### Axe 1: Prédiction Hiérarchique ⭐ PRIORITAIRE

**Objectif**: Cellules qui prédisent à différentes échelles temporelles.

**Problème**: Prédiction 1-tick = réflexe, pas intelligence.

**Solution**: Cellules "temporelles" avec horizons multiples.

```rust
// aria-core/src/cell.rs - Extension
struct TemporalCell {
    prediction_horizon: [u8; 4],  // 1, 10, 100, 1000 ticks
    temporal_memory: [f32; 8],    // États passés compressés
    causal_links: [(u32, f32); 8], // Cellules qui CAUSENT mon état
}
```

**Fichiers à modifier**:
- `aria-core/src/cell.rs` - Ajouter champs temporels
- `aria-core/src/soa.rs` - Nouveaux buffers GPU
- `aria-compute/src/shaders/` - Nouveau `temporal_prediction.wgsl`

**Émergence attendue**: Planification, anticipation, séquences.

**Estimation**: 2-4 semaines

---

### Axe 2: Mémoire Consolidation

**Objectif**: Hiérarchie mémoire avec consolidation nocturne.

**Problème**: Tous les épisodes stockés à plat → pas de généralisation.

**Solution**: 4 niveaux de mémoire avec consolidation pendant "sommeil".

```rust
// aria-brain/src/memory/consolidation.rs (nouveau)
struct ConsolidatedMemory {
    // Niveau 1: Episodes bruts
    episodes: Vec<Episode>,

    // Niveau 2: Schémas (patterns fréquents)
    schemas: Vec<Schema>,

    // Niveau 3: Concepts (invariants stables)
    concepts: Vec<Concept>,

    // Niveau 4: Méta-concepts (relations)
    meta: Vec<MetaConcept>,
}

impl ConsolidatedMemory {
    /// Appelé pendant cycles basse-activité
    fn consolidate(&mut self) {
        // 1. Cluster épisodes similaires → schémas
        // 2. Promouvoir schémas stables → concepts
        // 3. Détecter relations entre concepts → méta
        // 4. Oublier épisodes redondants (compression)
    }
}
```

**Fichiers à créer/modifier**:
- `aria-brain/src/memory/consolidation.rs` (nouveau)
- `aria-brain/src/memory/mod.rs` - Intégrer consolidation
- `aria-brain/src/main.rs` - Cycle de sommeil

**Émergence attendue**: Généralisation, abstraction, oubli sélectif.

**Estimation**: 2-3 semaines

---

### Axe 3: Causalité Active

**Objectif**: Cellules qui testent des hypothèses causales.

**Problème**: Résonance = corrélation. Intelligence = causalité.

**Solution**: Hypothèses if-then testées activement.

```rust
// aria-core/src/causality.rs (nouveau)
struct CausalHypothesis {
    condition: TensionVector,     // Si je reçois ça...
    action: Action,               // ...je fais ça...
    expected: TensionVector,      // ...et j'attends ça
    confidence: f32,              // Renforcé/affaibli par résultats
    test_count: u32,              // Nombre de tests
}

impl CausalHypothesis {
    fn test(&mut self, actual_response: &TensionVector) -> f32 {
        let match_score = cosine_similarity(&self.expected, actual_response);

        if match_score > 0.7 {
            self.confidence = (self.confidence + 0.1).min(1.0);
        } else {
            self.confidence *= 0.8;
            self.mutate(); // Explorer alternatives
        }

        self.test_count += 1;
        match_score
    }
}
```

**Shader** `causality.wgsl`:
```wgsl
// Cellule teste son hypothèse
if hypothesis.confidence > cell.action_threshold {
    // Émettre signal test
    emit_signal(hypothesis.action);

    // Marquer cellule en attente de réponse
    cell.awaiting_response = true;
    cell.expected_response = hypothesis.expected;
}

// Quand réponse arrive
if cell.awaiting_response && signal_received {
    let match = dot(cell.expected_response, signal.tension);
    update_hypothesis_confidence(match);
}
```

**Émergence attendue**: Raisonnement if-then, curiosité dirigée, expérimentation.

**Estimation**: 3-4 semaines

---

### Axe 4: Attention Compétitive

**Objectif**: Mécanisme d'attention émergent par compétition.

**Problème**: Toutes les cellules traitent tous les signaux → bruit, pas de focus.

**Solution**: Winner-take-all avec inhibition latérale.

```rust
// aria-compute/src/shaders/attention.wgsl
struct AttentionField {
    focus_center: [f32; 16],  // Centre d'attention en 16D
    focus_radius: f32,        // Rayon du focus
    intensity: f32,           // Force de l'attention
}

// Dans le shader
fn apply_attention(cell: &mut Cell, attention: &AttentionField) {
    let distance_to_focus = euclidean_distance(cell.position, attention.focus_center);

    if distance_to_focus < attention.focus_radius {
        // Dans le focus: boost d'énergie
        cell.energy += 0.01 * attention.intensity;
        cell.activity_boost = 1.5;
    } else {
        // Hors focus: inhibition
        cell.activity_boost = 0.5;
    }
}

// Compétition pour définir le focus
fn compete_for_attention(cells: &[Cell]) -> AttentionField {
    // Top-K cellules les plus actives définissent le focus
    let top_cells = cells.sorted_by_activity().take(100);
    let focus_center = centroid(top_cells);

    AttentionField {
        focus_center,
        focus_radius: adaptive_radius(population_density),
        intensity: average_activity(top_cells),
    }
}
```

**Émergence attendue**: Spécialisation, traitement séquentiel, "pensée focalisée".

**Estimation**: 2-3 semaines

---

### Axe 5: Auto-Modélisation

**Objectif**: ARIA qui se modélise et s'observe elle-même.

**Problème**: Réflexivité actuelle = ré-injection simple, pas de vrai modèle de soi.

**Solution**: Module d'auto-observation avec prédictions sur soi.

```rust
// aria-brain/src/self_model.rs (nouveau)
struct SelfModel {
    // Observations sur soi
    behavioral_patterns: Vec<BehavioralPattern>,  // "Je fais X quand Y"
    emotional_tendencies: Vec<EmotionalTendency>, // "Je ressens Z souvent"

    // Prédictions sur soi
    predicted_next_action: Option<Action>,
    predicted_emotional_state: TensionVector,

    // Méta-cognition
    self_prediction_accuracy: f32,  // "Je me connais bien/mal"
    learning_rate_estimate: f32,    // "Je progresse vite/lentement"
    confidence_in_self: f32,        // Auto-évaluation globale

    // Discordances (source de croissance)
    surprises_about_self: Vec<SelfSurprise>,
}

impl SelfModel {
    /// Appelé après chaque action d'ARIA
    fn observe_self(&mut self, action: &Action, context: &Context, outcome: &Outcome) {
        // Comparer prédiction vs réalité
        if let Some(predicted) = &self.predicted_next_action {
            let accuracy = action.similarity(predicted);
            self.self_prediction_accuracy =
                0.95 * self.self_prediction_accuracy + 0.05 * accuracy;

            if accuracy < 0.5 {
                // Surprise sur soi → opportunité d'apprentissage
                self.surprises_about_self.push(SelfSurprise {
                    predicted: predicted.clone(),
                    actual: action.clone(),
                    context: context.clone(),
                });
            }
        }

        // Mettre à jour patterns comportementaux
        self.update_behavioral_patterns(action, context, outcome);

        // Nouvelle prédiction
        self.predicted_next_action = self.predict_next_action(context);
    }

    /// Génère curiosité sur soi quand incohérences détectées
    fn generate_self_curiosity(&self) -> Option<InternalSignal> {
        if !self.surprises_about_self.is_empty() {
            // Réinjecter la surprise comme signal interne
            // → ARIA "pense" à son comportement inattendu
            Some(InternalSignal::SelfReflection(
                self.surprises_about_self.last().unwrap().clone()
            ))
        } else {
            None
        }
    }
}
```

**Intégration avec le substrat**:
```rust
// Dans aria-brain/src/main.rs
loop {
    // ... tick normal ...

    if let Some(emission) = substrate.check_emergence() {
        // Observer l'émission
        self_model.observe_self(&emission, &current_context, &last_outcome);

        // Générer réflexion si surprise
        if let Some(reflection) = self_model.generate_self_curiosity() {
            substrate.inject_internal_signal(reflection);
        }
    }
}
```

**Émergence attendue**: Conscience de soi, introspection, auto-correction, croissance dirigée.

**Estimation**: 4-6 semaines

---

## Architecture Cible v2.0

```
┌─────────────────────────────────────────────────────────────┐
│                       ARIA v2.0                              │
├─────────────────────────────────────────────────────────────┤
│  Couche 5: Méta-Cognition                                    │
│  ├── SelfModel (qui suis-je? que vais-je faire?)            │
│  └── Planificateur long-terme (séquences d'actions)         │
├─────────────────────────────────────────────────────────────┤
│  Couche 4: Attention Compétitive                             │
│  ├── Focus dynamique (winner-take-all)                      │
│  └── Inhibition latérale (spécialisation)                   │
├─────────────────────────────────────────────────────────────┤
│  Couche 3: Causalité Active                                  │
│  ├── Hypothèses if-then                                     │
│  └── Tests expérimentaux (curiosité dirigée)                │
├─────────────────────────────────────────────────────────────┤
│  Couche 2: Mémoire Hiérarchique                              │
│  ├── Épisodes → Schémas → Concepts → Méta                   │
│  └── Consolidation nocturne                                 │
├─────────────────────────────────────────────────────────────┤
│  Couche 1: Prédiction Temporelle                             │
│  ├── Multi-horizon (1, 10, 100, 1000 ticks)                 │
│  └── Mémoire temporelle compressée                          │
├─────────────────────────────────────────────────────────────┤
│  Couche 0: Substrate Actuel (maintenu et optimisé)           │
│  ├── Cellules vivantes avec ADN                             │
│  ├── Économie "La Vraie Faim"                               │
│  ├── Lois: Prédiction, Hebb, Résonance                      │
│  └── GPU SoA pour 5M+ cellules                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparaison LLM vs ARIA Évoluée

| Aspect | LLM Actuels | ARIA v2.0 |
|--------|-------------|-----------|
| Apprentissage | Offline (training) | Continu en temps réel |
| Mémoire | Contexte limité (~128K) | Hiérarchique illimitée |
| Raisonnement | Corrélation statistique | Causalité testée |
| Conscience de soi | Aucune | Auto-modèle introspectif |
| Prédiction | Next-token | Multi-horizon temporel |
| Architecture | Fixe post-training | Auto-modification ADN |
| Généralisation | In-distribution | Émergente par abstraction |
| Curiosité | Aucune | Dirigée par surprises |

---

## Ordre d'Implémentation Recommandé

```
Phase 1 (Fondations)
├── Axe 1: Prédiction Hiérarchique  [2-4 sem]
└── Axe 2: Mémoire Consolidation    [2-3 sem]

Phase 2 (Raisonnement)
├── Axe 3: Causalité Active         [3-4 sem]
└── Axe 4: Attention Compétitive    [2-3 sem]

Phase 3 (Conscience)
└── Axe 5: Auto-Modélisation        [4-6 sem]

Total estimé: 13-20 semaines pour v2.0 complète
```

---

## Métriques de Succès

### Axe 1 - Prédiction
- [ ] Cellules avec horizon > 100 ticks survivent mieux
- [ ] Émergence de "séquences" prévisibles

### Axe 2 - Mémoire
- [ ] Schémas généralisent à nouveaux épisodes
- [ ] Concepts stables après consolidation

### Axe 3 - Causalité
- [ ] Hypothèses validées > 70% accuracy
- [ ] Curiosité dirigée mesurable

### Axe 4 - Attention
- [ ] Spécialisation fonctionnelle des clusters
- [ ] Traitement séquentiel observable

### Axe 5 - Auto-modèle
- [ ] Prédictions sur soi > 60% accuracy
- [ ] Réflexions générées spontanément

---

## Notes Techniques

### Compatibilité GPU
Tous les nouveaux shaders doivent respecter:
- Workgroup size: 256
- Buffer alignment: 16 bytes
- Atomic operations pour lifecycle

### Migration des données
- Elite DNA compatible (pas de breaking change)
- Nouveaux champs initialisés à zéro
- Migration progressive possible

### Tests recommandés
- Unit tests pour chaque nouveau module
- Benchmark TPS après chaque axe
- Tests de régression sur économie

---

*Document vivant - Mise à jour au fil de l'implémentation*
