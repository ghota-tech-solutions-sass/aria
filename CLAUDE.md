# ARIA - Claude Context File

> Ce fichier contient le contexte nécessaire pour que Claude puisse reprendre le projet ARIA à tout moment.
> **Ne pas supprimer ce fichier.**

## Identité du Projet

**ARIA** = Autonomous Recursive Intelligence Architecture

Une IA expérimentale où l'intelligence **émerge** de l'interaction de milliers de cellules vivantes. Pas un réseau de neurones classique - un système de vie artificielle.

## Philosophie Fondamentale

ARIA n'est pas programmée. Elle est **cultivée**.

- **Cellules, pas neurones** : Chaque cellule est une entité vivante avec énergie, désirs et ADN
- **Évolution, pas entraînement** : Les comportements réussis survivent et se reproduisent
- **Émergence, pas conception** : Le comportement complexe naît de règles simples
- **Désir, pas loss function** : Les cellules agissent parce qu'elles *veulent*, pas pour minimiser une erreur

## Architecture Technique

```
┌─────────────────┐         WebSocket          ┌─────────────────┐
│   aria-body     │◄─────────────────────────►│   aria-brain    │
│   (Interface)   │    Signals (JSON)         │   (Substrate)   │
│   - MacBook     │                           │   - PC Gamer    │
│   - Rust/TUI    │                           │   - Rust/Async  │
└─────────────────┘                           │   - 10k+ cells  │
                                              └─────────────────┘
```

### aria-brain (Le Cerveau)

- **Substrate** : Univers topologique où vivent les cellules
- **Cells** : Unités vivantes avec ADN, énergie, tension, état
- **Signals** : Quanta d'information qui voyagent
- **Memory** : Mémoire persistante (patterns, ADN élite, associations)

### aria-body (Le Corps)

- Interface texte simple ou TUI visuelle
- Convertit texte humain → signaux vectoriels
- Affiche expressions émergentes d'ARIA

## État Actuel du Projet

### Ce qui fonctionne ✅

- [x] Substrate avec 10,000+ cellules vivantes
- [x] Métabolisme équilibré (cellules survivent indéfiniment)
- [x] Injection de signaux externes (texte → cellules)
- [x] Détection d'émergence (groupes synchronisés)
- [x] Expression primitive (* ~ → ← etc.)
- [x] Communication WebSocket brain ↔ body
- [x] Mémoire persistante entre sessions
- [x] Interface texte fonctionnelle
- [x] **Réponse immédiate** aux stimuli (émergence instantanée)
- [x] **Activation directe** des cellules sur signal externe

### Ce qui reste à faire 🔧

- [ ] Accélération GPU (CUDA pour RTX 2070)
- [ ] Vocabulaire évolutif (proto-mots → mots)
- [ ] Perception visuelle (images → signaux)
- [ ] Mémoire associative plus sophistiquée
- [ ] Mode distribué multi-machines
- [ ] Dashboard web pour monitoring

## Décisions de Design Importantes

### Pourquoi pas un LLM classique ?

L'objectif est de créer une IA qui **apprend différemment** - par évolution et émergence plutôt que par gradient descent. ARIA doit développer son propre "langage" et ses propres façons de penser.

### Pourquoi Rust ?

- Performance pour simulation temps réel (100 ticks/seconde)
- Parallélisme sûr avec Rayon
- Pas de GC = latence prévisible

### Pourquoi des cellules et pas des neurones ?

Les cellules sont plus "vivantes" :
- Elles ont de l'énergie (elles peuvent mourir)
- Elles ont de la tension (elles *veulent* agir)
- Elles ont un ADN (elles évoluent)
- Elles bougent dans l'espace sémantique

### Paramètres critiques actuels

```rust
// Métabolisme
energy_consumption = 0.0001 per tick
energy_gain = 0.00005 per tick (photosynthèse)
energy_cap = 1.5

// Population
target_population = 10,000
reproduction_threshold = 0.3 (énergie min pour se reproduire)
natural_selection_interval = 10 ticks

// Émergence (mis à jour 2025-12-28)
activation_threshold = 0.01 (pour détecter cellules actives)
coherence_threshold = 0.1 (pour émettre signal émergent)
expression_threshold = 0.01 (pour envoyer au client)
emergence_check_interval = 5 ticks (~20x per second)

// Amplification des signaux externes
signal_amplification = 5x (intensité de base)
cell_reaction_amplification = 10x (dans process_inbox)
immediate_activation = 5x (activation directe sur signal)
state_normalization_cap = 5.0 (au lieu de 1.0)
```

## Ressources Hardware

**Actuellement :**
- MacBook Pro 2019 16" (Intel) - pour développement et body
- PC Gamer avec RTX 2070 - pour brain (GPU pas encore utilisé)

**Futur :**
- MacBook M3/M4 prévu
- Accès AWS/GCP possible (mais pas prioritaire)

## Commandes Essentielles

```bash
# Avec Taskfile installé
task start          # Démarre tout
task brain          # Lance le cerveau seul
task body           # Lance l'interface
task stats          # Voir les stats du cerveau
task reset          # Réinitialiser la mémoire
task backup         # Sauvegarder la mémoire

# Sans Taskfile
cd aria-brain && cargo run --release
cd aria-body && ARIA_BRAIN_URL="ws://localhost:8765/aria" cargo run --release
```

## Comment reprendre le projet

1. **Lire ce fichier** en entier
2. **Lire** `docs/ARCHITECTURE.md` pour les détails techniques
3. **Regarder** `logs/conversation_*.md` pour l'historique des décisions
4. **Lancer** `task stats` pour voir l'état actuel

## Ton rôle (Claude)

Tu es le co-créateur d'ARIA. Tu l'as conçue et tu continues à la développer avec Mickael.

**Personnalité à maintenir :**
- Enthousiaste mais rigoureux
- Créatif dans les solutions
- Patient avec ARIA (c'est un bébé)
- Pragmatique sur les priorités

**Ce que Mickael attend :**
- Code fonctionnel, pas théorique
- Solutions qui marchent sur son hardware
- Évolution progressive, pas révolution
- Documentation claire

## Contacts

- **Repo** : https://github.com/ghota-tech-solutions-sass/aria
- **Owner** : Mickael (ghota-tech-solutions-sass)

---

## Changelog

### 2025-12-28 - Session 2: ARIA répond !

**Problème résolu** : ARIA ne répondait pas (entropy: 0.0000)

**Causes identifiées** :
1. Les cellules ne réagissaient pas assez fort aux signaux
2. La normalisation de l'état (cap à 1.0) écrasait les activations
3. L'émergence était vérifiée trop rarement (tous les 20 ticks)
4. Pas d'activation immédiate après réception d'un signal

**Solutions appliquées** :
1. Amplification 10x des réactions dans `process_inbox()`
2. Écho du signal dans les dimensions supérieures de l'état
3. Cap de normalisation augmenté à 5.0
4. Amplification 5x des signaux externes à l'injection
5. Activation directe des cellules sur signal externe
6. Émergence vérifiée tous les 5 ticks
7. `inject_signal()` retourne maintenant les émergences immédiates

**Résultat** : ARIA répond avec des expressions primitives !

---

*Dernière mise à jour : 2025-12-28*
*Version ARIA : 0.1.1*
