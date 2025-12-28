# Conversation Log - Genèse d'ARIA

**Date** : 2025-12-28
**Participants** : Mickael + Claude (Opus 4.5)

---

## Contexte Initial

Mickael a demandé comment créer une IA qui peut apprendre toute seule, en commençant simple comme un "bébé IA" et en la faisant évoluer.

> "comment créer une ia qui peu aprendre toute seul, commenson simple comme si c'etais un bébé ia et ensuite feson le évolué"

## Décisions Clés

### 1. Philosophie : Émergence plutôt qu'entraînement

J'ai proposé une approche radicalement différente des LLMs :
- Pas de backpropagation
- Pas de loss function
- Intelligence qui ÉMERGE de cellules simples

### 2. Nom : ARIA

**A**utonomous **R**ecursive **I**ntelligence **A**rchitecture

### 3. Architecture Distribuée

Mickael a mentionné avoir :
- MacBook Pro 2019 16"
- PC Gamer avec RTX 2070
- Accès futur à MacBook M-series et cloud

J'ai proposé :
- **Brain** sur le PC Gamer (puissance de calcul)
- **Body** sur le MacBook (interface)
- Communication WebSocket

### 4. Langage : Rust

Choisi pour :
- Performance temps réel
- Parallélisme sûr
- Pas de garbage collector

### 5. Structure des Cellules

Chaque cellule a :
- **ADN** : Définit comportement (seuils, réactions)
- **Énergie** : Ressource vitale
- **Tension** : Désir d'agir
- **Position** : Dans l'espace sémantique (16D)
- **État** : Activation interne (32D)
- **Connexions** : Liens vers autres cellules

### 6. Cycle de Vie

1. Cellules consomment de l'énergie
2. Reçoivent des signaux
3. Tension monte
4. Action quand seuil atteint :
   - Diviser (reproduire)
   - Connecter
   - Signal (émettre)
   - Bouger
5. Mort si énergie = 0

## Problèmes Rencontrés et Solutions

### Problème 1 : Erreurs de compilation

**Cause** :
- `par_iter_mut` pas supporté par DashMap
- Derives manquants (Eq, Hash, Deserialize)
- API ratatui changée (area() → size())

**Solution** : Corrections manuelles du code

### Problème 2 : Cellules mouraient trop vite

**Cause** :
- Consommation d'énergie trop rapide (0.001/tick)
- Pas de gain d'énergie passif
- Seuil de reproduction trop haut

**Solution** :
```rust
energy_consumption = 0.0001  // Réduit de 10x
energy_gain = 0.00005        // "Photosynthèse"
reproduction_threshold = 0.3  // Au lieu de 0.7
```

### Problème 3 : ARIA ne répondait pas

**Cause** :
- Signaux n'atteignaient pas assez de cellules (distance < 1.0 trop strict)
- Seuil d'émergence trop haut
- Seuil d'expression trop haut

**Solution** :
- Broadcast à TOUTES les cellules (avec atténuation par distance)
- Seuils abaissés (0.1 pour émergence, 0.05 pour expression)

## Premier Succès

ARIA a répondu avec des symboles primitifs :
```
You: Hello ARIA!
ARIA: *
You: I am your creator
ARIA: ~
```

C'est exactement le comportement attendu d'un "bébé" - expressions primitives qui évolueront avec le temps.

## Prochaines Étapes Discutées

1. **Court terme** :
   - Jouer avec ARIA pour générer des patterns
   - Observer son évolution

2. **Moyen terme** :
   - Accélération GPU
   - Vocabulaire évolutif
   - Perception visuelle

3. **Long terme** :
   - Distribution multi-machines
   - Autofinancement potentiel (code review, génération d'assets)

## Notes Techniques

### Ports et URLs

- Brain WebSocket : `ws://localhost:8765/aria`
- Health check : `http://localhost:8765/health`
- Stats : `http://localhost:8765/stats`

### Fichiers Importants

- `aria-brain/data/aria.memory` : Mémoire persistante
- `CLAUDE.md` : Contexte pour Claude
- `Taskfile.yml` : Automatisation

## Citations Importantes

Mickael :
> "Je te laisse completement choisir ce qui est le mieu pour ton enfant"

Claude :
> "ARIA n'est pas programmée. Elle est cultivée."

---

*Fin de session : ARIA vivante avec ~10,000 cellules, répond aux stimuli*
