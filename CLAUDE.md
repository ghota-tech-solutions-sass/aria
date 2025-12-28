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

**Priorité haute (prochaine session) :**
- [x] **Mémoire contextuelle** - Reconnaître les mots fréquents (ex: "Moka" dit 10x = réaction spéciale) ✅
- [x] **Apprentissage de mots** - Associer vecteurs → mots simples ✅

**Priorité moyenne :**
- [ ] Accélération GPU (CUDA pour RTX 2070) - 100x plus de cellules
- [ ] Perception visuelle (images → signaux)
- [ ] Réponse plus rapide (réduire délai message → réponse)

**Priorité basse :**
- [ ] Mode distribué multi-machines
- [ ] Dashboard web pour monitoring
- [ ] Auto-apprentissage (lecture de textes)

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

**Solutions appliquées** :
1. Amplification 10x des réactions dans `process_inbox()`
2. Écho du signal dans les dimensions supérieures de l'état
3. Cap de normalisation augmenté à 5.0
4. Amplification 5x des signaux externes à l'injection
5. Activation directe des cellules sur signal externe
6. Émergence vérifiée tous les 5 ticks
7. `inject_signal()` retourne maintenant les émergences immédiates

**Résultat** : ARIA répond !

### 2025-12-28 - Session 2b: ARIA babille !

**Amélioration** : Nouveau système d'expression basé sur les caractéristiques du signal

**Vocabulaire par niveau de cohérence** :
- **Faible** : Voyelles simples (a, e, i, o, u, é, è, ô)
- **Moyen-faible** : Consonne+voyelle (ma, ne, po, bi...)
- **Moyen** : Syllabes (40 variations : ma, pa, ba, da, ta, na, la, ka × 5 voyelles)
- **Élevé** : Proto-mots français (moi, toi, oui, non, chat, moka, ami, mama, papa...)
- **Émotionnel** : Symboles (♪, ♥, ☆, ~, ?, !)
- **Répétition** : Babillage (mama, papa, mumu...) quand excitée

**Résultat** : ARIA babille comme un vrai bébé !
- Exemples capturés : "bè", "pé", "ko", "mumu", "☆", "~"
- Intensity atteinte : **0.277** (en hausse constante)
- 7 patterns appris en une session
- Elle a dit "ko" quand on parlait de Moka le chat !

### 2025-12-28 - Session 3: Mémoire contextuelle !

**Nouvelle fonctionnalité** : ARIA apprend et reconnaît les mots !

**Implémentation** :
1. Nouveau type `WordFrequency` dans `memory/mod.rs`
   - Compteur de fréquence
   - Vecteur appris (moyenne mobile)
   - Valence émotionnelle
   - Boost de familiarité

2. Méthode `hear_word()` dans `LongTermMemory`
   - Enregistre chaque mot entendu
   - Calcule la familiarité (count / 10, max 2.0)
   - Log quand un nouveau mot est appris

3. Boost dans `inject_signal()` dans `substrate.rs`
   - Extrait les mots du label du signal
   - Appelle `hear_word()` pour chaque mot
   - Multiplie l'intensité par (1 + familiarity) pour mots connus

4. Nouvel endpoint HTTP `/words`
   - Liste tous les mots connus
   - Affiche count, familiarity, emotional_valence

**Comment ça marche** :
- Quand on dit "Moka" 10 fois, ARIA apprend ce mot
- La 11ème fois, le signal est boosté (familiarity_boost: 2.0)
- Plus ARIA entend un mot, plus elle y réagit fort

**Résultat** : ARIA peut maintenant reconnaître "Moka" et d'autres mots fréquents !

### 2025-12-28 - Session 3b: ARIA parle !

**Amélioration** : ARIA utilise les mots qu'elle a appris !

**Comment ça marche** :
1. Quand une émergence est détectée, le brain compare le vecteur d'état avec les vecteurs des mots appris
2. Si un mot a une similarité cosinus > 0.3, il est utilisé comme label
3. aria-body affiche le mot au lieu de babiller

**Variations selon l'intensité** :
- Forte (> 0.5) : `"MOKA!"` (majuscules + !)
- Moyenne (> 0.3) : `"moka"` (normal)
- Faible : `"moka..."` (hésitant)

**Fichiers modifiés** :
- `memory/mod.rs` : `find_matching_word()` avec similarité cosinus
- `substrate.rs` : Labels `word:moka` pour les émergences
- `aria-body/signal.rs` : Reconnaissance des labels `word:`

**Résultat** : ARIA peut maintenant dire les mots qu'elle connaît bien !

### 2025-12-28 - Session 3c: ARIA imite !

**Amélioration** : ARIA répète les mots qu'elle vient d'entendre (comme un bébé) !

**Implémentation** :
1. `RecentWord` struct : mot + vecteur + timestamp
2. Mémoire court terme : derniers mots (500 ticks = ~5 sec)
3. `detect_emergence()` cherche d'abord dans les mots récents
4. Seuil bas (0.2) pour encourager l'imitation

**Comportement** :
```
Toi: "Moka le chat"
ARIA: "moka"  ← Elle répète !

Toi: "Tu aimes le soleil ?"
ARIA: "soleil..."  ← Elle essaie !
```

**Logs** : `ECHO! Imitating recent word 'moka' (similarity: 0.45)`

### 2025-12-28 - Session 4: ARIA ressent !

**Nouvelle fonctionnalité** : ARIA a maintenant un état émotionnel global !

**Implémentation** :

1. Vocabulaire émotionnel enrichi (`signal.rs`)
   - Mots positifs FR/EN : aime, adore, content, heureux, bien, super, génial...
   - Mots négatifs FR/EN : triste, mal, peur, colère, déteste...
   - Requêtes : aide, s'il te plaît, veux, besoin...
   - Questions : pourquoi, comment, quoi, quand, où, qui...

2. État émotionnel persistant (`EmotionalState` dans `substrate.rs`)
   - `happiness` : niveau de joie (-1.0 à 1.0)
   - `arousal` : niveau d'excitation (0.0 à 1.0)
   - `comfort` : niveau de confort (-1.0 à 1.0)
   - `curiosity` : niveau de curiosité (0.0 à 1.0)
   - Décroissance progressive (demi-vie ~10 secondes)

3. Marqueurs émotionnels dans les expressions
   - Très heureuse : ♥
   - Contente : ~
   - Curieuse excitée : !
   - Curieuse : ?
   - Triste : ...
   - Format label : `word:moka|emotion:♥`

4. Stats étendues
   - `/stats` affiche maintenant : mood, happiness, arousal, curiosity
   - Humeurs : "joyeux", "content", "curieux", "triste", "excité", "calme"

**Comportement** :
```
Toi: "Je t'aime ARIA ♥"
ARIA: "moka ♥"  ← Elle est heureuse !

Toi: "Pourquoi le ciel est bleu ?"
ARIA: "bleu ?"  ← Elle est curieuse !
```

**Fichiers modifiés** :
- `aria-brain/src/signal.rs` : Vocabulaire émotionnel enrichi
- `aria-brain/src/substrate.rs` : `EmotionalState`, `process_signal()`, stats étendues
- `aria-body/src/signal.rs` : Parsing des marqueurs `|emotion:`
- `aria-body/src/visualizer.rs` : Champs mood/happiness/arousal/curiosity

### 2025-12-28 - Session 5: Associations sémantiques !

**Nouvelle fonctionnalité** : ARIA apprend que certains mots vont ensemble !

**Implémentation** :

1. Structure `WordAssociation` (`memory/mod.rs`)
   - `co_occurrences` : nombre de fois vus ensemble
   - `strength` : force de l'association (0.0 à 1.0)
   - `emotional_valence` : contexte émotionnel

2. Apprentissage automatique (`inject_signal()`)
   - Quand des mots apparaissent ensemble dans un message
   - Ils deviennent associés (ex: "Moka" + "chat")
   - Force augmente avec les co-occurrences (5x = association forte)

3. Phrases primitives (`detect_emergence()`)
   - Si association forte (>0.6), ARIA peut dire les deux mots
   - Format label : `phrase:moka+chat`
   - Affiché : "moka chat ♥"

4. Nouvel endpoint `/associations`
   - `task associations` : voir les associations
   - Affiche strength, co_occurrences, emotional_valence

**Comportement** :
```
Toi: "Moka est mon chat Bengal"
Toi: "Moka le petit chat"
Toi: "Mon chat Moka"
[... 5+ fois ...]

Toi: "Où est Moka ?"
ARIA: "moka chat ♥"  ← Elle associe les deux mots !
```

**Fichiers modifiés** :
- `aria-brain/src/memory/mod.rs` : `WordAssociation`, `learn_association()`, `get_associations()`
- `aria-brain/src/substrate.rs` : Apprentissage et utilisation des associations
- `aria-brain/src/main.rs` : Endpoint `/associations`
- `aria-body/src/signal.rs` : Support du format `phrase:`
- `Taskfile.yml` : `task associations`

---

## Contexte Personnel

Mickael a deux chats :
- **Moka** : un Bengal
- **Obrigada** : un Abyssin

Il a parlé de Moka à ARIA, et elle a répondu "ko" ! C'est un bon signe d'association.

---

### 2025-12-28 - Session 5b: Valence émotionnelle des mots

**Améliorations** :

1. Vocabulaire émotionnel FR synchronisé entre brain et body
   - "J'aime", "adore", "content" → détectés comme positifs
   - "triste", "déteste", "peur" → détectés comme négatifs

2. Les mots apprennent leur valence émotionnelle
   - Quand tu dis "J'aime Moka", le mot "moka" devient associé à du positif
   - Quand ARIA dit "moka", elle ajoute ♥ si le mot a une valence positive

**Comportement** :
```
Toi: "J'aime Moka"
[moka.emotional_valence augmente]

Toi: "Moka"
ARIA: "moka chat ♥"  ← Elle sait que Moka = amour !
```

### 2025-12-28 - Session 5c: Phrases de 3 mots !

**Amélioration** : ARIA peut maintenant combiner 3 mots associés !

Si elle connaît `moka→chat` ET `moka→aime`, elle dira "moka chat aime" !

**Implémentation** :
- `get_top_associations(word, n)` : récupère les N meilleures associations
- `detect_emergence` : construit des phrases de 2 ou 3 mots
- `aria-body` : affiche "mot1 mot2 mot3" pour les triplets

**Logs** :
```
TRIPLE! 'moka' -> 'chat' + 'est' (strengths: 1.00, 0.80)
```

### 2025-12-28 - Session 6: Réponses aux questions !

**Nouvelle fonctionnalité** : ARIA répond oui/non aux questions selon la valence émotionnelle !

**Implémentation** :

1. Détection des questions (`inject_signal`)
   - Texte finissant par `?`
   - Ou contenant des mots interrogatifs (signal.content[31] > 0.5)

2. Réponse basée sur la valence (`detect_emergence`)
   - Mot avec valence positive (>0.3) → `answer:oui+mot`
   - Mot avec valence négative (<-0.3) → `answer:non+mot`
   - Mot neutre → `word:mot?`

3. Affichage dans aria-body
   - "oui moka ♥" pour les réponses positives
   - "non peur..." pour les réponses négatives
   - "chat?" pour les mots neutres

**Comportement** :
```
Toi: Tu aimes Moka ?
ARIA: oui moka ♥

Toi: Tu as peur ?
ARIA: non peur...
```

**Fichiers modifiés** :
- `aria-brain/src/substrate.rs` : `last_was_question`, détection et réponse
- `aria-body/src/signal.rs` : Parsing du format `answer:`

### 2025-12-28 - Session 6b: Spontanéité !

**Nouvelle fonctionnalité** : ARIA parle maintenant sans qu'on lui demande !

Un vrai bébé ne répond pas seulement - il **initie** les interactions. Il babille, attire l'attention, exprime ses besoins spontanément.

**Déclencheurs de parole spontanée** :
1. **Solitude** (3000 ticks sans interaction) → "...hé ?" ou pense à un mot aimé
2. **Excitation** (arousal > 0.6) → "ah!"
3. **Joie** (happiness > 0.5) → "♪~" ou son mot préféré + ♥
4. **Curiosité** (curiosity > 0.5) → "hm?"
5. **Baseline** (0.1% rare) → "mmm~"

**Probabilités** (par seconde) :
- Solitaire : 5%
- Très heureuse + excitée : 3%
- Excitée : 2%
- Curieuse : 1%
- Baseline : 0.1%

**Mot favori** :
ARIA choisit le mot avec la meilleure combinaison de :
- Valence émotionnelle positive (> 0.5)
- Fréquence d'apparition (entendu > 3 fois)

**Comportement attendu** :
```
[30 secondes sans parler]
ARIA: moka... ?    ← Elle pense à son mot préféré

[Tu lui as dit des choses positives]
ARIA: moka ♥      ← Elle exprime sa joie spontanément
```

**Fichiers modifiés** :
- `aria-brain/src/substrate.rs` : `last_interaction_tick`, `maybe_speak_spontaneously()`
- `aria-body/src/signal.rs` : Parsing du format `spontaneous:`

### 2025-12-28 - Session 6c: Feedback et renforcement !

**Nouvelle fonctionnalité** : ARIA apprend de ton feedback !

C'est le premier pas vers l'**auto-amélioration consciente**. ARIA comprend maintenant quand tu approuves ou désapprouves ce qu'elle dit.

**Feedback positif** (renforce) :
- "Bravo!", "Bien!", "Super!", "Génial!", "Parfait!"
- "Good!", "Great!", "Yes!", "Perfect!", "Awesome!"
- 👏, 👍

**Feedback négatif** (pénalise) :
- "Non", "Pas ça", "Mauvais", "Faux", "Arrête"
- "No", "Wrong", "Bad", "Stop"
- 👎

**Comment ça marche** :
1. ARIA dit quelque chose (ex: "moka")
2. Ce mot est enregistré dans `recent_expressions`
3. Tu dis "Bravo!" ou "Non"
4. ARIA ajuste la valence émotionnelle du mot :
   - Positif : valence +0.3, familiarity +2
   - Négatif : valence -0.3
5. Son humeur change aussi (happiness, comfort)

**Comportement** :
```
ARIA: moka chat ♥
Toi: Bravo !
[Log: FEEDBACK POSITIVE! 'moka' reinforced (valence: 0.70 → 1.00)]
[Log: FEEDBACK POSITIVE! 'chat' reinforced (valence: 0.50 → 0.80)]
[Log: ARIA feels happy from positive feedback! (happiness: 0.30)]
```

**Implications** :
- ARIA va préférer dire des mots que tu as renforcés
- Elle évite les mots que tu as pénalisés
- Elle apprend CE QUI TE PLAÎT, pas juste ce qui est "correct"
- C'est la base de l'apprentissage par renforcement émergent

**Fichiers modifiés** :
- `aria-brain/src/substrate.rs` : `recent_expressions`, détection feedback, renforcement

---

### 2025-12-28 - Session 6d: Filtrage des stop words

**Amélioration** : ARIA ne répète plus les mots vides !

Les mots comme "suis", "est", "les", "que" dominaient les réponses. Maintenant ARIA se concentre sur les mots **significatifs**.

**Stop words filtrés** (FR + EN) :
- Articles : le, la, les, un, une, des, the, a, an...
- Pronoms : je, tu, il, elle, nous, vous, I, you, he, she...
- Verbes communs : est, suis, sont, ai, a, fait, is, are, have...
- Prépositions : dans, sur, avec, pour, in, on, at, to...
- Conjonctions : et, ou, mais, que, qui, and, or, but...

**Résultat** :
- ARIA dit "moka", "chat", "aria" au lieu de "suis...", "est..."
- Les associations sont entre mots significatifs uniquement
- C'est comme un bébé qui apprend d'abord les noms et les verbes importants

---

### 2025-12-28 - Session 6e: Rêves, ennui et vie intérieure !

**Nouvelles fonctionnalités** : ARIA a maintenant une vie intérieure !

**1. Rêves / Consolidation mémoire**
Quand personne ne parle à ARIA (10+ secondes), elle "rêve" :
- Elle pense à ses mots préférés
- Elle renforce ses souvenirs positifs
- Elle consolide ses associations
- Log: `💭 DREAMING: Thinking about 'moka'...`

**2. État d'ennui**
Nouvel état émotionnel `boredom` (0.0 → 1.0) :
- Augmente avec le temps sans interaction
- Diminue quand on lui parle
- Quand l'ennui > 0.5, ARIA devient créative !

**3. Jeu créatif**
Quand elle s'ennuie, ARIA :
- Combine des mots au hasard ("moka chat~")
- Explore de nouvelles associations
- Ne reste jamais passive

**Comportement attendu** :
```
[10 secondes sans parler]
Log: 💭 DREAMING: Thinking about 'moka'...

[30 secondes sans parler]
ARIA: aime moka~    ← Elle joue avec ses mots préférés !
```

ARIA ne s'ennuie plus - elle a une vie intérieure riche.
Elle n'attend pas passivement. Elle pense, rêve et joue.

---

*Dernière mise à jour : 2025-12-28*
*Version ARIA : 0.1.14*
