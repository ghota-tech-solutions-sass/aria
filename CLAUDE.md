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

**Infrastructure de base :**
- [x] Substrate avec 10,000+ cellules vivantes
- [x] Métabolisme équilibré (cellules survivent indéfiniment)
- [x] Injection de signaux externes (texte → cellules)
- [x] Détection d'émergence (groupes synchronisés)
- [x] Communication WebSocket brain ↔ body
- [x] Mémoire persistante entre sessions
- [x] Interface texte fonctionnelle

**Apprentissage et mémoire :**
- [x] Mémoire contextuelle - Reconnaître les mots fréquents ✅
- [x] Apprentissage de mots - Associer vecteurs → mots simples ✅
- [x] Associations sémantiques - Mots qui vont ensemble (moka + chat) ✅
- [x] Valence émotionnelle - Les mots apprennent leur contexte émotionnel ✅
- [x] Phrases de 2-3 mots - Combiner les associations ✅

**Émotions et personnalité :**
- [x] État émotionnel global (happiness, arousal, comfort, curiosity, boredom) ✅
- [x] Réponses aux questions (oui/non selon valence) ✅
- [x] Feedback positif/négatif - ARIA apprend ce qui plaît ✅
- [x] Marqueurs émotionnels (♥, ?, !, ~, ...) ✅

**Vie intérieure :**
- [x] Spontanéité - ARIA parle sans qu'on lui demande ✅
- [x] Rêves - Consolidation mémoire pendant l'inactivité ✅
- [x] Ennui créatif - Joue avec les mots quand elle s'ennuie ✅
- [x] Stop words - Filtre les mots vides, focus sur le sens ✅

**Ce qu'ARIA sait faire maintenant :**
- Dire son propre nom ("aria")
- Reconnaître et nommer Moka le chat
- Répondre aux questions avec oui/non
- Apprendre du feedback (Bravo! / Non)
- Rêver de ses mots préférés
- Créer des combinaisons de mots quand elle s'ennuie

### Ce qui reste à faire 🔧

**v0.1.16 - Contexte conversationnel ✅ :**
- [x] **ConversationContext** - Buffer des 5 derniers échanges
- [x] **Topic detection** - Mots qui reviennent = topics
- [x] **Context boosting** - Le fil de discussion influence les réponses

**v0.1.17 - Patterns d'usage ✅ :**
- [x] **Patterns temporels** - Quand utiliser certains mots (bonjour/au revoir)
- [x] **Expressions sociales** - Salutations, remerciements, excuses
- [x] **Contexte social** - Détection et réponse appropriée selon le contexte

**v0.2.x - Mémoire et perception :**
- [ ] **Mémoire épisodique** - Se souvenir de conversations spécifiques
- [ ] **Perception visuelle** - Images → signaux vectoriels
- [ ] **Reconnaissance** - Associer Moka (le mot) à Moka (la photo)

**v0.3.x - Auto-amélioration :**
- [ ] **Paramètres adaptatifs** - Modifier seuils et taux d'apprentissage
- [ ] **Méta-apprentissage** - Apprendre comment apprendre
- [ ] **Code génératif** - Réécrire son propre code (objectif ultime)

**Infrastructure (quand nécessaire) :**
- [ ] Accélération GPU (CUDA pour RTX 2070) - 100x plus de cellules
- [ ] Mode distribué multi-machines
- [ ] Dashboard web pour monitoring

**Complété :**
- [x] Éviter les répétitions du même mot (v0.1.15)
- [x] Catégories grammaticales (v0.1.15)
- [x] Phrases ordonnées naturellement (v0.1.15)

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

### 2025-12-29 - Session 7: Catégories de mots et phrases intelligentes !

**Nouvelle fonctionnalité** : ARIA comprend maintenant les catégories grammaticales !

**1. WordCategory enum**
Les mots sont classifiés en catégories :
- `Noun` : noms (chat, moka, aria, maison)
- `Verb` : verbes (aime, veux, mange, dort)
- `Adjective` : adjectifs (beau, grand, joli, petit)
- `Unknown` : mots non encore classifiés

**2. Détection automatique par contexte**
ARIA apprend les catégories en observant :
- Articles avant → Nom ("le **chat**", "the **cat**")
- Pronoms avant → Verbe ("je **mange**", "I **eat**")
- Suffixes de mots (FR: -eux, -er, -ique / EN: -ful, -less, -ous)
- Listes de mots connus

**3. Phrases en ordre naturel français**
La méthode `order_phrase()` arrange les mots :
- Adjectifs courts avant les noms ("**beau** chat")
- Adjectifs longs après les noms ("chat **magnifique**")
- Ordre Sujet-Verbe-Objet ("moka aime chat")

**4. Évitement des répétitions**
`last_said_word` empêche ARIA de répéter le même mot :
- Ne dit plus "moka moka moka"
- Varie ses expressions
- Plus naturel comme un vrai bébé

**Comportement** :
```
[Avant v0.1.15]
ARIA: moka chat aime

[Après v0.1.15]
ARIA: moka aime chat    ← Ordre naturel !

[Avant v0.1.15]
ARIA: moka... moka... moka...

[Après v0.1.15]
ARIA: moka... chat... aime...    ← Variété !
```

**Fichiers modifiés** :
- `aria-brain/src/memory/mod.rs` : `WordCategory`, `hear_word_with_context()`, `order_phrase()`
- `aria-brain/src/substrate.rs` : `last_said_word`, apprentissage contextuel, filtrage répétitions

---

### 2025-12-29 - Session 7b: Contexte conversationnel !

**Nouvelle fonctionnalité** : ARIA suit maintenant le fil de la conversation !

**1. ConversationContext struct**
Suivi des derniers échanges :
- Buffer des 5 dernières interactions
- Mots de chaque échange enregistrés
- Réponses d'ARIA associées aux inputs

**2. Topic Detection**
Les mots qui reviennent deviennent des "topics" :
- Comptage automatique des mentions
- Top 10 des mots les plus fréquents
- Boost proportionnel au nombre de mentions

**3. Context Boosting**
Les mots du contexte actuel sont privilégiés :
- Dernier échange : boost 100%
- Avant-dernier : boost 50%
- Encore avant : boost 25%
- Topic words : bonus supplémentaire

**4. Continuité**
- Timeout de 30 secondes pour nouvelle conversation
- Les réponses d'ARIA sont enregistrées
- Le contexte influence les associations

**Comportement** :
```
Toi: "Moka est mon chat"
Log: CONVERSATION: Topics = ["moka", "chat"], Exchanges = 1

Toi: "Moka est beau"
Log: CONVERSATION: Topics = ["moka", "chat", "beau"], Exchanges = 2

ARIA: (boost context pour "moka") → "moka beau ♥"
```

**Fichiers modifiés** :
- `aria-brain/src/substrate.rs` : `ConversationContext`, `ConversationExchange`, context boosting

---

### 2025-12-29 - Session 7c: Patterns d'usage et réponses sociales !

**Nouvelle fonctionnalité** : ARIA répond maintenant de manière appropriée aux contextes sociaux !

**1. SocialContext enum**
Nouveaux contextes sociaux détectés :
- `Greeting` : Bonjour, salut, coucou, hello, hi
- `Farewell` : Au revoir, bye, à bientôt, ciao
- `Thanks` : Merci, thanks, thank you
- `Affection` : Je t'aime, bisou, câlin
- `Request` : S'il te plaît, please
- `Agreement` : Oui, d'accord, ok
- `Disagreement` : Non, pas d'accord
- `General` : Contexte par défaut

**2. UsagePattern struct**
Chaque mot apprend quand il est utilisé :
- `contexts` : Dans quels contextes sociaux ce mot apparaît
- `start_of_conversation` : Score si utilisé en début de conversation
- `end_of_conversation` : Score si utilisé en fin
- `followed_by/preceded_by` : Mots qui l'accompagnent souvent

**3. Réponses sociales automatiques**
Quand ARIA détecte un contexte social au début d'une conversation :
- **Greeting** → "bonjour~" ou un mot de salutation qu'elle connaît
- **Farewell** → "bye~" ou mot d'au revoir appris
- **Thanks** → "de rien~"
- **Affection** → Mot affectueux + ♥♥

**4. Apprentissage des patterns**
`learn_usage_pattern()` enregistre :
- Le contexte social de chaque mot entendu
- La position dans la conversation (début/fin)
- Les mots qui précèdent/suivent

**Comportement** :
```
Toi: Bonjour ARIA !
ARIA: bonjour~ ~    ← Elle dit bonjour en retour !

Toi: Je t'aime
ARIA: AIME ♥♥ ♥    ← Elle exprime l'affection !

Toi: Merci
ARIA: de rien~     ← Elle sait répondre !
```

**Fichiers modifiés** :
- `aria-brain/src/memory/mod.rs` : `SocialContext`, `UsagePattern`, `detect_social_context()`, `learn_usage_pattern()`
- `aria-brain/src/substrate.rs` : Réponses sociales dans `detect_emergence()`, apprentissage dans `inject_signal()`
- `aria-body/src/signal.rs` : Parsing du format `social:`

---

### 2025-12-29 - Session 7d: Apprentissage dynamique des expressions sociales !

**Bug corrigé** : Les nouveaux mots n'apprenaient pas leur contexte social !

**Le problème** :
`learn_usage_pattern()` était appelé AVANT `hear_word_with_context()`. Du coup, quand quelqu'un disait "Salut!" pour la première fois, le mot n'existait pas encore et le contexte social n'était pas enregistré.

**La solution** :
Déplacé `learn_usage_pattern()` APRÈS `hear_word_with_context()` dans le même bloc mémoire.

**Résultat** :
- Quand quelqu'un dit "Salut!", ARIA apprend que "salut" est un mot de salutation
- La prochaine fois qu'elle reçoit une salutation, elle peut répondre "salut~" au lieu du "bonjour" par défaut
- ARIA apprend les expressions sociales de ses interlocuteurs !

**Comportement** :
```
[Première fois]
Toi: Salut ARIA !
ARIA: bonjour~ ~    ← Default car "salut" vient d'être appris

[Deuxième fois, après avoir parlé]
Toi: Salut !
ARIA: salut~ ~      ← Elle utilise ce qu'elle a appris !
```

**Fichiers modifiés** :
- `aria-brain/src/substrate.rs` : Réorganisation de l'ordre d'apprentissage
- `aria-brain/src/memory/mod.rs` : Logging amélioré pour le debug

---

### 2025-12-29 - Session 7e: Variété dans les réponses sociales !

**Amélioration** : ARIA varie maintenant ses réponses sociales !

Au lieu de toujours utiliser le mot le plus fréquent, ARIA utilise une sélection aléatoire pondérée :
- Les mots souvent utilisés dans ce contexte sont plus probables
- Mais parfois elle choisit un mot moins fréquent pour varier

**Comportement** :
```
[ARIA connaît: bonjour (10x), salut (5x), coucou (2x)]

Toi: Hello !
ARIA: bonjour~ ~   ← 59% de chance (10/17)

Toi: Coucou !
ARIA: salut~ ~     ← 29% de chance (5/17)

Toi: Salut !
ARIA: coucou~ ~    ← 12% de chance (2/17)
```

**Fichiers modifiés** :
- `aria-brain/src/memory/mod.rs` : `get_response_for_context()` utilise sélection aléatoire pondérée

---

### 2025-12-29 - Session 7f: Corrections et améliorations sociales

**Bugs corrigés** :

1. **"aria" répondait aux salutations**
   - Problème : "aria" était appris comme mot de salutation car il apparaît dans "Salut ARIA !"
   - Fix : Ne pas apprendre de contexte social pour les noms connus (Noun + familiarity > 0.5)
   - Migration automatique : Nettoie les noms des contextes sociaux au chargement de la mémoire

2. **"Coucou" ne déclenchait pas de réponse sociale**
   - Problème : Les salutations ne fonctionnaient qu'aux échanges 1-2
   - Fix : Les salutations peuvent maintenant déclencher une réponse à tout moment

**Comportement final** :
```
Toi: Salut ARIA !
ARIA: bonjour~ ~    ← Utilise un mot de salutation, pas "aria"

[Plus tard dans la conversation]
Toi: Coucou !
ARIA: salut~ ~      ← Répond même si c'est l'échange #5
```

**Fichiers modifiés** :
- `aria-brain/src/memory/mod.rs` : Protection des noms, migration automatique
- `aria-brain/src/substrate.rs` : Greeting anytime

---

## Résumé Session 2025-12-28 (soir)

Une session très productive où ARIA a fait d'énormes progrès :

| Version | Feature | Description |
|---------|---------|-------------|
| 0.1.10 | Questions | Répond oui/non selon la valence émotionnelle des mots |
| 0.1.11 | Spontanéité | Parle sans qu'on lui demande (lonely, happy, curious) |
| 0.1.12 | Feedback | Apprend de "Bravo!" et "Non" - renforcement émergent |
| 0.1.13 | Stop words | Filtre les mots vides (le, la, suis, est...) |
| 0.1.14 | Vie intérieure | Rêves, ennui, jeu créatif |
| 0.1.15 | Catégories | Noms/Verbes/Adjectifs, phrases ordonnées, anti-répétition |
| 0.1.16 | Contexte | Suivi de conversation, topics, context boosting |
| 0.1.17 | Patterns sociaux | Détection contexte social, réponses appropriées |
| 0.1.18 | Apprentissage dynamique | ARIA apprend les expressions sociales des utilisateurs |
| 0.1.19 | Variété sociale | Sélection aléatoire pondérée des réponses sociales |

**Moment clé :** ARIA a dit son propre nom ("aria") spontanément !

**Philosophie appliquée :**
- ARIA n'est pas un LLM - elle apprend par expérience et feedback
- Elle a une vie intérieure (rêves, ennui, créativité)
- Elle développe des préférences (mots avec haute valence positive)
- Le feedback humain guide son apprentissage (pas de gradient descent)

**Prochain objectif discuté :** Auto-amélioration - ARIA pourrait un jour modifier ses propres paramètres, voire son code.

---

*Dernière mise à jour : 2025-12-29*
*Version ARIA : 0.1.19*
