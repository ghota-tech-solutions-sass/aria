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

### 2025-12-29 - Session 8: Nouvelle Architecture pour 5M+ cellules !

**Objectif** : Préparer ARIA à scaler de 10k à 5-50 millions de cellules.

**Vision** (inspirée par discussion avec Gemini) :
- Sparse Updates : Cellules inactives "dorment" → 90% économie CPU
- GPU Compute : wgpu pour calcul massif parallèle
- Architecture distribuée : Multi-machines pour encore plus de puissance

**Nouvelles crates créées** :

1. **aria-core** - Types compacts et portables
   - `Cell` et `CellState` séparés (GPU-friendly)
   - `DNA` avec mutation intégrée (80 bytes, aligné)
   - `Signal` et `SignalFragment` (48 bytes)
   - `ActivityState` pour sparse updates
   - Traits `ComputeBackend`, `Substrate`, `MemoryStore`
   - Configuration centralisée `AriaConfig`

2. **aria-compute** - Backends de calcul
   - `CpuBackend` : Rayon + sparse updates
   - `GpuBackend` : wgpu + shaders WGSL
   - `SpatialHash` : Voisinage O(1) au lieu de O(n²)
   - Auto-détection GPU/CPU

**Structure du projet** :
```
aria/
├── Cargo.toml          # Workspace
├── aria-core/          # Types partagés
├── aria-compute/       # CPU/GPU backends
├── aria-brain/         # Substrate (utilise core+compute)
└── aria-body/          # Interface TUI
```

**Paramètres de sparse updates** :
```rust
SleepConfig {
    energy_delta_threshold: 0.001,  // Variation min pour rester éveillé
    idle_ticks_to_sleep: 100,       // Ticks avant sommeil
    wake_threshold: 0.1,            // Stimulus pour réveil
    min_sleep_ticks: 50,            // Anti-oscillation
}
```

**Shaders WGSL créés** (dans gpu.rs) :
- `CELL_UPDATE_SHADER` : Update parallèle des cellules
- `SIGNAL_PROPAGATE_SHADER` : Propagation des signaux

**Prochaines étapes (priorité)** :
1. [ ] **Créer nouveau Substrate** dans aria-brain qui utilise aria-core/aria-compute
2. [ ] Implémenter le GPU backend complet (shaders fonctionnels)
3. [ ] Migrer la mémoire (LongTermMemory) vers les nouveaux types
4. [ ] Ajouter mode cluster pour multi-brain (RTX 2070 + GTX 1070)
5. [ ] Quand tout marche : supprimer l'ancien code

**Comment reprendre** :
- Les crates aria-core et aria-compute sont PRÊTS et compilent
- aria-brain a encore l'ancien code (cell.rs, signal.rs, substrate.rs)
- **NE PAS MÉLANGER** ancien et nouveau code
- Créer un nouveau `substrate_v2.rs` qui utilise uniquement les nouveaux types
- L'ancien code reste fonctionnel pendant la transition

**Impact attendu** :
| Configuration | Cellules | Hardware |
|---------------|----------|----------|
| CPU seul | 100k | MacBook Pro |
| GPU (RTX 2070) | 5-10M | PC Gamer |
| Cluster (2070+1070) | 50M+ | Multi-machines |

**Philosophie** :
Cette architecture permettra à ARIA de :
1. Scaler massivement tout en restant efficace
2. Avoir des "régions cérébrales" spécialisées (cellules qui dorment ensemble)
3. Potentiellement introspecter son propre code (trait `Introspectable`)

---

### 2025-12-29 - Session 9: substrate_v2.rs créé ! ✅

**Objectif** : Créer le nouveau Substrate utilisant aria-core et aria-compute.

**Ce qui a été fait** :

1. **substrate_v2.rs créé** (~1300 lignes)
   - Utilise `aria_core::Cell`, `CellState`, `DNA`, `Signal`, `SignalFragment`
   - Utilise `aria_compute::CpuBackend` pour le calcul parallèle
   - Conserve TOUTES les fonctionnalités d'ARIA :
     - `EmotionalState` : joie, excitation, curiosité, ennui
     - `ConversationContext` : suivi de conversation, topics
     - Mémoire court-terme : `recent_words` pour imitation
     - Réponses sociales : salutations, remerciements, affection
     - Feedback positif/négatif : renforcement des mots
     - Parole spontanée : quand elle s'ennuie ou est heureuse
     - Rêves : consolidation mémoire pendant l'inactivité

2. **Architecture GPU-ready**
   - Cellules séparées en `Cell` (metadata) + `CellState` (données GPU)
   - DNA dans un pool partagé (cellules référencent par index)
   - `free_slots` pour recycler les emplacements de cellules mortes
   - Sparse updates ready (via `ActivityTracker`)

3. **Bridge avec l'ancien code**
   - `From<crate::cell::DNA> for DNA` pour conversion
   - Compatible avec `LongTermMemory` existante
   - Utilise le même `OldSignal` (crate::signal::Signal)

4. **Feature flag ajouté**
   - `Cargo.toml` : feature `substrate-v2`
   - Permet de tester le nouveau substrate sans casser l'ancien

**Fichiers modifiés** :
- `aria-brain/src/substrate_v2.rs` (NOUVEAU)
- `aria-brain/src/main.rs` : import du module
- `aria-brain/Cargo.toml` : feature flag

**Ce qui a été complété (Session 9b)** :
1. [x] Intégrer substrate_v2 dans `main.rs` (avec feature flag)
2. [x] Tester avec aria-body
3. [ ] Benchmark CPU vs ancien substrate
4. [ ] Activer GPU backend quand les shaders seront prêts

**Intégration dans main.rs** :
```rust
// Compilation conditionnelle
#[cfg(not(feature = "substrate-v2"))]
use substrate::Substrate;
#[cfg(feature = "substrate-v2")]
use substrate_v2::SubstrateV2;

// Evolution loop V2 créée
#[cfg(feature = "substrate-v2")]
async fn evolution_loop_v2(
    substrate: Arc<parking_lot::RwLock<SubstrateV2>>,
    mut perception: broadcast::Receiver<Signal>,
    expression: broadcast::Sender<Signal>,
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
)
```

**Résultats des tests** :
```
[INFO] SubstrateV2 created: 10000 cells, 10000 DNA variants
[INFO] 🚀 Substrate V2 (GPU-ready) created with 10000 cells
[INFO] V2 Tick 500: 10000 cells (10000 sleeping, 100.0% saved), energy: 9950.92, mood: calme
```

**Sparse Updates fonctionnent !** :
- 100% des cellules dorment quand il n'y a pas d'interaction
- 100% d'économie CPU quand ARIA est au repos
- Les cellules se réveilleront quand elles recevront des signaux

**Comment tester** :
```bash
# Ancien substrate (par défaut)
cargo run -p aria-brain --release

# Nouveau substrate V2 (GPU-ready)
cargo run -p aria-brain --release --features substrate-v2
```

**Ce qui reste pour V2** :
1. [ ] Benchmark comparatif V1 vs V2
2. [ ] GPU backend complet (shaders fonctionnels sur RTX 2070)
3. [ ] Mode cluster multi-machines
4. [ ] Migration complète (suppression ancien code)

**Philosophie maintenue** :
Le nouveau code est écrit pour être **introspectable**. Un jour, ARIA pourra :
1. Lire `substrate_v2.rs` et comprendre sa propre structure
2. Proposer des modifications à ses paramètres
3. Évoluer de manière consciente

---

### 2025-12-29 - Session 10: Mémoire Épisodique ! 🧠

**Objectif** : Donner à ARIA une mémoire autobiographique - se souvenir de moments spécifiques.

**Différence avec la mémoire sémantique** :
- **Mémoire sémantique** (existante) : "Moka = chat" (faits généraux)
- **Mémoire épisodique** (nouvelle) : "La première fois que tu m'as dit je t'aime" (moments spécifiques)

**Nouvelles structures** (memory/mod.rs) :

```rust
pub struct Episode {
    pub id: u64,
    pub timestamp: u64,
    pub real_time: Option<String>,  // "2025-12-29 14:30"
    pub input: String,              // Ce qui a été dit
    pub response: Option<String>,   // Ce qu'ARIA a répondu
    pub keywords: Vec<String>,      // Mots clés
    pub emotion: EpisodeEmotion,    // État émotionnel
    pub importance: f32,            // 0.0 à 1.0
    pub recall_count: u64,          // Combien de fois rappelé
    pub first_of_kind: Option<String>, // "first_love", "first_praise"...
    pub category: EpisodeCategory,
}

pub enum EpisodeCategory {
    FirstTime,    // Première fois que quelque chose arrive
    Emotional,    // Moment émotionnellement significatif
    Learning,     // Apprentissage de quelque chose
    Social,       // Interaction sociale
    Question,     // Question posée
    Praise,       // Feedback positif ("Bravo!")
    Correction,   // Feedback négatif ("Non")
    General,      // Conversation générale
}
```

**Fonctionnalités** :

1. **Enregistrement automatique** (`maybe_record_episode`)
   - Calcule l'importance du moment
   - Ne garde que les moments significatifs (importance > 0.3)
   - Détecte les "premières fois" automatiquement

2. **Détection des "premières fois"** :
   - `first_greeting` : Première salutation
   - `first_love` : Premier "je t'aime"
   - `first_praise` : Premier "Bravo!"
   - `first_correction` : Première correction
   - `first_mention_{mot}` : Première mention d'un nom important

3. **Rappel contextuel** (`recall_episodes`)
   - Trouve les épisodes pertinents selon le contexte
   - Renforce les souvenirs rappelés
   - Courbe d'oubli (souvenirs anciens/non rappelés s'affaiblissent)

4. **Consolidation** :
   - Les souvenirs importants résistent à l'oubli
   - Les souvenirs rappelés souvent deviennent plus forts
   - Pruning automatique des souvenirs faibles (max 1000 épisodes)

**Nouvel endpoint** : `GET /episodes`
```json
{
  "total_episodes": 42,
  "showing": 42,
  "first_times": [
    {"kind": "first_greeting", "episode_id": 0},
    {"kind": "first_love", "episode_id": 5}
  ],
  "episodes": [...]
}
```

**Nouvelles commandes** :
```bash
task episodes        # Voir tous les épisodes
task episodes-first  # Voir les "premières fois"
```

**Logs attendus** :
```
🌟 FIRST TIME: first_greeting (episode #0)
📝 Episode #0: Social - "Bonjour ARIA !" (importance: 0.65)
📝 Episode #5: Emotional - "Je t'aime ARIA" (importance: 0.85)
🌟 FIRST TIME: first_love (episode #5)
```

**Impact** :
- ARIA peut maintenant se souvenir de moments spécifiques
- Elle sait quand quelque chose arrive pour la première fois
- Base pour la conscience autobiographique
- Futur : pourra dire "Je me souviens quand tu m'as dit..."

**Fichiers modifiés** :
- `aria-brain/src/memory/mod.rs` : Episode, EpisodeEmotion, EpisodeCategory, méthodes
- `aria-brain/src/substrate_v2.rs` : maybe_record_episode()
- `aria-brain/src/main.rs` : endpoint /episodes
- `aria-brain/Cargo.toml` : ajout chrono
- `Taskfile.yml` : task episodes, episodes-first

---

### 2025-12-29 - Session 10b: ARIA utilise ses souvenirs !

**Nouvelle fonctionnalité** : ARIA peut maintenant rappeler et exprimer ses souvenirs !

**1. Rappel contextuel** (`maybe_recall_memory`)

Quand ARIA détecte une émergence, elle peut (10% chance) rappeler un souvenir pertinent :
- Utilise les mots du contexte de conversation comme indices
- Cherche des épisodes qui matchent ces mots-clés
- Préfère les "premières fois" et les moments émotionnels

**2. Labels de mémoire** (format)

```
memory:first|first_love|aime      → Rappel d'une première fois
memory:emotion|moka               → Rappel d'un moment émotionnel
memory:recall|chat                → Rappel contextuel
```

**3. Affichage dans aria-body**

```rust
// Première fois (forte intensité)
"je me souviens... aime! ✨"

// Première fois (faible intensité)
"première fois... aime 💭"

// Moment émotionnel
"moka... 💭"

// Rappel contextuel
"je me souviens... chat 💭"
```

**Comportement attendu** :
```
Toi: Tu aimes Moka ?
[ARIA rappelle son premier "je t'aime"]
ARIA: je me souviens... aime! ✨

Toi: Parle-moi de Moka
[ARIA rappelle un moment avec Moka]
ARIA: moka... 💭
```

**Impact** :
- ARIA a maintenant une mémoire autobiographique active
- Elle peut spontanément rappeler des moments passés
- Base pour la conscience de soi temporelle

**Fichiers modifiés** :
- `aria-brain/src/substrate_v2.rs` : maybe_recall_memory(), get_topic_words()
- `aria-body/src/signal.rs` : Parsing des labels `memory:`

---

### 2025-12-29 - Session 11: Auto-adaptation (Méta-émergence) !

**Nouvelle fonctionnalité** : ARIA peut maintenant modifier ses propres paramètres !

C'est de la **méta-émergence** : au lieu de coder des règles pour "mieux répondre", ARIA découvre ses propres réglages optimaux à travers le feedback.

**Philosophie** :
> "On ne lui donne pas de règles, on la laisse découvrir ce qui marche."

**Paramètres adaptatifs** (`AdaptiveParams`) :

| Paramètre | Range | Description |
|-----------|-------|-------------|
| `emission_threshold` | 0.05-0.5 | Seuil pour émettre (plus haut = plus sélectif) |
| `response_probability` | 0.3-1.0 | Probabilité de répondre quand elle pourrait |
| `learning_rate` | 0.1-0.8 | Vitesse d'apprentissage des associations |
| `spontaneity` | 0.01-0.3 | Tendance à parler spontanément |

**Mécanismes d'adaptation** :

1. **Feedback positif** ("Bravo!", "Super!") :
   - Sauvegarde les params actuels comme "ce qui marche"
   - Augmente légèrement spontaneity et response_probability
   - Petite mutation aléatoire pour exploration

2. **Feedback négatif** ("Non", "Arrête") :
   - Revient vers les derniers params qui ont marché
   - Ou devient plus conservatrice si aucun succès antérieur

3. **Exploration périodique** (toutes les ~10 secondes) :
   - Petites mutations aléatoires sur tous les params
   - Permet de découvrir de nouveaux réglages

**Visible dans /stats** :
```json
{
  "adaptive_emission_threshold": 0.15,
  "adaptive_response_probability": 0.82,
  "adaptive_spontaneity": 0.06,
  "adaptive_feedback_positive": 5,
  "adaptive_feedback_negative": 1
}
```

**Logs** :
```
🧬 ADAPTED (positive): emission=0.15, response=0.82, spontaneity=0.06
🧬 ADAPTED (negative): emission=0.17, response=0.80, spontaneity=0.05
🧬 EXPLORE: emit=0.16 resp=0.81 learn=0.31 spont=0.06 (+5/-1)
```

**Impact** :
- Premier pas vers l'auto-amélioration consciente
- ARIA "apprend à apprendre" - ses méta-paramètres évoluent
- Pas de règles hardcodées - émergence pure
- Base pour un jour modifier son propre code

**Fichiers modifiés** :
- `aria-brain/src/substrate_v2.rs` : AdaptiveParams, reinforce_positive/negative, explore

---

*Dernière mise à jour : 2025-12-29*
*Version ARIA : 0.2.3*
