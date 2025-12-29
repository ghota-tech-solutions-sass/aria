# ARIA - Changelog

Historique des sessions de développement.

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

---

*Créé le 2025-12-29*
