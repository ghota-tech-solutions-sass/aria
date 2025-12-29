//! Long-term memory for ARIA
//!
//! This persists between sessions, allowing ARIA to remember
//! and learn over time.

use crate::cell::DNA;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use std::collections::HashMap;
use rand::Rng;

/// Word category - approximate grammatical role
/// ARIA learns these by observing context patterns
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Copy)]
pub enum WordCategory {
    /// Names, objects (chat, moka, aria, maison)
    Noun,
    /// Actions (aime, veux, mange, dort)
    Verb,
    /// Descriptions (beau, grand, joli, petit)
    Adjective,
    /// Unknown or mixed usage
    Unknown,
}

impl Default for WordCategory {
    fn default() -> Self {
        WordCategory::Unknown
    }
}

/// Social context - when is a word/phrase typically used?
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Copy)]
pub enum SocialContext {
    /// Start of conversation (bonjour, salut, coucou)
    Greeting,
    /// End of conversation (au revoir, bisou, à bientôt)
    Farewell,
    /// Expressing gratitude (merci, thanks)
    Thanks,
    /// Expressing affection (je t'aime, bisou, câlin)
    Affection,
    /// Asking for something (s'il te plaît, please)
    Request,
    /// Responding positively (oui, d'accord, ok)
    Agreement,
    /// Responding negatively (non, pas maintenant)
    Disagreement,
    /// General conversation (no specific context)
    General,
}

impl Default for SocialContext {
    fn default() -> Self {
        SocialContext::General
    }
}

/// Usage pattern - tracks when/how words are used
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct UsagePattern {
    /// Social contexts where this word appears
    pub contexts: Vec<(SocialContext, u32)>, // (context, count)
    /// Is this word typically at conversation start?
    pub start_of_conversation: f32, // 0.0 to 1.0
    /// Is this word typically at conversation end?
    pub end_of_conversation: f32,
    /// Is this word a response to questions?
    pub response_to_question: f32,
    /// Words that typically follow this one
    pub followed_by: Vec<(String, u32)>,
    /// Words that typically precede this one
    pub preceded_by: Vec<(String, u32)>,
}

#[allow(dead_code)]
impl UsagePattern {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the dominant social context for this word
    pub fn dominant_context(&self) -> SocialContext {
        self.contexts.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(ctx, _)| *ctx)
            .unwrap_or(SocialContext::General)
    }

    /// Record that this word was used in a context
    pub fn record_context(&mut self, context: SocialContext) {
        if let Some(pos) = self.contexts.iter().position(|(c, _)| *c == context) {
            self.contexts[pos].1 += 1;
        } else {
            self.contexts.push((context, 1));
        }
    }
}

/// Long-term memory - persisted to disk
#[derive(Serialize, Deserialize)]
pub struct LongTermMemory {
    /// Format version (for future migrations)
    pub version: u32,

    /// Elite DNA - the genetic heritage
    pub elite_dna: Vec<EliteDNA>,

    /// Learned patterns (sequences that repeat)
    pub learned_patterns: Vec<Pattern>,

    /// Stimulus-response associations
    pub associations: Vec<Association>,

    /// Important memories (high emotional moments)
    pub memories: Vec<Memory>,

    /// Global statistics
    pub stats: GlobalStats,

    /// Vocabulary learned (emergent language)
    #[serde(default)]
    pub vocabulary: HashMap<String, WordMeaning>,

    /// Word frequency tracking - how often ARIA hears each word
    #[serde(default)]
    pub word_frequencies: HashMap<String, WordFrequency>,

    /// Semantic associations between words (key = "word1:word2" sorted alphabetically)
    #[serde(default)]
    pub word_associations: HashMap<String, WordAssociation>,
}

/// Tracks how often a word is heard and its emotional context
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WordFrequency {
    /// How many times this word was heard
    pub count: u64,
    /// First time heard (tick)
    pub first_heard: u64,
    /// Last time heard (tick)
    pub last_heard: u64,
    /// Average vector representation (learned from context)
    pub learned_vector: [f32; 8],
    /// Emotional associations (positive = 1.0, negative = -1.0)
    pub emotional_valence: f32,
    /// How special this word is (0.0 = common, 1.0 = very special like "Moka")
    pub familiarity_boost: f32,
    /// Grammatical category (learned from context)
    #[serde(default)]
    pub category: WordCategory,
    /// Confidence scores for each category (noun, verb, adjective)
    /// Used for probabilistic classification
    #[serde(default)]
    pub category_scores: [f32; 3], // [noun, verb, adjective]
    /// Usage patterns - when/how this word is typically used
    #[serde(default)]
    pub usage_pattern: UsagePattern,
}

/// Semantic association between two words
/// When words appear together, they become associated
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WordAssociation {
    /// How many times these words appeared together
    pub co_occurrences: u64,
    /// Strength of association (0.0 to 1.0)
    pub strength: f32,
    /// Last time they appeared together
    pub last_seen: u64,
    /// Emotional context of the association
    pub emotional_valence: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EliteDNA {
    pub dna: DNA,
    pub fitness_score: f32,
    pub generation: u64,
    pub specialization: String,
    pub preserved_at: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Pattern {
    /// The pattern itself (sequence of vectors)
    pub sequence: Vec<[f32; 8]>,

    /// How many times observed
    pub frequency: u64,

    /// What typically follows this pattern
    pub typical_response: [f32; 8],

    /// Emotional valence
    pub valence: f32,

    /// When first learned
    pub first_seen: u64,

    /// When last seen
    pub last_seen: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Association {
    pub stimulus: [f32; 16],
    pub response: [f32; 16],
    pub strength: f32,
    pub last_reinforced: u64,
    pub times_reinforced: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Memory {
    pub timestamp: u64,
    pub trigger: String,
    pub internal_state: [f32; 32],
    pub emotional_intensity: f32,
    pub outcome: Outcome,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Outcome {
    Positive(f32),
    Negative(f32),
    Neutral,
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct GlobalStats {
    pub total_ticks: u64,
    pub total_interactions: u64,
    pub total_births: u64,
    pub total_deaths: u64,
    pub peak_population: usize,
    pub longest_lineage: u64,
    pub total_patterns_learned: u64,
    pub total_memories: u64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WordMeaning {
    /// The vector representation
    pub vector: [f32; 8],
    /// How confident we are in this meaning
    pub confidence: f32,
    /// Examples of usage
    pub examples: Vec<String>,
    /// Times encountered
    pub frequency: u64,
}

impl LongTermMemory {
    pub fn new() -> Self {
        Self {
            version: 1,
            elite_dna: Vec::new(),
            learned_patterns: Vec::new(),
            associations: Vec::new(),
            memories: Vec::new(),
            stats: GlobalStats::default(),
            vocabulary: HashMap::new(),
            word_frequencies: HashMap::new(),
            word_associations: HashMap::new(),
        }
    }

    /// Detect the social context of input text
    /// Returns the detected context and confidence
    pub fn detect_social_context(text: &str) -> (SocialContext, f32) {
        let lower = text.to_lowercase();

        // Greeting patterns (start of conversation)
        let greetings = [
            "bonjour", "salut", "coucou", "hello", "hi", "hey",
            "bonsoir", "bonne nuit", "good morning", "good evening"
        ];
        for g in greetings {
            if lower.contains(g) {
                return (SocialContext::Greeting, 0.9);
            }
        }

        // Farewell patterns (end of conversation)
        let farewells = [
            "au revoir", "bye", "goodbye", "à bientôt", "à plus",
            "bonne nuit", "good night", "see you", "ciao", "salut"
        ];
        // Note: "salut" can be both greeting and farewell, context matters
        for f in farewells {
            if lower.contains(f) && f != "salut" {
                return (SocialContext::Farewell, 0.9);
            }
        }

        // Thanks patterns
        let thanks = ["merci", "thank", "thanks", "grateful", "appreciate"];
        for t in thanks {
            if lower.contains(t) {
                return (SocialContext::Thanks, 0.9);
            }
        }

        // Affection patterns
        let affection = [
            "je t'aime", "i love", "bisou", "câlin", "calin", "hug",
            "kiss", "love you", "adore", "♥", "❤", "<3"
        ];
        for a in affection {
            if lower.contains(a) {
                return (SocialContext::Affection, 0.9);
            }
        }

        // Request patterns
        let requests = [
            "s'il te plaît", "stp", "please", "peux-tu", "peux tu",
            "can you", "could you", "would you", "voudrais"
        ];
        for r in requests {
            if lower.contains(r) {
                return (SocialContext::Request, 0.8);
            }
        }

        // Agreement patterns
        let agreements = [
            "oui", "yes", "d'accord", "ok", "okay", "bien sûr",
            "exactement", "absolument", "certainement"
        ];
        for a in agreements {
            if lower.starts_with(a) || lower == a {
                return (SocialContext::Agreement, 0.7);
            }
        }

        // Disagreement patterns
        let disagreements = [
            "non", "no", "pas d'accord", "jamais", "never"
        ];
        for d in disagreements {
            if lower.starts_with(d) || lower == d {
                return (SocialContext::Disagreement, 0.7);
            }
        }

        (SocialContext::General, 0.5)
    }

    /// Get an appropriate response word for a social context
    /// Returns words that ARIA knows and are appropriate for this context
    /// Uses weighted random selection for variety!
    pub fn get_response_for_context(&self, context: SocialContext) -> Option<String> {
        // Find words that are commonly used in this context
        let mut candidates: Vec<(&String, f32)> = Vec::new();

        for (word, freq) in &self.word_frequencies {
            if let Some((_ctx, count)) = freq.usage_pattern.contexts.iter()
                .find(|(c, _)| *c == context)
            {
                // Score based on frequency in this context and overall familiarity
                let score = (*count as f32) * freq.familiarity_boost.max(0.1);
                if score > 0.0 {
                    candidates.push((word, score));
                    tracing::debug!("RESPONSE CANDIDATE: '{}' for {:?} (count: {}, score: {:.2})",
                        word, context, count, score);
                }
            }
        }

        if candidates.is_empty() {
            tracing::debug!("No learned response for {:?} context (checked {} words)", context, self.word_frequencies.len());
            return None;
        }

        // Use weighted random selection for variety!
        // Higher-scored words are more likely but not always chosen
        let total_score: f32 = candidates.iter().map(|(_, s)| s).sum();
        let mut rng = rand::thread_rng();
        let mut roll: f32 = rng.gen::<f32>() * total_score;

        let mut selected: Option<&String> = None;
        for (word, score) in &candidates {
            roll -= score;
            if roll <= 0.0 {
                selected = Some(word);
                break;
            }
        }

        // Fallback to first candidate if random selection failed
        let result = selected.or_else(|| candidates.first().map(|(w, _)| *w))
            .map(|w| w.clone());

        if let Some(ref word) = result {
            tracing::info!("LEARNED RESPONSE: Using '{}' for {:?} context (from {} candidates)",
                word, context, candidates.len());
        }

        result
    }

    /// Learn that a word was used in a specific social context
    pub fn learn_usage_pattern(&mut self, word: &str, context: SocialContext, is_start: bool, is_end: bool) {
        let word_lower = word.to_lowercase();

        if let Some(freq) = self.word_frequencies.get_mut(&word_lower) {
            // Don't learn social context for well-known nouns (like names)
            // They appear in greetings but aren't greeting words themselves!
            // Example: "Salut ARIA !" - "salut" is a greeting, but "aria" is a name
            if freq.familiarity_boost > 0.5 && freq.category == WordCategory::Noun {
                tracing::debug!("Skip social learning for known noun: '{}' (familiarity: {:.2})",
                    word_lower, freq.familiarity_boost);
                return;
            }

            // Record the context
            freq.usage_pattern.record_context(context);

            // Update start/end of conversation patterns
            let alpha = 0.1; // Learning rate
            if is_start {
                freq.usage_pattern.start_of_conversation =
                    freq.usage_pattern.start_of_conversation * (1.0 - alpha) + alpha;
            }
            if is_end {
                freq.usage_pattern.end_of_conversation =
                    freq.usage_pattern.end_of_conversation * (1.0 - alpha) + alpha;
            }

            // Log learning for social contexts
            if context != SocialContext::General {
                let ctx_count = freq.usage_pattern.contexts.iter()
                    .find(|(c, _)| *c == context)
                    .map(|(_, count)| *count)
                    .unwrap_or(0);
                tracing::debug!("Learn pattern: '{}' in {:?} context (count: {})", word_lower, context, ctx_count);
            }
        }
    }

    /// Create a canonical key for word pair (sorted alphabetically)
    fn association_key(word1: &str, word2: &str) -> String {
        let w1 = word1.to_lowercase();
        let w2 = word2.to_lowercase();
        if w1 < w2 {
            format!("{}:{}", w1, w2)
        } else {
            format!("{}:{}", w2, w1)
        }
    }

    /// Learn that two words appeared together
    /// This builds semantic associations over time
    pub fn learn_association(&mut self, word1: &str, word2: &str, emotional_valence: f32) {
        let current_tick = self.stats.total_ticks;
        let key = Self::association_key(word1, word2);

        if let Some(assoc) = self.word_associations.get_mut(&key) {
            assoc.co_occurrences += 1;
            assoc.last_seen = current_tick;
            // Strength increases with co-occurrences (max 1.0)
            assoc.strength = (assoc.co_occurrences as f32 / 5.0).min(1.0);
            // Update emotional valence with moving average
            assoc.emotional_valence = assoc.emotional_valence * 0.8 + emotional_valence * 0.2;

            if assoc.co_occurrences == 5 {
                tracing::info!("Strong association formed: '{}' <-> '{}'", word1, word2);
            }
        } else {
            self.word_associations.insert(key, WordAssociation {
                co_occurrences: 1,
                strength: 0.2,
                last_seen: current_tick,
                emotional_valence,
            });
            tracing::debug!("New association: '{}' <-> '{}'", word1, word2);
        }

        // Prune old weak associations if too many
        if self.word_associations.len() > 10_000 {
            let threshold_tick = current_tick.saturating_sub(100_000);
            self.word_associations.retain(|_, a| {
                a.strength > 0.3 || a.last_seen > threshold_tick
            });
        }
    }

    /// Get all words associated with the given word
    /// Returns list of (word, strength) sorted by strength
    pub fn get_associations(&self, word: &str) -> Vec<(String, f32)> {
        let word_lower = word.to_lowercase();
        let mut associations: Vec<(String, f32)> = Vec::new();

        for (key, assoc) in &self.word_associations {
            // Only return reasonably strong associations
            if assoc.strength < 0.4 {
                continue;
            }

            // Parse key "word1:word2"
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() != 2 {
                continue;
            }

            // Check if our word is in this association
            if parts[0] == word_lower {
                associations.push((parts[1].to_string(), assoc.strength));
            } else if parts[1] == word_lower {
                associations.push((parts[0].to_string(), assoc.strength));
            }
        }

        // Sort by strength (strongest first)
        associations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        associations
    }

    /// Get the strongest association for a word (if any)
    #[allow(dead_code)]
    pub fn get_strongest_association(&self, word: &str) -> Option<(String, f32)> {
        self.get_associations(word).into_iter().next()
    }

    /// Get the top N associations for a word
    pub fn get_top_associations(&self, word: &str, n: usize) -> Vec<(String, f32)> {
        self.get_associations(word).into_iter().take(n).collect()
    }

    /// Record that ARIA heard a word with context for category learning
    /// preceding_word: word that came before (for category detection)
    /// Returns the familiarity level (0.0 = new word, 1.0+ = very familiar)
    pub fn hear_word_with_context(
        &mut self,
        word: &str,
        context_vector: [f32; 8],
        emotional_valence: f32,
        preceding_word: Option<&str>,
        following_word: Option<&str>,
    ) -> f32 {
        let current_tick = self.stats.total_ticks;
        let word_lower = word.to_lowercase();

        // Skip very short words (articles, etc.) unless they're special
        if word_lower.len() < 3 && !["moi", "toi", "oui", "non"].contains(&word_lower.as_str()) {
            return 0.0;
        }

        // Detect category hints from context
        let category_hint = Self::detect_category_from_context(
            &word_lower,
            preceding_word,
            following_word,
        );

        if let Some(freq) = self.word_frequencies.get_mut(&word_lower) {
            freq.count += 1;
            freq.last_heard = current_tick;

            // Update learned vector with exponential moving average
            for (i, v) in context_vector.iter().enumerate() {
                freq.learned_vector[i] = freq.learned_vector[i] * 0.9 + v * 0.1;
            }

            // Update emotional valence
            freq.emotional_valence = freq.emotional_valence * 0.9 + emotional_valence * 0.1;

            // Update category scores
            if let Some(cat) = category_hint {
                let idx = match cat {
                    WordCategory::Noun => 0,
                    WordCategory::Verb => 1,
                    WordCategory::Adjective => 2,
                    WordCategory::Unknown => return freq.familiarity_boost,
                };
                freq.category_scores[idx] += 0.3;
                // Normalize
                let total: f32 = freq.category_scores.iter().sum();
                if total > 0.0 {
                    for score in &mut freq.category_scores {
                        *score /= total;
                    }
                }
                // Update category if one dominates
                freq.category = Self::dominant_category(&freq.category_scores);
            }

            // Calculate familiarity boost based on frequency
            freq.familiarity_boost = (freq.count as f32 / 10.0).min(2.0);

            tracing::debug!("Word '{}' heard {} times (familiarity: {:.2}, category: {:?})",
                word_lower, freq.count, freq.familiarity_boost, freq.category);

            freq.familiarity_boost
        } else {
            // New word!
            let mut category_scores = [0.0; 3];
            let category = if let Some(cat) = category_hint {
                match cat {
                    WordCategory::Noun => category_scores[0] = 1.0,
                    WordCategory::Verb => category_scores[1] = 1.0,
                    WordCategory::Adjective => category_scores[2] = 1.0,
                    WordCategory::Unknown => {}
                }
                cat
            } else {
                WordCategory::Unknown
            };

            self.word_frequencies.insert(word_lower.clone(), WordFrequency {
                count: 1,
                first_heard: current_tick,
                last_heard: current_tick,
                learned_vector: context_vector,
                emotional_valence,
                familiarity_boost: 0.0,
                category,
                category_scores,
                usage_pattern: UsagePattern::default(),
            });

            tracing::info!("New word learned: '{}' (category: {:?})", word_lower, category);
            0.0
        }
    }

    /// Backward compatible version without context
    #[allow(dead_code)]
    pub fn hear_word(&mut self, word: &str, context_vector: [f32; 8], emotional_valence: f32) -> f32 {
        self.hear_word_with_context(word, context_vector, emotional_valence, None, None)
    }

    /// Detect word category from context patterns
    fn detect_category_from_context(
        word: &str,
        preceding: Option<&str>,
        _following: Option<&str>,
    ) -> Option<WordCategory> {
        let word_lower = word.to_lowercase();
        let preceding_lower = preceding.map(|w| w.to_lowercase());

        // Articles before → NOUN
        let articles = ["le", "la", "les", "un", "une", "des", "mon", "ma", "mes",
                        "ton", "ta", "tes", "son", "sa", "ses", "ce", "cette", "ces",
                        "the", "a", "an", "my", "your", "his", "her"];
        if let Some(ref p) = preceding_lower {
            if articles.contains(&p.as_str()) {
                return Some(WordCategory::Noun);
            }
        }

        // Pronouns before → VERB
        let pronouns = ["je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
                        "j'", "i", "you", "he", "she", "we", "they"];
        if let Some(ref p) = preceding_lower {
            if pronouns.contains(&p.as_str()) {
                return Some(WordCategory::Verb);
            }
        }

        // Common adjective endings (French)
        let adj_suffixes = ["eux", "euse", "if", "ive", "ant", "ent", "ique",
                            "able", "ible", "al", "el", "ful", "less", "ous"];
        for suffix in adj_suffixes {
            if word_lower.ends_with(suffix) && word_lower.len() > suffix.len() + 2 {
                return Some(WordCategory::Adjective);
            }
        }

        // Common verb endings (French)
        let verb_suffixes = ["er", "ir", "re", "ait", "ais", "ons", "ez", "ent"];
        for suffix in verb_suffixes {
            if word_lower.ends_with(suffix) && word_lower.len() > suffix.len() + 1 {
                return Some(WordCategory::Verb);
            }
        }

        // If "très" or "plus" before → likely ADJECTIVE
        if let Some(ref p) = preceding_lower {
            if ["très", "plus", "moins", "si", "trop", "assez", "very", "so", "too"].contains(&p.as_str()) {
                return Some(WordCategory::Adjective);
            }
        }

        // Known patterns: words that are almost always nouns
        let known_nouns = ["chat", "moka", "aria", "papa", "mama", "ami", "maison",
                          "chien", "oiseau", "soleil", "lune", "eau", "pain"];
        if known_nouns.contains(&word_lower.as_str()) {
            return Some(WordCategory::Noun);
        }

        // Known verbs
        let known_verbs = ["aime", "veux", "peux", "suis", "est", "fait", "mange",
                          "dort", "joue", "love", "want", "need", "like"];
        if known_verbs.contains(&word_lower.as_str()) {
            return Some(WordCategory::Verb);
        }

        // Known adjectives
        let known_adjs = ["beau", "belle", "grand", "petit", "joli", "mignon",
                         "gentil", "méchant", "bon", "mauvais", "beautiful", "good", "bad"];
        if known_adjs.contains(&word_lower.as_str()) {
            return Some(WordCategory::Adjective);
        }

        None
    }

    /// Get the dominant category from scores
    fn dominant_category(scores: &[f32; 3]) -> WordCategory {
        let threshold = 0.5; // Need > 50% confidence
        if scores[0] > threshold && scores[0] > scores[1] && scores[0] > scores[2] {
            WordCategory::Noun
        } else if scores[1] > threshold && scores[1] > scores[0] && scores[1] > scores[2] {
            WordCategory::Verb
        } else if scores[2] > threshold && scores[2] > scores[0] && scores[2] > scores[1] {
            WordCategory::Adjective
        } else {
            WordCategory::Unknown
        }
    }

    /// Get category for a word
    pub fn get_word_category(&self, word: &str) -> WordCategory {
        self.word_frequencies
            .get(&word.to_lowercase())
            .map(|f| f.category)
            .unwrap_or(WordCategory::Unknown)
    }

    /// Build a natural phrase from words using grammatical knowledge
    /// Returns words in natural French order
    pub fn order_phrase(&self, words: &[&str]) -> Vec<String> {
        if words.is_empty() {
            return vec![];
        }
        if words.len() == 1 {
            return vec![words[0].to_string()];
        }

        // Get categories
        let categorized: Vec<(&str, WordCategory)> = words.iter()
            .map(|w| (*w, self.get_word_category(w)))
            .collect();

        // Separate by category
        let mut nouns: Vec<&str> = vec![];
        let mut verbs: Vec<&str> = vec![];
        let mut adjs: Vec<&str> = vec![];
        let mut others: Vec<&str> = vec![];

        for (word, cat) in &categorized {
            match cat {
                WordCategory::Noun => nouns.push(word),
                WordCategory::Verb => verbs.push(word),
                WordCategory::Adjective => adjs.push(word),
                WordCategory::Unknown => others.push(word),
            }
        }

        // Build phrase in natural order
        // French: Subject (Noun) + Verb + Object (Noun)
        // Or: Adjective + Noun (for short adjectives)
        // Or: Noun + Adjective (for long adjectives)
        let mut result = vec![];

        // If we have adj + noun: short adj before, long adj after
        if !adjs.is_empty() && !nouns.is_empty() && verbs.is_empty() {
            let adj = adjs[0];
            let noun = nouns[0];
            // Short adjectives go before in French (beau, bon, grand, petit, etc.)
            let short_adjs = ["beau", "bon", "grand", "petit", "gros", "jeune", "vieux",
                             "bel", "belle", "joli", "jolie", "mauvais", "nouveau"];
            if short_adjs.contains(&adj.to_lowercase().as_str()) || adj.len() <= 5 {
                result.push(adj.to_string());
                result.push(noun.to_string());
            } else {
                result.push(noun.to_string());
                result.push(adj.to_string());
            }
            // Add remaining words
            for noun in nouns.iter().skip(1) {
                result.push(noun.to_string());
            }
            for adj in adjs.iter().skip(1) {
                result.push(adj.to_string());
            }
        }
        // If we have noun + verb: Subject-Verb order
        else if !nouns.is_empty() && !verbs.is_empty() {
            result.push(nouns[0].to_string()); // Subject
            result.push(verbs[0].to_string()); // Verb
            // Add remaining nouns as objects
            for noun in nouns.iter().skip(1) {
                result.push(noun.to_string());
            }
            // Add adjectives at end
            for adj in &adjs {
                result.push(adj.to_string());
            }
        }
        // Default: keep original order but move verbs after first noun
        else {
            // Just return in original order for now
            result = words.iter().map(|s| s.to_string()).collect();
        }

        // Add others at the end
        for other in &others {
            if !result.contains(&other.to_string()) {
                result.push(other.to_string());
            }
        }

        result
    }

    /// Get the familiarity level for a word (0.0 = unknown, 1.0+ = familiar)
    #[allow(dead_code)]
    pub fn get_familiarity(&self, word: &str) -> f32 {
        let word_lower = word.to_lowercase();
        self.word_frequencies
            .get(&word_lower)
            .map(|f| f.familiarity_boost)
            .unwrap_or(0.0)
    }

    /// Get all words with high familiarity (ARIA knows these well)
    pub fn get_familiar_words(&self, min_familiarity: f32) -> Vec<(&String, &WordFrequency)> {
        self.word_frequencies
            .iter()
            .filter(|(_, freq)| freq.familiarity_boost >= min_familiarity)
            .collect()
    }

    /// Get the learned vector for a word (if known)
    #[allow(dead_code)]
    pub fn get_word_vector(&self, word: &str) -> Option<[f32; 8]> {
        let word_lower = word.to_lowercase();
        self.word_frequencies
            .get(&word_lower)
            .map(|f| f.learned_vector)
    }

    /// Find the word whose learned vector is most similar to the given vector
    /// Returns (word, similarity) if a good match is found (similarity > threshold)
    pub fn find_matching_word(&self, vector: &[f32; 8], min_familiarity: f32) -> Option<(String, f32)> {
        let mut best_match: Option<(String, f32)> = None;

        for (word, freq) in &self.word_frequencies {
            // Only consider words we've heard enough times
            if freq.familiarity_boost < min_familiarity {
                continue;
            }

            // Calculate cosine similarity
            let similarity = Self::vector_similarity_8(vector, &freq.learned_vector);

            if similarity > 0.3 {
                match &best_match {
                    Some((_, best_sim)) if similarity > *best_sim => {
                        best_match = Some((word.clone(), similarity));
                    }
                    None => {
                        best_match = Some((word.clone(), similarity));
                    }
                    _ => {}
                }
            }
        }

        best_match
    }

    fn vector_similarity_8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < 0.001 || mag_b < 0.001 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    pub fn load_or_create(path: &Path) -> Self {
        if path.exists() {
            match fs::read(path) {
                Ok(data) => {
                    match bincode::deserialize(&data) {
                        Ok(memory) => {
                            let mem: LongTermMemory = memory;
                            tracing::info!(
                                "Memory loaded: {} memories, {} patterns, {} elite DNA",
                                mem.memories.len(),
                                mem.learned_patterns.len(),
                                mem.elite_dna.len()
                            );
                            return mem;
                        }
                        Err(e) => {
                            tracing::warn!("Memory corrupted, starting fresh: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Cannot read memory file: {}", e);
                }
            }
        }

        tracing::info!("Creating new memory");
        Self::new()
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let data = bincode::serialize(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    /// Remember an important moment
    #[allow(dead_code)]
    pub fn remember(&mut self, trigger: &str, state: [f32; 32], intensity: f32, outcome: Outcome) {
        // Only remember significant moments
        if intensity > 0.3 {
            self.memories.push(Memory {
                timestamp: self.stats.total_ticks,
                trigger: trigger.to_string(),
                internal_state: state,
                emotional_intensity: intensity,
                outcome,
            });

            self.stats.total_memories += 1;

            // Keep only the most intense memories
            if self.memories.len() > 10_000 {
                self.memories.sort_by(|a, b| {
                    b.emotional_intensity.partial_cmp(&a.emotional_intensity).unwrap()
                });
                self.memories.truncate(5_000);
            }
        }
    }

    /// Learn a pattern
    pub fn learn_pattern(&mut self, sequence: Vec<[f32; 8]>, response: [f32; 8], valence: f32) {
        let current_tick = self.stats.total_ticks;

        // Check if pattern exists
        for pattern in &mut self.learned_patterns {
            if Self::sequences_similar(&pattern.sequence, &sequence) {
                pattern.frequency += 1;
                pattern.last_seen = current_tick;

                // Update response with moving average
                for (i, r) in response.iter().enumerate() {
                    pattern.typical_response[i] =
                        pattern.typical_response[i] * 0.9 + r * 0.1;
                }
                pattern.valence = pattern.valence * 0.9 + valence * 0.1;
                return;
            }
        }

        // New pattern
        self.learned_patterns.push(Pattern {
            sequence,
            frequency: 1,
            typical_response: response,
            valence,
            first_seen: current_tick,
            last_seen: current_tick,
        });

        self.stats.total_patterns_learned += 1;

        // Prune old, infrequent patterns
        if self.learned_patterns.len() > 5_000 {
            self.learned_patterns.retain(|p| {
                p.frequency > 5 || (current_tick - p.last_seen) < 100_000
            });
        }
    }

    fn sequences_similar(a: &[[f32; 8]], b: &[[f32; 8]]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        let total_distance: f32 = a.iter().zip(b.iter())
            .map(|(va, vb)| {
                va.iter().zip(vb.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .sum();

        total_distance / (a.len() as f32) < 0.5
    }

    /// Preserve elite DNA
    pub fn preserve_elite(&mut self, dna: DNA, fitness: f32, generation: u64, spec: &str) {
        self.elite_dna.push(EliteDNA {
            dna,
            fitness_score: fitness,
            generation,
            specialization: spec.to_string(),
            preserved_at: self.stats.total_ticks,
        });

        // Keep top 100 by fitness
        self.elite_dna.sort_by(|a, b| {
            b.fitness_score.partial_cmp(&a.fitness_score).unwrap()
        });
        self.elite_dna.truncate(100);

        // Track longest lineage
        if generation > self.stats.longest_lineage {
            self.stats.longest_lineage = generation;
        }
    }

    /// Create an association between stimulus and response
    #[allow(dead_code)]
    pub fn associate(&mut self, stimulus: [f32; 16], response: [f32; 16], strength: f32) {
        let current_tick = self.stats.total_ticks;

        // Check if association exists
        for assoc in &mut self.associations {
            let stim_sim = Self::vector_similarity_16(&assoc.stimulus, &stimulus);
            if stim_sim > 0.9 {
                // Reinforce existing association
                assoc.strength = (assoc.strength + strength) / 2.0;
                assoc.last_reinforced = current_tick;
                assoc.times_reinforced += 1;

                // Update response with moving average
                for (i, r) in response.iter().enumerate() {
                    assoc.response[i] = assoc.response[i] * 0.8 + r * 0.2;
                }
                return;
            }
        }

        // New association
        self.associations.push(Association {
            stimulus,
            response,
            strength,
            last_reinforced: current_tick,
            times_reinforced: 1,
        });

        // Prune weak associations
        if self.associations.len() > 1_000 {
            self.associations.retain(|a| a.strength > 0.1 || a.times_reinforced > 10);
        }
    }

    fn vector_similarity_16(a: &[f32; 16], b: &[f32; 16]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < 0.001 || mag_b < 0.001 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    /// Learn a word meaning
    #[allow(dead_code)]
    pub fn learn_word(&mut self, word: &str, vector: [f32; 8], example: &str) {
        if let Some(meaning) = self.vocabulary.get_mut(word) {
            // Update existing meaning
            meaning.frequency += 1;
            meaning.confidence = (meaning.confidence + 0.1).min(1.0);

            // Blend vectors
            for (i, v) in vector.iter().enumerate() {
                meaning.vector[i] = meaning.vector[i] * 0.9 + v * 0.1;
            }

            // Add example
            if meaning.examples.len() < 10 {
                meaning.examples.push(example.to_string());
            }
        } else {
            // New word
            self.vocabulary.insert(word.to_string(), WordMeaning {
                vector,
                confidence: 0.1,
                examples: vec![example.to_string()],
                frequency: 1,
            });
        }
    }

    /// Find response for a similar stimulus
    #[allow(dead_code)]
    pub fn recall(&self, stimulus: &[f32; 16]) -> Option<([f32; 16], f32)> {
        let mut best_match: Option<(&Association, f32)> = None;

        for assoc in &self.associations {
            let sim = Self::vector_similarity_16(&assoc.stimulus, stimulus);
            if sim > 0.5 {
                match best_match {
                    Some((_, best_sim)) if sim > best_sim => {
                        best_match = Some((assoc, sim));
                    }
                    None => {
                        best_match = Some((assoc, sim));
                    }
                    _ => {}
                }
            }
        }

        best_match.map(|(assoc, sim)| (assoc.response, sim * assoc.strength))
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_memory_creation() {
        let memory = LongTermMemory::new();
        assert_eq!(memory.version, 1);
        assert!(memory.memories.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.memory");

        let mut memory = LongTermMemory::new();
        memory.remember("test", [0.0; 32], 0.5, Outcome::Positive(1.0));

        memory.save(&path).unwrap();

        let loaded = LongTermMemory::load_or_create(&path);
        assert_eq!(loaded.memories.len(), 1);
    }

    #[test]
    fn test_pattern_learning() {
        let mut memory = LongTermMemory::new();

        let seq = vec![[0.5; 8]];
        memory.learn_pattern(seq.clone(), [0.5; 8], 0.5);
        memory.learn_pattern(seq.clone(), [0.5; 8], 0.6);

        assert_eq!(memory.learned_patterns.len(), 1);
        assert_eq!(memory.learned_patterns[0].frequency, 2);
    }
}
