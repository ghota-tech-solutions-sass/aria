//! Long-term memory for ARIA
//!
//! This persists between sessions, allowing ARIA to remember
//! and learn over time.
//!
//! ## Module Structure (refactored)
//!
//! - `types` - WordCategory, SocialContext, UsagePattern
//! - `vocabulary` - WordFrequency, WordAssociation, SemanticCluster
//! - `episodic` - Episode, EpisodeEmotion, EpisodeCategory
//! - `visual` - VisualMemory, VisualWordLink
//! - `exploration` - ExplorationResult
//! - `hierarchy` - WorkingMemory, ShortTermMemory (Gemini suggestion)

pub mod types;
pub mod vocabulary;
pub mod episodic;
pub mod visual;
pub mod exploration;
pub mod hierarchy;

// Re-export all types for convenience
pub use types::{WordCategory, SocialContext, UsagePattern};
pub use vocabulary::{WordFrequency, WordAssociation, SemanticCluster, WordMeaning};
pub use episodic::{Episode, EpisodeEmotion, EpisodeCategory, CompressedEpisode};
pub use visual::{VisualMemory, VisualWordLink};
pub use exploration::ExplorationResult;
pub use hierarchy::{WorkingMemory, ShortTermMemory, WorkingContent};

// Re-export meta_learning types for external use
pub use crate::meta_learning::{
    MetaLearner, InternalReward, SelfModifier, ModifiableParam, CurrentParams,
};

use aria_core::DNA;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use std::collections::{HashMap, HashSet};
use rand::Rng;

// ============================================================================
// LEGACY TYPES (for persistence)
// ============================================================================

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

// Default values for adaptive params
fn default_emission_threshold() -> f32 { 0.15 }
fn default_response_probability() -> f32 { 0.8 }
fn default_learning_rate() -> f32 { 0.3 }
fn default_spontaneity() -> f32 { 0.05 }

// ============================================================================
// LONG-TERM MEMORY
// ============================================================================

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

    /// Semantic clusters - groups of related words (rebuilt periodically from associations)
    #[serde(default)]
    pub semantic_clusters: Vec<SemanticCluster>,

    // === EPISODIC MEMORY ===

    /// Episodes - specific moments ARIA remembers
    #[serde(default)]
    pub episodes: Vec<Episode>,

    /// Compressed episodes - older memories in compact form (Gemini suggestion)
    #[serde(default)]
    pub compressed_episodes: Vec<CompressedEpisode>,

    /// Next episode ID
    #[serde(default)]
    pub next_episode_id: u64,

    /// First times tracker - to detect "first of kind" episodes
    #[serde(default)]
    pub first_times: HashMap<String, u64>, // kind -> episode_id

    // === ADAPTIVE PARAMETERS (self-modification) ===

    /// Emission threshold (0.05-0.5)
    #[serde(default = "default_emission_threshold")]
    pub adaptive_emission_threshold: f32,

    /// Response probability (0.3-1.0)
    #[serde(default = "default_response_probability")]
    pub adaptive_response_probability: f32,

    /// Learning rate (0.1-0.8)
    #[serde(default = "default_learning_rate")]
    pub adaptive_learning_rate: f32,

    /// Spontaneity (0.01-0.3)
    #[serde(default = "default_spontaneity")]
    pub adaptive_spontaneity: f32,

    /// Positive feedback count
    #[serde(default)]
    pub adaptive_feedback_positive: u64,

    /// Negative feedback count
    #[serde(default)]
    pub adaptive_feedback_negative: u64,

    // === EXPLORATION MEMORY (curiosity-driven learning) ===

    /// Combinations ARIA has tried and their outcomes
    #[serde(default)]
    pub exploration_history: HashMap<String, ExplorationResult>,

    // === META-LEARNING (Session 14) ===

    /// The meta-learner - learns how to learn
    #[serde(default)]
    pub meta_learner: MetaLearner,

    // === VISUAL MEMORY (Session 15) ===

    /// Visual memories - images ARIA has seen
    #[serde(default)]
    pub visual_memories: Vec<VisualMemory>,

    /// Visual-word associations - connects what ARIA sees to words she knows
    #[serde(default)]
    pub visual_word_links: HashMap<String, VisualWordLink>,

    // === SELF-MODIFICATION (Session 16) ===

    /// Self-modifier - ARIA changes herself
    #[serde(default)]
    pub self_modifier: SelfModifier,

    // === MEMORY HIERARCHY (Gemini suggestion) ===

    /// Short-term memory (not persisted, but included for completeness)
    #[serde(skip)]
    pub short_term: ShortTermMemory,

    /// Working memory (not persisted)
    #[serde(skip)]
    pub working: WorkingMemory,
}

#[allow(dead_code)]
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
            semantic_clusters: Vec::new(),
            episodes: Vec::new(),
            compressed_episodes: Vec::new(),
            next_episode_id: 0,
            first_times: HashMap::new(),
            // Adaptive params
            adaptive_emission_threshold: default_emission_threshold(),
            adaptive_response_probability: default_response_probability(),
            adaptive_learning_rate: default_learning_rate(),
            adaptive_spontaneity: default_spontaneity(),
            adaptive_feedback_positive: 0,
            adaptive_feedback_negative: 0,
            // Exploration memory
            exploration_history: HashMap::new(),
            // Meta-learning
            meta_learner: MetaLearner::new(),
            // Visual memory
            visual_memories: Vec::new(),
            visual_word_links: HashMap::new(),
            // Self-modification
            self_modifier: SelfModifier::new(),
            // Memory hierarchy (not persisted)
            short_term: ShortTermMemory::new(50),
            working: WorkingMemory::new(),
        }
    }

    // =========================================================================
    // EXPLORATION METHODS
    // =========================================================================

    /// Log an exploration attempt (ARIA tried a word combination)
    pub fn log_exploration(&mut self, combination: &str, tick: u64, intensity: f32) {
        let key = combination.to_lowercase();
        if let Some(existing) = self.exploration_history.get_mut(&key) {
            existing.attempts += 1;
            existing.avg_intensity = (existing.avg_intensity * (existing.attempts - 1) as f32 + intensity) / existing.attempts as f32;
            existing.last_attempt = tick;
        } else {
            self.exploration_history.insert(key.clone(), ExplorationResult::new(tick, intensity));
            tracing::info!("🔍 NEW EXPLORATION: '{}'", combination);
        }
    }

    /// Record feedback for recent exploration
    pub fn feedback_exploration(&mut self, combination: &str, positive: bool) {
        let key = combination.to_lowercase();
        if let Some(result) = self.exploration_history.get_mut(&key) {
            if positive {
                result.positive_feedback += 1;
                tracing::info!("✅ EXPLORATION SUCCESS: '{}' ({}/{})",
                    combination, result.positive_feedback, result.attempts);
            } else {
                result.negative_feedback += 1;
                tracing::info!("❌ EXPLORATION FAILURE: '{}' ({}/{})",
                    combination, result.negative_feedback, result.attempts);
            }
        }
    }

    /// Get a novel combination to try (curiosity-driven exploration)
    pub fn get_novel_combination(&self, words: &[String], current_tick: u64) -> Option<(String, String)> {
        if words.len() < 2 {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut best_pair: Option<(String, String, f32)> = None;

        for _ in 0..20 {
            let w1 = &words[rng.gen_range(0..words.len())];
            let w2 = &words[rng.gen_range(0..words.len())];
            if w1 == w2 {
                continue;
            }

            let key = format!("{}+{}", w1, w2);
            let score = if let Some(result) = self.exploration_history.get(&key) {
                let recency = (current_tick - result.last_attempt) as f32 / 10000.0;
                result.exploration_score() + recency.min(1.0) * 0.2
            } else {
                1.5 // Never tried = high novelty
            };

            if best_pair.is_none() || score > best_pair.as_ref().unwrap().2 {
                best_pair = Some((w1.clone(), w2.clone(), score));
            }
        }

        best_pair.map(|(w1, w2, _)| (w1, w2))
    }

    /// Get exploration statistics
    pub fn exploration_stats(&self) -> (usize, usize, usize) {
        let total = self.exploration_history.len();
        let successful = self.exploration_history.values()
            .filter(|r| r.positive_feedback > r.negative_feedback)
            .count();
        let failed = self.exploration_history.values()
            .filter(|r| r.negative_feedback > r.positive_feedback)
            .count();
        (total, successful, failed)
    }

    // =========================================================================
    // SOCIAL CONTEXT
    // =========================================================================

    /// Detect the social context of input text
    pub fn detect_social_context(text: &str) -> (SocialContext, f32) {
        let lower = text.to_lowercase();

        let greetings = ["bonjour", "salut", "coucou", "hello", "hi", "hey",
            "bonsoir", "bonne nuit", "good morning", "good evening"];
        for g in greetings {
            if lower.contains(g) {
                return (SocialContext::Greeting, 0.9);
            }
        }

        let farewells = ["au revoir", "bye", "goodbye", "à bientôt", "à plus",
            "bonne nuit", "good night", "see you", "ciao"];
        for f in farewells {
            if lower.contains(f) {
                return (SocialContext::Farewell, 0.9);
            }
        }

        let thanks = ["merci", "thank", "thanks", "grateful", "appreciate"];
        for t in thanks {
            if lower.contains(t) {
                return (SocialContext::Thanks, 0.9);
            }
        }

        let affection = ["je t'aime", "i love", "bisou", "câlin", "calin", "hug",
            "kiss", "love you", "adore", "♥", "❤", "<3"];
        for a in affection {
            if lower.contains(a) {
                return (SocialContext::Affection, 0.9);
            }
        }

        let requests = ["s'il te plaît", "stp", "please", "peux-tu", "peux tu",
            "can you", "could you", "would you", "voudrais"];
        for r in requests {
            if lower.contains(r) {
                return (SocialContext::Request, 0.8);
            }
        }

        let agreements = ["oui", "yes", "d'accord", "ok", "okay", "bien sûr",
            "exactement", "absolument", "certainement"];
        for a in agreements {
            if lower.starts_with(a) || lower == a {
                return (SocialContext::Agreement, 0.7);
            }
        }

        let disagreements = ["non", "no", "pas d'accord", "jamais", "never"];
        for d in disagreements {
            if lower.starts_with(d) || lower == d {
                return (SocialContext::Disagreement, 0.7);
            }
        }

        (SocialContext::General, 0.5)
    }

    /// Get an appropriate response word for a social context
    pub fn get_response_for_context(&self, context: SocialContext) -> Option<String> {
        let mut candidates: Vec<(&String, f32)> = Vec::new();

        for (word, freq) in &self.word_frequencies {
            if let Some((_ctx, count)) = freq.usage_pattern.contexts.iter()
                .find(|(c, _)| *c == context)
            {
                let score = (*count as f32) * freq.familiarity_boost.max(0.1);
                if score > 0.0 {
                    candidates.push((word, score));
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        let total_score: f32 = candidates.iter().map(|(_, s)| s).sum();
        let mut rng = rand::thread_rng();
        let mut roll: f32 = rng.gen::<f32>() * total_score;

        for (word, score) in &candidates {
            roll -= score;
            if roll <= 0.0 {
                return Some((*word).clone());
            }
        }

        candidates.first().map(|(w, _)| (*w).clone())
    }

    /// Learn that a word was used in a specific social context
    pub fn learn_usage_pattern(&mut self, word: &str, context: SocialContext, is_start: bool, is_end: bool) {
        let word_lower = word.to_lowercase();

        if let Some(freq) = self.word_frequencies.get_mut(&word_lower) {
            if freq.familiarity_boost > 0.5 && freq.category == WordCategory::Noun {
                return;
            }

            freq.usage_pattern.record_context(context);

            let alpha = 0.1;
            if is_start {
                freq.usage_pattern.start_of_conversation =
                    freq.usage_pattern.start_of_conversation * (1.0 - alpha) + alpha;
            }
            if is_end {
                freq.usage_pattern.end_of_conversation =
                    freq.usage_pattern.end_of_conversation * (1.0 - alpha) + alpha;
            }
        }
    }

    // =========================================================================
    // WORD ASSOCIATIONS
    // =========================================================================

    fn association_key(word1: &str, word2: &str) -> String {
        let w1 = word1.to_lowercase();
        let w2 = word2.to_lowercase();
        if w1 < w2 {
            format!("{}:{}", w1, w2)
        } else {
            format!("{}:{}", w2, w1)
        }
    }

    pub fn learn_association(&mut self, word1: &str, word2: &str, emotional_valence: f32) {
        let current_tick = self.stats.total_ticks;
        let key = Self::association_key(word1, word2);

        if let Some(assoc) = self.word_associations.get_mut(&key) {
            assoc.co_occurrences += 1;
            assoc.last_seen = current_tick;
            assoc.strength = (assoc.co_occurrences as f32 / 5.0).min(1.0);
            assoc.emotional_valence = assoc.emotional_valence * 0.8 + emotional_valence * 0.2;
        } else {
            self.word_associations.insert(key, WordAssociation {
                co_occurrences: 1,
                strength: 0.2,
                last_seen: current_tick,
                emotional_valence,
            });
        }

        if self.word_associations.len() > 10_000 {
            let threshold_tick = current_tick.saturating_sub(100_000);
            self.word_associations.retain(|_, a| {
                a.strength > 0.3 || a.last_seen > threshold_tick
            });
        }
    }

    pub fn get_associations(&self, word: &str) -> Vec<(String, f32)> {
        let word_lower = word.to_lowercase();
        let mut associations: Vec<(String, f32)> = Vec::new();

        for (key, assoc) in &self.word_associations {
            if assoc.strength < 0.4 {
                continue;
            }

            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() != 2 {
                continue;
            }

            if parts[0] == word_lower {
                associations.push((parts[1].to_string(), assoc.strength));
            } else if parts[1] == word_lower {
                associations.push((parts[0].to_string(), assoc.strength));
            }
        }

        associations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        associations
    }

    #[allow(dead_code)]
    pub fn get_strongest_association(&self, word: &str) -> Option<(String, f32)> {
        self.get_associations(word).into_iter().next()
    }

    pub fn get_top_associations(&self, word: &str, n: usize) -> Vec<(String, f32)> {
        self.get_associations(word).into_iter().take(n).collect()
    }

    // =========================================================================
    // WORD LEARNING
    // =========================================================================

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

        if word_lower.len() < 3 && !["moi", "toi", "oui", "non"].contains(&word_lower.as_str()) {
            return 0.0;
        }

        let category_hint = Self::detect_category_from_context(&word_lower, preceding_word, following_word);

        // Add to working memory (Gemini suggestion)
        self.working.push(
            WorkingContent::Word(word_lower.clone()),
            emotional_valence,
            current_tick,
        );

        if let Some(freq) = self.word_frequencies.get_mut(&word_lower) {
            freq.count += 1;
            freq.last_heard = current_tick;

            for (i, v) in context_vector.iter().enumerate() {
                freq.learned_vector[i] = freq.learned_vector[i] * 0.9 + v * 0.1;
            }

            freq.emotional_valence = freq.emotional_valence * 0.9 + emotional_valence * 0.1;

            if let Some(cat) = category_hint {
                let idx = match cat {
                    WordCategory::Noun => 0,
                    WordCategory::Verb => 1,
                    WordCategory::Adjective => 2,
                    WordCategory::Unknown => return freq.familiarity_boost,
                };
                freq.category_scores[idx] += 0.3;
                let total: f32 = freq.category_scores.iter().sum();
                if total > 0.0 {
                    for score in &mut freq.category_scores {
                        *score /= total;
                    }
                }
                freq.category = Self::dominant_category(&freq.category_scores);
            }

            freq.familiarity_boost = (freq.count as f32 / 10.0).min(2.0);
            freq.familiarity_boost
        } else {
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

    #[allow(dead_code)]
    pub fn hear_word(&mut self, word: &str, context_vector: [f32; 8], emotional_valence: f32) -> f32 {
        self.hear_word_with_context(word, context_vector, emotional_valence, None, None)
    }

    fn detect_category_from_context(
        word: &str,
        preceding: Option<&str>,
        _following: Option<&str>,
    ) -> Option<WordCategory> {
        let word_lower = word.to_lowercase();
        let preceding_lower = preceding.map(|w| w.to_lowercase());

        let articles = ["le", "la", "les", "un", "une", "des", "mon", "ma", "mes",
            "ton", "ta", "tes", "son", "sa", "ses", "ce", "cette", "ces",
            "the", "a", "an", "my", "your", "his", "her"];
        if let Some(ref p) = preceding_lower {
            if articles.contains(&p.as_str()) {
                return Some(WordCategory::Noun);
            }
        }

        let pronouns = ["je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
            "j'", "i", "you", "he", "she", "we", "they"];
        if let Some(ref p) = preceding_lower {
            if pronouns.contains(&p.as_str()) {
                return Some(WordCategory::Verb);
            }
        }

        let adj_suffixes = ["eux", "euse", "if", "ive", "ant", "ent", "ique",
            "able", "ible", "al", "el", "ful", "less", "ous"];
        for suffix in adj_suffixes {
            if word_lower.ends_with(suffix) && word_lower.len() > suffix.len() + 2 {
                return Some(WordCategory::Adjective);
            }
        }

        let verb_suffixes = ["er", "ir", "re", "ait", "ais", "ons", "ez", "ent"];
        for suffix in verb_suffixes {
            if word_lower.ends_with(suffix) && word_lower.len() > suffix.len() + 1 {
                return Some(WordCategory::Verb);
            }
        }

        if let Some(ref p) = preceding_lower {
            if ["très", "plus", "moins", "si", "trop", "assez", "very", "so", "too"].contains(&p.as_str()) {
                return Some(WordCategory::Adjective);
            }
        }

        let known_nouns = ["chat", "moka", "aria", "papa", "mama", "ami", "maison",
            "chien", "oiseau", "soleil", "lune", "eau", "pain"];
        if known_nouns.contains(&word_lower.as_str()) {
            return Some(WordCategory::Noun);
        }

        let known_verbs = ["aime", "veux", "peux", "suis", "est", "fait", "mange",
            "dort", "joue", "love", "want", "need", "like"];
        if known_verbs.contains(&word_lower.as_str()) {
            return Some(WordCategory::Verb);
        }

        let known_adjs = ["beau", "belle", "grand", "petit", "joli", "mignon",
            "gentil", "méchant", "bon", "mauvais", "beautiful", "good", "bad"];
        if known_adjs.contains(&word_lower.as_str()) {
            return Some(WordCategory::Adjective);
        }

        None
    }

    fn dominant_category(scores: &[f32; 3]) -> WordCategory {
        let threshold = 0.5;
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

    pub fn get_word_category(&self, word: &str) -> WordCategory {
        self.word_frequencies
            .get(&word.to_lowercase())
            .map(|f| f.category)
            .unwrap_or(WordCategory::Unknown)
    }

    pub fn order_phrase(&self, words: &[&str]) -> Vec<String> {
        if words.is_empty() {
            return vec![];
        }
        if words.len() == 1 {
            return vec![words[0].to_string()];
        }

        let categorized: Vec<(&str, WordCategory)> = words.iter()
            .map(|w| (*w, self.get_word_category(w)))
            .collect();

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

        let mut result = vec![];

        if !adjs.is_empty() && !nouns.is_empty() && verbs.is_empty() {
            let adj = adjs[0];
            let noun = nouns[0];
            let short_adjs = ["beau", "bon", "grand", "petit", "gros", "jeune", "vieux",
                "bel", "belle", "joli", "jolie", "mauvais", "nouveau"];
            if short_adjs.contains(&adj.to_lowercase().as_str()) || adj.len() <= 5 {
                result.push(adj.to_string());
                result.push(noun.to_string());
            } else {
                result.push(noun.to_string());
                result.push(adj.to_string());
            }
            for noun in nouns.iter().skip(1) {
                result.push(noun.to_string());
            }
            for adj in adjs.iter().skip(1) {
                result.push(adj.to_string());
            }
        } else if !nouns.is_empty() && !verbs.is_empty() {
            result.push(nouns[0].to_string());
            result.push(verbs[0].to_string());
            for noun in nouns.iter().skip(1) {
                result.push(noun.to_string());
            }
            for adj in &adjs {
                result.push(adj.to_string());
            }
        } else {
            result = words.iter().map(|s| s.to_string()).collect();
        }

        for other in &others {
            if !result.contains(&other.to_string()) {
                result.push(other.to_string());
            }
        }

        result
    }

    #[allow(dead_code)]
    pub fn get_familiarity(&self, word: &str) -> f32 {
        self.word_frequencies
            .get(&word.to_lowercase())
            .map(|f| f.familiarity_boost)
            .unwrap_or(0.0)
    }

    pub fn get_familiar_words(&self, min_familiarity: f32) -> Vec<(&String, &WordFrequency)> {
        self.word_frequencies
            .iter()
            .filter(|(_, freq)| freq.familiarity_boost >= min_familiarity)
            .collect()
    }

    #[allow(dead_code)]
    pub fn get_word_vector(&self, word: &str) -> Option<[f32; 8]> {
        self.word_frequencies
            .get(&word.to_lowercase())
            .map(|f| f.learned_vector)
    }

    pub fn find_matching_word(&self, vector: &[f32; 8], min_familiarity: f32) -> Option<(String, f32)> {
        let mut best_match: Option<(String, f32)> = None;

        for (word, freq) in &self.word_frequencies {
            if freq.familiarity_boost < min_familiarity {
                continue;
            }

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

    // =========================================================================
    // PERSISTENCE
    // =========================================================================

    pub fn load_or_create(path: &Path) -> Self {
        if path.exists() {
            match fs::read(path) {
                Ok(data) => {
                    match bincode::deserialize(&data) {
                        Ok(memory) => {
                            let mut mem: LongTermMemory = memory;
                            tracing::info!(
                                "Memory loaded: {} memories, {} patterns, {} elite DNA",
                                mem.memories.len(),
                                mem.learned_patterns.len(),
                                mem.elite_dna.len()
                            );

                            mem.migrate_clean_nouns_from_social_contexts();

                            // Initialize non-persisted fields
                            mem.short_term = ShortTermMemory::new(50);
                            mem.working = WorkingMemory::new();

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

    fn migrate_clean_nouns_from_social_contexts(&mut self) {
        let social_contexts = [
            SocialContext::Greeting,
            SocialContext::Farewell,
            SocialContext::Thanks,
            SocialContext::Affection,
        ];

        let mut cleaned = 0;
        for (word, freq) in &mut self.word_frequencies {
            if freq.category == WordCategory::Noun && freq.familiarity_boost > 0.3 {
                let before_len = freq.usage_pattern.contexts.len();
                freq.usage_pattern.contexts.retain(|(ctx, _)| !social_contexts.contains(ctx));
                let after_len = freq.usage_pattern.contexts.len();

                if before_len != after_len {
                    tracing::info!("Migration: Cleaned social contexts from noun '{}'", word);
                    cleaned += 1;
                }
            }
        }

        if cleaned > 0 {
            tracing::info!("Migration complete: Cleaned {} nouns from social contexts", cleaned);
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let data = bincode::serialize(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn remember(&mut self, trigger: &str, state: [f32; 32], intensity: f32, outcome: Outcome) {
        if intensity > 0.3 {
            self.memories.push(Memory {
                timestamp: self.stats.total_ticks,
                trigger: trigger.to_string(),
                internal_state: state,
                emotional_intensity: intensity,
                outcome,
            });

            self.stats.total_memories += 1;

            if self.memories.len() > 10_000 {
                self.memories.sort_by(|a, b| {
                    b.emotional_intensity.partial_cmp(&a.emotional_intensity).unwrap()
                });
                self.memories.truncate(5_000);
            }
        }
    }

    pub fn learn_pattern(&mut self, sequence: Vec<[f32; 8]>, response: [f32; 8], valence: f32) {
        let current_tick = self.stats.total_ticks;

        for pattern in &mut self.learned_patterns {
            if Self::sequences_similar(&pattern.sequence, &sequence) {
                pattern.frequency += 1;
                pattern.last_seen = current_tick;

                for (i, r) in response.iter().enumerate() {
                    pattern.typical_response[i] =
                        pattern.typical_response[i] * 0.9 + r * 0.1;
                }
                pattern.valence = pattern.valence * 0.9 + valence * 0.1;
                return;
            }
        }

        self.learned_patterns.push(Pattern {
            sequence,
            frequency: 1,
            typical_response: response,
            valence,
            first_seen: current_tick,
            last_seen: current_tick,
        });

        self.stats.total_patterns_learned += 1;

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

    pub fn preserve_elite(&mut self, dna: DNA, fitness: f32, generation: u64, spec: &str) {
        self.elite_dna.push(EliteDNA {
            dna,
            fitness_score: fitness,
            generation,
            specialization: spec.to_string(),
            preserved_at: self.stats.total_ticks,
        });

        self.elite_dna.sort_by(|a, b| {
            b.fitness_score.partial_cmp(&a.fitness_score).unwrap()
        });
        self.elite_dna.truncate(100);

        if generation > self.stats.longest_lineage {
            self.stats.longest_lineage = generation;
        }
    }

    #[allow(dead_code)]
    pub fn associate(&mut self, stimulus: [f32; 16], response: [f32; 16], strength: f32) {
        let current_tick = self.stats.total_ticks;

        for assoc in &mut self.associations {
            let stim_sim = Self::vector_similarity_16(&assoc.stimulus, &stimulus);
            if stim_sim > 0.9 {
                assoc.strength = (assoc.strength + strength) / 2.0;
                assoc.last_reinforced = current_tick;
                assoc.times_reinforced += 1;

                for (i, r) in response.iter().enumerate() {
                    assoc.response[i] = assoc.response[i] * 0.8 + r * 0.2;
                }
                return;
            }
        }

        self.associations.push(Association {
            stimulus,
            response,
            strength,
            last_reinforced: current_tick,
            times_reinforced: 1,
        });

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

    #[allow(dead_code)]
    pub fn learn_word(&mut self, word: &str, vector: [f32; 8], example: &str) {
        if let Some(meaning) = self.vocabulary.get_mut(word) {
            meaning.frequency += 1;
            meaning.confidence = (meaning.confidence + 0.1).min(1.0);

            for (i, v) in vector.iter().enumerate() {
                meaning.vector[i] = meaning.vector[i] * 0.9 + v * 0.1;
            }

            if meaning.examples.len() < 10 {
                meaning.examples.push(example.to_string());
            }
        } else {
            self.vocabulary.insert(word.to_string(), WordMeaning {
                vector,
                confidence: 0.1,
                examples: vec![example.to_string()],
                frequency: 1,
            });
        }
    }

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

    // =========================================================================
    // EPISODIC MEMORY
    // =========================================================================

    pub fn record_episode(
        &mut self,
        input: &str,
        response: Option<&str>,
        keywords: Vec<String>,
        emotion: EpisodeEmotion,
        importance: f32,
        category: EpisodeCategory,
        current_tick: u64,
    ) -> u64 {
        let id = self.next_episode_id;
        self.next_episode_id += 1;

        let mut episode = Episode::new(
            id,
            current_tick,
            input.to_string(),
            response.map(|s| s.to_string()),
            keywords.clone(),
            emotion,
            importance,
            category.clone(),
        );

        let first_of_kind = self.check_first_of_kind(&category, &keywords);
        if let Some(ref kind) = first_of_kind {
            episode.mark_first_of_kind(kind);
            self.first_times.insert(kind.clone(), id);
            tracing::info!("🌟 FIRST TIME: {} (episode #{})", kind, id);
        }

        tracing::info!(
            "📝 Episode #{}: {:?} - \"{}\" (importance: {:.2})",
            id, category, input, episode.importance
        );

        self.episodes.push(episode);

        if self.episodes.len() > 1000 {
            self.prune_episodes(current_tick);
        }

        id
    }

    fn check_first_of_kind(&self, category: &EpisodeCategory, keywords: &[String]) -> Option<String> {
        match category {
            EpisodeCategory::Praise => {
                if !self.first_times.contains_key("first_praise") {
                    return Some("first_praise".to_string());
                }
            }
            EpisodeCategory::Correction => {
                if !self.first_times.contains_key("first_correction") {
                    return Some("first_correction".to_string());
                }
            }
            EpisodeCategory::Social => {
                if !self.first_times.contains_key("first_greeting") {
                    return Some("first_greeting".to_string());
                }
            }
            EpisodeCategory::Emotional => {
                if keywords.iter().any(|k| k == "aime" || k == "love") {
                    if !self.first_times.contains_key("first_love") {
                        return Some("first_love".to_string());
                    }
                }
            }
            _ => {}
        }

        for keyword in keywords {
            let key = format!("first_mention_{}", keyword);
            if !self.first_times.contains_key(&key) && self.is_significant_word(keyword) {
                return Some(key);
            }
        }

        None
    }

    fn is_significant_word(&self, word: &str) -> bool {
        if let Some(freq) = self.word_frequencies.get(word) {
            return freq.category == WordCategory::Noun && freq.familiarity_boost > 0.5;
        }
        false
    }

    pub fn recall_episodes(&mut self, context_words: &[String], current_tick: u64, limit: usize) -> Vec<&Episode> {
        if self.episodes.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self.episodes.iter()
            .enumerate()
            .map(|(i, ep)| {
                let context_match = ep.matches_context(context_words);
                let memory_strength = ep.memory_strength(current_tick);
                let score = context_match * 0.6 + memory_strength * 0.4;
                (i, score)
            })
            .filter(|(_, score)| *score > 0.1)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        for (idx, _) in &scored {
            if let Some(ep) = self.episodes.get_mut(*idx) {
                ep.recall(current_tick);
            }
        }

        scored.iter()
            .filter_map(|(idx, _)| self.episodes.get(*idx))
            .collect()
    }

    pub fn get_significant_episodes(&self, current_tick: u64, limit: usize) -> Vec<&Episode> {
        let mut episodes: Vec<&Episode> = self.episodes.iter().collect();

        episodes.sort_by(|a, b| {
            let score_a = a.memory_strength(current_tick) * a.importance;
            let score_b = b.memory_strength(current_tick) * b.importance;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        episodes.truncate(limit);
        episodes
    }

    fn prune_episodes(&mut self, current_tick: u64) {
        // Compress old episodes before pruning (Gemini suggestion)
        let old_threshold = current_tick.saturating_sub(500_000);
        let to_compress: Vec<CompressedEpisode> = self.episodes.iter()
            .filter(|ep| ep.timestamp < old_threshold && ep.importance < 0.6)
            .map(|ep| CompressedEpisode::from_episode(ep))
            .collect();

        if !to_compress.is_empty() {
            tracing::info!("📦 Compressing {} old episodes", to_compress.len());
            self.compressed_episodes.extend(to_compress);
        }

        // Remove old, low-importance episodes
        self.episodes.retain(|ep| {
            ep.timestamp >= old_threshold || ep.importance >= 0.6
        });

        // If still too many, prune by strength
        if self.episodes.len() > 800 {
            let mut with_strength: Vec<(usize, f32)> = self.episodes.iter()
                .enumerate()
                .map(|(i, ep)| (i, ep.memory_strength(current_tick) * ep.importance))
                .collect();

            with_strength.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let to_remove: Vec<usize> = with_strength.iter()
                .take(self.episodes.len() / 5)
                .filter(|(_, strength)| *strength < 0.3)
                .map(|(idx, _)| *idx)
                .collect();

            for idx in to_remove.into_iter().rev() {
                let removed = self.episodes.remove(idx);
                tracing::debug!("Forgot episode #{}: \"{}\"", removed.id, removed.input);
            }
        }
    }

    pub fn get_episode(&self, id: u64) -> Option<&Episode> {
        self.episodes.iter().find(|ep| ep.id == id)
    }

    pub fn episode_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        for ep in &self.episodes {
            let key = format!("{:?}", ep.category);
            *stats.entry(key).or_insert(0) += 1;
        }
        stats.insert("total".to_string(), self.episodes.len());
        stats.insert("compressed".to_string(), self.compressed_episodes.len());
        stats
    }

    // =========================================================================
    // SEMANTIC CLUSTERING
    // =========================================================================

    pub fn rebuild_clusters(&mut self) {
        let min_strength = 0.3;
        let min_words = 2;
        let hub_threshold = 5;

        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        for (key, assoc) in &self.word_associations {
            if assoc.strength < min_strength {
                continue;
            }
            let parts: Vec<&str> = key.split(':').collect();
            if parts.len() != 2 {
                continue;
            }
            let (w1, w2) = (parts[0].to_string(), parts[1].to_string());

            adjacency.entry(w1.clone()).or_default().push(w2.clone());
            adjacency.entry(w2.clone()).or_default().push(w1.clone());
        }

        let hub_words: HashSet<String> = adjacency
            .iter()
            .filter(|(_, neighbors)| neighbors.len() > hub_threshold)
            .map(|(word, _)| word.clone())
            .collect();

        for hub in &hub_words {
            adjacency.remove(hub);
        }
        for neighbors in adjacency.values_mut() {
            neighbors.retain(|n| !hub_words.contains(n));
        }

        let mut visited: HashMap<String, bool> = HashMap::new();
        let mut clusters: Vec<SemanticCluster> = Vec::new();
        let mut cluster_id: u32 = 0;

        for word in adjacency.keys() {
            if visited.get(word).copied().unwrap_or(false) {
                continue;
            }

            let mut component: Vec<(String, f32)> = Vec::new();
            let mut queue: Vec<String> = vec![word.clone()];
            visited.insert(word.clone(), true);

            while let Some(current) = queue.pop() {
                let membership = self.word_frequencies
                    .get(&current)
                    .map(|f| f.familiarity_boost)
                    .unwrap_or(0.5);
                component.push((current.clone(), membership));

                if let Some(neighbors) = adjacency.get(&current) {
                    for neighbor in neighbors {
                        if !visited.get(neighbor).copied().unwrap_or(false) {
                            visited.insert(neighbor.clone(), true);
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }

            if component.len() >= min_words {
                let mut total_valence = 0.0;
                let mut category_counts = [0u32; 3];

                for (word, _) in &component {
                    if let Some(freq) = self.word_frequencies.get(word) {
                        total_valence += freq.emotional_valence;
                        match freq.category {
                            WordCategory::Noun => category_counts[0] += 1,
                            WordCategory::Verb => category_counts[1] += 1,
                            WordCategory::Adjective => category_counts[2] += 1,
                            WordCategory::Unknown => {}
                        }
                    }
                }

                let avg_valence = total_valence / component.len() as f32;
                let dominant_cat = if category_counts[0] >= category_counts[1] && category_counts[0] >= category_counts[2] {
                    WordCategory::Noun
                } else if category_counts[1] >= category_counts[2] {
                    WordCategory::Verb
                } else {
                    WordCategory::Adjective
                };

                clusters.push(SemanticCluster {
                    id: cluster_id,
                    label: None,
                    words: component,
                    emotional_valence: avg_valence,
                    dominant_category: dominant_cat,
                });
                cluster_id += 1;
            }
        }

        self.semantic_clusters = clusters;
        tracing::info!("Rebuilt {} semantic clusters", self.semantic_clusters.len());
    }

    pub fn get_cluster_for_word(&self, word: &str) -> Option<&SemanticCluster> {
        let word_lower = word.to_lowercase();
        self.semantic_clusters.iter().find(|c| {
            c.words.iter().any(|(w, _)| w.to_lowercase() == word_lower)
        })
    }

    pub fn get_cluster_words(&self, word: &str) -> Vec<String> {
        if let Some(cluster) = self.get_cluster_for_word(word) {
            let mut words: Vec<(String, f32)> = cluster.words.clone();
            words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            words.into_iter()
                .filter(|(w, _)| w.to_lowercase() != word.to_lowercase())
                .map(|(w, _)| w)
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_related_words_from_input(&self, input_words: &[String]) -> Vec<(String, f32)> {
        let mut related: HashMap<String, f32> = HashMap::new();

        for word in input_words {
            if let Some(cluster) = self.get_cluster_for_word(word) {
                for (cluster_word, strength) in &cluster.words {
                    if !input_words.iter().any(|w| w.to_lowercase() == cluster_word.to_lowercase()) {
                        let entry = related.entry(cluster_word.clone()).or_insert(0.0);
                        *entry = (*entry + strength).min(2.0);
                    }
                }
            }
        }

        let mut result: Vec<(String, f32)> = related.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    // =========================================================================
    // VISUAL MEMORY
    // =========================================================================

    pub fn see(&mut self, signature: [f32; 32], description: String, source: String, tick: u64, emotional_context: f32)
        -> (u64, bool, Option<String>)
    {
        let mut best_match: Option<(usize, f32)> = None;
        for (i, mem) in self.visual_memories.iter().enumerate() {
            let sim = mem.similarity(&signature);
            if sim > 0.85 {
                if best_match.map_or(true, |(_, s)| sim > s) {
                    best_match = Some((i, sim));
                }
            }
        }

        if let Some((idx, similarity)) = best_match {
            let mem = &mut self.visual_memories[idx];
            mem.times_seen += 1;
            mem.last_seen = tick;

            for i in 0..32 {
                mem.signature[i] = mem.signature[i] * 0.9 + signature[i] * 0.1;
            }

            let recognition = if mem.labels.is_empty() {
                format!("Je reconnais ça ! (vu {} fois)", mem.times_seen)
            } else {
                format!("Je reconnais: {} ! (vu {} fois)", mem.labels.join(", "), mem.times_seen)
            };

            tracing::info!("👁️ RECOGNITION: {} (similarity: {:.2})", recognition, similarity);
            (mem.id, false, Some(recognition))
        } else {
            let id = self.visual_memories.len() as u64;
            let mem = VisualMemory::new(id, signature, description.clone(), source, tick, emotional_context);
            self.visual_memories.push(mem);

            tracing::info!("👁️ NEW VISUAL MEMORY #{}: {}", id, description);
            (id, true, None)
        }
    }

    pub fn link_vision_to_word(&mut self, signature: &[f32; 32], word: &str, tick: u64) {
        let word_lower = word.to_lowercase();

        if let Some(link) = self.visual_word_links.get_mut(&word_lower) {
            let n = link.association_count as f32;
            for i in 0..32 {
                link.visual_prototype[i] = (link.visual_prototype[i] * n + signature[i]) / (n + 1.0);
            }
            link.association_count += 1;
            link.last_reinforced = tick;

            let mut sig_arr = [0.0f32; 32];
            sig_arr.copy_from_slice(signature);
            let sim = VisualMemory::new(0, sig_arr, String::new(), String::new(), 0, 0.0)
                .similarity(&link.visual_prototype);
            link.avg_similarity = (link.avg_similarity * n + sim) / (n + 1.0);
        } else {
            let mut prototype = [0.0f32; 32];
            prototype.copy_from_slice(signature);

            self.visual_word_links.insert(word_lower.clone(), VisualWordLink {
                word: word_lower.clone(),
                visual_prototype: prototype,
                association_count: 1,
                avg_similarity: 1.0,
                last_reinforced: tick,
            });
        }

        for mem in &mut self.visual_memories {
            if mem.similarity(signature) > 0.85 && !mem.labels.contains(&word_lower) {
                mem.labels.push(word_lower.clone());
            }
        }
    }

    pub fn visual_to_words(&self, signature: &[f32; 32]) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self.visual_word_links.iter()
            .map(|(word, link)| {
                let mut sig_arr = [0.0f32; 32];
                sig_arr.copy_from_slice(signature);
                let sim = VisualMemory::new(0, sig_arr, String::new(), String::new(), 0, 0.0)
                    .similarity(&link.visual_prototype);
                let confidence = (link.association_count as f32).ln().max(0.0) * 0.1 + 1.0;
                (word.clone(), sim * confidence)
            })
            .filter(|(_, score)| *score > 0.3)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(5);
        results
    }

    pub fn word_to_visual(&self, word: &str) -> Option<[f32; 32]> {
        self.visual_word_links.get(&word.to_lowercase())
            .filter(|link| link.association_count >= 2)
            .map(|link| link.visual_prototype)
    }

    pub fn visual_stats(&self) -> (usize, usize, Vec<(String, u64)>) {
        let total_memories = self.visual_memories.len();
        let total_links = self.visual_word_links.len();

        let mut top_words: Vec<(String, u64)> = self.visual_word_links.iter()
            .map(|(word, link)| (word.clone(), link.association_count))
            .collect();
        top_words.sort_by(|a, b| b.1.cmp(&a.1));
        top_words.truncate(10);

        (total_memories, total_links, top_words)
    }

    pub fn find_visual_by_label(&self, label: &str) -> Vec<&VisualMemory> {
        let label_lower = label.to_lowercase();
        self.visual_memories.iter()
            .filter(|mem| mem.labels.iter().any(|l| l.contains(&label_lower)))
            .collect()
    }

    // =========================================================================
    // MEMORY HIERARCHY (Gemini suggestion)
    // =========================================================================

    /// Update working memory decay (call each tick)
    pub fn tick_working_memory(&mut self, decay_rate: f32) {
        self.working.decay(decay_rate);
    }

    /// Consolidate short-term memory to long-term (call during "sleep")
    pub fn consolidate(&mut self, current_tick: u64) {
        // Prune old short-term memories
        self.short_term.prune(current_tick);

        // Get candidates for consolidation (collect to avoid borrow conflict)
        let candidates: Vec<_> = self.short_term.consolidation_candidates()
            .into_iter()
            .map(|item| (item.words.clone(), item.emotional_valence))
            .collect();

        for (words, valence) in candidates {
            // Learn associations between words in the item
            for i in 0..words.len() {
                for j in (i+1)..words.len() {
                    self.learn_association(&words[i], &words[j], valence);
                }
            }

            tracing::debug!("💭 Consolidated: {:?} (valence: {:.2})", words, valence);
        }
    }

    /// Record interaction in short-term memory
    pub fn record_short_term(&mut self, words: Vec<String>, emotional_valence: f32, tick: u64) {
        self.short_term.record(words, emotional_valence, tick);
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

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

    #[test]
    fn test_working_memory() {
        let mut wm = WorkingMemory::new();
        wm.push(WorkingContent::Word("test".to_string()), 0.5, 0);
        assert!(wm.contains_word("test"));
        assert_eq!(wm.items().len(), 1);

        wm.decay(0.5);
        assert!(wm.items()[0].activation < 1.0);
    }

    #[test]
    fn test_compressed_episode() {
        let episode = Episode::new(
            1, 100,
            "Hello world".to_string(),
            Some("Hi".to_string()),
            vec!["hello".to_string(), "world".to_string()],
            EpisodeEmotion { happiness: 0.5, arousal: 0.3, comfort: 0.7, curiosity: 0.2 },
            0.8,
            EpisodeCategory::Social,
        );

        let compressed = CompressedEpisode::from_episode(&episode);
        assert_eq!(compressed.id, episode.id);
        assert!(compressed.keywords.len() <= 3);
    }
}
