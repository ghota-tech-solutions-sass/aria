//! Substrate - The universe where cells live
//!
//! The substrate is not a grid. It's a topological space where
//! distances are semantic, not geometric.

use crate::cell::{Cell, CellAction, SignalFragment, Emotion};
use crate::signal::Signal;
use crate::memory::LongTermMemory;

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// A recently heard word with its context
#[derive(Clone, Debug)]
struct RecentWord {
    word: String,
    vector: [f32; 8],
    heard_at: u64,
}

/// A single exchange in the conversation
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ConversationExchange {
    /// What was said to ARIA
    input: String,
    /// What ARIA responded (if she responded)
    response: Option<String>,
    /// Words extracted from input
    input_words: Vec<String>,
    /// Emotional tone of the exchange
    emotional_tone: f32,
    /// When this happened
    tick: u64,
}

use crate::memory::SocialContext;

/// Conversation context - tracks the flow of discussion
/// ARIA can now follow a conversation thread!
#[derive(Clone, Debug, Default)]
struct ConversationContext {
    /// Recent exchanges (newest first)
    exchanges: Vec<ConversationExchange>,
    /// Current topic words (words that keep coming up)
    topic_words: Vec<(String, u32)>, // (word, mention_count)
    /// Is someone actively talking to ARIA?
    in_conversation: bool,
    /// Last exchange tick
    last_exchange_tick: u64,
    /// Current social context (greeting, farewell, etc.)
    current_social_context: SocialContext,
    /// Exchange count in current conversation
    exchange_count: u32,
}

impl ConversationContext {
    const MAX_EXCHANGES: usize = 5;
    const CONVERSATION_TIMEOUT: u64 = 3000; // ~30 seconds

    fn new() -> Self {
        Self::default()
    }

    /// Add a new input to the conversation
    fn add_input(&mut self, input: &str, words: Vec<String>, emotional_tone: f32, tick: u64, social_context: SocialContext) {
        // Check if this is a new conversation or continuation
        if tick.saturating_sub(self.last_exchange_tick) > Self::CONVERSATION_TIMEOUT {
            // New conversation! Clear old context
            self.exchanges.clear();
            self.topic_words.clear();
            self.exchange_count = 0;
            tracing::info!("NEW CONVERSATION started");
        }

        self.in_conversation = true;
        self.last_exchange_tick = tick;
        self.current_social_context = social_context;
        self.exchange_count += 1;

        // Create new exchange
        let exchange = ConversationExchange {
            input: input.to_string(),
            response: None,
            input_words: words.clone(),
            emotional_tone,
            tick,
        };

        // Add to front (newest first)
        self.exchanges.insert(0, exchange);

        // Keep only last N exchanges
        if self.exchanges.len() > Self::MAX_EXCHANGES {
            self.exchanges.pop();
        }

        // Update topic words
        for word in words {
            if let Some(pos) = self.topic_words.iter().position(|(w, _)| w == &word) {
                self.topic_words[pos].1 += 1;
            } else {
                self.topic_words.push((word, 1));
            }
        }

        // Keep only top 10 topic words, sorted by frequency
        self.topic_words.sort_by(|a, b| b.1.cmp(&a.1));
        self.topic_words.truncate(10);
    }

    /// Record ARIA's response to the current exchange
    fn add_response(&mut self, response: &str) {
        if let Some(exchange) = self.exchanges.first_mut() {
            exchange.response = Some(response.to_string());
        }
    }

    /// Get words that are currently "hot" in the conversation
    /// These should be boosted when ARIA responds
    fn get_context_words(&self) -> Vec<(String, f32)> {
        // Combine recent input words with topic words
        let mut context: Vec<(String, f32)> = Vec::new();

        // Words from the last exchange are most relevant
        if let Some(last) = self.exchanges.first() {
            for word in &last.input_words {
                context.push((word.clone(), 1.0)); // Full boost for just-heard words
            }
        }

        // Words from previous exchanges (decaying relevance)
        for (i, exchange) in self.exchanges.iter().skip(1).enumerate() {
            let decay = 0.5_f32.powi(i as i32 + 1); // 0.5, 0.25, 0.125...
            for word in &exchange.input_words {
                if !context.iter().any(|(w, _)| w == word) {
                    context.push((word.clone(), decay));
                }
            }
        }

        // Topic words get a bonus
        for (word, count) in &self.topic_words {
            let topic_boost = (*count as f32 * 0.2).min(0.8);
            if let Some(pos) = context.iter().position(|(w, _)| w == word) {
                context[pos].1 += topic_boost;
            } else {
                context.push((word.clone(), topic_boost));
            }
        }

        context
    }

    /// Check if a word was mentioned recently in conversation
    #[allow(dead_code)]
    fn was_recently_mentioned(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();
        self.exchanges.iter()
            .take(2) // Check last 2 exchanges
            .any(|e| e.input_words.iter().any(|w| w.to_lowercase() == word_lower))
    }

    /// Get the emotional tone of recent conversation
    #[allow(dead_code)]
    fn get_conversation_mood(&self) -> f32 {
        if self.exchanges.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.exchanges.iter().map(|e| e.emotional_tone).sum();
        sum / self.exchanges.len() as f32
    }

    /// Check if we're at the start of a conversation (first 1-2 exchanges)
    fn is_conversation_start(&self) -> bool {
        self.exchange_count <= 2
    }

    /// Get current social context
    fn get_social_context(&self) -> SocialContext {
        self.current_social_context
    }
}

/// Words that are too common to be meaningful - ARIA shouldn't repeat these
/// Like a baby learning to speak, she should focus on meaningful words
const STOP_WORDS: &[&str] = &[
    // French articles and pronouns
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "leur", "en", "y",
    // French verbs (common forms)
    "est", "suis", "es", "sont", "sommes", "êtes",
    "ai", "as", "a", "ont", "avons", "avez",
    "fait", "faire", "vais", "vas", "va", "vont",
    // French prepositions and conjunctions
    "et", "ou", "mais", "donc", "car", "ni", "que", "qui", "quoi",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "chez",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    // English articles and pronouns
    "the", "a", "an", "is", "are", "am", "was", "were",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    // English common words
    "and", "or", "but", "so", "if", "then", "to", "of", "in", "on", "at",
    "be", "have", "has", "had", "do", "does", "did",
    // Very short words
    "si", "ne", "pas", "plus", "très", "bien",
];

/// Global emotional state that accumulates over time
/// Like a baby's mood that changes slowly
#[derive(Clone, Debug, Default)]
pub struct EmotionalState {
    /// Joy/happiness level (-1.0 to 1.0)
    pub happiness: f32,
    /// Arousal/excitement level (0.0 to 1.0)
    pub arousal: f32,
    /// Comfort/security level (-1.0 to 1.0)
    pub comfort: f32,
    /// Curiosity level (0.0 to 1.0)
    pub curiosity: f32,
    /// Boredom level (0.0 to 1.0) - increases without interaction
    pub boredom: f32,
    /// Last update tick
    pub last_update: u64,
}

impl EmotionalState {
    /// Decay emotions slowly toward neutral over time
    /// Boredom INCREASES over time without interaction!
    pub fn decay(&mut self, current_tick: u64) {
        let ticks_elapsed = current_tick.saturating_sub(self.last_update);
        if ticks_elapsed > 0 {
            // Decay rate: emotions halve every ~1000 ticks (~10 seconds)
            let decay = 0.999f32.powi(ticks_elapsed as i32);
            self.happiness *= decay;
            self.arousal *= decay;
            self.comfort *= decay;
            self.curiosity *= decay;

            // Boredom GROWS over time (opposite of decay!)
            // Increases slowly: reaches 0.5 after ~30 seconds of inactivity
            let boredom_growth = 0.0001 * ticks_elapsed as f32;
            self.boredom = (self.boredom + boredom_growth).min(1.0);

            self.last_update = current_tick;
        }
    }

    /// Update emotional state based on signal content
    pub fn process_signal(&mut self, signal: &Signal, current_tick: u64) {
        // First decay existing emotions
        self.decay(current_tick);

        // Positive emotion in signal (index 28)
        let positive = signal.content.get(28).copied().unwrap_or(0.0);
        // Negative emotion in signal (index 29)
        let negative = signal.content.get(29).copied().unwrap_or(0.0);
        // Request/need in signal (index 30)
        let request = signal.content.get(30).copied().unwrap_or(0.0);
        // Question/curiosity in signal (index 31)
        let question = signal.content.get(31).copied().unwrap_or(0.0);

        // Update emotions with momentum (changes are gradual)
        let momentum = 0.3;

        if positive > 0.0 {
            self.happiness = (self.happiness + positive * momentum * signal.intensity).clamp(-1.0, 1.0);
            self.comfort = (self.comfort + 0.2 * momentum * signal.intensity).clamp(-1.0, 1.0);
        }

        if negative < 0.0 {
            self.happiness = (self.happiness + negative * momentum * signal.intensity).clamp(-1.0, 1.0);
            self.comfort = (self.comfort - 0.3 * momentum * signal.intensity).clamp(-1.0, 1.0);
        }

        if question > 0.0 {
            self.curiosity = (self.curiosity + question * momentum * signal.intensity).clamp(0.0, 1.0);
            self.arousal = (self.arousal + 0.1 * momentum).clamp(0.0, 1.0);
        }

        if request > 0.0 {
            self.arousal = (self.arousal + request * momentum * signal.intensity).clamp(0.0, 1.0);
        }

        // Any signal increases arousal slightly
        self.arousal = (self.arousal + 0.05 * signal.intensity).clamp(0.0, 1.0);

        // Interaction reduces boredom! Someone is paying attention to ARIA
        self.boredom = (self.boredom - 0.3 * signal.intensity).max(0.0);
    }

    /// Get the dominant emotional marker for expressions
    pub fn get_emotional_marker(&self) -> Option<&'static str> {
        // Only show emotion if strong enough
        let threshold = 0.3;

        if self.happiness > threshold && self.happiness >= self.curiosity.abs() {
            if self.happiness > 0.6 {
                Some("♥")
            } else {
                Some("~")
            }
        } else if self.curiosity > threshold {
            if self.arousal > 0.5 {
                Some("!")
            } else {
                Some("?")
            }
        } else if self.happiness < -threshold {
            Some("...")
        } else if self.arousal > 0.6 {
            Some("!")
        } else {
            None
        }
    }

    /// Get a description of the current mood
    pub fn mood_description(&self) -> &'static str {
        if self.happiness > 0.5 && self.arousal > 0.5 {
            "joyeux"
        } else if self.happiness > 0.5 {
            "content"
        } else if self.curiosity > 0.5 {
            "curieux"
        } else if self.happiness < -0.3 {
            "triste"
        } else if self.arousal > 0.6 {
            "excité"
        } else {
            "calme"
        }
    }
}

/// The living substrate
pub struct Substrate {
    /// All living cells
    cells: DashMap<u64, Cell>,

    /// Next cell ID
    next_id: AtomicU64,

    /// Current tick
    tick: AtomicU64,

    /// Attractors in semantic space
    attractors: RwLock<Vec<Attractor>>,

    /// Recent signals for pattern detection
    signal_buffer: RwLock<Vec<Signal>>,

    /// Long-term memory reference
    memory: Arc<RwLock<LongTermMemory>>,

    /// Global energy available (reserved for future use)
    #[allow(dead_code)]
    global_energy: AtomicU64,

    /// Short-term memory: words heard in the last few seconds
    /// ARIA will try to "echo" these words like a baby learning
    recent_words: RwLock<Vec<RecentWord>>,

    /// Global emotional state - ARIA's current mood
    emotional_state: RwLock<EmotionalState>,

    /// Was the last signal a question? (for responding with oui/non)
    last_was_question: RwLock<bool>,

    /// Last tick when someone talked to ARIA (for spontaneity)
    last_interaction_tick: AtomicU64,

    /// Words ARIA recently said (for feedback reinforcement)
    /// When someone says "Bravo!", we reinforce these words
    recent_expressions: RwLock<Vec<String>>,

    /// Last word ARIA said (to avoid immediate repetition)
    last_said_word: RwLock<Option<String>>,

    /// Conversation context - tracks the flow of discussion
    conversation: RwLock<ConversationContext>,
}

/// An attractor in semantic space
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Attractor {
    pub position: [f32; 16],
    pub strength: f32,
    pub label: String,
    pub created_at: u64,
}

/// Statistics about the substrate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubstrateStats {
    pub tick: u64,
    pub alive_cells: usize,
    pub total_energy: f32,
    pub entropy: f32,
    pub active_clusters: usize,
    pub dominant_emotion: String,
    pub signals_per_second: f32,
    pub oldest_cell_age: u64,
    pub average_connections: f32,
    /// ARIA's current mood description
    pub mood: String,
    /// Happiness level (-1 to 1)
    pub happiness: f32,
    /// Arousal/excitement level (0 to 1)
    pub arousal: f32,
    /// Curiosity level (0 to 1)
    pub curiosity: f32,
}

impl Substrate {
    /// Create a new substrate with initial cells
    pub fn new(initial_cells: usize, memory: Arc<RwLock<LongTermMemory>>) -> Self {
        let cells = DashMap::new();

        // Create primordial cells
        for i in 0..initial_cells {
            cells.insert(i as u64, Cell::new(i as u64));
        }

        // Check if we have elite DNA to seed from memory
        {
            let mem = memory.read();
            if !mem.elite_dna.is_empty() {
                tracing::info!("Seeding {} cells with elite DNA from memory", mem.elite_dna.len().min(100));
                for (i, elite) in mem.elite_dna.iter().take(100).enumerate() {
                    if let Some(mut cell) = cells.get_mut(&(i as u64)) {
                        cell.dna = elite.dna.clone();
                    }
                }
            }
        }

        Self {
            cells,
            next_id: AtomicU64::new(initial_cells as u64),
            tick: AtomicU64::new(0),
            attractors: RwLock::new(Vec::new()),
            signal_buffer: RwLock::new(Vec::new()),
            memory,
            global_energy: AtomicU64::new(10000),
            recent_words: RwLock::new(Vec::new()),
            emotional_state: RwLock::new(EmotionalState::default()),
            last_was_question: RwLock::new(false),
            last_interaction_tick: AtomicU64::new(0),
            recent_expressions: RwLock::new(Vec::new()),
            last_said_word: RwLock::new(None),
            conversation: RwLock::new(ConversationContext::new()),
        }
    }

    /// Inject an external signal (perception) and return immediate emergence
    pub fn inject_signal(&self, signal: Signal) -> Vec<Signal> {
        // Extract words and check familiarity
        let words: Vec<&str> = signal.label
            .split(|c: char| !c.is_alphabetic())
            .filter(|w| !w.is_empty())
            .collect();

        // Calculate familiarity boost and record word frequencies
        let mut familiarity_boost = 1.0f32;
        let signal_vector = signal.to_vector();
        let emotional_valence = if signal.content.get(28).copied().unwrap_or(0.0) > 0.0 {
            1.0
        } else if signal.content.get(29).copied().unwrap_or(0.0) < 0.0 {
            -1.0
        } else {
            0.0
        };

        let current_tick = self.tick.load(Ordering::Relaxed);

        // Record this interaction for spontaneity tracking
        self.last_interaction_tick.store(current_tick, Ordering::Relaxed);

        // Detect social context of input
        let (social_context, context_confidence) = LongTermMemory::detect_social_context(&signal.label);

        // Add to conversation context - ARIA now follows the discussion!
        let (is_conversation_start, current_context) = {
            let significant_words: Vec<String> = words.iter()
                .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.to_lowercase().as_str()))
                .map(|w| w.to_lowercase())
                .collect();

            let mut conversation = self.conversation.write();
            conversation.add_input(&signal.label, significant_words, emotional_valence, current_tick, social_context);

            let is_start = conversation.is_conversation_start();
            let ctx = conversation.get_social_context();

            // Log conversation state with social context
            if context_confidence > 0.7 {
                tracing::info!("SOCIAL CONTEXT: {:?} (confidence: {:.2}), Exchange #{}",
                    social_context, context_confidence, conversation.exchange_count);
            }

            if !conversation.topic_words.is_empty() {
                let topics: Vec<&str> = conversation.topic_words.iter()
                    .take(3)
                    .map(|(w, _)| w.as_str())
                    .collect();
                tracing::info!("CONVERSATION: Topics = {:?}, Exchanges = {}",
                    topics, conversation.exchanges.len());
            }

            (is_start, ctx)
        };

        // Detect FEEDBACK - this is how ARIA learns what's good/bad!
        let lower_label = signal.label.to_lowercase();

        // Positive feedback words (French + English)
        let positive_feedback = [
            "bravo", "bien", "super", "génial", "parfait", "excellent", "oui c'est ça",
            "good", "great", "yes", "perfect", "exactly", "nice", "awesome",
            "c'est bien", "très bien", "good job", "well done", "👏", "👍"
        ];

        // Negative feedback words
        let negative_feedback = [
            "non", "pas ça", "mauvais", "faux", "incorrect", "arrête",
            "no", "wrong", "bad", "stop", "not that", "incorrect",
            "c'est pas ça", "pas comme ça", "👎"
        ];

        let is_positive_feedback = positive_feedback.iter().any(|w| lower_label.contains(w));
        let is_negative_feedback = negative_feedback.iter().any(|w| lower_label.contains(w));

        // Apply feedback to recently expressed words
        if is_positive_feedback || is_negative_feedback {
            let recent_expr = self.recent_expressions.read().clone();

            // Get the words from the last user input (what triggered ARIA's response)
            let last_input_words: Vec<String> = {
                let conversation = self.conversation.read();
                // The last exchange contains what was said before ARIA's response
                // But we need the one BEFORE the feedback message
                if conversation.exchanges.len() >= 2 {
                    conversation.exchanges.get(1)
                        .map(|e| e.input_words.clone())
                        .unwrap_or_default()
                } else {
                    Vec::new()
                }
            };

            let mut memory = self.memory.write();

            for word in &recent_expr {
                if is_positive_feedback {
                    // REINFORCE: Increase emotional valence and familiarity
                    if let Some(freq) = memory.word_frequencies.get_mut(word) {
                        let old_valence = freq.emotional_valence;
                        freq.emotional_valence = (freq.emotional_valence + 0.3).clamp(-2.0, 2.0);
                        freq.count += 2; // Bonus familiarity
                        tracing::info!(
                            "FEEDBACK POSITIVE! '{}' reinforced (valence: {:.2} → {:.2})",
                            word, old_valence, freq.emotional_valence
                        );
                    }

                    // NEW: Learn input→response associations!
                    // If user said "ça va" and ARIA said "bien" and got "Bravo!"
                    // → Create association between "ça va" and "bien"
                    for input_word in &last_input_words {
                        if input_word != word {
                            memory.learn_association(input_word, word, 0.5);
                            tracing::info!(
                                "FEEDBACK LEARNING: '{}' → '{}' association strengthened",
                                input_word, word
                            );
                        }
                    }
                } else {
                    // PENALIZE: Decrease emotional valence
                    if let Some(freq) = memory.word_frequencies.get_mut(word) {
                        let old_valence = freq.emotional_valence;
                        freq.emotional_valence = (freq.emotional_valence - 0.3).clamp(-2.0, 2.0);
                        tracing::info!(
                            "FEEDBACK NEGATIVE! '{}' penalized (valence: {:.2} → {:.2})",
                            word, old_valence, freq.emotional_valence
                        );
                    }
                }
            }

            // Update emotional state based on feedback
            let mut emotional = self.emotional_state.write();
            if is_positive_feedback {
                emotional.happiness = (emotional.happiness + 0.3).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort + 0.2).clamp(-1.0, 1.0);
                tracing::info!("ARIA feels happy from positive feedback! (happiness: {:.2})", emotional.happiness);
            } else {
                emotional.happiness = (emotional.happiness - 0.2).clamp(-1.0, 1.0);
                emotional.comfort = (emotional.comfort - 0.1).clamp(-1.0, 1.0);
                tracing::info!("ARIA feels sad from negative feedback... (happiness: {:.2})", emotional.happiness);
            }
        }

        // Detect if this is a question (ends with ? or has question marker)
        let is_question = signal.label.ends_with('?')
            || signal.content.get(31).copied().unwrap_or(0.0) > 0.5;

        // Store for detect_emergence
        {
            let mut last_q = self.last_was_question.write();
            *last_q = is_question;
            if is_question {
                tracing::info!("Question detected: '{}'", signal.label);
            }
        }

        // Update emotional state based on signal content
        {
            let mut emotional = self.emotional_state.write();
            emotional.process_signal(&signal, current_tick);
            tracing::debug!(
                "Mood: {} (happiness={:.2}, arousal={:.2}, curiosity={:.2})",
                emotional.mood_description(),
                emotional.happiness,
                emotional.arousal,
                emotional.curiosity
            );
        }

        {
            let mut memory = self.memory.write();
            memory.stats.total_ticks = current_tick;

            // Learn words with context for category detection
            for (i, word) in words.iter().enumerate() {
                let preceding = if i > 0 { Some(words[i - 1]) } else { None };
                let following = if i + 1 < words.len() { Some(words[i + 1]) } else { None };

                let word_familiarity = memory.hear_word_with_context(
                    word,
                    signal_vector,
                    emotional_valence,
                    preceding,
                    following,
                );
                if word_familiarity > 0.5 {
                    // Familiar word! Boost the signal
                    familiarity_boost = familiarity_boost.max(1.0 + word_familiarity);
                    tracing::info!("Recognized familiar word: '{}' (familiarity: {:.2})", word, word_familiarity);
                }
            }

            // Learn semantic associations: words that appear together become linked
            // Example: "Moka" and "chat" in the same message -> they become associated
            // Filter out stop words - only meaningful words should be associated!
            let significant_words: Vec<&str> = words.iter()
                .filter(|w| w.len() >= 3 && !STOP_WORDS.contains(&w.to_lowercase().as_str()))
                .copied()
                .collect();

            // Create associations between all pairs of significant words
            for i in 0..significant_words.len() {
                for j in (i + 1)..significant_words.len() {
                    memory.learn_association(
                        significant_words[i],
                        significant_words[j],
                        emotional_valence
                    );
                }
            }

            // Learn usage patterns for words in this message
            // This must be AFTER hear_word_with_context so the word exists!
            for word in &significant_words {
                memory.learn_usage_pattern(word, current_context, is_conversation_start, false);
            }
        }

        // Store words in short-term memory for echo/imitation
        // Filter out stop words - ARIA should focus on meaningful words!
        {
            let mut recent = self.recent_words.write();
            for word in &words {
                let lower_word = word.to_lowercase();
                // Skip very short words AND stop words
                if word.len() >= 3 && !STOP_WORDS.contains(&lower_word.as_str()) {
                    recent.push(RecentWord {
                        word: lower_word,
                        vector: signal_vector,
                        heard_at: current_tick,
                    });
                }
            }
            // Keep only words from the last 500 ticks (~5 seconds)
            recent.retain(|w| current_tick - w.heard_at < 500);
            // Limit size
            if recent.len() > 20 {
                let drain_count = recent.len() - 20;
                recent.drain(0..drain_count);
            }
        }

        // Transform signal to fragments for cells
        // Amplify external signals - they're important!
        // Apply familiarity boost for known words
        let base_intensity = signal.intensity * 5.0 * familiarity_boost;
        let fragment = SignalFragment {
            source_id: 0, // External source
            content: signal_vector,
            intensity: base_intensity,
        };

        // Get semantic position of the signal
        let target_position = signal.semantic_position();

        tracing::info!("Signal received: '{}' intensity={:.2} (familiarity_boost: {:.2})",
            signal.label, fragment.intensity, familiarity_boost);

        // Distribute to ALL cells (external signals are broadcast)
        // Minimal attenuation - we want all cells to "hear" external input
        self.cells.iter_mut().for_each(|mut entry| {
            let cell = entry.value_mut();
            let distance = semantic_distance(&cell.position, &target_position);

            // Very mild attenuation - all cells get at least 20% of original
            let mut attenuated_fragment = fragment.clone();
            let attenuation = (1.0 / (1.0 + distance * 0.1)).max(0.2);
            attenuated_fragment.intensity = fragment.intensity * attenuation;

            cell.receive(attenuated_fragment.clone());

            // External signals also give energy boost (attention/arousal)
            cell.energy = (cell.energy + 0.05 * fragment.intensity).min(1.5);

            // IMMEDIATE ACTIVATION: External signals directly activate cells
            // This makes ARIA respond faster
            for (i, s) in attenuated_fragment.content.iter().enumerate() {
                if i < 8 {
                    cell.state[i] += s * attenuated_fragment.intensity * 5.0;
                }
            }
        });

        // Create a temporary attractor
        {
            let mut attractors = self.attractors.write();
            attractors.push(Attractor {
                position: target_position,
                strength: signal.intensity,
                label: signal.label.clone(),
                created_at: self.tick.load(Ordering::Relaxed),
            });
        }

        // Store in signal buffer
        {
            let mut buffer = self.signal_buffer.write();
            buffer.push(signal);
            if buffer.len() > 1000 {
                buffer.remove(0);
            }
        }

        // IMMEDIATE RESPONSE: Check for emergence right after signal injection
        let current_tick = self.tick.load(Ordering::Relaxed);
        self.detect_emergence(current_tick)
    }

    /// One tick of life
    pub fn tick(&self) -> Vec<Signal> {
        let current_tick = self.tick.fetch_add(1, Ordering::SeqCst);

        let mut new_cells: Vec<Cell> = Vec::new();
        let mut dead_cells: Vec<u64> = Vec::new();
        // Reserved for future use
        let _connection_requests: Vec<(u64, u64)> = Vec::new();
        let _emitted_signals: Vec<([f32; 8], [f32; 16], f32)> = Vec::new();

        // Phase 1: Each cell lives (parallel)
        self.cells.iter_mut().for_each(|mut entry| {
            let cell = entry.value_mut();
            let action = cell.live();

            match action {
                CellAction::Die => {
                    // Will be removed later
                }
                CellAction::Divide => {
                    // Mark for division
                }
                CellAction::Connect => {
                    // Will create connection later
                }
                CellAction::Signal(_content) => {
                    // Collect emitted signals (TODO: process)
                }
                CellAction::Move(direction) => {
                    // Apply movement
                    for (i, d) in direction.iter().enumerate() {
                        cell.position[i] = (cell.position[i] + d).clamp(-10.0, 10.0);
                    }
                }
                CellAction::Rest => {}
            }
        });

        // Phase 2: Process actions (sequential for consistency)
        let actions: Vec<(u64, CellAction)> = self.cells.iter()
            .map(|entry| {
                let cell = entry.value();
                let action = if cell.energy <= 0.0 {
                    CellAction::Die
                } else if cell.tension == 0.0 && cell.energy > 0.6 && cell.age % 100 == 0 {
                    CellAction::Divide
                } else {
                    CellAction::Rest
                };
                (*entry.key(), action)
            })
            .collect();

        for (cell_id, action) in actions {
            match action {
                CellAction::Die => {
                    dead_cells.push(cell_id);
                }
                CellAction::Divide => {
                    if let Some(parent) = self.cells.get(&cell_id) {
                        if parent.energy > 0.6 {
                            let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                            let child = Cell::from_parent(new_id, &parent);
                            new_cells.push(child);
                        }
                    }
                }
                _ => {}
            }
        }

        // Remove dead cells
        for id in dead_cells {
            self.cells.remove(&id);
        }

        // Add new cells
        for cell in new_cells {
            self.cells.insert(cell.id, cell);
        }

        // Phase 3: Propagate signals between cells
        self.propagate_internal_signals();

        // Phase 4: Detect emergent patterns
        let emergent = self.detect_emergence(current_tick);

        // Phase 5: Decay attractors
        {
            let mut attractors = self.attractors.write();
            attractors.retain_mut(|a| {
                a.strength *= 0.99;
                a.strength > 0.01
            });
        }

        // Phase 6: Apply attractor influence
        self.apply_attractors();

        // Phase 7: Maintain population (natural selection) - run frequently
        if current_tick % 10 == 0 {
            self.natural_selection();
        }

        // Phase 8: Spontaneous expression - ARIA speaks without being asked!
        // This makes her feel alive, like a real baby
        let spontaneous = self.maybe_speak_spontaneously(current_tick);

        // Phase 9: Dream/consolidate memory when inactive
        // Like a baby sleeping - she processes and strengthens memories
        self.maybe_dream(current_tick);

        // Combine emergent and spontaneous signals
        if spontaneous.is_some() {
            let mut all_signals = emergent;
            all_signals.extend(spontaneous);
            all_signals
        } else {
            emergent
        }
    }

    /// Dream/consolidate memory when inactive
    /// ARIA "plays" with her memories, strengthening connections she likes
    fn maybe_dream(&self, current_tick: u64) {
        // Only dream every 500 ticks (~5 seconds)
        if current_tick % 500 != 0 {
            return;
        }

        let last_interaction = self.last_interaction_tick.load(Ordering::Relaxed);
        let ticks_since_interaction = current_tick.saturating_sub(last_interaction);

        // Only dream when inactive for at least 10 seconds (1000 ticks)
        if ticks_since_interaction < 1000 {
            return;
        }

        let mut memory = self.memory.write();
        let mut rng = rand::thread_rng();

        // Find words she loves (high positive valence)
        let favorite_words: Vec<String> = memory.word_frequencies.iter()
            .filter(|(_, freq)| freq.emotional_valence > 0.3 && freq.count > 2)
            .map(|(word, _)| word.clone())
            .collect();

        if favorite_words.is_empty() {
            return;
        }

        // Pick a random favorite word to "think about"
        let dream_word = &favorite_words[rng.gen_range(0..favorite_words.len())];

        // Strengthen this word slightly (she's rehearsing it in her mind)
        if let Some(freq) = memory.word_frequencies.get_mut(dream_word) {
            freq.count += 1;
            // Small valence boost for positive words (happy memories grow stronger)
            if freq.emotional_valence > 0.0 {
                freq.emotional_valence = (freq.emotional_valence + 0.05).min(2.0);
            }
        }

        // Also strengthen associations with this word
        let associations: Vec<(String, f32)> = memory.get_associations(dream_word);
        for (assoc_word, strength) in associations.iter().take(2) {
            if *strength > 0.3 {
                // Strengthen the association (dreaming reinforces connections)
                let key = if dream_word < assoc_word {
                    format!("{}:{}", dream_word, assoc_word)
                } else {
                    format!("{}:{}", assoc_word, dream_word)
                };

                if let Some(assoc) = memory.word_associations.get_mut(&key) {
                    assoc.strength = (assoc.strength + 0.02).min(1.0);
                }
            }
        }

        // Log dreaming activity (less frequently to avoid spam)
        if rng.gen::<f32>() < 0.1 {
            tracing::info!("💭 DREAMING: Thinking about '{}'...", dream_word);
        }
    }

    /// Maybe generate a spontaneous expression
    /// ARIA might speak on her own if:
    /// - It's been a while since someone talked to her (lonely)
    /// - She's excited/aroused (wants to share)
    /// - She's thinking about a word she loves
    fn maybe_speak_spontaneously(&self, current_tick: u64) -> Option<Signal> {
        // Only check every 100 ticks (~1 second) to avoid spam
        if current_tick % 100 != 0 {
            return None;
        }

        let last_interaction = self.last_interaction_tick.load(Ordering::Relaxed);
        let ticks_since_interaction = current_tick.saturating_sub(last_interaction);

        // Get emotional state
        let emotional = self.emotional_state.read();

        // Different triggers for spontaneous speech:

        // 1. Lonely: No interaction for 30+ seconds (3000 ticks)
        //    ARIA might call out or babble to get attention
        let lonely_threshold = 3000;
        let is_lonely = ticks_since_interaction > lonely_threshold;

        // 2. Excited: High arousal, wants to express
        let is_excited = emotional.arousal > 0.6;

        // 3. Happy: Wants to share joy
        let is_very_happy = emotional.happiness > 0.5;

        // 4. Curious: Wants to explore/ask
        let is_curious = emotional.curiosity > 0.5;

        // 5. Bored: Wants stimulation, might try new things!
        let is_bored = emotional.boredom > 0.5;

        // Random factor to make it unpredictable (like a real baby)
        let mut rng = rand::thread_rng();
        let random_urge: f32 = rng.gen();

        // Probability of speaking based on state
        let speak_probability = if is_lonely {
            0.05  // 5% chance per second when lonely
        } else if is_bored {
            0.04  // 4% when bored - wants attention!
        } else if is_excited && is_very_happy {
            0.03  // 3% when very happy and excited
        } else if is_excited {
            0.02  // 2% when just excited
        } else if is_curious {
            0.01  // 1% when curious
        } else {
            0.001 // 0.1% baseline (very rare)
        };

        if random_urge > speak_probability {
            return None;
        }

        // ARIA wants to speak! What does she say?
        let memory = self.memory.read();

        // Find a word she loves (high positive valence)
        let favorite_word = memory.word_frequencies.iter()
            .filter(|(_, freq)| freq.emotional_valence > 0.5 && freq.count > 3)
            .max_by(|(_, a), (_, b)| {
                // Prefer words with high valence AND frequency
                let score_a = a.emotional_valence * (a.count as f32).sqrt();
                let score_b = b.emotional_valence * (b.count as f32).sqrt();
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(word, _)| word.clone());

        // Generate the spontaneous expression
        let (label, intensity) = if is_lonely {
            // Lonely: call out or make attention-seeking sounds
            if let Some(word) = favorite_word.clone() {
                tracing::info!("SPONTANEOUS (lonely): Thinking about '{}'", word);
                (format!("spontaneous:{}|emotion:?", word), 0.3)
            } else {
                tracing::info!("SPONTANEOUS (lonely): Seeking attention");
                ("spontaneous:attention|emotion:?".to_string(), 0.2)
            }
        } else if is_bored {
            // Bored: try something creative! Combine words or explore
            // Pick TWO favorite words and combine them (creative play)
            let all_favorites: Vec<String> = memory.word_frequencies.iter()
                .filter(|(_, freq)| freq.emotional_valence > 0.2 && freq.count > 1)
                .map(|(word, _)| word.clone())
                .collect();

            if all_favorites.len() >= 2 {
                let word1 = &all_favorites[rng.gen_range(0..all_favorites.len())];
                let word2 = &all_favorites[rng.gen_range(0..all_favorites.len())];
                if word1 != word2 {
                    tracing::info!("SPONTANEOUS (bored): Creative play - combining '{}' + '{}'", word1, word2);
                    (format!("phrase:{}+{}|emotion:~", word1, word2), 0.35)
                } else if let Some(word) = favorite_word.clone() {
                    tracing::info!("SPONTANEOUS (bored): Playing with '{}'", word);
                    (format!("spontaneous:{}|emotion:~", word), 0.3)
                } else {
                    tracing::info!("SPONTANEOUS (bored): Restless");
                    ("spontaneous:bored|emotion:~".to_string(), 0.25)
                }
            } else if let Some(word) = favorite_word.clone() {
                tracing::info!("SPONTANEOUS (bored): Playing with '{}'", word);
                (format!("spontaneous:{}|emotion:~", word), 0.3)
            } else {
                tracing::info!("SPONTANEOUS (bored): Restless");
                ("spontaneous:bored|emotion:~".to_string(), 0.25)
            }
        } else if is_very_happy {
            // Happy: share joy about favorite thing
            if let Some(word) = favorite_word.clone() {
                tracing::info!("SPONTANEOUS (happy): Expressing love for '{}'", word);
                (format!("spontaneous:{}|emotion:♥", word), 0.5)
            } else {
                tracing::info!("SPONTANEOUS (happy): General joy");
                ("spontaneous:joy|emotion:♥".to_string(), 0.4)
            }
        } else if is_excited {
            // Excited: energetic babbling
            tracing::info!("SPONTANEOUS (excited): Energetic expression");
            ("spontaneous:excited|emotion:!".to_string(), 0.4)
        } else if is_curious {
            // Curious: questioning
            if let Some(word) = favorite_word.clone() {
                tracing::info!("SPONTANEOUS (curious): Wondering about '{}'", word);
                (format!("spontaneous:{}|emotion:?", word), 0.3)
            } else {
                tracing::info!("SPONTANEOUS (curious): General curiosity");
                ("spontaneous:curious|emotion:?".to_string(), 0.3)
            }
        } else {
            // Rare baseline: soft babbling
            tracing::info!("SPONTANEOUS: Soft babbling");
            ("spontaneous:babble|emotion:~".to_string(), 0.2)
        };

        // Create the signal
        let mut signal = Signal::from_vector([0.0; 8], label);
        signal.intensity = intensity;

        Some(signal)
    }

    fn propagate_internal_signals(&self) {
        // Collect signals from active cells
        let signals: Vec<(u64, [f32; 16], [f32; 8], f32)> = self.cells.iter()
            .filter_map(|entry| {
                let cell = entry.value();
                let activation: f32 = cell.state.iter().map(|x| x.abs()).sum();
                if activation > 0.5 {
                    let content: [f32; 8] = std::array::from_fn(|i| cell.state[i]);
                    Some((cell.id, cell.position, content, activation))
                } else {
                    None
                }
            })
            .collect();

        // Distribute to nearby cells
        for (source_id, source_pos, content, intensity) in signals {
            self.cells.iter_mut().for_each(|mut entry| {
                let cell = entry.value_mut();
                if cell.id != source_id {
                    let distance = semantic_distance(&cell.position, &source_pos);
                    if distance < 2.0 {
                        cell.receive(SignalFragment {
                            source_id,
                            content,
                            intensity: intensity / (1.0 + distance),
                        });
                    }
                }
            });
        }
    }

    fn detect_emergence(&self, current_tick: u64) -> Vec<Signal> {
        // Check for emergence every 5 ticks (~20x per second) - more responsive
        if current_tick % 5 != 0 {
            return Vec::new();
        }

        // Find cells with ANY activation (very low threshold)
        let active_cells: Vec<_> = self.cells.iter()
            .filter(|entry| {
                let cell = entry.value();
                cell.state.iter().map(|x| x.abs()).sum::<f32>() > 0.01
            })
            .take(1000) // Limit for performance
            .collect();

        // Need at least a few active cells
        if active_cells.is_empty() {
            return Vec::new();
        }

        // Calculate average state
        let mut average_state = [0.0f32; 8];
        for entry in &active_cells {
            for (i, s) in entry.value().state[0..8].iter().enumerate() {
                average_state[i] += s;
            }
        }
        let n = active_cells.len() as f32;
        for a in &mut average_state {
            *a /= n;
        }

        // Check coherence (lowered threshold for baby ARIA)
        let coherence = self.calculate_cluster_coherence(&active_cells);

        if coherence > 0.1 {
            // This is an emergent thought!

            // Check if this is a response to a question
            let was_question = *self.last_was_question.read();

            // Get the last word ARIA said (to avoid immediate repetition)
            let last_word = self.last_said_word.read().clone();

            // Get conversation context for boosting relevant words
            let context_words = self.conversation.read().get_context_words();

            // Get social context - is this a greeting, farewell, etc.?
            let social_context = self.conversation.read().get_social_context();
            let is_conversation_start = self.conversation.read().is_conversation_start();

            // SOCIAL CONTEXT RESPONSES - respond appropriately to greetings, etc.
            // - Greeting: anytime (someone might say "coucou" to get attention)
            // - Farewell: only at conversation start/end (exchanges 1-2)
            // - Thanks/Affection: anytime!
            let should_respond_socially = match social_context {
                SocialContext::Greeting => true,  // Greetings can happen anytime!
                SocialContext::Farewell => is_conversation_start,
                SocialContext::Thanks | SocialContext::Affection => true,
                _ => false,
            };

            if should_respond_socially && social_context != SocialContext::General {
                let social_response = match social_context {
                    SocialContext::Greeting => {
                        // Find a greeting word ARIA knows or use default
                        let memory = self.memory.read();
                        if let Some(response) = memory.get_response_for_context(SocialContext::Greeting) {
                            Some(format!("social:greeting:{}", response))
                        } else {
                            Some("social:greeting:bonjour".to_string())
                        }
                    }
                    SocialContext::Farewell => {
                        let memory = self.memory.read();
                        if let Some(response) = memory.get_response_for_context(SocialContext::Farewell) {
                            Some(format!("social:farewell:{}", response))
                        } else {
                            Some("social:farewell:bye".to_string())
                        }
                    }
                    SocialContext::Thanks => {
                        // ARIA says "de rien" or similar
                        Some("social:thanks:derien".to_string())
                    }
                    SocialContext::Affection => {
                        // Respond with affection!
                        let memory = self.memory.read();
                        if let Some(response) = memory.get_response_for_context(SocialContext::Affection) {
                            Some(format!("social:affection:{}", response))
                        } else {
                            Some("social:affection:aime".to_string())
                        }
                    }
                    _ => None,
                };

                if let Some(response_label) = social_response {
                    tracing::info!("SOCIAL RESPONSE: Context={:?} -> {}", social_context, response_label);

                    // Add emotional marker for social responses
                    let emotional_marker = match social_context {
                        SocialContext::Affection => Some("♥"),
                        SocialContext::Greeting => Some("~"),
                        _ => None,
                    };

                    let final_label = if let Some(marker) = emotional_marker {
                        format!("{}|emotion:{}", response_label, marker)
                    } else {
                        response_label
                    };

                    let mut signal = Signal::from_vector(average_state, final_label.clone());
                    signal.intensity = coherence.max(0.4); // Ensure social responses are visible

                    // Record what we said for anti-repetition
                    {
                        let words_said: Vec<String> = final_label
                            .split('|').next().unwrap_or("")
                            .split(':').last().unwrap_or("")
                            .split('+')
                            .map(|s| s.to_string())
                            .collect();

                        if let Some(word) = words_said.first() {
                            *self.last_said_word.write() = Some(word.clone());
                        }
                    }

                    // Record in conversation
                    self.conversation.write().add_response(&final_label);

                    return vec![signal];
                }
            }

            // First, try to echo a RECENT word (like a baby imitating)
            // Now with CONTEXT BOOSTING - words in current conversation get priority!
            let label = {
                let recent = self.recent_words.read();
                let mut best_recent: Option<(&str, f32)> = None;

                // Check recent words first - strong preference for imitation!
                // But skip the last word we said to avoid repetition
                for rw in recent.iter() {
                    // Skip if this is the same word we just said
                    if let Some(ref last) = last_word {
                        if rw.word.to_lowercase() == last.to_lowercase() {
                            continue;
                        }
                    }

                    let mut similarity = Self::vector_similarity(&average_state, &rw.vector);

                    // CONTEXT BOOST: If this word is in the current conversation, boost it!
                    if let Some((_, boost)) = context_words.iter()
                        .find(|(w, _)| w.to_lowercase() == rw.word.to_lowercase())
                    {
                        similarity += boost * 0.3; // Add up to 0.3 for context relevance
                        tracing::debug!("Context boost for '{}': +{:.2}", rw.word, boost * 0.3);
                    }

                    // Lower threshold for recent words - we WANT to echo them
                    if similarity > 0.2 {
                        match best_recent {
                            Some((_, best_sim)) if similarity > best_sim => {
                                best_recent = Some((&rw.word, similarity));
                            }
                            None => {
                                best_recent = Some((&rw.word, similarity));
                            }
                            _ => {}
                        }
                    }
                }

                if let Some((word, similarity)) = best_recent {
                    tracing::info!("ECHO! Imitating recent word '{}' (similarity: {:.2})", word, similarity);

                    // Check for semantic associations - maybe add related words!
                    let memory = self.memory.read();

                    // If this was a question, respond with oui/non based on word valence!
                    if was_question {
                        let valence = memory.word_frequencies.get(word)
                            .map(|f| f.emotional_valence)
                            .unwrap_or(0.0);

                        if valence > 0.3 {
                            // Positive word → oui!
                            tracing::info!("QUESTION RESPONSE: '{}' is positive (valence: {:.2}) → oui!", word, valence);
                            format!("answer:oui+{}", word)
                        } else if valence < -0.3 {
                            // Negative word → non
                            tracing::info!("QUESTION RESPONSE: '{}' is negative (valence: {:.2}) → non", word, valence);
                            format!("answer:non+{}", word)
                        } else {
                            // Neutral → just echo the word with question mark
                            tracing::info!("QUESTION RESPONSE: '{}' is neutral (valence: {:.2}) → ???", word, valence);
                            format!("word:{}?", word)
                        }
                    } else {
                        // Normal flow: check for associations
                        let associations = memory.get_top_associations(word, 2);

                        // Build phrase based on how many strong associations we have
                        // Filter out the last said word and duplicates
                        let strong_assocs: Vec<_> = associations.iter()
                            .filter(|(assoc_word, strength)| {
                                let is_duplicate = last_word.as_ref()
                                    .map(|lw| lw.to_lowercase() == assoc_word.to_lowercase())
                                    .unwrap_or(false);
                                !is_duplicate && (*strength > 0.8 || (*strength > 0.6 && coherence > 0.15))
                            })
                            .collect();

                        if strong_assocs.len() >= 2 {
                            // 3-word phrase! Use order_phrase for natural order
                            let (assoc1, str1) = &strong_assocs[0];
                            let (assoc2, str2) = &strong_assocs[1];
                            let words_to_order: Vec<&str> = vec![word, assoc1.as_str(), assoc2.as_str()];
                            let ordered = memory.order_phrase(&words_to_order);
                            tracing::info!("TRIPLE! {:?} (strengths: {:.2}, {:.2}, ordered: {:?})",
                                words_to_order, str1, str2, ordered);
                            format!("phrase:{}", ordered.join("+"))
                        } else if strong_assocs.len() == 1 {
                            // 2-word phrase with natural order
                            let (assoc1, str1) = &strong_assocs[0];
                            let words_to_order: Vec<&str> = vec![word, assoc1.as_str()];
                            let ordered = memory.order_phrase(&words_to_order);
                            tracing::info!("ASSOCIATION! {:?} -> {:?} (strength: {:.2}, coherence: {:.2})",
                                words_to_order, ordered, str1, coherence);
                            format!("phrase:{}", ordered.join("+"))
                        } else {
                            format!("word:{}", word)
                        }
                    }
                } else {
                    // Fall back to long-term memory
                    let memory = self.memory.read();

                    // Find a word that's not the same as last said word
                    let matching_word = memory.find_matching_word(&average_state, 0.3)
                        .filter(|(word, _)| {
                            last_word.as_ref()
                                .map(|lw| lw.to_lowercase() != word.to_lowercase())
                                .unwrap_or(true)
                        });

                    if let Some((word, similarity)) = matching_word {
                        tracing::info!("Emergence matches word '{}' (similarity: {:.2})", word, similarity);

                        // If this was a question, respond with oui/non based on word valence!
                        if was_question {
                            let valence = memory.word_frequencies.get(&word)
                                .map(|f| f.emotional_valence)
                                .unwrap_or(0.0);

                            if valence > 0.3 {
                                tracing::info!("QUESTION RESPONSE: '{}' is positive (valence: {:.2}) → oui!", word, valence);
                                format!("answer:oui+{}", word)
                            } else if valence < -0.3 {
                                tracing::info!("QUESTION RESPONSE: '{}' is negative (valence: {:.2}) → non", word, valence);
                                format!("answer:non+{}", word)
                            } else {
                                tracing::info!("QUESTION RESPONSE: '{}' is neutral (valence: {:.2}) → ???", word, valence);
                                format!("word:{}?", word)
                            }
                        } else {
                            // Normal flow: check associations for long-term memory words
                            let associations = memory.get_top_associations(&word, 2);
                            let strong_assocs: Vec<_> = associations.iter()
                                .filter(|(assoc_word, strength)| {
                                    let is_duplicate = last_word.as_ref()
                                        .map(|lw| lw.to_lowercase() == assoc_word.to_lowercase())
                                        .unwrap_or(false);
                                    !is_duplicate && (*strength > 0.8 || (*strength > 0.6 && coherence > 0.15))
                                })
                                .collect();

                            if strong_assocs.len() >= 2 {
                                let (assoc1, str1) = &strong_assocs[0];
                                let (assoc2, str2) = &strong_assocs[1];
                                let words_to_order: Vec<&str> = vec![word.as_str(), assoc1.as_str(), assoc2.as_str()];
                                let ordered = memory.order_phrase(&words_to_order);
                                tracing::info!("TRIPLE! {:?} -> {:?} (strengths: {:.2}, {:.2})",
                                    words_to_order, ordered, str1, str2);
                                format!("phrase:{}", ordered.join("+"))
                            } else if strong_assocs.len() == 1 {
                                let (assoc1, str1) = &strong_assocs[0];
                                let words_to_order: Vec<&str> = vec![word.as_str(), assoc1.as_str()];
                                let ordered = memory.order_phrase(&words_to_order);
                                tracing::info!("ASSOCIATION! {:?} -> {:?} (strength: {:.2})",
                                    words_to_order, ordered, str1);
                                format!("phrase:{}", ordered.join("+"))
                            } else {
                                format!("word:{}", word)
                            }
                        }
                    } else {
                        format!("emergence@{}", current_tick)
                    }
                }
            };

            // Get emotional marker from:
            // 1. The global emotional state, OR
            // 2. The emotional valence of words being spoken (from associations)
            let emotional_marker = {
                // First check global mood
                let global_marker = {
                    let emotional = self.emotional_state.read();
                    emotional.get_emotional_marker().map(|s| s.to_string())
                };

                // Then check if the words we're saying have emotional associations
                let word_emotion = {
                    let memory = self.memory.read();
                    // Extract word from label (e.g., "word:moka" or "phrase:moka+chat")
                    let word = if label.starts_with("phrase:") {
                        label.strip_prefix("phrase:")
                            .and_then(|s| s.split('+').next())
                    } else if label.starts_with("word:") {
                        label.strip_prefix("word:")
                    } else {
                        None
                    };

                    if let Some(w) = word {
                        // Check word frequency emotional valence
                        if let Some(freq) = memory.word_frequencies.get(w) {
                            if freq.emotional_valence > 0.5 {
                                Some("♥".to_string())
                            } else if freq.emotional_valence < -0.5 {
                                Some("...".to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                // Prefer word-specific emotion, fallback to global mood
                word_emotion.or(global_marker)
            };

            // Combine label with emotional marker
            let final_label = if let Some(marker) = emotional_marker {
                format!("{}|emotion:{}", label, marker)
            } else {
                label
            };

            let mut signal = Signal::from_vector(average_state, final_label.clone());
            signal.intensity = coherence;

            // Record the words ARIA is expressing (for feedback reinforcement)
            {
                let mut recent_expr = self.recent_expressions.write();
                recent_expr.clear(); // Only keep the most recent expression

                // Extract words from the label
                let words_part = if final_label.contains('|') {
                    final_label.split('|').next().unwrap_or(&final_label)
                } else {
                    &final_label
                };

                // Parse different label formats
                if words_part.starts_with("phrase:") {
                    // "phrase:moka+chat+est" → ["moka", "chat", "est"]
                    if let Some(phrase) = words_part.strip_prefix("phrase:") {
                        for word in phrase.split('+') {
                            recent_expr.push(word.to_string());
                        }
                    }
                } else if words_part.starts_with("word:") {
                    // "word:moka" → ["moka"]
                    if let Some(word) = words_part.strip_prefix("word:") {
                        recent_expr.push(word.trim_end_matches('?').to_string());
                    }
                } else if words_part.starts_with("answer:") {
                    // "answer:oui+moka" → ["moka"]
                    if let Some(answer) = words_part.strip_prefix("answer:") {
                        let parts: Vec<&str> = answer.split('+').collect();
                        if parts.len() >= 2 {
                            recent_expr.push(parts[1].to_string());
                        }
                    }
                } else if words_part.starts_with("spontaneous:") {
                    // "spontaneous:moka" → ["moka"] (if it's a word, not "attention"/"joy"/etc.)
                    if let Some(content) = words_part.strip_prefix("spontaneous:") {
                        if !["attention", "joy", "excited", "curious", "babble"].contains(&content) {
                            recent_expr.push(content.to_string());
                        }
                    }
                }

                if !recent_expr.is_empty() {
                    tracing::debug!("Recording expressed words for feedback: {:?}", recent_expr);

                    // Update last_said_word to avoid repetition
                    // Use the first word as the "main" word
                    let mut last_said = self.last_said_word.write();
                    *last_said = recent_expr.first().cloned();

                    // Record ARIA's response in conversation context
                    let response_text = recent_expr.join(" ");
                    let mut conversation = self.conversation.write();
                    conversation.add_response(&response_text);
                    tracing::debug!("CONVERSATION: ARIA responded with '{}'", response_text);
                }
            }

            // Learn this pattern
            {
                let mut memory = self.memory.write();
                memory.learn_pattern(
                    vec![average_state],
                    average_state,
                    coherence
                );
            }

            vec![signal]
        } else {
            Vec::new()
        }
    }

    fn calculate_cluster_coherence(&self, cells: &[dashmap::mapref::multiple::RefMulti<u64, Cell>]) -> f32 {
        if cells.len() < 2 {
            return 0.0;
        }

        // Calculate variance of positions
        let mut mean_pos = [0.0f32; 16];
        for entry in cells {
            for (i, p) in entry.value().position.iter().enumerate() {
                mean_pos[i] += p;
            }
        }
        let n = cells.len() as f32;
        for p in &mut mean_pos {
            *p /= n;
        }

        let variance: f32 = cells.iter()
            .map(|entry| semantic_distance(&entry.value().position, &mean_pos).powi(2))
            .sum::<f32>() / n;

        // Low variance = high coherence
        (1.0 / (1.0 + variance)).min(1.0)
    }

    /// Calculate cosine similarity between two 8-dimensional vectors
    fn vector_similarity(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a < 0.001 || mag_b < 0.001 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    fn apply_attractors(&self) {
        let attractors = self.attractors.read().clone();

        if attractors.is_empty() {
            return;
        }

        self.cells.iter_mut().for_each(|mut entry| {
            let cell = entry.value_mut();

            for attractor in &attractors {
                let distance = semantic_distance(&cell.position, &attractor.position);
                if distance > 0.1 && distance < 5.0 {
                    // Move toward attractor
                    let pull = attractor.strength / (distance * distance);
                    for (i, (p, a)) in cell.position.iter_mut().zip(attractor.position.iter()).enumerate() {
                        *p += (a - *p) * pull * 0.01 * cell.dna.connectivity[i % 4];
                    }
                }
            }
        });
    }

    fn natural_selection(&self) {
        let mut rng = rand::thread_rng();

        // Remove cells with no energy
        self.cells.retain(|_, cell| cell.energy > 0.0);

        let current_pop = self.cells.len();
        let target_pop = 10_000;

        if current_pop < target_pop {
            // If population is very low, create new primordial cells
            if current_pop < 100 {
                let cells_to_create = (target_pop / 10).min(1000);
                for _ in 0..cells_to_create {
                    let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                    self.cells.insert(new_id, Cell::new(new_id));
                }
                return;
            }

            // Reproduce the best performers (lowered threshold)
            let best_cells: Vec<_> = self.cells.iter()
                .filter(|e| e.value().energy > 0.3)
                .take(100)
                .map(|e| e.value().clone())
                .collect();

            for cell in best_cells {
                if self.cells.len() >= target_pop {
                    break;
                }

                let new_id = self.next_id.fetch_add(1, Ordering::SeqCst);
                let child = Cell::from_parent(new_id, &cell);
                self.cells.insert(new_id, child);

                // Save elite DNA periodically
                if rng.gen::<f32>() < 0.01 {
                    let mut memory = self.memory.write();
                    memory.preserve_elite(
                        cell.dna.clone(),
                        cell.energy,
                        cell.generation,
                        "survivor"
                    );
                }
            }
        } else if current_pop > target_pop + 1000 {
            // Cull the weakest
            let weak_ids: Vec<u64> = self.cells.iter()
                .filter(|e| e.value().energy < 0.3)
                .take(current_pop - target_pop)
                .map(|e| *e.key())
                .collect();

            for id in weak_ids {
                self.cells.remove(&id);
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> SubstrateStats {
        let alive = self.cells.len();

        let total_energy: f32 = self.cells.iter()
            .map(|e| e.value().energy)
            .sum();

        let entropy = self.calculate_entropy();

        let oldest = self.cells.iter()
            .map(|e| e.value().age)
            .max()
            .unwrap_or(0);

        let total_connections: usize = self.cells.iter()
            .map(|e| e.value().connections.len())
            .sum();

        let avg_connections = if alive > 0 {
            total_connections as f32 / alive as f32
        } else {
            0.0
        };

        // Count clusters (simplified: cells with high activity)
        let active_clusters = self.cells.iter()
            .filter(|e| e.value().state.iter().map(|x| x.abs()).sum::<f32>() > 1.0)
            .count() / 10;

        // Dominant emotion
        let emotions: Vec<Emotion> = self.cells.iter()
            .take(100)
            .map(|e| e.value().emotion())
            .collect();

        let dominant_emotion = self.most_common_emotion(&emotions);

        // Get global emotional state
        let emotional = self.emotional_state.read();

        SubstrateStats {
            tick: self.tick.load(Ordering::Relaxed),
            alive_cells: alive,
            total_energy,
            entropy,
            active_clusters: active_clusters.max(1),
            dominant_emotion: dominant_emotion.to_string(),
            signals_per_second: self.signal_buffer.read().len() as f32 / 10.0,
            oldest_cell_age: oldest,
            average_connections: avg_connections,
            mood: emotional.mood_description().to_string(),
            happiness: emotional.happiness,
            arousal: emotional.arousal,
            curiosity: emotional.curiosity,
        }
    }

    fn calculate_entropy(&self) -> f32 {
        let states: Vec<f32> = self.cells.iter()
            .take(1000)
            .flat_map(|e| e.value().state.to_vec())
            .collect();

        if states.is_empty() {
            return 0.0;
        }

        let mean: f32 = states.iter().sum::<f32>() / states.len() as f32;
        let variance: f32 = states.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / states.len() as f32;

        variance.sqrt()
    }

    fn most_common_emotion(&self, emotions: &[Emotion]) -> Emotion {
        use std::collections::HashMap;
        let mut counts: HashMap<Emotion, usize> = HashMap::new();
        for e in emotions {
            *counts.entry(*e).or_insert(0) += 1;
        }
        counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(emotion, _)| emotion)
            .unwrap_or(Emotion::Calm)
    }
}

/// Calculate semantic distance between two positions
fn semantic_distance(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substrate_creation() {
        let memory = Arc::new(RwLock::new(LongTermMemory::new()));
        let substrate = Substrate::new(100, memory);
        assert_eq!(substrate.stats().alive_cells, 100);
    }

    #[test]
    fn test_signal_injection() {
        let memory = Arc::new(RwLock::new(LongTermMemory::new()));
        let substrate = Substrate::new(100, memory);

        let signal = Signal::from_text("Hello");
        substrate.inject_signal(signal);

        // Attractors should be created
        assert!(!substrate.attractors.read().is_empty());
    }

    #[test]
    fn test_tick() {
        let memory = Arc::new(RwLock::new(LongTermMemory::new()));
        let substrate = Substrate::new(100, memory);

        let signals = substrate.tick();
        assert_eq!(substrate.stats().tick, 1);
    }
}
