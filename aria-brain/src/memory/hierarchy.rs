//! Memory Hierarchy - Gemini suggestion
//!
//! Implements a proper memory hierarchy:
//! - Working Memory: Active thoughts (very short term, ~5-10 items)
//! - Short-Term Memory: Recent interactions (minutes)
//! - Long-Term Memory: Persistent knowledge (days/forever)
//!
//! Memory flows: Working -> Short-Term -> Long-Term
//! Consolidation happens during "sleep" (idle periods)

use serde::{Deserialize, Serialize};

/// Working memory capacity (like human ~7±2 items)
pub const WORKING_MEMORY_CAPACITY: usize = 7;

/// Short-term memory duration (in ticks)
pub const SHORT_TERM_DURATION: u64 = 3000; // ~1-2 minutes

/// Working memory item - currently active thought
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkingMemoryItem {
    /// Content type
    pub content: WorkingContent,
    /// Activation level (higher = more active)
    pub activation: f32,
    /// When this entered working memory
    pub entered_at: u64,
    /// Associated emotional valence
    pub emotional_valence: f32,
}

/// What can be in working memory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum WorkingContent {
    /// A word being processed
    Word(String),
    /// A visual signature being processed
    Visual([f32; 32]),
    /// An emotional state
    Emotion { happiness: f32, arousal: f32 },
    /// A goal being pursued
    Goal(String),
    /// A memory being recalled
    RecalledMemory(u64), // episode ID
}

/// Working memory - what ARIA is thinking about RIGHT NOW
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// Current items (limited capacity)
    items: Vec<WorkingMemoryItem>,
    /// Total items that have passed through
    pub total_processed: u64,
}

#[allow(dead_code)]
impl WorkingMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item to working memory
    /// If full, removes least activated item
    pub fn push(&mut self, content: WorkingContent, emotional_valence: f32, tick: u64) {
        let item = WorkingMemoryItem {
            content,
            activation: 1.0,
            entered_at: tick,
            emotional_valence,
        };

        // If at capacity, remove least activated
        if self.items.len() >= WORKING_MEMORY_CAPACITY {
            // Find minimum activation index
            let min_idx = self.items.iter()
                .enumerate()
                .min_by(|a, b| a.1.activation.partial_cmp(&b.1.activation).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.items.remove(min_idx);
        }

        self.items.push(item);
        self.total_processed += 1;
    }

    /// Decay activation levels (called each tick)
    pub fn decay(&mut self, decay_rate: f32) {
        for item in &mut self.items {
            item.activation *= 1.0 - decay_rate;
        }
        // Remove items with very low activation
        self.items.retain(|item| item.activation > 0.1);
    }

    /// Boost activation of an item (attention)
    pub fn attend(&mut self, content_matches: impl Fn(&WorkingContent) -> bool) {
        for item in &mut self.items {
            if content_matches(&item.content) {
                item.activation = (item.activation + 0.3).min(1.0);
            }
        }
    }

    /// Get most activated items
    pub fn most_active(&self, n: usize) -> Vec<&WorkingMemoryItem> {
        let mut sorted: Vec<&WorkingMemoryItem> = self.items.iter().collect();
        sorted.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap());
        sorted.truncate(n);
        sorted
    }

    /// Get all current items
    pub fn items(&self) -> &[WorkingMemoryItem] {
        &self.items
    }

    /// Check if a word is in working memory
    pub fn contains_word(&self, word: &str) -> bool {
        self.items.iter().any(|item| {
            matches!(&item.content, WorkingContent::Word(w) if w.to_lowercase() == word.to_lowercase())
        })
    }

    /// Get average emotional valence of working memory
    pub fn average_valence(&self) -> f32 {
        if self.items.is_empty() {
            return 0.0;
        }
        self.items.iter().map(|i| i.emotional_valence).sum::<f32>() / self.items.len() as f32
    }

    /// Clear working memory
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

/// Short-term memory item
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShortTermItem {
    /// Words from the interaction
    pub words: Vec<String>,
    /// Emotional context
    pub emotional_valence: f32,
    /// When this occurred
    pub timestamp: u64,
    /// Rehearsal count (repeated in mind = stronger)
    pub rehearsals: u32,
}

/// Short-term memory - recent interactions
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShortTermMemory {
    /// Recent items
    items: Vec<ShortTermItem>,
    /// Maximum capacity
    capacity: usize,
}

#[allow(dead_code)]
impl ShortTermMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::new(),
            capacity,
        }
    }

    /// Add an interaction to short-term memory
    pub fn record(&mut self, words: Vec<String>, emotional_valence: f32, tick: u64) {
        let item = ShortTermItem {
            words,
            emotional_valence,
            timestamp: tick,
            rehearsals: 0,
        };

        self.items.push(item);

        // Trim to capacity
        if self.items.len() > self.capacity {
            self.items.remove(0);
        }
    }

    /// Prune old items
    pub fn prune(&mut self, current_tick: u64) {
        self.items.retain(|item| {
            current_tick.saturating_sub(item.timestamp) < SHORT_TERM_DURATION
        });
    }

    /// Rehearse an item (strengthens it for consolidation)
    pub fn rehearse(&mut self, word: &str) {
        for item in &mut self.items {
            if item.words.iter().any(|w| w.to_lowercase() == word.to_lowercase()) {
                item.rehearsals += 1;
            }
        }
    }

    /// Get items worth consolidating to long-term memory
    /// High emotional valence or high rehearsal count
    pub fn consolidation_candidates(&self) -> Vec<&ShortTermItem> {
        self.items.iter()
            .filter(|item| item.emotional_valence.abs() > 0.5 || item.rehearsals > 2)
            .collect()
    }

    /// Get all recent words
    pub fn recent_words(&self) -> Vec<&String> {
        self.items.iter().flat_map(|item| &item.words).collect()
    }

    /// Clear short-term memory
    pub fn clear(&mut self) {
        self.items.clear();
    }
}
