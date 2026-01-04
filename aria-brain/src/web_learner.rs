//! Web-based autonomous learning for ARIA
//!
//! This module allows ARIA to fetch web content, extract knowledge,
//! and learn from it autonomously.

use aria_core::TensionVector;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{info, warn};

/// Maximum number of knowledge items to keep
const MAX_KNOWLEDGE_ITEMS: usize = 10_000;

/// Sources ARIA can learn from
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSource {
    pub url: String,
    pub category: KnowledgeCategory,
    pub last_fetched: Option<u64>,
    pub fetch_count: u32,
}

/// Categories of knowledge
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeCategory {
    Science,
    Philosophy,
    Language,
    Mathematics,
    Art,
    History,
    General,
}

/// A piece of extracted knowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeItem {
    pub content: String,
    pub source_url: String,
    pub category: KnowledgeCategory,
    pub tension_vector: TensionVector,
    pub learned_at: u64,
    pub reinforcement_count: u32,
}

/// Pending content to be injected into substrate
#[derive(Debug, Clone)]
pub struct PendingInjection {
    pub tension: TensionVector,
    pub label: String,
    pub intensity: f32,
}

/// Web learner state
#[derive(Debug, Serialize, Deserialize)]
pub struct WebLearner {
    /// Knowledge sources to learn from
    pub sources: Vec<KnowledgeSource>,
    /// Extracted knowledge items
    #[serde(default)]
    pub knowledge: VecDeque<KnowledgeItem>,
    /// Current learning tick
    pub learning_tick: u64,
    /// Total items learned
    pub total_learned: u64,
    /// Is currently fetching
    #[serde(skip)]
    pub is_fetching: bool,
    /// Pending injections queue
    #[serde(skip)]
    pub pending_injections: VecDeque<PendingInjection>,
}

impl Default for WebLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl WebLearner {
    pub fn new() -> Self {
        Self {
            sources: Self::default_sources(),
            knowledge: VecDeque::new(),
            learning_tick: 0,
            total_learned: 0,
            is_fetching: false,
            pending_injections: VecDeque::new(),
        }
    }

    /// Default knowledge sources - curated for learning
    fn default_sources() -> Vec<KnowledgeSource> {
        vec![
            // Simple Wikipedia for accessible knowledge
            KnowledgeSource {
                url: "https://simple.wikipedia.org/wiki/Special:Random".to_string(),
                category: KnowledgeCategory::General,
                last_fetched: None,
                fetch_count: 0,
            },
            // Wikiquote for wisdom
            KnowledgeSource {
                url: "https://en.wikiquote.org/wiki/Special:Random".to_string(),
                category: KnowledgeCategory::Philosophy,
                last_fetched: None,
                fetch_count: 0,
            },
        ]
    }

    /// Convert text to tension vector using semantic hashing
    pub fn text_to_tension(text: &str) -> TensionVector {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut tension = [0.0f32; 8];

        // Split into words and create semantic fingerprint
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, chunk) in words.chunks(3).enumerate() {
            let combined = chunk.join(" ");
            let mut hasher = DefaultHasher::new();
            combined.hash(&mut hasher);
            let hash = hasher.finish();

            // Map to [-1, 1] range
            for j in 0..8 {
                let byte = ((hash >> (j * 8)) & 0xFF) as f32;
                let contribution = (byte / 255.0) * 2.0 - 1.0;
                tension[j] += contribution * (1.0 / (i as f32 + 1.0).sqrt());
            }
        }

        // Normalize
        let magnitude: f32 = tension.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for t in &mut tension {
                *t /= magnitude;
            }
        }

        tension
    }

    /// Extract knowledge from HTML content
    fn extract_knowledge_from_html(html: &str, source_url: &str, category: &KnowledgeCategory) -> Vec<KnowledgeItem> {
        let mut items = Vec::new();

        // Simple HTML text extraction (remove tags)
        let text = Self::strip_html_tags(html);

        // Debug: log extracted text length
        info!("📖 Extracted {} chars from HTML ({} chars)", text.len(), html.len());

        // Split into sentences
        let all_sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?').collect();
        let sentences: Vec<&str> = all_sentences.iter()
            .filter(|s| s.len() > 20 && s.len() < 500)
            .take(50) // Limit per page
            .copied()
            .collect();

        info!("📖 Found {} sentences ({} after length filter)", all_sentences.len(), sentences.len());

        for sentence in sentences {
            let clean = sentence.trim();
            // Skip error messages, robot policy notices, code fragments, and CSS
            let lower = clean.to_lowercase();
            let has_css = clean.contains('{') || clean.contains('}')
                || lower.contains("font-weight")
                || lower.contains("line-height")
                || lower.contains("box-sizing")
                || lower.contains("text-align")
                || lower.contains("padding:")
                || lower.contains("margin:")
                || lower.contains("position:")
                || lower.contains("display:")
                || lower.contains("border:")
                || lower.contains("background:")
                || lower.contains("width:")
                || lower.contains("height:");

            if has_css
                || lower.contains("user-agent")
                || lower.contains("robot")
                || lower.contains("phabricator")
                || lower.contains("error")
                || lower.contains("blocked")
                || lower.contains("javascript")
                || lower.contains("stylesheet")
                || lower.contains("cookie")
                || lower.contains("privacy policy")
                || lower.contains("terms of use")
                || lower.contains("from wikipedia")  // Skip Wikipedia meta-content
                || lower.contains("wikipedia, the free")
                || lower.contains("wikimedia foundation")
                || lower.contains("creative commons")
                || lower.contains("edit this page")
                || lower.contains("last edited")
                || lower.contains("retrieved from")
                || clean.chars().filter(|c| c.is_alphabetic()).count() < 15  // Must have real words
            {
                continue;
            }
            if clean.len() > 20 {
                let tension_vector = Self::text_to_tension(clean);
                items.push(KnowledgeItem {
                    content: clean.to_string(),
                    source_url: source_url.to_string(),
                    category: category.clone(),
                    tension_vector,
                    learned_at: 0, // Will be set when added
                    reinforcement_count: 0,
                });
            }
        }

        items
    }

    /// Strip HTML tags from content - extracts clean text using scraper library
    fn strip_html_tags(html: &str) -> String {
        let document = Html::parse_document(html);

        // Select only content paragraphs (skip navigation, scripts, etc.)
        let content_selector = Selector::parse("p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote")
            .unwrap_or_else(|_| Selector::parse("p").unwrap());

        let mut texts = Vec::new();

        for element in document.select(&content_selector) {
            let text: String = element.text().collect::<Vec<_>>().join(" ");
            let cleaned = text.trim();
            if !cleaned.is_empty() {
                texts.push(cleaned.to_string());
            }
        }

        texts.join(" ")
    }

    /// Fetch and learn from a source (async)
    pub async fn fetch_and_learn(&mut self, source_idx: usize, current_tick: u64) -> Option<Vec<PendingInjection>> {
        if source_idx >= self.sources.len() {
            return None;
        }

        self.is_fetching = true;
        let source = &mut self.sources[source_idx];
        let url = source.url.clone();
        let category = source.category.clone();

        // Fetch content with browser-like User-Agent (bot detection bypass)
        let client = reqwest::Client::new();
        match client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0 (compatible; ARIA/0.9.7; Learning System)")
            .send()
            .await
        {
            Ok(response) => {
                if let Ok(html) = response.text().await {
                    // Extract knowledge
                    let mut items = Self::extract_knowledge_from_html(&html, &url, &category);

                    // Set timestamps
                    for item in &mut items {
                        item.learned_at = current_tick;
                    }

                    // Update source stats
                    source.last_fetched = Some(current_tick);
                    source.fetch_count += 1;

                    // Create injections for substrate
                    // Session 34: Reduced from 10 items @ 0.3 to 5 items @ 0.15 to prevent population explosion
                    let injections: Vec<PendingInjection> = items.iter()
                        .take(5) // Inject up to 5 items per fetch
                        .map(|item| PendingInjection {
                            tension: item.tension_vector,
                            label: format!("web:{}", item.content.chars().take(50).collect::<String>()),
                            intensity: 0.15, // Half intensity for web learning
                        })
                        .collect();

                    // Store knowledge
                    for item in items {
                        self.knowledge.push_back(item);
                        self.total_learned += 1;
                    }

                    // Trim if too large
                    while self.knowledge.len() > MAX_KNOWLEDGE_ITEMS {
                        self.knowledge.pop_front();
                    }

                    info!(
                        "📚 LEARNED from {}: {} new items (total: {})",
                        url,
                        injections.len(),
                        self.total_learned
                    );

                    self.is_fetching = false;
                    return Some(injections);
                }
            }
            Err(e) => {
                warn!("❌ Web fetch failed: {}", e);
            }
        }

        self.is_fetching = false;
        None
    }

    /// Get next pending injection
    pub fn get_next_injection(&mut self) -> Option<PendingInjection> {
        self.pending_injections.pop_front()
    }

    /// Queue injections
    pub fn queue_injections(&mut self, injections: Vec<PendingInjection>) {
        for inj in injections {
            self.pending_injections.push_back(inj);
        }
    }

    /// Tick - autonomous learning
    pub fn tick(&mut self) {
        self.learning_tick += 1;
    }

    /// Get stats
    pub fn stats(&self) -> WebLearnerStats {
        WebLearnerStats {
            sources_count: self.sources.len(),
            knowledge_count: self.knowledge.len(),
            total_learned: self.total_learned,
            pending_injections: self.pending_injections.len(),
            is_fetching: self.is_fetching,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WebLearnerStats {
    pub sources_count: usize,
    pub knowledge_count: usize,
    pub total_learned: u64,
    pub pending_injections: usize,
    pub is_fetching: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_to_tension() {
        let tension = WebLearner::text_to_tension("The quick brown fox jumps over the lazy dog");
        assert!(tension.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_strip_html() {
        let html = "<html><body><p>Hello world</p><script>evil()</script></body></html>";
        let text = WebLearner::strip_html_tags(html);
        assert!(text.contains("Hello world"));
        assert!(!text.contains("evil"));
    }
}
