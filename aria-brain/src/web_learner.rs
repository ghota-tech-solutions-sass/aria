//! Web-based autonomous learning for ARIA
//!
//! This module allows ARIA to fetch web content, extract knowledge,
//! and learn from it autonomously.

use aria_core::TensionVector;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::mpsc;
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

    /// Add a new knowledge source
    pub fn add_source(&mut self, url: String, category: KnowledgeCategory) {
        self.sources.push(KnowledgeSource {
            url,
            category,
            last_fetched: None,
            fetch_count: 0,
        });
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

        // Split into sentences
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| s.len() > 20 && s.len() < 500)
            .take(50) // Limit per page
            .collect();

        for sentence in sentences {
            let clean = sentence.trim();
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

    /// Strip HTML tags from content
    fn strip_html_tags(html: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;

        for c in html.chars() {
            match c {
                '<' => {
                    in_tag = true;
                    // Check for script/style tags
                    let lower = html.to_lowercase();
                    if lower.contains("<script") { in_script = true; }
                    if lower.contains("<style") { in_style = true; }
                }
                '>' => {
                    in_tag = false;
                    if html.to_lowercase().contains("</script>") { in_script = false; }
                    if html.to_lowercase().contains("</style>") { in_style = false; }
                }
                _ if !in_tag && !in_script && !in_style => {
                    result.push(c);
                }
                _ => {}
            }
        }

        // Clean up whitespace
        result.split_whitespace().collect::<Vec<_>>().join(" ")
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

        // Fetch content
        match reqwest::get(&url).await {
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
                    let injections: Vec<PendingInjection> = items.iter()
                        .take(10) // Inject up to 10 items per fetch
                        .map(|item| PendingInjection {
                            tension: item.tension_vector,
                            label: format!("web:{}", item.content.chars().take(50).collect::<String>()),
                            intensity: 0.3,
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

    /// Find related knowledge by tension similarity
    pub fn find_related(&self, query_tension: &TensionVector, limit: usize) -> Vec<&KnowledgeItem> {
        let mut scored: Vec<(f32, &KnowledgeItem)> = self.knowledge.iter()
            .map(|item| {
                let similarity = Self::cosine_similarity(&item.tension_vector, query_tension);
                (similarity, item)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(_, item)| item).collect()
    }

    fn cosine_similarity(a: &TensionVector, b: &TensionVector) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a > 0.0 && mag_b > 0.0 {
            dot / (mag_a * mag_b)
        } else {
            0.0
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
