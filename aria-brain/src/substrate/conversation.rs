//! Conversation context tracking for ARIA
//!
//! ARIA maintains context about the current conversation:
//! - Recent exchanges (input/response pairs)
//! - Topic words that recur (for context boosting)
//! - Social context (greeting, farewell, etc.)

use crate::memory::SocialContext;

/// A single exchange in conversation (input + optional response)
#[derive(Clone, Debug)]
pub struct ConversationExchange {
    /// The user's input text
    pub input: String,
    /// ARIA's response (if any)
    pub response: Option<String>,
    /// Significant words from the input
    pub input_words: Vec<String>,
    /// Emotional tone of the input (-1.0 to 1.0)
    pub emotional_tone: f32,
    /// Tick when this exchange occurred
    pub tick: u64,
}

/// Conversation tracking - ARIA follows the discussion thread
#[derive(Clone, Debug, Default)]
pub struct ConversationContext {
    /// Recent exchanges (most recent first)
    pub exchanges: Vec<ConversationExchange>,
    /// Words that recur in conversation (word, count)
    pub topic_words: Vec<(String, u32)>,
    /// Whether we're currently in a conversation
    pub in_conversation: bool,
    /// Tick of the last exchange
    pub last_exchange_tick: u64,
    /// Current social context
    pub current_social_context: SocialContext,
    /// Number of exchanges in this conversation
    pub exchange_count: u32,
}

impl ConversationContext {
    /// Maximum number of exchanges to keep in memory
    pub const MAX_EXCHANGES: usize = 5;
    /// Ticks before considering a conversation "timed out"
    pub const CONVERSATION_TIMEOUT: u64 = 3000; // ~30 seconds

    /// Create a new conversation context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a new input to the conversation
    pub fn add_input(
        &mut self,
        input: &str,
        words: Vec<String>,
        tone: f32,
        tick: u64,
        context: SocialContext,
    ) {
        // Check for conversation timeout - start fresh if too long since last exchange
        if tick.saturating_sub(self.last_exchange_tick) > Self::CONVERSATION_TIMEOUT {
            self.exchanges.clear();
            self.topic_words.clear();
            self.exchange_count = 0;
            tracing::info!("NEW CONVERSATION started");
        }

        self.in_conversation = true;
        self.last_exchange_tick = tick;
        self.current_social_context = context;
        self.exchange_count += 1;

        // Add exchange at the front (most recent first)
        self.exchanges.insert(0, ConversationExchange {
            input: input.to_string(),
            response: None,
            input_words: words.clone(),
            emotional_tone: tone,
            tick,
        });

        // Keep only the last MAX_EXCHANGES
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

        // Sort by frequency and keep top 10
        self.topic_words.sort_by(|a, b| b.1.cmp(&a.1));
        self.topic_words.truncate(10);
    }

    /// Add a response to the current exchange
    pub fn add_response(&mut self, response: &str) {
        if let Some(exchange) = self.exchanges.first_mut() {
            exchange.response = Some(response.to_string());
        }
    }

    /// Get context words with their relevance scores
    /// More recent words have higher scores
    pub fn get_context_words(&self) -> Vec<(String, f32)> {
        let mut context: Vec<(String, f32)> = Vec::new();

        // Most recent exchange gets full weight
        if let Some(last) = self.exchanges.first() {
            for word in &last.input_words {
                context.push((word.clone(), 1.0));
            }
        }

        // Older exchanges get exponentially decaying weight
        for (i, exchange) in self.exchanges.iter().skip(1).enumerate() {
            let decay = 0.5_f32.powi(i as i32 + 1);
            for word in &exchange.input_words {
                if !context.iter().any(|(w, _)| w == word) {
                    context.push((word.clone(), decay));
                }
            }
        }

        // Topic words get bonus based on frequency
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

    /// Get just the topic words as strings (for memory recall)
    pub fn get_topic_words(&self) -> Vec<String> {
        self.topic_words.iter().map(|(w, _)| w.clone()).collect()
    }

    /// Check if this is the start of a conversation (exchanges 1-2)
    pub fn is_conversation_start(&self) -> bool {
        self.exchange_count <= 2
    }

    /// Get the current social context
    pub fn get_social_context(&self) -> SocialContext {
        self.current_social_context
    }

    /// Get the last input words (if any)
    pub fn last_input_words(&self) -> Option<&[String]> {
        self.exchanges.first().map(|e| e.input_words.as_slice())
    }

    /// Get the last response (if any)
    pub fn last_response(&self) -> Option<&str> {
        self.exchanges.first().and_then(|e| e.response.as_deref())
    }

    /// Check if we're in an active conversation
    pub fn is_active(&self, current_tick: u64) -> bool {
        self.in_conversation
            && current_tick.saturating_sub(self.last_exchange_tick) < Self::CONVERSATION_TIMEOUT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_context() {
        let mut ctx = ConversationContext::new();

        ctx.add_input(
            "hello world",
            vec!["hello".into(), "world".into()],
            0.5,
            100,
            SocialContext::Greeting,
        );

        assert!(ctx.is_conversation_start());
        assert_eq!(ctx.get_social_context(), SocialContext::Greeting);

        let words = ctx.get_context_words();
        assert!(!words.is_empty());
        assert!(words.iter().any(|(w, _)| w == "hello"));
    }

    #[test]
    fn test_topic_tracking() {
        let mut ctx = ConversationContext::new();

        // Add same word multiple times
        ctx.add_input("moka", vec!["moka".into()], 0.5, 100, SocialContext::General);
        ctx.add_input("moka chat", vec!["moka".into(), "chat".into()], 0.5, 200, SocialContext::General);
        ctx.add_input("moka dort", vec!["moka".into(), "dort".into()], 0.5, 300, SocialContext::General);

        let topics = ctx.get_topic_words();
        assert!(topics.contains(&"moka".to_string()));

        // moka should be first (most frequent)
        assert_eq!(topics.first(), Some(&"moka".to_string()));
    }

    #[test]
    fn test_conversation_timeout() {
        let mut ctx = ConversationContext::new();

        ctx.add_input("hello", vec!["hello".into()], 0.5, 100, SocialContext::Greeting);
        assert_eq!(ctx.exchange_count, 1);

        // Timeout (more than 3000 ticks later)
        ctx.add_input("new topic", vec!["new".into(), "topic".into()], 0.0, 4000, SocialContext::General);

        // Should have reset
        assert_eq!(ctx.exchange_count, 1);
        assert!(!ctx.get_topic_words().contains(&"hello".to_string()));
    }
}
