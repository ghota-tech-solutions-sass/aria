//! Memory Types - Word categories and social contexts
//!
//! These types define how ARIA categorizes and contextualizes words.

use serde::{Deserialize, Serialize};

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
