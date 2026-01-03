//! Memory Types - Social contexts for episodic memory
//!
//! NOTE: WordCategory and UsagePattern removed in Session 31 (Physical Intelligence)

use serde::{Deserialize, Serialize};

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
