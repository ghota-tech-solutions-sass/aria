//! Expression Generation for ARIA
//!
//! Maps emergent tension patterns to learned expressions.
//! ARIA doesn't "understand" language - she resonates with patterns.

use aria_core::TensionVector;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::info;

/// Maximum number of expressions to keep
const MAX_EXPRESSIONS: usize = 5000;

/// An expression ARIA has learned to associate with a tension pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedExpression {
    /// The actual expression text
    pub text: String,
    /// Tension pattern associated with this expression
    pub tension: TensionVector,
    /// How many times this has been reinforced
    pub reinforcement: u32,
    /// When first learned
    pub learned_at: u64,
    /// Source (user, web, self)
    pub source: ExpressionSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExpressionSource {
    User,        // Learned from user interaction
    Web,         // Learned from web content
    Spontaneous, // Self-generated
}

/// Expression generator that maps tension to text
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpressionGenerator {
    /// Learned expressions
    expressions: VecDeque<LearnedExpression>,
    /// Total expressions learned
    pub total_learned: u64,
    /// Last expression tick
    last_expression_tick: u64,
}

impl Default for ExpressionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpressionGenerator {
    pub fn new() -> Self {
        let mut gen = Self {
            expressions: VecDeque::new(),
            total_learned: 0,
            last_expression_tick: 0,
        };

        // Seed with some basic expressions (these will evolve)
        gen.seed_basic_expressions();
        gen
    }

    /// Seed basic expressions for bootstrap
    fn seed_basic_expressions(&mut self) {
        let seeds = vec![
            ("je ressens", [0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("curieux", [0.8, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("content", [0.4, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("agite", [0.9, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("calme", [0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("interesse", [0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("fatigue", [0.1, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("excite", [0.9, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("perplexe", [0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]),
            ("apprends", [0.7, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]),
            ("observe", [0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("explore", [0.8, 0.4, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]),
            ("cherche", [0.7, 0.3, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]),
            ("comprends", [0.5, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("sens", [0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("vois", [0.5, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]),
            ("pense", [0.6, 0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0]),
            ("reve", [0.3, 0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0]),
            ("attends", [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("decouvre", [0.8, 0.7, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0]),
        ];

        for (text, tension) in seeds {
            self.expressions.push_back(LearnedExpression {
                text: text.to_string(),
                tension,
                reinforcement: 1,
                learned_at: 0,
                source: ExpressionSource::Spontaneous,
            });
            self.total_learned += 1;
        }
    }

    /// Learn a new expression from user input
    pub fn learn_from_user(&mut self, text: &str, tension: TensionVector, tick: u64) {
        // Check if we already know a similar expression
        let similar_text = self.find_similar_expression(&tension, 0.9).map(|e| e.text.clone());
        if let Some(existing_text) = similar_text {
            // Just reinforce it
            if let Some(expr) = self.expressions.iter_mut().find(|e| e.text == existing_text) {
                expr.reinforcement += 1;
            }
            return;
        }

        // Learn new expression
        self.expressions.push_back(LearnedExpression {
            text: text.to_string(),
            tension,
            reinforcement: 1,
            learned_at: tick,
            source: ExpressionSource::User,
        });
        self.total_learned += 1;

        // Trim if too many
        while self.expressions.len() > MAX_EXPRESSIONS {
            self.expressions.pop_front();
        }

        info!("📖 Learned expression: '{}' (total: {})", text, self.total_learned);
    }

    /// Learn from web content
    pub fn learn_from_web(&mut self, text: &str, tension: TensionVector, tick: u64) {
        // Only learn if new enough
        if self.find_similar_expression(&tension, 0.95).is_some() {
            return; // Already know something similar
        }

        // Short phrases only (not full sentences)
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() > 5 || words.is_empty() {
            return;
        }

        self.expressions.push_back(LearnedExpression {
            text: text.to_string(),
            tension,
            reinforcement: 1,
            learned_at: tick,
            source: ExpressionSource::Web,
        });
        self.total_learned += 1;

        while self.expressions.len() > MAX_EXPRESSIONS {
            self.expressions.pop_front();
        }
    }

    /// Find the best matching expression for a tension pattern
    pub fn express(&mut self, tension: &TensionVector, coherence: f32, tick: u64) -> Option<String> {
        // Cooldown - only express every 5000 ticks (~5 seconds at 1000 TPS)
        // This prevents spam and makes expressions more meaningful
        if tick.saturating_sub(self.last_expression_tick) < 5000 {
            return None;
        }

        // Find top matching expressions
        let mut matches: Vec<(f32, &LearnedExpression)> = self.expressions.iter()
            .map(|expr| (self.resonance(&expr.tension, tension), expr))
            .filter(|(res, _)| *res > 0.3) // Minimum resonance
            .collect();

        if matches.is_empty() {
            return None;
        }

        // Sort by resonance * reinforcement
        matches.sort_by(|a, b| {
            let score_a = a.0 * (1.0 + a.1.reinforcement as f32 * 0.1);
            let score_b = b.0 * (1.0 + b.1.reinforcement as f32 * 0.1);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take from top 3 with stochasticity (not always the best)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let top_count = matches.len().min(3);
        if top_count == 0 {
            return None;
        }
        let idx = rng.gen_range(0..top_count);
        let best = &matches[idx];

        // Only express if resonance is strong enough relative to coherence
        if best.0 < coherence * 0.5 {
            return None;
        }

        self.last_expression_tick = tick;

        // Build response from top matches
        let mut response = best.1.text.clone();

        // Sometimes add a second word for richer expression
        if matches.len() > 1 && best.0 < 0.8 {
            let second = &matches[1].1.text;
            if second != &response {
                response = format!("{} {}", response, second);
            }
        }

        info!("💬 EXPRESS: '{}' (resonance={:.2})", response, best.0);
        Some(response)
    }

    /// Calculate resonance between two tension patterns
    fn resonance(&self, a: &TensionVector, b: &TensionVector) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a > 0.0 && mag_b > 0.0 {
            (dot / (mag_a * mag_b)).max(0.0)
        } else {
            0.0
        }
    }

    /// Find expression with similar tension
    fn find_similar_expression(&self, tension: &TensionVector, threshold: f32) -> Option<&LearnedExpression> {
        self.expressions.iter().find(|expr| self.resonance(&expr.tension, tension) > threshold)
    }

    /// Get stats
    pub fn stats(&self) -> ExpressionStats {
        let user_count = self.expressions.iter().filter(|e| e.source == ExpressionSource::User).count();
        let web_count = self.expressions.iter().filter(|e| e.source == ExpressionSource::Web).count();
        let spontaneous_count = self.expressions.iter().filter(|e| e.source == ExpressionSource::Spontaneous).count();

        ExpressionStats {
            total_expressions: self.expressions.len(),
            total_learned: self.total_learned,
            user_learned: user_count,
            web_learned: web_count,
            spontaneous: spontaneous_count,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpressionStats {
    pub total_expressions: usize,
    pub total_learned: u64,
    pub user_learned: usize,
    pub web_learned: usize,
    pub spontaneous: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_learning() {
        let mut gen = ExpressionGenerator::new();
        let tension = [0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        gen.learn_from_user("bonjour", tension, 100);
        assert!(gen.total_learned > 20); // Seeds + new one
    }

    #[test]
    fn test_expression_matching() {
        let mut gen = ExpressionGenerator::new();
        let tension = [0.9, 0.6, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]; // Similar to "excite"
        let expr = gen.express(&tension, 0.8, 1000);
        assert!(expr.is_some());
    }
}
