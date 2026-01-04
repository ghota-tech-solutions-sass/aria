//! Learning-related endpoints: web learning, expression, self-modification

use std::sync::Arc;
use warp::Filter;

use crate::expression::ExpressionGenerator;
use crate::memory::LongTermMemory;
use crate::substrate::Substrate;
use crate::web_learner::WebLearner;

/// Combine all learning routes
pub fn routes(
    web_learner: Arc<tokio::sync::RwLock<WebLearner>>,
    expression_gen: Arc<parking_lot::RwLock<ExpressionGenerator>>,
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
    substrate: Arc<parking_lot::RwLock<Substrate>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    learn(web_learner)
        .or(express(expression_gen))
        .or(self_modification(memory, substrate))
}

/// GET /learn - Web learner stats
fn learn(
    web_learner: Arc<tokio::sync::RwLock<WebLearner>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("learn").and_then(move || {
        let wl = web_learner.clone();
        async move {
            let stats = wl.read().await.stats();
            Ok::<_, warp::Rejection>(warp::reply::json(&stats))
        }
    })
}

/// GET /express - Expression generator stats
fn express(
    expression_gen: Arc<parking_lot::RwLock<ExpressionGenerator>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("express").map(move || {
        let stats = expression_gen.read().stats();
        warp::reply::json(&stats)
    })
}

/// GET /self - Self-modification status
fn self_modification(
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
    substrate: Arc<parking_lot::RwLock<Substrate>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("self").map(move || {
        let mem = memory.read();
        let params = {
            let sub = substrate.read();
            let p = sub.stats();
            serde_json::json!({
                "emission_threshold": p.adaptive_emission_threshold,
                "response_probability": p.adaptive_response_probability,
                "spontaneity": p.adaptive_spontaneity,
                "feedback_positive": p.adaptive_feedback_positive,
                "feedback_negative": p.adaptive_feedback_negative
            })
        };

        let modifier = &mem.self_modifier;
        let recent_mods: Vec<serde_json::Value> = modifier
            .modification_history
            .iter()
            .rev()
            .take(10)
            .map(|m| {
                serde_json::json!({
                    "param": m.param.name(),
                    "from": m.current_value,
                    "to": m.new_value,
                    "reasoning": m.reasoning,
                    "confidence": m.confidence,
                    "tick": m.proposed_at,
                    "evaluated": m.evaluated,
                    "successful": m.was_successful
                })
            })
            .collect();

        warp::reply::json(&serde_json::json!({
            "self_modification": {
                "enabled": true,
                "total_modifications": modifier.total_modifications,
                "successful_modifications": modifier.successful_modifications,
                "success_rate": if modifier.total_modifications > 0 {
                    modifier.successful_modifications as f32 / modifier.total_modifications as f32
                } else { 0.0 },
                "last_check_tick": modifier.last_modification_tick,
                "check_interval": modifier.modification_interval
            },
            "current_params": params,
            "recent_modifications": recent_mods,
            "meta_learning": {
                "competence_level": mem.meta_learner.progress.competence_level,
                "learning_quality": mem.meta_learner.progress.learning_quality,
                "trend": format!("{:?}", mem.meta_learner.progress.trend),
                "total_evaluations": mem.meta_learner.total_evaluations
            }
        }))
    })
}
