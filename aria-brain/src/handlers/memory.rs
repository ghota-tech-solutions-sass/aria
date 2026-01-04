//! Memory-related endpoints: episodes, meta-learning, visual stats

use std::sync::Arc;
use warp::Filter;

use crate::memory::LongTermMemory;

/// Combine all memory routes
pub fn routes(
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    episodes(memory.clone())
        .or(meta(memory.clone()))
        .or(visual_stats(memory))
}

/// GET /episodes - Episodic memories
fn episodes(
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("episodes").map(move || {
        let mem = memory.read();
        let episodes_info: Vec<serde_json::Value> = mem
            .episodes
            .iter()
            .rev()
            .take(50)
            .map(|ep| {
                serde_json::json!({
                    "id": ep.id,
                    "timestamp": ep.timestamp,
                    "real_time": ep.real_time,
                    "input": ep.input,
                    "response": ep.response,
                    "keywords": ep.keywords,
                    "category": format!("{:?}", ep.category),
                    "importance": ep.importance,
                    "recall_count": ep.recall_count,
                    "first_of_kind": ep.first_of_kind,
                    "emotion": {
                        "happiness": ep.emotion.happiness,
                        "arousal": ep.emotion.arousal,
                        "comfort": ep.emotion.comfort,
                        "curiosity": ep.emotion.curiosity
                    }
                })
            })
            .collect();

        let first_times: Vec<serde_json::Value> = mem
            .first_times
            .iter()
            .map(|(kind, id)| {
                serde_json::json!({
                    "kind": kind,
                    "episode_id": id
                })
            })
            .collect();

        warp::reply::json(&serde_json::json!({
            "total_episodes": mem.episodes.len(),
            "showing": episodes_info.len(),
            "first_times": first_times,
            "episodes": episodes_info
        }))
    })
}

/// GET /meta - Meta-learning progress
fn meta(
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("meta").map(move || {
        let mem = memory.read();
        let ml = &mem.meta_learner;

        let strategies: Vec<serde_json::Value> = ml
            .strategies
            .iter()
            .map(|(st, strategy)| {
                serde_json::json!({
                    "type": st.name(),
                    "usage_count": strategy.usage_count,
                    "avg_reward": strategy.avg_reward,
                    "best_reward": strategy.best_reward,
                    "recent_avg": strategy.recent_avg()
                })
            })
            .collect();

        let goals: Vec<serde_json::Value> = ml
            .goals
            .iter()
            .map(|goal| {
                serde_json::json!({
                    "id": goal.id,
                    "description": goal.description,
                    "progress": goal.progress,
                    "completed": goal.completed
                })
            })
            .collect();

        let progress = &ml.progress;

        warp::reply::json(&serde_json::json!({
            "total_evaluations": ml.total_evaluations,
            "current_strategy": ml.current_strategy.name(),
            "exploration_rate": ml.config.exploration_rate,
            "strategies": strategies,
            "goals": goals,
            "progress": {
                "learning_quality": progress.learning_quality,
                "competence_level": progress.competence_level,
                "trend": format!("{:?}", progress.trend),
                "recent_successes": progress.recent_successes,
                "recent_failures": progress.recent_failures,
                "status": progress.status_description()
            }
        }))
    })
}

/// GET /visual - Visual memory stats
fn visual_stats(
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("visual").map(move || {
        let mem = memory.read();
        let total_memories = mem.visual_stats();

        let recent_memories: Vec<serde_json::Value> = mem
            .visual_memories
            .iter()
            .rev()
            .take(10)
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "description": m.description,
                    "times_seen": m.times_seen,
                    "source": m.source
                })
            })
            .collect();

        warp::reply::json(&serde_json::json!({
            "total_visual_memories": total_memories,
            "recent_memories": recent_memories
        }))
    })
}
