//! Stats and substrate view endpoints

use std::sync::Arc;
use warp::Filter;

use crate::substrate::Substrate;

/// Combine all stats routes
pub fn routes(
    substrate: Arc<parking_lot::RwLock<Substrate>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    stats(substrate.clone())
        .or(substrate_view(substrate))
        .or(deprecated_words())
        .or(deprecated_associations())
        .or(deprecated_clusters())
}

/// GET /stats - Population and energy statistics
fn stats(
    substrate: Arc<parking_lot::RwLock<Substrate>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("stats").map(move || {
        let s = substrate.read().stats();
        warp::reply::json(&s)
    })
}

/// GET /substrate - Spatial activity view for TUI visualization
fn substrate_view(
    substrate: Arc<parking_lot::RwLock<Substrate>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("substrate").map(move || {
        let view = substrate.read().spatial_view();
        warp::reply::json(&view)
    })
}

/// GET /words - Deprecated in Session 31
fn deprecated_words() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone
{
    warp::path("words").map(move || {
        warp::reply::json(&serde_json::json!({
            "message": "Vocabulary removed in Session 31 (Physical Intelligence)",
            "total_words": 0,
            "words": []
        }))
    })
}

/// GET /associations - Deprecated in Session 31
fn deprecated_associations(
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("associations").map(move || {
        warp::reply::json(&serde_json::json!({
            "message": "Word associations removed in Session 31 (Physical Intelligence)",
            "total_associations": 0,
            "associations": []
        }))
    })
}

/// GET /clusters - Deprecated in Session 31
fn deprecated_clusters(
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("clusters").map(move || {
        warp::reply::json(&serde_json::json!({
            "message": "Semantic clusters removed in Session 31 (Physical Intelligence)",
            "total_clusters": 0,
            "clusters": []
        }))
    })
}
