//! Health check endpoint

use warp::Filter;

/// GET /health - Simple health check
pub fn route() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("health").map(|| warp::reply::json(&serde_json::json!({"status": "alive"})))
}
