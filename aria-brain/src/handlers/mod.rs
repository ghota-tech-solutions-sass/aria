//! HTTP/WebSocket route handlers for ARIA Brain
//!
//! Organized by functional area:
//! - `health`: Health check endpoint
//! - `stats`: Substrate and population stats
//! - `memory`: Episodes, meta-learning, visual memory
//! - `learning`: Web learning, expression generation
//! - `vision`: Visual perception (POST)
//! - `websocket`: WebSocket communication

pub mod health;
pub mod learning;
pub mod memory;
pub mod stats;
pub mod vision;
pub mod websocket;

use std::sync::Arc;
use tokio::sync::broadcast;
use warp::Filter;

use crate::expression::ExpressionGenerator;
use crate::memory::LongTermMemory;
use crate::signal::Signal;
use crate::substrate::Substrate;
use crate::web_learner::WebLearner;

/// Shared application state for handlers
#[derive(Clone)]
pub struct AppState {
    pub substrate: Arc<parking_lot::RwLock<Substrate>>,
    pub memory: Arc<parking_lot::RwLock<LongTermMemory>>,
    pub web_learner: Arc<tokio::sync::RwLock<WebLearner>>,
    pub expression_gen: Arc<parking_lot::RwLock<ExpressionGenerator>>,
}

/// Compose all routes into a single filter
pub fn routes(
    state: AppState,
    perception_tx: broadcast::Sender<Signal>,
    expression_tx: broadcast::Sender<Signal>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    // WebSocket route
    let ws = websocket::route(perception_tx.clone(), expression_tx);

    // Health check
    let health = health::route();

    // Stats endpoints
    let stats_routes = stats::routes(state.substrate.clone());

    // Memory endpoints
    let memory_routes = memory::routes(state.memory.clone());

    // Learning endpoints
    let learning_routes = learning::routes(
        state.web_learner.clone(),
        state.expression_gen.clone(),
        state.memory.clone(),
        state.substrate.clone(),
    );

    // Vision endpoint
    let vision = vision::route(
        perception_tx,
        state.memory.clone(),
        state.substrate.clone(),
    );

    // Compose all routes
    ws.or(health)
        .or(stats_routes)
        .or(memory_routes)
        .or(learning_routes)
        .or(vision)
}
