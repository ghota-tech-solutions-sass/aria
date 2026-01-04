//! WebSocket communication handler

use futures::{SinkExt, StreamExt};
use tokio::sync::broadcast;
use tracing::{info, warn};
use warp::Filter;

use crate::signal::Signal;

/// WebSocket route at /aria
pub fn route(
    perception_tx: broadcast::Sender<Signal>,
    expression_tx: broadcast::Sender<Signal>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("aria").and(warp::ws()).map(move |ws: warp::ws::Ws| {
        let perception_tx = perception_tx.clone();
        let expression_rx = expression_tx.subscribe();

        ws.on_upgrade(move |socket| handle_connection(socket, perception_tx, expression_rx))
    })
}

/// Handle a WebSocket connection
async fn handle_connection(
    ws: warp::ws::WebSocket,
    perception_tx: broadcast::Sender<Signal>,
    mut expression_rx: broadcast::Receiver<Signal>,
) {
    let (mut ws_tx, mut ws_rx) = ws.split();

    info!("New connection established");

    // Task to forward expressions to the client
    let send_task = tokio::spawn(async move {
        while let Ok(signal) = expression_rx.recv().await {
            let json = serde_json::to_string(&signal).unwrap_or_default();
            if ws_tx.send(warp::ws::Message::text(json)).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages
    while let Some(result) = ws_rx.next().await {
        match result {
            Ok(msg) => {
                if let Ok(text) = msg.to_str() {
                    // Try to parse as a signal
                    if let Ok(signal) = serde_json::from_str::<Signal>(text) {
                        let _ = perception_tx.send(signal);
                    }
                }
            }
            Err(e) => {
                warn!("WebSocket error: {}", e);
                break;
            }
        }
    }

    send_task.abort();
    info!("Connection closed");
}
