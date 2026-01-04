//! Visual perception endpoint

use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, warn};
use warp::Filter;

use crate::memory::LongTermMemory;
use crate::signal::{Signal, SignalType};
use crate::substrate::Substrate;
use crate::vision::{VisualPerception, VisualSignal};

/// POST /vision - Process base64 image
pub fn route(
    perception_tx: broadcast::Sender<Signal>,
    memory: Arc<parking_lot::RwLock<LongTermMemory>>,
    substrate: Arc<parking_lot::RwLock<Substrate>>,
) -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path("vision")
        .and(warp::post())
        .and(warp::body::json())
        .map(move |body: serde_json::Value| {
            // Extract base64 image from body
            let b64 = body.get("image").and_then(|v| v.as_str()).unwrap_or("");

            let source = body
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("upload")
                .to_string();

            // NOTE: Labels removed in Session 31 (Physical Intelligence)
            let _labels: Vec<String> = body
                .get("labels")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| s.to_string())
                        .collect()
                })
                .unwrap_or_default();

            if b64.is_empty() {
                return warp::reply::json(&serde_json::json!({
                    "error": "Missing 'image' field with base64 data"
                }));
            }

            // Process the image
            let vision_processor = VisualPerception::new();
            match vision_processor.process_base64(b64) {
                Ok(features) => {
                    let visual_signal = VisualSignal::new(features.clone(), source.clone());

                    // Get current tick from substrate
                    let current_tick = substrate.read().current_tick();

                    // Convert 32-element vector to fixed array
                    let mut signature = [0.0f32; 32];
                    for (i, v) in visual_signal.vector.iter().take(32).enumerate() {
                        signature[i] = *v;
                    }

                    // Store in visual memory and check for recognition
                    let (memory_id, is_new, recognition) = {
                        let mut mem = memory.write();
                        mem.see(
                            signature,
                            visual_signal.description.clone(),
                            source,
                            current_tick,
                            features.emotional_valence,
                        )
                    };

                    // Convert to internal signal and send to substrate
                    let signal = Signal {
                        content: visual_signal.vector.to_vec(),
                        intensity: visual_signal.intensity,
                        label: format!("vision:{}", visual_signal.description),
                        signal_type: SignalType::Visual,
                        timestamp: current_tick,
                    };

                    info!(
                        "👁️ VISION: {} (intensity: {:.2}, memory_id: {}, new: {})",
                        visual_signal.description, visual_signal.intensity, memory_id, is_new
                    );

                    if let Err(e) = perception_tx.send(signal) {
                        warn!("Failed to send visual signal: {}", e);
                    }

                    warp::reply::json(&serde_json::json!({
                        "success": true,
                        "description": visual_signal.description,
                        "intensity": visual_signal.intensity,
                        "memory": {
                            "id": memory_id,
                            "is_new": is_new,
                            "recognition": recognition
                        },
                        "features": {
                            "brightness": features.brightness,
                            "warmth": features.warmth,
                            "saturation": features.saturation,
                            "complexity": features.edge_density,
                            "emotional_valence": features.emotional_valence,
                            "nature_score": features.nature_score,
                            "face_likelihood": features.face_likelihood
                        }
                    }))
                }
                Err(e) => warp::reply::json(&serde_json::json!({
                    "error": format!("Failed to process image: {}", e)
                })),
            }
        })
}
