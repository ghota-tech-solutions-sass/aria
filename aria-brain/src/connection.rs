//! Connection management for ARIA
//!
//! Handles WebSocket connections and message protocol.

use serde::{Deserialize, Serialize};

/// Messages that can be sent/received over the WebSocket
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    /// A signal (perception or expression)
    #[serde(rename = "signal")]
    Signal(crate::signal::Signal),

    /// Request for stats
    #[serde(rename = "stats")]
    StatsRequest,

    /// Stats response
    #[serde(rename = "stats_response")]
    StatsResponse(crate::substrate::SubstrateStats),

    /// Ping/pong for keepalive
    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "pong")]
    Pong,

    /// Control commands
    #[serde(rename = "command")]
    Command(Command),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum Command {
    /// Pause evolution
    #[serde(rename = "pause")]
    Pause,

    /// Resume evolution
    #[serde(rename = "resume")]
    Resume,

    /// Save memory now
    #[serde(rename = "save")]
    Save,

    /// Reset to initial state
    #[serde(rename = "reset")]
    Reset { keep_memory: bool },

    /// Inject energy
    #[serde(rename = "inject_energy")]
    InjectEnergy { amount: f32 },

    /// Change population target
    #[serde(rename = "set_population")]
    SetPopulation { target: usize },
}
