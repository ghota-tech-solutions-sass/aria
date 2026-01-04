//! WGSL shader templates for ARIA GPU backend
//!
//! Shaders are organized by functional category:
//! - `cell_update`: Core cell metabolism and state updates
//! - `signal`: Signal propagation and spatial signals
//! - `lifecycle`: Birth, death, and lifecycle management
//! - `prediction`: Prediction law (generate & evaluate)
//! - `hebbian`: Hebbian learning and spatial attraction
//! - `cluster`: Cluster statistics and hysteresis
//! - `dispatch`: Sparse dispatch and compaction
//! - `utility`: Sleeping drain and other utilities

pub mod cell_update;
pub mod cluster;
pub mod dispatch;
pub mod hebbian;
pub mod lifecycle;
pub mod prediction;
pub mod signal;
pub mod utility;

// Re-export all shader constants for backward compatibility
pub use cell_update::CELL_UPDATE_TEMPLATE;
pub use cluster::{CLUSTER_HYSTERESIS_SHADER, CLUSTER_STATS_SHADER, CLUSTER_SYNTHESIS_TEMPLATE};
pub use dispatch::{COMPACT_SHADER, PREPARE_DISPATCH_SHADER};
pub use hebbian::{HEBBIAN_ATTRACTION_SHADER, HEBBIAN_CENTROID_SHADER, HEBBIAN_SHADER};
pub use lifecycle::{BIRTH_SHADER, DEATH_SHADER, RESET_LIFECYCLE_COUNTERS_SHADER};
pub use prediction::{PREDICTION_EVALUATE_SHADER, PREDICTION_GENERATE_SHADER};
pub use signal::{SIGNAL_TEMPLATE, SPATIAL_SIGNAL_TEMPLATE};
pub use utility::SLEEPING_DRAIN_SHADER;
