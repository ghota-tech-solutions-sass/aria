//! # Error Types for ARIA
//!
//! Unified error handling across all ARIA crates.

use thiserror::Error;

/// Main error type for ARIA operations
#[derive(Error, Debug)]
pub enum AriaError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Cell-related error
    #[error("Cell error: {0}")]
    Cell(String),

    /// Signal processing error
    #[error("Signal error: {0}")]
    Signal(String),

    /// Compute backend error
    #[error("Compute error: {0}")]
    Compute(String),

    /// GPU-specific error
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Memory/storage error
    #[error("Memory error: {0}")]
    Memory(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// Cluster synchronization error
    #[error("Cluster sync error: {0}")]
    ClusterSync(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Cell not found
    #[error("Cell {0} not found")]
    CellNotFound(u64),

    /// Population limit reached
    #[error("Population limit reached: {current} >= {max}")]
    PopulationLimit { current: u64, max: u64 },

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Result type for ARIA operations
pub type AriaResult<T> = Result<T, AriaError>;

impl AriaError {
    /// Create a config error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create a compute error
    pub fn compute(msg: impl Into<String>) -> Self {
        Self::Compute(msg.into())
    }

    /// Create a GPU error
    pub fn gpu(msg: impl Into<String>) -> Self {
        Self::Gpu(msg.into())
    }

    /// Create a network error
    pub fn network(msg: impl Into<String>) -> Self {
        Self::Network(msg.into())
    }

    /// Create a memory error
    pub fn memory(msg: impl Into<String>) -> Self {
        Self::Memory(msg.into())
    }
}
