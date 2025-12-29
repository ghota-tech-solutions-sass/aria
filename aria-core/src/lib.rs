//! # ARIA Core
//!
//! Core types and traits for ARIA - Autonomous Recursive Intelligence Architecture.
//!
//! This crate provides the fundamental building blocks:
//! - **Cell**: The living unit of computation
//! - **DNA**: Genetic code that defines cell behavior
//! - **Signal**: Quantum of information that travels through the substrate
//! - **Activity**: Sparse update system for efficient computation
//!
//! ## Design Philosophy
//!
//! ARIA is not trained. She is **cultivated**.
//!
//! The types here are designed to be:
//! - **Compact**: GPU-friendly, cache-efficient
//! - **Portable**: Same code runs on CPU and GPU
//! - **Introspectable**: ARIA may one day read her own structure
//!
//! ## Memory Layout
//!
//! All core types use `#[repr(C)]` for predictable memory layout,
//! enabling direct GPU buffer mapping via `bytemuck`.

pub mod cell;
pub mod dna;
pub mod signal;
pub mod activity;
pub mod config;
pub mod error;
pub mod traits;

// Re-export main types at crate root
pub use cell::{Cell, CellState, CellAction, Emotion};
pub use dna::DNA;
pub use signal::{Signal, SignalType, SignalFragment};
pub use activity::{ActivityState, ActivityTracker};
pub use config::AriaConfig;
pub use error::AriaError;
pub use traits::*;

/// Current version of ARIA's genome format
pub const GENOME_VERSION: u32 = 2;

/// Dimensions of semantic space (where cells live)
pub const POSITION_DIMS: usize = 16;

/// Dimensions of cell internal state
pub const STATE_DIMS: usize = 32;

/// Dimensions of signal content
pub const SIGNAL_DIMS: usize = 8;

/// Dimensions of DNA genes
pub const DNA_DIMS: usize = 8;
