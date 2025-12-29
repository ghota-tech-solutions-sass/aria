//! # Spatial Hashing
//!
//! Efficient neighbor lookup for cells in semantic space.
//!
//! Instead of checking all N cells for neighbors (O(N²)),
//! spatial hashing groups cells into grid regions, making
//! neighbor lookups O(1) on average.
//!
//! ## How It Works
//!
//! 1. Semantic space is divided into a grid of regions
//! 2. Each cell is assigned to a region based on its position
//! 3. To find neighbors, only check cells in the same/adjacent regions
//!
//! ## Benefits
//!
//! - O(1) neighbor lookup
//! - Cache-friendly memory access
//! - Enables region-based sleep/wake

use std::collections::HashMap;

use aria_core::POSITION_DIMS;

/// Spatial hash for efficient neighbor lookup
pub struct SpatialHash {
    /// Grid resolution (cells per dimension)
    #[allow(dead_code)]
    resolution: usize,

    /// Cell indices by region
    regions: HashMap<RegionKey, Vec<usize>>,

    /// Region size in semantic space
    region_size: f32,
}

/// Key for a region in the spatial hash
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegionKey {
    /// Coordinates in the grid (first 4 dimensions for efficiency)
    coords: [i32; 4],
}

impl SpatialHash {
    /// Create a new spatial hash with given resolution
    pub fn new(resolution: usize) -> Self {
        // Map semantic space [-1, 1] to grid
        let region_size = 2.0 / resolution as f32;

        Self {
            resolution,
            regions: HashMap::new(),
            region_size,
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.regions.clear();
    }

    /// Insert a cell into the hash
    pub fn insert(&mut self, cell_index: usize, position: &[f32; POSITION_DIMS]) {
        let key = self.position_to_key(position);
        self.regions.entry(key).or_default().push(cell_index);
    }

    /// Remove a cell from the hash
    pub fn remove(&mut self, cell_index: usize, position: &[f32; POSITION_DIMS]) {
        let key = self.position_to_key(position);
        if let Some(cells) = self.regions.get_mut(&key) {
            cells.retain(|&i| i != cell_index);
            if cells.is_empty() {
                self.regions.remove(&key);
            }
        }
    }

    /// Update a cell's position in the hash
    pub fn update(
        &mut self,
        cell_index: usize,
        old_position: &[f32; POSITION_DIMS],
        new_position: &[f32; POSITION_DIMS],
    ) {
        let old_key = self.position_to_key(old_position);
        let new_key = self.position_to_key(new_position);

        if old_key != new_key {
            self.remove(cell_index, old_position);
            self.insert(cell_index, new_position);
        }
    }

    /// Get all cell indices near a position
    pub fn get_nearby(&self, position: &[f32; POSITION_DIMS]) -> Vec<usize> {
        let key = self.position_to_key(position);
        let mut result = Vec::new();

        // Check current region and all neighbors (3^4 = 81 regions max)
        for d0 in -1..=1 {
            for d1 in -1..=1 {
                for d2 in -1..=1 {
                    for d3 in -1..=1 {
                        let neighbor_key = RegionKey {
                            coords: [
                                key.coords[0] + d0,
                                key.coords[1] + d1,
                                key.coords[2] + d2,
                                key.coords[3] + d3,
                            ],
                        };

                        if let Some(cells) = self.regions.get(&neighbor_key) {
                            result.extend(cells.iter().copied());
                        }
                    }
                }
            }
        }

        result
    }

    /// Get all cell indices in a specific region
    pub fn get_region(&self, key: &RegionKey) -> Option<&Vec<usize>> {
        self.regions.get(key)
    }

    /// Get all non-empty regions
    pub fn active_regions(&self) -> impl Iterator<Item = (&RegionKey, &Vec<usize>)> {
        self.regions.iter()
    }

    /// Get the number of active regions
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Get the total number of cells indexed
    pub fn cell_count(&self) -> usize {
        self.regions.values().map(|v| v.len()).sum()
    }

    /// Convert a position to a region key
    fn position_to_key(&self, position: &[f32; POSITION_DIMS]) -> RegionKey {
        RegionKey {
            coords: [
                ((position[0] + 1.0) / self.region_size).floor() as i32,
                ((position[1] + 1.0) / self.region_size).floor() as i32,
                ((position[2] + 1.0) / self.region_size).floor() as i32,
                ((position[3] + 1.0) / self.region_size).floor() as i32,
            ],
        }
    }

    /// Get the center position of a region
    pub fn region_center(&self, key: &RegionKey) -> [f32; POSITION_DIMS] {
        let mut center = [0.0f32; POSITION_DIMS];
        for (i, &coord) in key.coords.iter().enumerate() {
            center[i] = (coord as f32 + 0.5) * self.region_size - 1.0;
        }
        center
    }
}

/// Statistics about spatial distribution
#[derive(Debug, Default)]
pub struct SpatialStats {
    /// Total regions with cells
    pub active_regions: usize,

    /// Average cells per region
    pub avg_cells_per_region: f32,

    /// Maximum cells in a single region
    pub max_cells_in_region: usize,

    /// Regions with only 1 cell
    pub lonely_regions: usize,

    /// Regions with more than 100 cells
    pub crowded_regions: usize,
}

impl SpatialHash {
    /// Calculate statistics about the spatial distribution
    pub fn stats(&self) -> SpatialStats {
        let active_regions = self.regions.len();

        if active_regions == 0 {
            return SpatialStats::default();
        }

        let total_cells: usize = self.regions.values().map(|v| v.len()).sum();
        let max_cells = self.regions.values().map(|v| v.len()).max().unwrap_or(0);
        let lonely = self.regions.values().filter(|v| v.len() == 1).count();
        let crowded = self.regions.values().filter(|v| v.len() > 100).count();

        SpatialStats {
            active_regions,
            avg_cells_per_region: total_cells as f32 / active_regions as f32,
            max_cells_in_region: max_cells,
            lonely_regions: lonely,
            crowded_regions: crowded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_hash() {
        let mut hash = SpatialHash::new(10);

        // Insert some cells
        let pos1 = [0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pos2 = [0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pos3 = [-0.9, -0.9, -0.9, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        hash.insert(0, &pos1);
        hash.insert(1, &pos2);
        hash.insert(2, &pos3);

        // pos1 and pos2 should be in the same region
        let nearby = hash.get_nearby(&pos1);
        assert!(nearby.contains(&0));
        assert!(nearby.contains(&1));

        // pos3 is far away
        assert!(!nearby.contains(&2));
    }

    #[test]
    fn test_spatial_stats() {
        let mut hash = SpatialHash::new(10);

        // Insert cells
        for i in 0..100 {
            let pos = [
                (i as f32 / 100.0) * 2.0 - 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ];
            hash.insert(i, &pos);
        }

        let stats = hash.stats();
        assert!(stats.active_regions > 0);
        assert!(stats.avg_cells_per_region > 0.0);
    }
}
