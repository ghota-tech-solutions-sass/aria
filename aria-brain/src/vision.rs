//! Visual Perception System for ARIA (Session 15)
//!
//! This module gives ARIA the ability to "see" images.
//! Images are converted into semantic vectors that can be processed
//! by the substrate, just like text signals.
//!
//! Features extracted:
//! - Color information (dominant colors, warmth, saturation)
//! - Brightness and contrast
//! - Complexity (edge density, texture)
//! - Spatial distribution (where is the subject?)
//! - Emotional tone (warm/cold, bright/dark)

use image::{DynamicImage, GenericImageView, Pixel};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Visual features extracted from an image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualFeatures {
    // === Color features (0-7) ===
    /// Average red component (0-1)
    pub avg_red: f32,
    /// Average green component (0-1)
    pub avg_green: f32,
    /// Average blue component (0-1)
    pub avg_blue: f32,
    /// Color variance (how diverse are the colors)
    pub color_variance: f32,
    /// Warmth (-1 cold to 1 warm)
    pub warmth: f32,
    /// Saturation (0 gray to 1 vivid)
    pub saturation: f32,
    /// Dominant hue (0-1 mapped from 0-360 degrees)
    pub dominant_hue: f32,
    /// Color complexity (number of distinct color clusters)
    pub color_complexity: f32,

    // === Brightness features (8-11) ===
    /// Average brightness (0-1)
    pub brightness: f32,
    /// Contrast (brightness variance)
    pub contrast: f32,
    /// Dark ratio (% of dark pixels)
    pub dark_ratio: f32,
    /// Bright ratio (% of bright pixels)
    pub bright_ratio: f32,

    // === Texture features (12-15) ===
    /// Edge density (0-1, how many edges)
    pub edge_density: f32,
    /// Smoothness (0-1, inverse of texture)
    pub smoothness: f32,
    /// Horizontal patterns
    pub horizontal_energy: f32,
    /// Vertical patterns
    pub vertical_energy: f32,

    // === Spatial features (16-23) ===
    /// Brightness in top-left quadrant
    pub quadrant_tl: f32,
    /// Brightness in top-right quadrant
    pub quadrant_tr: f32,
    /// Brightness in bottom-left quadrant
    pub quadrant_bl: f32,
    /// Brightness in bottom-right quadrant
    pub quadrant_br: f32,
    /// Center brightness (middle region)
    pub center_brightness: f32,
    /// Edge brightness (outer region)
    pub edge_brightness: f32,
    /// Horizontal balance (-1 left to 1 right)
    pub horizontal_balance: f32,
    /// Vertical balance (-1 top to 1 bottom)
    pub vertical_balance: f32,

    // === Emotional/semantic features (24-31) ===
    /// Estimated emotional valence (-1 sad to 1 happy)
    pub emotional_valence: f32,
    /// Estimated arousal (0 calm to 1 exciting)
    pub arousal: f32,
    /// Nature score (green/blue dominant)
    pub nature_score: f32,
    /// Urban score (gray/brown dominant)
    pub urban_score: f32,
    /// Face likelihood (skin tones present)
    pub face_likelihood: f32,
    /// Sky likelihood (blue at top)
    pub sky_likelihood: f32,
    /// Symmetry score
    pub symmetry: f32,
    /// Overall visual interest
    pub visual_interest: f32,
}

impl Default for VisualFeatures {
    fn default() -> Self {
        Self {
            avg_red: 0.5,
            avg_green: 0.5,
            avg_blue: 0.5,
            color_variance: 0.0,
            warmth: 0.0,
            saturation: 0.5,
            dominant_hue: 0.0,
            color_complexity: 0.5,
            brightness: 0.5,
            contrast: 0.5,
            dark_ratio: 0.0,
            bright_ratio: 0.0,
            edge_density: 0.0,
            smoothness: 1.0,
            horizontal_energy: 0.0,
            vertical_energy: 0.0,
            quadrant_tl: 0.5,
            quadrant_tr: 0.5,
            quadrant_bl: 0.5,
            quadrant_br: 0.5,
            center_brightness: 0.5,
            edge_brightness: 0.5,
            horizontal_balance: 0.0,
            vertical_balance: 0.0,
            emotional_valence: 0.0,
            arousal: 0.5,
            nature_score: 0.0,
            urban_score: 0.0,
            face_likelihood: 0.0,
            sky_likelihood: 0.0,
            symmetry: 0.5,
            visual_interest: 0.5,
        }
    }
}

impl VisualFeatures {
    /// Convert features to a 32-dimensional vector for the substrate
    pub fn to_vector(&self) -> [f32; 32] {
        [
            // Color features (0-7)
            self.avg_red,
            self.avg_green,
            self.avg_blue,
            self.color_variance,
            self.warmth * 0.5 + 0.5, // Normalize to 0-1
            self.saturation,
            self.dominant_hue,
            self.color_complexity,
            // Brightness features (8-11)
            self.brightness,
            self.contrast,
            self.dark_ratio,
            self.bright_ratio,
            // Texture features (12-15)
            self.edge_density,
            self.smoothness,
            self.horizontal_energy,
            self.vertical_energy,
            // Spatial features (16-23)
            self.quadrant_tl,
            self.quadrant_tr,
            self.quadrant_bl,
            self.quadrant_br,
            self.center_brightness,
            self.edge_brightness,
            self.horizontal_balance * 0.5 + 0.5, // Normalize to 0-1
            self.vertical_balance * 0.5 + 0.5,
            // Emotional features (24-31)
            self.emotional_valence * 0.5 + 0.5,
            self.arousal,
            self.nature_score,
            self.urban_score,
            self.face_likelihood,
            self.sky_likelihood,
            self.symmetry,
            self.visual_interest,
        ]
    }

    /// Get a textual description of what ARIA "sees"
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();

        // Brightness description
        if self.brightness > 0.7 {
            parts.push("lumineux");
        } else if self.brightness < 0.3 {
            parts.push("sombre");
        }

        // Color description
        if self.warmth > 0.3 {
            parts.push("chaud");
        } else if self.warmth < -0.3 {
            parts.push("froid");
        }

        if self.saturation > 0.6 {
            parts.push("coloré");
        } else if self.saturation < 0.2 {
            parts.push("gris");
        }

        // Complexity
        if self.edge_density > 0.5 {
            parts.push("complexe");
        } else if self.smoothness > 0.7 {
            parts.push("simple");
        }

        // Nature/urban
        if self.nature_score > 0.5 {
            parts.push("naturel");
        } else if self.urban_score > 0.5 {
            parts.push("urbain");
        }

        // Face
        if self.face_likelihood > 0.5 {
            parts.push("visage?");
        }

        // Emotional
        if self.emotional_valence > 0.3 {
            parts.push("joyeux");
        } else if self.emotional_valence < -0.3 {
            parts.push("triste");
        }

        if parts.is_empty() {
            "image".to_string()
        } else {
            parts.join(" ")
        }
    }
}

/// Visual perception system
pub struct VisualPerception {
    /// Target size for processing (smaller = faster)
    target_size: u32,
}

impl Default for VisualPerception {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualPerception {
    pub fn new() -> Self {
        Self {
            target_size: 64, // Process at 64x64 for speed
        }
    }

    /// Load and process an image from a file path
    pub fn process_file<P: AsRef<Path>>(&self, path: P) -> Result<VisualFeatures, String> {
        let img = image::open(path).map_err(|e| format!("Failed to open image: {}", e))?;
        Ok(self.process_image(&img))
    }

    /// Process an image from bytes
    pub fn process_bytes(&self, bytes: &[u8]) -> Result<VisualFeatures, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("Failed to decode image: {}", e))?;
        Ok(self.process_image(&img))
    }

    /// Process an image from base64 string
    pub fn process_base64(&self, b64: &str) -> Result<VisualFeatures, String> {
        // Remove data URL prefix if present
        let b64_clean = if b64.contains(',') {
            b64.split(',').nth(1).unwrap_or(b64)
        } else {
            b64
        };

        let bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            b64_clean,
        ).map_err(|e| format!("Failed to decode base64: {}", e))?;

        self.process_bytes(&bytes)
    }

    /// Process a DynamicImage and extract features
    pub fn process_image(&self, img: &DynamicImage) -> VisualFeatures {
        // Resize for faster processing
        let img = img.resize_exact(
            self.target_size,
            self.target_size,
            image::imageops::FilterType::Triangle,
        );

        let (width, height) = img.dimensions();
        let total_pixels = (width * height) as f32;

        // Collect all pixel data
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut sum_brightness = 0.0f32;
        let mut dark_count = 0u32;
        let mut bright_count = 0u32;

        // Quadrant sums
        let half_w = width / 2;
        let half_h = height / 2;
        let mut quad_tl = 0.0f32;
        let mut quad_tr = 0.0f32;
        let mut quad_bl = 0.0f32;
        let mut quad_br = 0.0f32;
        let mut center_sum = 0.0f32;
        let mut edge_sum = 0.0f32;
        let mut center_count = 0u32;
        let mut edge_count = 0u32;

        // Edge detection (simple Sobel-like)
        let mut edge_sum_total = 0.0f32;
        let mut h_energy = 0.0f32;
        let mut v_energy = 0.0f32;

        // Skin tone detection
        let mut skin_pixels = 0u32;

        // Blue at top (sky detection)
        let mut top_blue = 0.0f32;
        let mut top_count = 0u32;

        // Green (nature detection)
        let mut green_dominance = 0.0f32;

        // Brightness variance for contrast
        let mut brightness_values: Vec<f32> = Vec::with_capacity((width * height) as usize);

        // Left/right balance
        let mut left_sum = 0.0f32;
        let mut right_sum = 0.0f32;
        let mut top_sum = 0.0f32;
        let mut bottom_sum = 0.0f32;

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let channels = pixel.channels();
                let r = channels[0] as f32 / 255.0;
                let g = channels[1] as f32 / 255.0;
                let b = channels[2] as f32 / 255.0;

                // Basic sums
                sum_r += r;
                sum_g += g;
                sum_b += b;

                // Brightness (perceived luminance)
                let brightness = 0.299 * r + 0.587 * g + 0.114 * b;
                sum_brightness += brightness;
                brightness_values.push(brightness);

                if brightness < 0.3 {
                    dark_count += 1;
                }
                if brightness > 0.7 {
                    bright_count += 1;
                }

                // Quadrants
                if x < half_w && y < half_h {
                    quad_tl += brightness;
                } else if x >= half_w && y < half_h {
                    quad_tr += brightness;
                } else if x < half_w && y >= half_h {
                    quad_bl += brightness;
                } else {
                    quad_br += brightness;
                }

                // Center vs edge
                let center_margin = width / 4;
                if x > center_margin && x < width - center_margin
                    && y > center_margin && y < height - center_margin
                {
                    center_sum += brightness;
                    center_count += 1;
                } else {
                    edge_sum += brightness;
                    edge_count += 1;
                }

                // Balance
                if x < half_w {
                    left_sum += brightness;
                } else {
                    right_sum += brightness;
                }
                if y < half_h {
                    top_sum += brightness;
                } else {
                    bottom_sum += brightness;
                }

                // Skin tone detection (approximate)
                if r > 0.4 && g > 0.25 && b > 0.15
                    && r > g && g > b
                    && (r - g) < 0.3
                {
                    skin_pixels += 1;
                }

                // Sky detection (blue at top)
                if y < height / 3 {
                    top_blue += b - (r + g) / 2.0;
                    top_count += 1;
                }

                // Nature detection (green dominance)
                if g > r && g > b {
                    green_dominance += g - (r + b) / 2.0;
                }

                // Simple edge detection (gradient magnitude)
                if x > 0 && y > 0 {
                    let prev_x = img.get_pixel(x - 1, y);
                    let prev_y = img.get_pixel(x, y - 1);
                    let prev_x_brightness = {
                        let c = prev_x.channels();
                        0.299 * c[0] as f32 / 255.0 + 0.587 * c[1] as f32 / 255.0 + 0.114 * c[2] as f32 / 255.0
                    };
                    let prev_y_brightness = {
                        let c = prev_y.channels();
                        0.299 * c[0] as f32 / 255.0 + 0.587 * c[1] as f32 / 255.0 + 0.114 * c[2] as f32 / 255.0
                    };

                    let dx = (brightness - prev_x_brightness).abs();
                    let dy = (brightness - prev_y_brightness).abs();

                    edge_sum_total += (dx * dx + dy * dy).sqrt();
                    h_energy += dx;
                    v_energy += dy;
                }
            }
        }

        // Calculate averages
        let avg_red = sum_r / total_pixels;
        let avg_green = sum_g / total_pixels;
        let avg_blue = sum_b / total_pixels;
        let brightness = sum_brightness / total_pixels;

        // Color variance
        let mut color_var = 0.0f32;
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                let channels = pixel.channels();
                let r = channels[0] as f32 / 255.0;
                let g = channels[1] as f32 / 255.0;
                let b = channels[2] as f32 / 255.0;

                color_var += (r - avg_red).powi(2) + (g - avg_green).powi(2) + (b - avg_blue).powi(2);
            }
        }
        let color_variance = (color_var / total_pixels / 3.0).sqrt().min(1.0);

        // Brightness contrast (standard deviation)
        let brightness_variance: f32 = brightness_values.iter()
            .map(|b| (b - brightness).powi(2))
            .sum::<f32>() / total_pixels;
        let contrast = brightness_variance.sqrt().min(1.0) * 2.0; // Scale to 0-1

        // Warmth (red-blue balance)
        let warmth = (avg_red - avg_blue).clamp(-1.0, 1.0);

        // Saturation (distance from gray)
        let max_rgb = avg_red.max(avg_green).max(avg_blue);
        let min_rgb = avg_red.min(avg_green).min(avg_blue);
        let saturation = if max_rgb > 0.0 {
            (max_rgb - min_rgb) / max_rgb
        } else {
            0.0
        };

        // Dominant hue (simplified)
        let dominant_hue = if max_rgb == min_rgb {
            0.0
        } else if max_rgb == avg_red {
            ((avg_green - avg_blue) / (max_rgb - min_rgb) / 6.0 + 1.0) % 1.0
        } else if max_rgb == avg_green {
            (avg_blue - avg_red) / (max_rgb - min_rgb) / 6.0 + 1.0 / 3.0
        } else {
            (avg_red - avg_green) / (max_rgb - min_rgb) / 6.0 + 2.0 / 3.0
        };

        // Quadrant normalization
        let quad_pixels = (half_w * half_h) as f32;
        let quadrant_tl = quad_tl / quad_pixels;
        let quadrant_tr = quad_tr / quad_pixels;
        let quadrant_bl = quad_bl / quad_pixels;
        let quadrant_br = quad_br / quad_pixels;

        // Center/edge brightness
        let center_brightness = if center_count > 0 {
            center_sum / center_count as f32
        } else {
            brightness
        };
        let edge_brightness = if edge_count > 0 {
            edge_sum / edge_count as f32
        } else {
            brightness
        };

        // Balance
        let half_pixels = total_pixels / 2.0;
        let horizontal_balance = ((right_sum - left_sum) / half_pixels).clamp(-1.0, 1.0);
        let vertical_balance = ((bottom_sum - top_sum) / half_pixels).clamp(-1.0, 1.0);

        // Edge density
        let edge_pixels = ((width - 1) * (height - 1)) as f32;
        let edge_density = (edge_sum_total / edge_pixels * 4.0).min(1.0);
        let smoothness = 1.0 - edge_density;
        let horizontal_energy = (h_energy / edge_pixels * 4.0).min(1.0);
        let vertical_energy = (v_energy / edge_pixels * 4.0).min(1.0);

        // Semantic features
        let dark_ratio = dark_count as f32 / total_pixels;
        let bright_ratio = bright_count as f32 / total_pixels;

        // Nature score (green dominance)
        let nature_score = (green_dominance / total_pixels * 10.0).clamp(0.0, 1.0);

        // Urban score (gray, low saturation)
        let urban_score = if saturation < 0.2 && brightness > 0.3 && brightness < 0.7 {
            1.0 - saturation
        } else {
            0.0
        };

        // Face likelihood (skin tones)
        let face_likelihood = (skin_pixels as f32 / total_pixels * 3.0).min(1.0);

        // Sky likelihood
        let sky_likelihood = if top_count > 0 {
            (top_blue / top_count as f32 * 2.0 + 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Symmetry (compare left/right quadrants)
        let symmetry = 1.0 - ((quadrant_tl - quadrant_tr).abs() + (quadrant_bl - quadrant_br).abs()) / 2.0;

        // Emotional valence (warm, bright = happy)
        let emotional_valence = (warmth * 0.3 + (brightness - 0.5) * 0.4 + saturation * 0.3).clamp(-1.0, 1.0);

        // Arousal (contrast, saturation = exciting)
        let arousal = (contrast * 0.5 + saturation * 0.3 + edge_density * 0.2).min(1.0);

        // Visual interest (combination of factors)
        let visual_interest = (color_variance * 0.3 + contrast * 0.3 + edge_density * 0.2 + saturation * 0.2).min(1.0);

        // Color complexity (simplified)
        let color_complexity = (color_variance * 2.0 + saturation * 0.5).min(1.0);

        VisualFeatures {
            avg_red,
            avg_green,
            avg_blue,
            color_variance,
            warmth,
            saturation,
            dominant_hue,
            color_complexity,
            brightness,
            contrast,
            dark_ratio,
            bright_ratio,
            edge_density,
            smoothness,
            horizontal_energy,
            vertical_energy,
            quadrant_tl,
            quadrant_tr,
            quadrant_bl,
            quadrant_br,
            center_brightness,
            edge_brightness,
            horizontal_balance,
            vertical_balance,
            emotional_valence,
            arousal,
            nature_score,
            urban_score,
            face_likelihood,
            sky_likelihood,
            symmetry,
            visual_interest,
        }
    }
}

/// A visual signal to inject into the substrate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualSignal {
    /// The extracted features
    pub features: VisualFeatures,
    /// Vector representation (32D)
    pub vector: [f32; 32],
    /// Textual description
    pub description: String,
    /// Intensity of the signal (based on visual interest)
    pub intensity: f32,
    /// Source of the image (path, "webcam", "upload", etc.)
    pub source: String,
}

impl VisualSignal {
    /// Create a new visual signal from features
    pub fn new(features: VisualFeatures, source: String) -> Self {
        let vector = features.to_vector();
        let description = features.describe();
        let intensity = features.visual_interest * 0.5 + 0.3; // Base intensity + interest bonus

        Self {
            features,
            vector,
            description,
            intensity,
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_to_vector() {
        let features = VisualFeatures::default();
        let vector = features.to_vector();

        assert_eq!(vector.len(), 32);
        // All values should be in 0-1 range
        for v in &vector {
            assert!(*v >= 0.0 && *v <= 1.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_describe() {
        let mut features = VisualFeatures::default();

        // Test bright image
        features.brightness = 0.8;
        assert!(features.describe().contains("lumineux"));

        // Test dark image
        features.brightness = 0.2;
        assert!(features.describe().contains("sombre"));

        // Test warm image
        features.brightness = 0.5;
        features.warmth = 0.5;
        assert!(features.describe().contains("chaud"));
    }
}
