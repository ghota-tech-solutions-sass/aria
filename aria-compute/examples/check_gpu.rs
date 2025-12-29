//! Quick GPU check utility
use aria_compute::{gpu_available, device_info};

fn main() {
    println!("=== ARIA GPU Check ===\n");
    println!("GPU Available: {}", gpu_available());
    println!("\nDevices found:");
    for device in device_info() {
        println!("  🎮 {} ", device.name);
        println!("     Vendor: {}", device.vendor);
        println!("     Type: {}", device.device_type);
        println!("     Backend: {}", device.backend);
        println!();
    }
}
