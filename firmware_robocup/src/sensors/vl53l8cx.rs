//! VL53L8CX 8x8-zone ToF depth sensor: raw I2C platform layer.
//!
//! ev3dev-lang-rust has no built-in driver for this sensor (custom I2C
//! device), so it is accessed directly via Linux's `/dev/i2c-*` character
//! device through the `i2cdev` crate.
//!
//! IMPORTANT / SCOPE NOTE: ST's VL53L8CX requires uploading a substantial
//! firmware blob into the sensor's internal RAM at init and driving it
//! through a proprietary ranging protocol (the "Ultra Lite Driver", ULD).
//! Reimplementing that protocol from scratch in pure Rust is out of scope
//! here (thousands of lines, ST-provided binary firmware blob). What is
//! implemented below is the *platform layer* ST's own driver expects an
//! integrator to provide (`RdMulti`/`WrMulti`/`WrByte` over I2C at a
//! 16-bit register address) -- functionally equivalent to `platform.c` in
//! ST's C driver. The recommended path to a working ranging driver is to
//! FFI-bind ST's official `VL53L8CX_ULD` C driver against this platform
//! layer (or a thin C shim calling back into these functions), rather than
//! reimplementing the ranging firmware protocol by hand.

use i2cdev::core::I2CDevice;
use i2cdev::linux::{LinuxI2CDevice, LinuxI2CError};

use crate::config::DEPTH_MAX_RANGE_MM;

/// Default 7-bit I2C address per ST's datasheet (may be reprogrammed).
pub const VL53L8CX_DEFAULT_ADDRESS: u16 = 0x29;

pub struct Vl53l8cxPlatform {
    dev: LinuxI2CDevice,
}

impl Vl53l8cxPlatform {
    pub fn new(i2c_bus_path: &str, address: u16) -> Result<Self, LinuxI2CError> {
        let dev = LinuxI2CDevice::new(i2c_bus_path, address)?;
        Ok(Self { dev })
    }

    /// Write `data` to a 16-bit register address (ST platform.c `WrMulti`).
    pub fn write_multi(&mut self, reg: u16, data: &[u8]) -> Result<(), LinuxI2CError> {
        let mut buf = Vec::with_capacity(2 + data.len());
        buf.push((reg >> 8) as u8);
        buf.push((reg & 0xFF) as u8);
        buf.extend_from_slice(data);
        self.dev.write(&buf)
    }

    /// Read `out.len()` bytes starting at a 16-bit register address
    /// (ST platform.c `RdMulti`).
    pub fn read_multi(&mut self, reg: u16, out: &mut [u8]) -> Result<(), LinuxI2CError> {
        let addr_buf = [(reg >> 8) as u8, (reg & 0xFF) as u8];
        self.dev.write(&addr_buf)?;
        self.dev.read(out)
    }

    pub fn write_byte(&mut self, reg: u16, value: u8) -> Result<(), LinuxI2CError> {
        self.write_multi(reg, &[value])
    }
}

/// 8x8 zone ranging result, distances in millimeters. Zones with no valid
/// return are reported at `DEPTH_MAX_RANGE_MM` (matches the simulated
/// rangefinder's cutoff behavior in assets/robot_asset.py).
pub struct DepthFrame {
    pub distances_mm: [u16; 64],
}

impl Default for DepthFrame {
    fn default() -> Self {
        Self {
            distances_mm: [DEPTH_MAX_RANGE_MM; 64],
        }
    }
}

/// Placeholder high-level driver. `start_ranging`/`read_frame` must be
/// implemented on top of ST's ULD (see module docs) before this compiles
/// against real hardware behavior -- the platform I/O primitives above are
/// ready to be wired into that driver's callback table.
pub struct Vl53l8cx {
    platform: Vl53l8cxPlatform,
}

impl Vl53l8cx {
    pub fn new(platform: Vl53l8cxPlatform) -> Self {
        Self { platform }
    }

    pub fn platform_mut(&mut self) -> &mut Vl53l8cxPlatform {
        &mut self.platform
    }
}
