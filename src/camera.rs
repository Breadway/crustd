use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::FourCC;

/// A single grayscale frame: raw luma bytes plus the dimensions they came from.
pub struct GrayFrame {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u8>,
}

impl GrayFrame {
    /// Mean luma across the whole frame — used both to normalize the RGB motion
    /// diff against global brightness changes, and as a cheap non-ML occupancy
    /// signal on IR frames (something close to the sensor reflects a lot more IR
    /// light back than an empty room, independent of whether a face is detected).
    pub fn mean_luma(&self) -> f32 {
        if self.luma.is_empty() {
            return 0.0;
        }
        self.luma.iter().map(|&b| b as u64).sum::<u64>() as f32 / self.luma.len() as f32
    }
}

/// Which physical sensor a device should be: used to auto-resolve a stable path
/// instead of trusting a possibly-renumbered `/dev/videoN` node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Rgb,
    Ir,
}

/// Resolves a configured device string to an actual path.
///
/// A value of `"auto:rgb"` / `"auto:ir"` probes the stable `/dev/v4l/by-path/*`
/// symlinks and picks by capture capability (supports MJPG/YUYV color formats =
/// RGB sensor; GREY-only = IR sensor) rather than trusting `/dev/videoN` numbering,
/// which isn't stable across reboots or external camera plug events. Any other
/// value is used as a literal path, for explicit overrides.
pub fn resolve_device(configured: &str, want: DeviceKind) -> Result<PathBuf> {
    if configured != "auto:rgb" && configured != "auto:ir" {
        return Ok(PathBuf::from(configured));
    }

    let by_path = Path::new("/dev/v4l/by-path");
    let entries = std::fs::read_dir(by_path)
        .with_context(|| format!("listing {by_path:?} for camera auto-detection"))?;

    let mut seen = std::collections::HashSet::new();
    for entry in entries.flatten() {
        let symlink_path = entry.path();
        let Ok(resolved) = std::fs::canonicalize(&symlink_path) else {
            continue;
        };
        if !seen.insert(resolved.clone()) {
            continue; // by-path lists each physical device under multiple aliases
        }

        let Ok(dev) = v4l::Device::with_path(&resolved) else {
            continue;
        };
        let Ok(formats) = Capture::enum_formats(&dev) else {
            continue;
        };

        let has_color = formats
            .iter()
            .any(|f| f.fourcc == FourCC::new(b"MJPG") || f.fourcc == FourCC::new(b"YUYV"));
        let has_grey = formats.iter().any(|f| f.fourcc == FourCC::new(b"GREY"));

        let matches = match want {
            DeviceKind::Rgb => has_color,
            DeviceKind::Ir => has_grey && !has_color,
        };
        if matches {
            return Ok(resolved);
        }
    }

    anyhow::bail!("no {want:?} camera found under {by_path:?}; set camera.rgb_device / camera.ir_device explicitly in crustd.toml")
}

/// Grabs one YUYV frame from the RGB camera and returns just the luma (Y) plane.
///
/// Opens the device, captures, and drops it before returning — nothing about this
/// camera is left streaming between polls.
pub fn grab_rgb_luma(device: &str, width: u32, height: u32) -> Result<GrayFrame> {
    let mut dev = v4l::Device::with_path(device)
        .with_context(|| format!("opening rgb camera {device}"))?;

    let mut fmt = Capture::format(&dev).context("reading current rgb format")?;
    fmt.width = width;
    fmt.height = height;
    fmt.fourcc = FourCC::new(b"YUYV");
    Capture::set_format(&dev, &fmt).context("setting rgb format")?;
    let fmt = Capture::format(&dev).context("reading confirmed rgb format")?;

    let mut stream = MmapStream::with_buffers(&mut dev, Type::VideoCapture, 2)
        .context("starting rgb capture stream")?;

    // First frame after (re)starting a stream is often stale/partially exposed; take the second.
    let _ = stream.next().context("priming rgb stream")?;
    let (buf, _meta) = stream.next().context("capturing rgb frame")?;

    let mut luma = Vec::with_capacity((fmt.width * fmt.height) as usize);
    for chunk in buf.chunks_exact(2) {
        luma.push(chunk[0]);
    }

    Ok(GrayFrame {
        width: fmt.width,
        height: fmt.height,
        luma,
    })
}

/// Grabs `count` frames from the IR camera in a single short burst and returns them.
///
/// The device is opened once for the whole burst and dropped immediately after —
/// callers must not hold the IR camera open outside of this function, since that's
/// what keeps the IR illuminator from flashing continuously.
pub fn grab_ir_burst(device: &str, count: u32) -> Result<Vec<GrayFrame>> {
    let mut dev =
        v4l::Device::with_path(device).with_context(|| format!("opening ir camera {device}"))?;

    let fmt = Capture::format(&dev).context("reading ir format")?;
    let mut stream = MmapStream::with_buffers(&mut dev, Type::VideoCapture, 2)
        .context("starting ir capture stream")?;

    // Discard the first frame post-open the same way as the RGB path.
    let _ = stream.next().context("priming ir stream")?;

    let mut frames = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (buf, _meta) = stream.next().context("capturing ir frame")?;
        frames.push(GrayFrame {
            width: fmt.width,
            height: fmt.height,
            luma: buf.to_vec(),
        });
    }

    Ok(frames)
}
