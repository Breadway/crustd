use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::FourCC;

/// Upper bound on how long a single `stream.next()` call may block waiting for a
/// frame. Without this, a wedged or contended device (another process holding the
/// sensor, a stalled UVC controller) blocks the calling thread forever — since
/// captures run inside `spawn_blocking`, that leaks a blocking-pool thread and can
/// stall the whole presence loop indefinitely.
const CAPTURE_TIMEOUT: Duration = Duration::from_millis(2000);

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
    let dev = v4l::Device::with_path(device)
        .with_context(|| format!("opening rgb camera {device}"))?;

    let requested_fourcc = FourCC::new(b"YUYV");
    let mut fmt = Capture::format(&dev).context("reading current rgb format")?;
    fmt.width = width;
    fmt.height = height;
    fmt.fourcc = requested_fourcc;
    Capture::set_format(&dev, &fmt).context("setting rgb format")?;
    let fmt = Capture::format(&dev).context("reading confirmed rgb format")?;

    // set_format is best-effort: the driver may silently substitute the closest
    // format/resolution it actually supports. Treating a substituted format as if
    // it were YUYV at the requested size would corrupt every downstream pixel
    // computation (or, for a resolution change, blow the motion-blob threshold out
    // of proportion) with nothing ever logging why. Fail loudly instead.
    if fmt.fourcc != requested_fourcc || fmt.width != width || fmt.height != height {
        anyhow::bail!(
            "rgb camera {device} does not support {width}x{height} {requested_fourcc}; \
             driver reported {}x{} {} instead",
            fmt.width,
            fmt.height,
            fmt.fourcc
        );
    }

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 2)
        .context("starting rgb capture stream")?;
    stream.set_timeout(CAPTURE_TIMEOUT);

    // First frame after (re)starting a stream is often stale/partially exposed; take the second.
    let _ = stream.next().context("priming rgb stream")?;
    let (buf, meta) = stream.next().context("capturing rgb frame")?;

    // The mmap buffer is sized for the driver's worst-case allocation, not the
    // actual frame — only `bytesused` bytes at its start are real pixel data for
    // this frame. Using the whole buffer would treat leftover memory from a prior
    // (possibly larger) allocation as image content.
    let expected = (fmt.width * fmt.height * 2) as usize;
    let bytesused = meta.bytesused as usize;
    if bytesused < expected {
        anyhow::bail!(
            "rgb frame from {device} truncated: got {bytesused} bytes, expected {expected} \
             for {}x{} YUYV",
            fmt.width,
            fmt.height
        );
    }
    let buf = &buf[..expected];

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
pub fn grab_ir_burst(device: &str, width: u32, height: u32, count: u32) -> Result<Vec<GrayFrame>> {
    let dev =
        v4l::Device::with_path(device).with_context(|| format!("opening ir camera {device}"))?;

    // V4L2 format is persistent device state — without explicitly setting it here,
    // this capture would silently inherit whatever pixel format/resolution the
    // previous consumer (another app, or a prior crustd run) last configured on the
    // node, and every downstream byte would be misinterpreted with no error raised.
    let requested_fourcc = FourCC::new(b"GREY");
    let mut fmt = Capture::format(&dev).context("reading current ir format")?;
    fmt.width = width;
    fmt.height = height;
    fmt.fourcc = requested_fourcc;
    Capture::set_format(&dev, &fmt).context("setting ir format")?;
    let fmt = Capture::format(&dev).context("reading confirmed ir format")?;

    if fmt.fourcc != requested_fourcc || fmt.width != width || fmt.height != height {
        anyhow::bail!(
            "ir camera {device} does not support {width}x{height} {requested_fourcc}; \
             driver reported {}x{} {} instead",
            fmt.width,
            fmt.height,
            fmt.fourcc
        );
    }

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, 2)
        .context("starting ir capture stream")?;
    stream.set_timeout(CAPTURE_TIMEOUT);

    // Discard the first frame post-open the same way as the RGB path.
    let _ = stream.next().context("priming ir stream")?;

    let expected = (fmt.width * fmt.height) as usize;
    let mut frames = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (buf, meta) = stream.next().context("capturing ir frame")?;
        let bytesused = meta.bytesused as usize;
        if bytesused < expected {
            anyhow::bail!(
                "ir frame from {device} truncated: got {bytesused} bytes, expected {expected} \
                 for {}x{} GREY",
                fmt.width,
                fmt.height
            );
        }
        frames.push(GrayFrame {
            width: fmt.width,
            height: fmt.height,
            luma: buf[..expected].to_vec(),
        });
    }

    Ok(frames)
}
