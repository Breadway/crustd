use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::FourCC;

use crate::config::EmitterConfig;

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

/// UVC request code from <linux/usb/video.h>: bRequest for UVC_SET_CUR.
const UVC_SET_CUR: u8 = 0x01;

/// `struct uvc_xu_control_query` from <linux/uvcvideo.h>, in the layout the C
/// compiler emits on this target (repr(C): unit@0, selector@1, query@2, size@4,
/// data@8; 16 bytes - confirmed against the kernel header by the irprobe tool
/// that discovered the emitter control).
#[repr(C)]
struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

/// UVCIOC_CTRL_QUERY from <linux/uvcvideo.h>: _IOWR('u', 0x21,
/// struct uvc_xu_control_query). Written out for the _IOC form of asm-generic/
/// ioctl.h: nr at bit 0, type at bit 8, size at bit 16, direction bits at bit
/// 30 (READ|WRITE = 3); 'u' = 0x75, size = 16 on x86_64 (matches the
/// 0xc0107521 the irprobe tool prints for this machine).
const UVCIOC_CTRL_QUERY: u64 =
    (3 << 30) | (0x75 << 8) | 0x21 | ((std::mem::size_of::<UvcXuControlQuery>() as u64) << 16);

extern "C" {
    fn ioctl(fd: i32, request: u64, arg: *mut UvcXuControlQuery) -> i32;
}

/// Low-level write of `value` to the emitter control on the device with fd
/// `fd`, via UVCIOC_CTRL_QUERY / UVC_SET_CUR. Returns the ioctl result.
fn xu_set_emitter_value(fd: i32, emitter: &EmitterConfig, value: &[u8]) -> std::io::Result<()> {
    let mut query = UvcXuControlQuery {
        unit: emitter.unit,
        selector: emitter.selector,
        query: UVC_SET_CUR,
        size: value.len() as u16,
        data: value.as_ptr() as *mut u8,
    };
    let rc = unsafe { ioctl(fd, UVCIOC_CTRL_QUERY, &mut query) };
    if rc != 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Writes `value` to the emitter's extension-unit control. On the 30c9:00ee
/// module the control lives on the IR node where the burst runs, so with
/// `emitter.device` empty the burst's own device is used; a configured
/// `emitter.device` opens that node instead. Returns the ioctl result (Err on
/// open or write failure; callers log and carry on).
fn ir_emitter_write(
    dev: &v4l::Device,
    emitter: &EmitterConfig,
    value: &[u8],
) -> std::io::Result<()> {
    if emitter.device.is_empty() {
        xu_set_emitter_value(dev.handle().fd(), emitter, value)
    } else {
        match v4l::Device::with_path(&emitter.device) {
            Ok(emitter_dev) => xu_set_emitter_value(emitter_dev.handle().fd(), emitter, value),
            Err(err) => Err(err),
        }
    }
}

/// Grabs `count` frames from the IR camera in a single short burst and returns them.
///
/// The device is opened once for the whole burst and dropped immediately after.
/// Callers must not hold the IR camera open outside of this function, since
/// that's what keeps the IR illuminator from flashing continuously.
///
/// When `emitter` is enabled, its ON value is written to the extension-unit
/// control while the device is open and the stream is not yet running - the
/// driver latches the value at stream start, so writing earlier (or later)
/// leaves the camera dark during the burst - and the OFF value is written after
/// the stream closes. Both writes are best-effort: a camera without an emitter
/// control at this unit/selector, or where the write fails, still gets a valid
/// (dark) burst - log a warning and continue, never fail.
pub fn grab_ir_burst(
    device: &str,
    width: u32,
    height: u32,
    count: u32,
    emitter: &EmitterConfig,
) -> Result<Vec<GrayFrame>> {
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
        if fmt.fourcc != requested_fourcc {
            anyhow::bail!(
                "ir camera {device} does not support {width}x{height} {requested_fourcc}; \
                 driver reported {}x{} {} instead",
                fmt.width,
                fmt.height,
                fmt.fourcc
            );
        }
        // Same pixel format, different size: the 30c9:00ee module supports
        // only 640x360 GREY while the presence defaults ask for 320x240. The
        // detector resizes to its own input size anyway (vision::preprocess),
        // so capture at whatever the driver actually supports instead of
        // failing every burst.
        tracing::debug!(
            device,
            requested_w = width,
            requested_h = height,
            got_w = fmt.width,
            got_h = fmt.height,
            "ir camera substituted a different frame size; using it"
        );
    }

    // Switch the emitter on before the stream opens: the driver latches the
    // control value at stream start, so this write is what makes the burst see
    // light. Best-effort - a camera without an emitter control at this
    // unit/selector, or where the write fails, still gets a valid (dark) burst:
    // warn and carry on.
    if emitter.enable && !emitter.on_value.is_empty() {
        if let Err(err) = ir_emitter_write(&dev, emitter, &emitter.on_value) {
            tracing::warn!(
                device,
                unit = emitter.unit,
                selector = emitter.selector,
                error = %err,
                "ir emitter on-write failed"
            );
        } else {
            tracing::debug!(
                device,
                unit = emitter.unit,
                selector = emitter.selector,
                "ir emitter armed before stream start"
            );
        }
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

    // The burst is closed: stop the stream before switching the emitter off so
    // the write isn't racing the stream-start latch. Same best-effort
    // semantics as the write above.
    drop(stream);
    if emitter.enable && !emitter.off_value.is_empty() {
        if let Err(err) = ir_emitter_write(&dev, emitter, &emitter.off_value) {
            tracing::warn!(
                device,
                unit = emitter.unit,
                selector = emitter.selector,
                error = %err,
                "ir emitter off-write failed"
            );
        }
    }

    Ok(frames)
}
