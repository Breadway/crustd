use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub camera: CameraConfig,
    pub motion: MotionConfig,
    pub presence: PresenceConfig,
    pub bread: BreadConfig,
    pub emitter: EmitterConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CameraConfig {
    /// Visible-light camera used for the always-on motion gate. Never opens the IR emitter.
    pub rgb_device: String,
    /// IR camera used only in short bursts for face detection/recognition.
    pub ir_device: String,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            // Auto-resolved by capture capability via /dev/v4l/by-path at startup —
            // see camera::resolve_device. Set to a literal /dev/videoN to override.
            rgb_device: "auto:rgb".to_string(),
            ir_device: "auto:ir".to_string(),
        }
    }
}

/// UVC extension-unit control that switches the IR emitter of the IR camera on.
/// Some Luxvisions hybrid RGB+IR modules ship a separate IR illuminator that
/// uvcvideo never powers up on its own; writing the ON value to this control
/// (before the IR stream opens - the driver latches the value at stream start)
/// makes it strobe. Defaults are the values discovered on this laptop's module
/// by probing its extension units, so an existing crustd.toml without an
/// [emitter] section still works. On a camera without this control, or where
/// the write fails, the write is skipped or logged as a warning and the burst
/// stays dark - never fatal.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmitterConfig {
    /// Device node that owns the emitter control. Empty = the IR camera node
    /// from [camera] ir_device.
    pub device: String,
    /// UVC extension unit id of the emitter control.
    pub unit: u8,
    /// UVC control selector of the emitter control.
    pub selector: u8,
    /// Value bytes written to the control to switch the emitter on. Empty =
    /// skip the write entirely.
    pub on_value: Vec<u8>,
    /// Value bytes written to the control after each burst closes. Empty =
    /// leave the control as-is (the driver resets it when the device closes).
    pub off_value: Vec<u8>,
    /// Master switch for the emitter writes.
    pub enable: bool,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            device: String::new(),
            // Discovered by probing /dev/video2's extension units on the
            // 30c9:00ee module: UVC unit 14, selector 6. ON = 01 03 02 00 00
            // 00 00 00 00 (byte 2 = 0x02 means on, per the status readback in
            // the reference emitter tools); OFF = 01 03 01 00 00 00 00 00 00.
            // The other units on this module (10, 13) are unrelated controls.
            unit: 14,
            selector: 6,
            on_value: vec![0x01, 0x03, 0x02, 0, 0, 0, 0, 0, 0],
            off_value: vec![0x01, 0x03, 0x01, 0, 0, 0, 0, 0, 0],
            enable: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MotionConfig {
    /// Poll interval when nothing has moved recently.
    pub idle_poll_ms: u64,
    /// Poll interval right after motion was seen, before it settles back down.
    pub active_poll_ms: u64,
    /// Ceiling for backoff when the room has been still for a long time.
    pub max_idle_poll_ms: u64,
    /// RGB poll interval while presence is already confirmed. This is the retention
    /// signal — typing, mouse movement, posture shifts — and it's cheap (no IR, no
    /// inference), so it can run continuously without a power or annoyance cost.
    pub present_poll_ms: u64,
    /// How long RGB has to be continuously still while present before we bother
    /// spending an IR burst to double-check someone's still there. A person actively
    /// working is essentially never RGB-still for this long, so in normal use this
    /// keeps the IR camera dark for the entire session.
    pub present_stillness_timeout_ms: u64,
    /// After an IR retention check comes back with no face, how soon to retry — much
    /// faster than `present_stillness_timeout_ms` so a real departure is confirmed
    /// (and presence released) within `release_frames` short retries instead of
    /// waiting a full stillness timeout each time. A positive check at any point
    /// resets back to the slow cadence.
    pub present_retry_after_miss_ms: u64,
    /// How long the compositor's seat (keyboard/mouse) must be idle, via
    /// ext-idle-notify-v1, before Present-mode falls back to the camera-based
    /// retention cascade at all. While input is active, no camera is opened —
    /// typing/mousing is treated as sufficient proof of presence on its own.
    pub input_idle_timeout_ms: u32,
    /// Per-pixel luma delta (0-255) that counts as "changed".
    pub diff_threshold: u8,
    /// Minimum count of changed pixels (out of 320x180) to call it motion.
    pub blob_min_pixels: usize,
    /// Minimum time between IR acquisition bursts while Absent and RGB motion is
    /// present but no face has been confirmed yet. Without a floor here, sustained
    /// non-face motion (a TV, a pet, trees through a window) would retrigger a burst
    /// on every `active_poll_ms` tick indefinitely.
    pub ir_burst_cooldown_ms: u64,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            idle_poll_ms: 2_000,
            active_poll_ms: 250,
            max_idle_poll_ms: 15_000,
            present_poll_ms: 1_000,
            present_stillness_timeout_ms: 45_000,
            present_retry_after_miss_ms: 8_000,
            input_idle_timeout_ms: 20_000,
            diff_threshold: 24,
            blob_min_pixels: 200,
            ir_burst_cooldown_ms: 3_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PresenceConfig {
    /// Path to the ONNX face-detection model used for the presence tier.
    pub face_detect_model: String,
    /// Consecutive positive IR bursts required before presence flips on.
    pub confirm_frames: u32,
    /// Consecutive negative IR bursts required before presence flips off.
    pub release_frames: u32,
    /// Minimum detector confidence to count a frame as "face seen".
    pub min_confidence: f32,
    /// Frames grabbed per IR burst (averaged/voted, then the camera is closed).
    pub ir_burst_frames: u32,
    /// Consecutive retention misses required before the recheck cadence escalates
    /// from the slow stillness-timeout interval to the fast retry interval. Kept
    /// above 1 so a single missed frontal-face check (person glanced down, turned to
    /// talk to someone) gets a second slow-cadence check before anything urgent
    /// happens, rather than immediately starting a fast countdown toward auto-lock.
    ///
    /// (An earlier version of this tried a "mean IR brightness = still occupied"
    /// fallback signal instead, on the theory that a nearby person reflects
    /// noticeably more IR than an empty room. Measured on real hardware, empty-room
    /// and occupied-room mean luma overlapped too much to use — the frame is
    /// dominated by background reflectance, not near-field subject reflectance.
    /// Discarded rather than shipped; this grace-period approach doesn't depend on
    /// that assumption.)
    pub retention_grace_misses: u32,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            face_detect_model: "~/.local/share/crustd/models/ultraface-rfb-320.onnx".to_string(),
            confirm_frames: 2,
            release_frames: 3,
            min_confidence: 0.7,
            ir_burst_frames: 3,
            retention_grace_misses: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BreadConfig {
    /// Empty = resolve $XDG_RUNTIME_DIR/bread/breadd.sock at runtime, matching breadd's default.
    pub socket_path: String,
    pub event_prefix: String,
}

impl Default for BreadConfig {
    fn default() -> Self {
        Self {
            socket_path: String::new(),
            event_prefix: "bread.presence".to_string(),
        }
    }
}

impl Config {
    pub fn load(path: &PathBuf) -> Result<Self> {
        let cfg = if !path.exists() {
            Self::default()
        } else {
            let raw = std::fs::read_to_string(path)?;
            toml::from_str(&raw)?
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Rejects config values that would otherwise produce a permanent hot-spin poll
    /// loop or a presence state that can never change (e.g. a zero poll interval, or
    /// a confidence floor of 0.0 that makes every frame "a face"). Called once after
    /// load so a bad config fails fast with a clear message instead of the daemon
    /// running in a silently broken mode.
    pub fn validate(&self) -> Result<()> {
        const MIN_POLL_MS: u64 = 50;

        let polls = [
            ("motion.idle_poll_ms", self.motion.idle_poll_ms),
            ("motion.active_poll_ms", self.motion.active_poll_ms),
            ("motion.max_idle_poll_ms", self.motion.max_idle_poll_ms),
            ("motion.present_poll_ms", self.motion.present_poll_ms),
            ("motion.present_stillness_timeout_ms", self.motion.present_stillness_timeout_ms),
            ("motion.present_retry_after_miss_ms", self.motion.present_retry_after_miss_ms),
            ("motion.ir_burst_cooldown_ms", self.motion.ir_burst_cooldown_ms),
        ];
        for (name, value) in polls {
            if value < MIN_POLL_MS {
                anyhow::bail!("{name} must be at least {MIN_POLL_MS}ms, got {value}ms");
            }
        }

        if self.presence.ir_burst_frames == 0 {
            anyhow::bail!("presence.ir_burst_frames must be at least 1");
        }

        if !(self.presence.min_confidence > 0.0 && self.presence.min_confidence <= 1.0) {
            anyhow::bail!(
                "presence.min_confidence must be in (0.0, 1.0], got {}",
                self.presence.min_confidence
            );
        }

        if self.motion.present_retry_after_miss_ms > self.motion.present_stillness_timeout_ms {
            anyhow::bail!(
                "motion.present_retry_after_miss_ms ({}) must not exceed \
                 motion.present_stillness_timeout_ms ({}) — the retry cadence is meant \
                 to be faster than the initial stillness timeout, not slower",
                self.motion.present_retry_after_miss_ms,
                self.motion.present_stillness_timeout_ms
            );
        }

        Ok(())
    }

    pub fn default_path() -> PathBuf {
        directories::ProjectDirs::from("dev", "breadway", "crustd")
            .map(|d| d.config_dir().join("crustd.toml"))
            .unwrap_or_else(|| PathBuf::from("crustd.toml"))
    }
}

pub fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}
