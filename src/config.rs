use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub camera: CameraConfig,
    pub motion: MotionConfig,
    pub presence: PresenceConfig,
    pub bread: BreadConfig,
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

impl Default for Config {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            motion: MotionConfig::default(),
            presence: PresenceConfig::default(),
            bread: BreadConfig::default(),
        }
    }
}

impl Config {
    pub fn load(path: &PathBuf) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
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
