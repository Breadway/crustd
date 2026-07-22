mod bread;
mod camera;
mod config;
mod idle;
mod lid;
mod motion;
mod presence;
mod vision;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{info, warn};

use bread::BreadClient;
use camera::DeviceKind;
use config::{Config, PresenceConfig};
use idle::IdleWatcher;
use motion::MotionGate;
use presence::{PresenceState, PresenceTracker};
use vision::{Backend, FaceDetector};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("crustd=info")),
        )
        .init();

    let config_path = Config::default_path();
    let cfg = Config::load(&config_path)?;
    info!(?config_path, "loaded config");

    let model_path = config::expand_home(&cfg.presence.face_detect_model);
    if !model_path.exists() {
        anyhow::bail!(
            "face detector model not found at {model_path:?}; place ultraface-rfb-320.onnx \
             there or point presence.face_detect_model at it in crustd.toml"
        );
    }

    let mut detector = FaceDetector::load(&model_path, Backend::Cpu)?;
    let mut motion_gate = MotionGate::new();
    let mut presence = PresenceTracker::new(&cfg.presence);
    let bread = BreadClient::new(&cfg.bread.socket_path, cfg.bread.event_prefix.clone());

    let rgb_device = camera::resolve_device(&cfg.camera.rgb_device, DeviceKind::Rgb)
        .context("resolving rgb camera device")?
        .to_string_lossy()
        .into_owned();
    let ir_device = camera::resolve_device(&cfg.camera.ir_device, DeviceKind::Ir)
        .context("resolving ir camera device")?
        .to_string_lossy()
        .into_owned();

    let idle_watcher = IdleWatcher::spawn(cfg.motion.input_idle_timeout_ms);
    if idle_watcher.is_none() {
        warn!(
            "compositor doesn't support ext-idle-notify-v1 (or connection failed); \
             falling back to camera-only retention, no compositor idle fast-path"
        );
    }

    let mut idle_interval = cfg.motion.idle_poll_ms;

    // While present, RGB motion (typing, mouse, posture shifts) — or, when available,
    // the compositor directly reporting active input — is the retention signal that
    // resets this clock. Only once things have been still for a while do we spend an
    // IR burst to double-check someone's actually still there.
    let mut last_rgb_motion_at = Instant::now();

    info!(
        rgb_device,
        ir_device,
        compositor_idle_signal = idle_watcher.is_some(),
        "crustd started; ir camera is opened only in short bursts, never streamed continuously"
    );

    let mut shutdown = std::pin::pin!(shutdown_signal());

    loop {
        let sleep_ms = if presence.state() == PresenceState::Present {
            cfg.motion.present_poll_ms
        } else {
            idle_interval
        };

        tokio::select! {
            _ = &mut shutdown => {
                info!("shutting down");
                break;
            }
            _ = tokio::time::sleep(Duration::from_millis(sleep_ms)) => {}
        }

        if lid::is_closed() {
            // Nothing a camera can see with the lid shut; don't bother opening one.
            tracing::trace!("lid closed, skipping this tick");
            continue;
        }

        if presence.state() == PresenceState::Present {
            if let Some(watcher) = &idle_watcher {
                if !watcher.is_idle() {
                    // Compositor confirms the seat has active input right now — that's
                    // sufficient proof of presence on its own. No camera opened at all.
                    last_rgb_motion_at = Instant::now();
                    continue;
                }
            }
        }

        // Both states share the same cheap RGB capture from here — IR is never
        // touched just to decide whether to look.
        let rgb_device_owned = rgb_device.clone();
        let capture = tokio::task::spawn_blocking(move || camera::grab_rgb_luma(&rgb_device_owned, 320, 180))
            .await
            .context("rgb capture task panicked");

        let frame = match capture {
            Ok(Ok(frame)) => frame,
            Ok(Err(err)) => {
                warn!(error = %err, "rgb capture failed");
                continue;
            }
            Err(err) => {
                warn!(error = %err, "rgb capture task join error");
                continue;
            }
        };

        let motion = motion_gate.observe(frame, &cfg.motion);
        tracing::trace!(changed_pixels = motion.changed_pixels, "rgb motion sample");

        if presence.state() == PresenceState::Present {
            if motion.is_motion {
                last_rgb_motion_at = Instant::now();
                continue;
            }

            // RGB (and, per above, compositor input) has been still; only escalate to
            // an IR retention check once it's been still long enough to matter. The
            // first `retention_grace_misses` misses still use the slow stillness
            // cadence — a single missed frontal-face check (glanced down, turned to
            // talk to someone) shouldn't immediately start an urgent countdown toward
            // auto-lock. Only after that grace period does it retry at the fast
            // cadence, and any positive check resets back to the slow cadence.
            let due_after = if presence.negative_streak() >= cfg.presence.retention_grace_misses {
                cfg.motion.present_retry_after_miss_ms
            } else {
                cfg.motion.present_stillness_timeout_ms
            };

            if last_rgb_motion_at.elapsed() < Duration::from_millis(due_after) {
                continue;
            }

            match run_ir_check(&ir_device, &cfg.presence, &mut detector).await {
                Ok(result) => {
                    let face_seen = result.max_confidence >= cfg.presence.min_confidence;
                    if face_seen {
                        last_rgb_motion_at = Instant::now();
                    }
                    handle_observation(&mut presence, &bread, face_seen).await;
                }
                Err(err) => warn!(error = %err, "ir retention check failed"),
            }
            continue;
        }

        // Absent: escalate to an IR acquisition burst as soon as RGB sees motion.
        // Acquisition deliberately requires an actual frontal face match (not the
        // looser occupancy signal) — a hand or mug near the sensor shouldn't count
        // as someone arriving. IR camera stays closed and dark the rest of the time.
        if motion.is_motion {
            idle_interval = cfg.motion.active_poll_ms;
            match run_ir_check(&ir_device, &cfg.presence, &mut detector).await {
                Ok(result) => {
                    let face_seen = result.max_confidence >= cfg.presence.min_confidence;
                    if face_seen {
                        last_rgb_motion_at = Instant::now();
                    }
                    handle_observation(&mut presence, &bread, face_seen).await;
                }
                Err(err) => warn!(error = %err, "ir burst check failed"),
            }
        } else {
            idle_interval = (idle_interval * 3 / 2).min(cfg.motion.max_idle_poll_ms);
        }
    }

    Ok(())
}

struct IrCheckResult {
    max_confidence: f32,
    max_mean_luma: f32,
}

/// Opens the IR camera for exactly one short burst, runs the tier-1 detector over
/// each frame, and closes it again before returning.
async fn run_ir_check(ir_device: &str, cfg: &PresenceConfig, detector: &mut FaceDetector) -> Result<IrCheckResult> {
    let device = ir_device.to_string();
    let burst_frames = cfg.ir_burst_frames;

    let frames = tokio::task::spawn_blocking(move || camera::grab_ir_burst(&device, burst_frames))
        .await
        .context("ir capture task panicked")??;

    let mut max_confidence = 0.0f32;
    let mut max_mean_luma = 0.0f32;
    for frame in &frames {
        let confidence = detector.max_face_confidence(frame)?;
        max_confidence = max_confidence.max(confidence);
        max_mean_luma = max_mean_luma.max(frame.mean_luma());
    }

    tracing::debug!(
        max_confidence,
        max_mean_luma,
        threshold = cfg.min_confidence,
        "ir burst result"
    );

    Ok(IrCheckResult { max_confidence, max_mean_luma })
}

async fn handle_observation(presence: &mut PresenceTracker, bread: &BreadClient, positive: bool) {
    if let Some(new_state) = presence.observe(positive) {
        info!(state = new_state.as_str(), "presence state changed");
        let data = serde_json::json!({ "state": new_state.as_str() });
        if let Err(err) = bread.emit("changed", data).await {
            warn!(error = %err, "failed to emit presence event to bread");
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut sig) = signal(SignalKind::terminate()) {
            sig.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
