use crate::camera::GrayFrame;
use crate::config::MotionConfig;

/// Cheap frame-differencing gate over the RGB camera. This is the only thing that
/// runs continuously — it never touches the IR camera, so nothing flashes while
/// the room is empty or still.
pub struct MotionGate {
    previous: Option<GrayFrame>,
}

pub struct MotionResult {
    pub changed_pixels: usize,
    pub is_motion: bool,
}

impl MotionGate {
    pub fn new() -> Self {
        Self { previous: None }
    }

    pub fn observe(&mut self, frame: GrayFrame, cfg: &MotionConfig) -> MotionResult {
        let changed_pixels = match &self.previous {
            None => 0,
            Some(prev) if prev.luma.len() != frame.luma.len() => 0,
            Some(prev) => {
                // Subtract the frame-wide brightness shift before thresholding, so a
                // light turning on/off (which moves every pixel by roughly the same
                // amount) doesn't register as motion — only localized change does.
                let mean_shift = frame.mean_luma() - prev.mean_luma();
                prev.luma
                    .iter()
                    .zip(frame.luma.iter())
                    .filter(|(&a, &b)| {
                        let adjusted = b as f32 - mean_shift;
                        (adjusted - a as f32).abs() >= cfg.diff_threshold as f32
                    })
                    .count()
            }
        };

        let is_motion = changed_pixels >= cfg.blob_min_pixels;
        self.previous = Some(frame);

        MotionResult {
            changed_pixels,
            is_motion,
        }
    }
}
