use anyhow::Result;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;

use crate::camera::GrayFrame;

const MODEL_WIDTH: usize = 320;
const MODEL_HEIGHT: usize = 240;

/// Which execution provider a `FaceDetector` was built against.
///
/// `Npu` is intentionally unimplemented: this hardware (RyzenAI-npu6 / Krackan) has
/// no matching xclbin overlay in the installed Vitis AI SDK (it only ships Phoenix
/// overlays), so there's nothing for the VitisAI EP to load yet. The seam is here so
/// that dropping in a Krackan xclbin later is a config change, not a rewrite: swap
/// this match arm to call `SessionBuilder::with_execution_providers` with the VitisAI
/// EP once `vaip_config.json` points at a compatible overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Npu,
}

/// Tier-1 presence face detector (UltraFace-RFB-320 class model). Runs only against
/// short IR bursts handed to it by the presence loop — it never touches the camera
/// itself.
pub struct FaceDetector {
    session: Session,
}

impl FaceDetector {
    pub fn load(model_path: &std::path::Path, backend: Backend) -> Result<Self> {
        if backend == Backend::Npu {
            anyhow::bail!(
                "NPU backend requested but no Krackan/Strix xclbin is available for the \
                 VitisAI execution provider on this system (only Phoenix overlays are \
                 installed under ryzenai-env). Falling back is the caller's job; this \
                 constructor refuses rather than silently running on CPU under a false name."
            );
        }

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("creating onnxruntime session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("setting graph optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("setting intra-op thread count: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("loading face detector model from {model_path:?}: {e}"))?;

        Ok(Self { session })
    }

    /// Resizes/normalizes the frame to the model's expected input and returns the
    /// highest face-class probability across all anchor boxes. Callers decide the
    /// confidence threshold; this just reports the peak score.
    pub fn max_face_confidence(&mut self, frame: &GrayFrame) -> Result<f32> {
        let tensor = preprocess(frame);

        let input = Value::from_array(([1usize, 3, MODEL_HEIGHT, MODEL_WIDTH], tensor))
            .map_err(|e| anyhow::anyhow!("building input tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs!["input" => input])
            .map_err(|e| anyhow::anyhow!("running face detector session: {e}"))?;

        // Indexing outputs by name (`outputs["scores"]`) panics if the loaded model
        // doesn't have an output by that name — a real possibility since the model
        // path is user/config-controlled. Look it up fallibly instead so a
        // mismatched model produces a normal error the caller can log and recover
        // from, rather than aborting the whole daemon.
        let scores_value = outputs
            .get("scores")
            .ok_or_else(|| anyhow::anyhow!("face detector model has no \"scores\" output"))?;

        let (shape, data) = scores_value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extracting scores output: {e}"))?;

        // scores is expected to be [1, N, 2]: column 0 = background prob, column 1 =
        // face prob. Verify the layout before treating pair[1] as the face score —
        // a differently-shaped export (e.g. channel-first [1, 2, N]) would otherwise
        // silently read the wrong values with no error.
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 2 {
            anyhow::bail!("face detector model output has unexpected shape {shape:?}, expected [1, N, 2]");
        }

        // The exported graph already applies softmax, so these are usable directly.
        let max_face_prob = data
            .chunks_exact(2)
            .map(|pair| pair[1])
            .fold(0.0f32, f32::max);

        Ok(max_face_prob)
    }
}

/// Nearest-neighbor stretch-resize to 320x240 + grayscale-replicated-to-RGB +
/// `(x - 127) / 128` normalization, matching the reference UltraFace preprocessing.
/// Nearest-neighbor is a deliberate v1 simplification (no extra image/resize
/// dependency) — a proper area-average downscale would be a bit more accurate but
/// isn't needed for a presence gate.
fn preprocess(frame: &GrayFrame) -> Vec<f32> {
    let mut out = vec![0f32; 3 * MODEL_HEIGHT * MODEL_WIDTH];
    let plane_len = MODEL_HEIGHT * MODEL_WIDTH;

    for y in 0..MODEL_HEIGHT {
        let src_y = y * frame.height as usize / MODEL_HEIGHT;
        for x in 0..MODEL_WIDTH {
            let src_x = x * frame.width as usize / MODEL_WIDTH;
            let src_idx = src_y * frame.width as usize + src_x;
            let luma = *frame.luma.get(src_idx).unwrap_or(&0) as f32;
            let normalized = (luma - 127.0) / 128.0;

            let dst_idx = y * MODEL_WIDTH + x;
            out[dst_idx] = normalized; // R
            out[plane_len + dst_idx] = normalized; // G
            out[2 * plane_len + dst_idx] = normalized; // B
        }
    }

    out
}
