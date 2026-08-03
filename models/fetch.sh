#!/usr/bin/env bash
# Downloads the tier-1 presence face detector (UltraFace-RFB-320, ~1.2MB, MIT-licensed)
# from the upstream Ultra-Light-Fast-Generic-Face-Detector-1MB repo.
#
# Pinned to a commit (the last one that touched this file, as of writing) rather
# than `master` — a mutable branch ref means a compromised/force-pushed upstream
# repo could silently substitute a different model, which crustd then loads and
# executes as an ONNX computation graph with no other integrity check anywhere in
# the chain. The sha256 check below is the actual guard; the pinned commit just
# keeps re-runs of this script reproducible.
#
# To update: bump MODEL_COMMIT to the new commit that changed the model, download
# once, verify the new file is what you expect, then update MODEL_SHA256 to match.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_COMMIT="0f9ca4a9fc80170fd505168fd1132b837141f7df"
MODEL_SHA256="34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017"

curl -fsSL -o ultraface-rfb-320.onnx \
  "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/${MODEL_COMMIT}/models/onnx/version-RFB-320.onnx"

echo "${MODEL_SHA256}  ultraface-rfb-320.onnx" | sha256sum -c -

mkdir -p ~/.local/share/crustd/models
cp ultraface-rfb-320.onnx ~/.local/share/crustd/models/ultraface-rfb-320.onnx
echo "installed to ~/.local/share/crustd/models/ultraface-rfb-320.onnx"
