#!/usr/bin/env bash
# Downloads the tier-1 presence face detector (UltraFace-RFB-320, ~1.2MB, MIT-licensed)
# from the upstream Ultra-Light-Fast-Generic-Face-Detector-1MB repo.
set -euo pipefail
cd "$(dirname "$0")"
curl -fsSL -o ultraface-rfb-320.onnx \
  "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320.onnx"
mkdir -p ~/.local/share/crustd/models
cp ultraface-rfb-320.onnx ~/.local/share/crustd/models/ultraface-rfb-320.onnx
echo "installed to ~/.local/share/crustd/models/ultraface-rfb-320.onnx"
