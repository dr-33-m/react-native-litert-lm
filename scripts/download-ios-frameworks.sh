#!/bin/bash
# download-ios-frameworks.sh
# Downloads the official prebuilt LiteRT-LM iOS framework (CLiteRTLM.xcframework)
# directly from the google-ai-edge/LiteRT-LM releases.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/ios/Frameworks"

LITERT_LM_VERSION="$(node -e "console.log(require('$PROJECT_ROOT/package.json').litertLm.iosGitTag)")"
RELEASE_URL="https://github.com/google-ai-edge/LiteRT-LM/releases/download/${LITERT_LM_VERSION}/CLiteRTLM.xcframework.zip"

# Skip if already present
if [ -d "$OUTPUT_DIR/CLiteRTLM.xcframework" ]; then
  echo "[LiteRT-LM] iOS CLiteRTLM.xcframework already present, skipping download."
  exit 0
fi

echo "[LiteRT-LM] Downloading prebuilt iOS engine from Google's release:"
echo "   ${RELEASE_URL}"

mkdir -p "$OUTPUT_DIR"
TMP_ZIP="$PROJECT_ROOT/.ios-frameworks-tmp.zip"

if curl -fsSL -o "$TMP_ZIP" "$RELEASE_URL"; then
  echo "[LiteRT-LM] Download successful, extracting to $OUTPUT_DIR..."
  rm -rf "$OUTPUT_DIR/CLiteRTLM.xcframework"
  unzip -o -q "$TMP_ZIP" -d "$OUTPUT_DIR"
  rm -f "$TMP_ZIP"
  echo "[LiteRT-LM] ✅ iOS frameworks successfully installed."
  exit 0
else
  rm -f "$TMP_ZIP"
  echo "Error: Failed to download LiteRT-LM iOS framework from ${RELEASE_URL}"
  exit 1
fi
