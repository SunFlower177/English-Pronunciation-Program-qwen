#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install -e ".[all]"

cat <<'MSG'
Installed Python dependencies.
System dependencies still required:
  - ffmpeg
  - espeak-ng
For Ubuntu/Debian:
  sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng

Charsiu is not published as a stable pip package. For paper-faithful CMU cue extraction:
  git clone https://github.com/lingjzhu/charsiu external/charsiu
  export CHARSIU_PATH=$PWD/external/charsiu
MSG
