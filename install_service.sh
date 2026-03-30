#!/bin/bash
# ParakeetTranscription - One-Command Installer for macOS
# Usage: ./install_service.sh
#
# Handles everything: brew deps, uv, venv, Python packages, and Launch Agent setup.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLIST_PATH="$HOME/Library/LaunchAgents/com.voicetranscription.plist"
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python3"

echo "============================================"
echo "  ParakeetTranscription Installer"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. Homebrew
# ------------------------------------------------------------------
if ! command -v brew &>/dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for this session
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "[ok] Homebrew found"
fi

# ------------------------------------------------------------------
# 2. System dependencies (portaudio, ffmpeg)
# ------------------------------------------------------------------
MISSING_DEPS=()
brew list portaudio &>/dev/null || MISSING_DEPS+=(portaudio)
brew list ffmpeg &>/dev/null || MISSING_DEPS+=(ffmpeg)

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "Installing system dependencies: ${MISSING_DEPS[*]}..."
    brew install "${MISSING_DEPS[@]}"
else
    echo "[ok] portaudio and ffmpeg installed"
fi

# ------------------------------------------------------------------
# 3. uv (Python package manager)
# ------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    brew install uv
else
    echo "[ok] uv found"
fi

# ------------------------------------------------------------------
# 4. Python virtual environment + dependencies
# ------------------------------------------------------------------
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    cd "$SCRIPT_DIR"
    uv venv --python 3.13
else
    echo "[ok] Virtual environment exists"
fi

echo "Installing Python dependencies..."
cd "$SCRIPT_DIR"
uv pip install -e .

echo "[ok] Dependencies installed"

# ------------------------------------------------------------------
# 5. Unload existing service if running
# ------------------------------------------------------------------
if launchctl list | grep -q com.voicetranscription; then
    echo "Stopping existing service..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
fi

# ------------------------------------------------------------------
# 6. Create Launch Agent
# ------------------------------------------------------------------
echo "Creating Launch Agent..."

cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voicetranscription</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_PATH</string>
        <string>$SCRIPT_DIR/whisper_flow_app.py</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$HOME/Library/Logs/VoiceTranscription.log</string>

    <key>StandardErrorPath</key>
    <string>$HOME/Library/Logs/VoiceTranscription.error.log</string>

    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

# ------------------------------------------------------------------
# 7. Start the service
# ------------------------------------------------------------------
echo "Starting service..."
launchctl load "$PLIST_PATH"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "The app is now running in your menu bar."
echo "Press Alt+R to start/stop transcription."
echo ""
echo "IMPORTANT - You need to grant two permissions:"
echo "  1. Microphone access"
echo "  2. Accessibility access (for global hotkey)"
echo ""
echo "If prompted, allow access in System Settings > Privacy & Security."
echo ""
echo "Commands:"
echo "  Stop:      launchctl unload $PLIST_PATH"
echo "  Start:     launchctl load $PLIST_PATH"
echo "  Uninstall: launchctl unload $PLIST_PATH && rm $PLIST_PATH"
echo "  Logs:      tail -f ~/Library/Logs/VoiceTranscription.log"
