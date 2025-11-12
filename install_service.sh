#!/bin/bash
# Install Voice Transcription as a macOS Launch Agent

set -e

# Get the current directory (absolute path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLIST_PATH="$HOME/Library/LaunchAgents/com.voicetranscription.plist"

# Get the Python interpreter path
PYTHON_PATH=$(which python3)

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
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
EOF

echo "✅ Launch Agent created at: $PLIST_PATH"
echo ""
echo "To start the service now:"
echo "  launchctl load $PLIST_PATH"
echo ""
echo "To stop the service:"
echo "  launchctl unload $PLIST_PATH"
echo ""
echo "To uninstall:"
echo "  launchctl unload $PLIST_PATH"
echo "  rm $PLIST_PATH"
echo ""
echo "Logs will be available at:"
echo "  $HOME/Library/Logs/VoiceTranscription.log"
echo "  $HOME/Library/Logs/VoiceTranscription.error.log"
