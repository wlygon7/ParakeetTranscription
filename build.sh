#!/bin/bash
# Build Voice Transcription app for macOS

set -e

echo "Installing PyInstaller..."
uv pip install pyinstaller

echo ""
echo "Building app bundle..."
pyinstaller build_app.spec --clean

echo ""
echo "✅ Build complete!"
echo ""
echo "Your app is ready at: dist/VoiceTranscription.app"
echo ""
echo "To install:"
echo "  cp -r dist/VoiceTranscription.app /Applications/"
echo ""
echo "To test:"
echo "  open dist/VoiceTranscription.app"
