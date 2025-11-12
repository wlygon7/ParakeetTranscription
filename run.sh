#!/bin/bash
# Launch the Voice Transcription menu bar app

# Check if dependencies are installed
if ! python3 -c "import rumps" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install parakeet-mlx pyaudio pynput pyperclip rumps
fi

# Run the app
python3 whisper_flow_app.py
