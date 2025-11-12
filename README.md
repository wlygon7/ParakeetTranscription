# Voice Transcription Tool

A hotkey-activated voice transcription tool inspired by Whisper Flow, powered by NVIDIA's Parakeet model optimized for Apple Silicon.

## Features

- Press a hotkey to start/stop recording
- Automatic transcription using state-of-the-art Parakeet TDT model
- Text automatically copied to clipboard
- Menu bar app with notifications
- Runs in background
- Optimized for Apple Silicon (M1/M2/M3) using MLX

## Quick Start - Menu Bar App (Recommended)

### Prerequisites

Install ffmpeg and portaudio:
```bash
brew install ffmpeg portaudio
```

### Install dependencies with uv

```bash
uv pip install parakeet-mlx pyaudio pynput pyperclip rumps
```

### Run the menu bar app

```bash
python whisper_flow_app.py
```

The app will:
- Show a microphone icon in your menu bar
- Load the model in the background (you'll see a notification when ready)
- Listen for **F12** hotkey globally
- Show notifications with transcribed text

### Usage

1. Press **F12** to start recording (icon turns red 🔴)
2. Speak your text
3. Press **F12** again to stop
4. You'll get a notification with the transcribed text
5. Text is automatically copied to clipboard - paste with Cmd+V

## Alternative - Command Line Version

For a simpler command-line version without menu bar:

```bash
python whisper_flow.py
```

This version prints output to the terminal instead of showing notifications.

## Running as Background Service (Recommended)

The easiest way to have the app always available is to install it as a Launch Agent:

### Install as background service

```bash
./install_service.sh
```

This will create a Launch Agent that:
- Starts automatically when you log in
- Runs in the background
- Restarts if it crashes
- Logs to `~/Library/Logs/VoiceTranscription.log`

### Start the service

```bash
launchctl load ~/Library/LaunchAgents/com.voicetranscription.plist
```

### Stop the service

```bash
launchctl unload ~/Library/LaunchAgents/com.voicetranscription.plist
```

### Uninstall the service

```bash
launchctl unload ~/Library/LaunchAgents/com.voicetranscription.plist
rm ~/Library/LaunchAgents/com.voicetranscription.plist
```

## Alternative: Build Standalone App

If you prefer a traditional .app bundle, you can build one with PyInstaller:

```bash
./build.sh
```

This creates `dist/VoiceTranscription.app` that you can:
- Move to `/Applications`
- Launch like any other Mac app
- Add to Login Items for auto-start

Note: Building with py2app may hit recursion limits due to MLX dependencies. PyInstaller works better for this use case.

## Configuration

Edit `whisper_flow.py` to change the hotkey:

```python
HOTKEY = keyboard.Key.f13  # Change to your preferred key
```

Common options:
- `keyboard.Key.f13`, `keyboard.Key.f14`, etc.
- `keyboard.Key.caps_lock`
- Use keyboard remapping tools to map unused keys to F13/F14

## Hotkey Setup Tips

On macOS, F13-F20 keys are ideal because they're rarely used. You can:

1. Use Karabiner-Elements to map CapsLock or another key to F13
2. Use BetterTouchTool to create custom keyboard shortcuts
3. On some keyboards, F13-F15 are accessible via Fn+F1, F2, F3

## Troubleshooting

### Microphone Permission
If recording doesn't work, ensure Terminal/Python has microphone access:
- System Preferences → Security & Privacy → Microphone

### PyAudio Installation Issues
If `uv pip install pyaudio` fails, ensure portaudio is installed:
```bash
brew install portaudio
```

### Model Download
First run will download the model (~2GB). Ensure you have internet connection.

## Technical Details

- Model: Parakeet TDT 0.6B v3 (600M parameters)
- Audio: 16kHz mono
- Framework: MLX (optimized for Apple Silicon)
- Average WER: ~6%

## Future Enhancements

- [ ] Real-time streaming transcription (show text as you speak)
- [ ] Direct text insertion at cursor (instead of clipboard)
- [ ] Visual feedback (menu bar icon)
- [ ] Multiple language support
- [ ] Custom vocabulary/commands
