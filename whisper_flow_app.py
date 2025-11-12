#!/usr/bin/env python3
"""
Voice Transcription Menu Bar App
Inspired by Whisper Flow, powered by NVIDIA Parakeet on Apple Silicon via MLX
"""

import pyaudio
import wave
import tempfile
import pyperclip
import threading
import rumps
import sys
from pathlib import Path
from pynput import keyboard
from pynput.keyboard import GlobalHotKeys
from parakeet_mlx import from_pretrained
from datetime import datetime
from Cocoa import NSEvent, NSSound
from Quartz import (
    CGEventMaskBit,
    kCGEventKeyDown,
    kCGEventFlagsChanged,
)
import Quartz

# Setup logging to file
import os
LOG_FILE = os.path.expanduser("~/Library/Logs/VoiceTranscription_debug.log")

def log(message):
    """Write log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    sys.stdout.flush()

# Log the PATH at startup
log(f"Starting Voice Transcription App")
log(f"PATH: {os.environ.get('PATH', 'NOT SET')}")

# Helper function for playing sounds
def play_sound(sound_name):
    """Play a macOS system sound"""
    try:
        sound = NSSound.soundNamed_(sound_name)
        if sound:
            sound.play()
    except Exception as e:
        log(f"Sound error: {e}")

# Helper function for safe notifications
def safe_notification(title, subtitle="", message=""):
    """Send notification, gracefully handle failures"""
    try:
        rumps.notification(title=title, subtitle=subtitle, message=message)
    except Exception as e:
        log(f"Notification: {title} - {subtitle} - {message}")
        log(f"(Notification error: {e})")

# Configuration
HOTKEY = keyboard.KeyCode.from_char('r')  # Will be used with Alt/Option modifier
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16


class VoiceTranscriber:
    def __init__(self, app):
        self.app = app
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.model = None
        self.model_loaded = False

    def load_model(self):
        """Load the model in background"""
        if self.model_loaded:
            return

        self.app.title = "⏳"
        log("Starting to load Parakeet model...")
        try:
            self.model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
            self.model_loaded = True
            self.app.title = "🎙️"
            log("✅ Model loaded successfully!")
            safe_notification(
                title="Parakeet Ready",
                subtitle="Voice transcription is ready",
                message="Press Alt+R to start recording"
            )
        except Exception as e:
            self.app.title = "❌"
            log(f"❌ Error loading model: {e}")
            import traceback
            log(traceback.format_exc())
            safe_notification(
                title="Error Loading Model",
                subtitle="Failed to load Parakeet",
                message=str(e)
            )

    def start_recording(self):
        """Start recording audio"""
        log("🎤 Start recording requested...")
        if not self.model_loaded:
            log("⚠️ Model not loaded yet")
            safe_notification(
                title="Model Not Ready",
                subtitle="Please wait",
                message="Model is still loading..."
            )
            return

        if self.is_recording:
            log("⚠️ Already recording")
            return

        self.is_recording = True
        self.frames = []
        self.app.title = "🔴"
        log("✅ Recording started!")
        play_sound("Tink")  # Light, minimal "start" sound

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        # Record in a separate thread
        def record():
            while self.is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break

        self.record_thread = threading.Thread(target=record, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and transcribe"""
        log("⏹️ Stop recording requested...")
        if not self.is_recording:
            log("⚠️ Not currently recording")
            return

        self.is_recording = False
        self.record_thread.join()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.app.title = "⚙️"
        log("⚙️ Transcribing...")
        play_sound("Tink")  # Same sound for stop

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name

            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(self.frames))

        # Transcribe in background
        def transcribe():
            try:
                result = self.model.transcribe(temp_path)
                transcribed_text = result.text.strip()
                log(f"📝 Transcription result: '{transcribed_text}'")

                if transcribed_text:
                    pyperclip.copy(transcribed_text)
                    log(f"✅ Copied to clipboard: {transcribed_text}")
                    play_sound("Glass")  # Success sound - subtle completion
                    safe_notification(
                        title="Transcribed",
                        subtitle=transcribed_text[:50] + ("..." if len(transcribed_text) > 50 else ""),
                        message="Copied to clipboard - paste with Cmd+V"
                    )
                else:
                    log("⚠️ No speech detected in audio")
                    play_sound("Funk")  # Warning sound
                    safe_notification(
                        title="No Speech Detected",
                        subtitle="Try speaking louder or closer to mic",
                        message=""
                    )
            except Exception as e:
                log(f"❌ Transcription error: {e}")
                import traceback
                log(traceback.format_exc())
                safe_notification(
                    title="Transcription Error",
                    subtitle=str(e),
                    message=""
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)
                self.app.title = "🎙️"

        threading.Thread(target=transcribe, daemon=True).start()

    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def cleanup(self):
        """Clean up resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class VoiceTranscriptionApp(rumps.App):
    def __init__(self):
        super(VoiceTranscriptionApp, self).__init__(
            "Voice Transcription",
            icon=None,
            title="⏳",
            quit_button=None
        )

        self.transcriber = VoiceTranscriber(self)
        self.last_hotkey_time = 0
        self.hotkey_debounce = 0.5  # 500ms debounce
        self.menu = [
            rumps.MenuItem("Status: Loading...", callback=None),
            rumps.separator,
            rumps.MenuItem("Hotkey: Alt+R", callback=None),
            rumps.MenuItem("Record", callback=self.toggle_recording),
            rumps.separator,
            rumps.MenuItem("Quit", callback=self.quit_app)
        ]

        # Load model in background thread
        threading.Thread(target=self.load_model, daemon=True).start()

        # Start keyboard listener
        self.start_keyboard_listener()

    def load_model(self):
        """Load the Parakeet model"""
        self.transcriber.load_model()
        self.menu["Status: Loading..."].title = "Status: Ready"

    def start_keyboard_listener(self):
        """Start listening for global hotkey Alt+R using macOS event tap"""
        def event_handler(proxy, event_type, event, refcon):
            try:
                # Get key code and flags
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                flags = Quartz.CGEventGetFlags(event)

                # Check if Alt is pressed (option key)
                alt_pressed = (flags & Quartz.kCGEventFlagMaskAlternate) != 0

                # Key code 15 is 'r'
                if event_type == kCGEventKeyDown and keycode == 15 and alt_pressed:
                    # Debounce: check if enough time has passed since last trigger
                    import time
                    current_time = time.time()
                    if current_time - self.last_hotkey_time > self.hotkey_debounce:
                        self.last_hotkey_time = current_time
                        log("Alt+R pressed - toggling recording")
                        # Use threading instead of Timer for better reliability
                        threading.Thread(target=self.transcriber.toggle_recording, daemon=True).start()
                    # Return None to suppress the event (prevent ® from appearing)
                    return None

            except Exception as e:
                log(f"Event handler error: {e}")
                import traceback
                log(traceback.format_exc())

            # Pass through all other events
            return event

        # Create event tap
        try:
            self.event_tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,
                CGEventMaskBit(kCGEventKeyDown),
                event_handler,
                None
            )

            if self.event_tap is None:
                log("Failed to create event tap - accessibility permissions needed!")
                return

            # Create run loop source and add to current run loop
            run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, self.event_tap, 0)
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetCurrent(),
                run_loop_source,
                Quartz.kCFRunLoopCommonModes
            )

            # Enable the event tap
            Quartz.CGEventTapEnable(self.event_tap, True)
            log("Hotkey listener started for Alt+R with event tap")

        except Exception as e:
            log(f"Failed to setup event tap: {e}")
            import traceback
            log(traceback.format_exc())

    @rumps.clicked("Record")
    def toggle_recording(self, _):
        """Toggle recording from menu"""
        self.transcriber.toggle_recording()

    def quit_app(self, _):
        """Clean up and quit"""
        self.transcriber.cleanup()
        if hasattr(self, 'listener'):
            self.listener.stop()
        rumps.quit_application()


def main():
    VoiceTranscriptionApp().run()


if __name__ == "__main__":
    main()
