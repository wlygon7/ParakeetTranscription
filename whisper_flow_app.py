#!/usr/bin/env python3
"""
Voice Transcription Menu Bar App
Alt+R: Quick transcription to clipboard

Powered by NVIDIA Parakeet on Apple Silicon via MLX
Optimized for low-latency real-time streaming transcription
"""

import pyaudio
import wave
import pyperclip
import threading
import rumps
import sys
from pathlib import Path
from parakeet_mlx import from_pretrained
from datetime import datetime
from Cocoa import NSSound
from Quartz import CGEventMaskBit, kCGEventKeyDown
import Quartz
import numpy as np
import mlx.core as mx
import torch
import time

# Setup logging to file
import os
import json

CONFIG_FILE = os.path.expanduser("~/.parakeet_config.json")
LOG_FILE = os.path.expanduser("~/Library/Logs/VoiceTranscription_debug.log")

# Model Constants
MODELS = {
    "fast": {
        "name": "Fast (0.6b)",
        "path": "mlx-community/parakeet-tdt-0.6b-v3"
    },
    "accurate": {
        "name": "Accurate (1.1b)",
        "path": "mlx-community/parakeet-tdt-1.1b"
    }
}

# Audio feed interval: accumulate this many chunks before feeding to stream
# 1024 frames at 16kHz = 64ms per chunk, 8 chunks = ~0.5s
STREAM_FEED_INTERVAL = 8


def log(message):
    """Write log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    sys.stdout.flush()


# Log the PATH at startup
log("Starting Voice Transcription App")
log(f"PATH: {os.environ.get('PATH', 'NOT SET')}")

# Configuration
RECORDING_ARCHIVE_PATH = Path("/Users/gassandrid/MEDIA/recordings")
QUICK_NOTES_PATH = RECORDING_ARCHIVE_PATH / "quick_notes"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

# Ensure directories exist
QUICK_NOTES_PATH.mkdir(parents=True, exist_ok=True)


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


class VoiceTranscriber:
    def __init__(self, app):
        self.app = app
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.model = None
        self.model_loaded = False

        # Load config
        self.config = self.load_config()
        self.current_model_key = self.config.get("model", "fast")

        # Device tracking by name for hot-swap resilience
        self.input_device_index = None
        self.input_device_name = self.config.get("device_name", None)
        self.vad_model = None
        self.vad_device = None

        # Streaming transcription context
        self.stream_ctx = None

        # Pre-initialize audio to reduce first-recording latency
        self._warmup_audio()

        # Resolve saved device name to index
        if self.input_device_name:
            self._resolve_device_by_name()

    def _warmup_audio(self):
        """Pre-initialize PyAudio stream to reduce first-recording latency"""
        try:
            log("Warming up audio subsystem...")
            test_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            test_stream.read(CHUNK, exception_on_overflow=False)
            test_stream.stop_stream()
            test_stream.close()
            log("Audio warmup complete")
        except Exception as e:
            log(f"Audio warmup error (non-fatal): {e}")

    def _resolve_device_by_name(self):
        """Find device index matching saved device name"""
        devices = self.get_input_devices()
        for device in devices:
            if device["name"] == self.input_device_name:
                self.input_device_index = device["index"]
                log(f"Resolved device '{self.input_device_name}' to index {self.input_device_index}")
                return
        log(f"Saved device '{self.input_device_name}' not found, using default")
        self.input_device_index = None

    def _reinit_audio(self):
        """Recreate PyAudio instance to get fresh device list"""
        try:
            self.audio.terminate()
        except Exception:
            pass
        self.audio = pyaudio.PyAudio()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            log(f"Error loading config: {e}")
        return {"model": "fast"}

    def save_config(self):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f)
        except Exception as e:
            log(f"Error saving config: {e}")

    def set_model(self, model_key):
        """Switch model and reload"""
        if model_key not in MODELS:
            return

        if model_key == self.current_model_key and self.model_loaded:
            return

        log(f"Switching model to {model_key}...")
        self.current_model_key = model_key
        self.config["model"] = model_key
        self.save_config()

        # Reload model
        self.model_loaded = False
        self.model = None
        threading.Thread(target=self.load_model, daemon=True).start()

    def get_input_devices(self):
        """Get list of available input devices"""
        devices = []
        try:
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get("deviceCount")

            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get("maxInputChannels") > 0:
                    devices.append({"index": i, "name": device_info.get("name")})
        except Exception as e:
            log(f"Error enumerating devices: {e}")

        return devices

    def set_input_device(self, device_index, device_name=None):
        """Set the input device for recording, saving by name for resilience"""
        self.input_device_index = device_index

        if device_name:
            self.input_device_name = device_name
        elif device_index is not None:
            try:
                self.input_device_name = self.audio.get_device_info_by_index(device_index).get("name")
            except Exception:
                self.input_device_name = None
        else:
            self.input_device_name = None

        self.config["device_name"] = self.input_device_name
        self.save_config()

        display_name = self.input_device_name or "Default"
        log(f"Input device set to: {display_name} (index: {device_index})")
        safe_notification(
            title="Input Device Changed", subtitle=display_name, message=""
        )

    def refresh_devices(self):
        """Re-enumerate audio devices after hardware changes"""
        self._reinit_audio()
        if self.input_device_name:
            self._resolve_device_by_name()
        return self.get_input_devices()

    def load_vad_model(self):
        """Load Silero VAD model with ONNX for 4-5x faster inference"""
        try:
            log("Loading Silero VAD model (ONNX mode)...")
            self.vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            (
                self.get_speech_timestamps,
                self.save_audio,
                self.read_audio,
                self.VADIterator,
                self.collect_chunks,
            ) = utils

            self.vad_device = torch.device("cpu")
            log("Silero VAD model loaded (ONNX)")
        except Exception as e:
            log(f"Error loading VAD model: {e}")
            import traceback
            log(traceback.format_exc())
            self.vad_model = None

    def load_model(self):
        """Load the Parakeet transcription model with streaming warmup"""
        if self.model_loaded:
            return

        self.app.title = "⏳"
        model_info = MODELS.get(self.current_model_key, MODELS["fast"])
        log(f"Loading Parakeet model: {model_info['name']} ({model_info['path']})...")

        try:
            self.model = from_pretrained(model_info['path'])

            # Warmup: run a streaming inference to JIT-compile the path we use
            log("Warming up model (streaming JIT compilation)...")
            self._warmup_model()

            self.model_loaded = True
            self.app.title = "🎙️"
            log(f"Parakeet model ({model_info['name']}) loaded and warmed up!")
            safe_notification(
                title="Model Ready",
                subtitle=f"Loaded {model_info['name']}",
                message="Press Alt+R to transcribe",
            )
        except Exception as e:
            self.app.title = "❌"
            log(f"Error loading Parakeet model: {e}")
            import traceback
            log(traceback.format_exc())
            safe_notification(
                title="Error Loading Model",
                subtitle="Failed to load Parakeet",
                message=str(e),
            )

        # Load VAD model in parallel
        threading.Thread(target=self.load_vad_model, daemon=True).start()

    def _warmup_model(self):
        """Warmup using streaming path to JIT-compile encoder + decoder"""
        try:
            warmup_samples = np.zeros(8000, dtype=np.float32)
            stream = self.model.transcribe_stream(context_size=(256, 256), depth=1)
            stream.__enter__()
            stream.add_audio(mx.array(warmup_samples))
            _ = stream.result
            stream.__exit__(None, None, None)
            log("Streaming model warmup complete")
        except Exception as e:
            log(f"Warmup error (non-fatal): {e}")

    def _open_audio_stream(self):
        """Open PyAudio stream with current device settings"""
        stream_params = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": SAMPLE_RATE,
            "input": True,
            "frames_per_buffer": CHUNK,
        }
        if self.input_device_index is not None:
            stream_params["input_device_index"] = self.input_device_index
        self.stream = self.audio.open(**stream_params)

    def start_recording(self):
        """Start recording with streaming transcription"""
        log("Start recording requested...")
        if not self.model_loaded:
            log("Model not loaded yet")
            safe_notification(
                title="Model Not Ready",
                subtitle="Please wait",
                message="Model is still loading...",
            )
            return

        if self.is_recording:
            log("Already recording")
            return

        self.is_recording = True
        self.frames = []
        self.app.title = "🔴"
        self.record_start_time = time.time()

        # Open streaming transcription context
        try:
            self.stream_ctx = self.model.transcribe_stream(
                context_size=(256, 256), depth=1
            )
            self.stream_ctx.__enter__()
            log("Streaming transcription context opened")
        except Exception as e:
            log(f"Error opening stream context: {e}")
            import traceback
            log(traceback.format_exc())
            self.stream_ctx = None

        # Open audio stream with device fallback
        device_name = self.input_device_name or "Default"
        try:
            self._open_audio_stream()
            log(f"Recording started using device: {device_name}")
        except Exception as e:
            log(f"Device '{device_name}' failed: {e}, trying default...")
            self._reinit_audio()
            self.input_device_index = None
            try:
                self._open_audio_stream()
                safe_notification(
                    title="Device Unavailable",
                    subtitle=f"Using default instead of {device_name}",
                    message="",
                )
                log("Recording started using default device (fallback)")
            except Exception as e2:
                log(f"Default device also failed: {e2}")
                import traceback
                log(traceback.format_exc())
                self.is_recording = False
                self.app.title = "❌"
                self._cleanup_stream_ctx()
                safe_notification(
                    title="Recording Error",
                    subtitle="Failed to start recording",
                    message=str(e2),
                )
                return

        play_sound("Tink")

        def record():
            """Recording loop: capture audio and feed to streaming transcription"""
            feed_counter = 0
            audio_accumulator = []

            while self.is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)

                    # Accumulate audio for streaming transcription
                    if self.stream_ctx is not None:
                        audio_accumulator.append(data)
                        feed_counter += 1

                        if feed_counter >= STREAM_FEED_INTERVAL:
                            combined = b"".join(audio_accumulator)
                            audio_float = np.frombuffer(
                                combined, dtype=np.int16
                            ).astype(np.float32) / 32768.0
                            try:
                                self.stream_ctx.add_audio(mx.array(audio_float))
                            except Exception as e:
                                log(f"Stream feed error: {e}")
                            audio_accumulator = []
                            feed_counter = 0
                except Exception as e:
                    log(f"Recording error: {e}")
                    break

            # Feed any remaining accumulated audio
            if self.stream_ctx is not None and audio_accumulator:
                combined = b"".join(audio_accumulator)
                audio_float = np.frombuffer(
                    combined, dtype=np.int16
                ).astype(np.float32) / 32768.0
                try:
                    self.stream_ctx.add_audio(mx.array(audio_float))
                except Exception as e:
                    log(f"Stream final feed error: {e}")

        self.record_thread = threading.Thread(target=record, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and get streaming transcription result"""
        log("Stop recording requested...")
        if not self.is_recording:
            log("Not currently recording")
            return

        self.is_recording = False

        try:
            self.record_thread.join(timeout=2.0)
        except Exception as e:
            log(f"Error joining record thread: {e}")

        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        except Exception as e:
            log(f"Error closing stream: {e}")

        self.app.title = "⚙️"
        duration = len(self.frames) * CHUNK / SAMPLE_RATE
        log(f"Processing {duration:.1f}s of audio...")
        play_sound("Tink")

        # Check if we have audio data
        if not self.frames:
            log("No audio data recorded")
            self._cleanup_stream_ctx()
            self.app.title = "🎙️"
            safe_notification(
                title="No Audio", subtitle="No audio was recorded", message=""
            )
            play_sound("Funk")
            return

        # Save audio to file for archival
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        audio_path = QUICK_NOTES_PATH / f"{timestamp}.wav"

        try:
            with wave.open(str(audio_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(self.frames))
            log(f"Audio saved: {audio_path}")
        except Exception as e:
            log(f"Error saving audio (non-fatal): {e}")

        # Get streaming transcription result (already processed during recording)
        transcribed_text = ""
        if self.stream_ctx is not None:
            try:
                result = self.stream_ctx.result
                transcribed_text = result.text.strip()
            except Exception as e:
                log(f"Stream result error: {e}")
                import traceback
                log(traceback.format_exc())

        self._cleanup_stream_ctx()

        elapsed = time.time() - self.record_start_time
        log(f"Transcription result ({elapsed:.2f}s total): '{transcribed_text}'")

        if transcribed_text:
            pyperclip.copy(transcribed_text)
            log("Copied to clipboard")
            play_sound("Glass")
            safe_notification(
                title="Transcribed",
                subtitle=transcribed_text[:50]
                + ("..." if len(transcribed_text) > 50 else ""),
                message="Copied to clipboard",
            )
        else:
            log("No speech detected")
            play_sound("Funk")
            safe_notification(
                title="No Speech Detected",
                subtitle="Try speaking louder",
                message="",
            )

        self.app.title = "🎙️"

    def _cleanup_stream_ctx(self):
        """Clean up streaming transcription context"""
        if self.stream_ctx is not None:
            try:
                self.stream_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self.stream_ctx = None

    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def cleanup(self):
        """Clean up resources"""
        self._cleanup_stream_ctx()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class VoiceTranscriptionApp(rumps.App):
    def __init__(self):
        super(VoiceTranscriptionApp, self).__init__(
            "Voice Transcription", icon=None, title="⏳", quit_button=None
        )

        self.transcriber = VoiceTranscriber(self)
        self.last_hotkey_time = 0
        self.hotkey_debounce = 0.5

        # Build menu with device selector
        self.device_menu_items = {}
        self.model_menu_items = {}
        self.build_menu()

        # Load model in background thread
        threading.Thread(target=self.load_model, daemon=True).start()

        # Start keyboard listener
        self.start_keyboard_listener()

    def build_menu(self):
        """Build the menu bar menu"""
        menu_items = [
            rumps.MenuItem("Status: Loading...", callback=None),
            rumps.separator,
            rumps.MenuItem("Alt+R: Transcribe to clipboard", callback=None),
            rumps.separator,
        ]

        # Add input device submenu
        device_menu = self._build_device_menu_items()
        menu_items.append(("Input Device", device_menu))

        # Add Model submenu
        model_menu = []
        for key, info in MODELS.items():
            item = rumps.MenuItem(
                info["name"],
                callback=lambda sender, k=key: self.select_model(k)
            )
            model_menu.append(item)
            self.model_menu_items[key] = item

        menu_items.append(("Model", model_menu))

        menu_items.append(rumps.separator)
        menu_items.append(rumps.MenuItem("Refresh Devices", callback=self.refresh_devices))
        menu_items.append(rumps.MenuItem("Restart App", callback=self.restart_app))
        menu_items.append(rumps.MenuItem("Quit", callback=self.quit_app))

        self.menu = menu_items

        # Update model checkmark
        self.update_model_menu()

    def _build_device_menu_items(self):
        """Build device menu items from current device list"""
        self.device_menu_items.clear()
        device_menu = []

        # Add "Default" option
        default_item = rumps.MenuItem(
            "Default (System)", callback=lambda _: self.select_device(None)
        )
        default_item.state = self.transcriber.input_device_index is None
        device_menu.append(default_item)
        self.device_menu_items[None] = default_item

        # Add all available input devices
        devices = self.transcriber.get_input_devices()
        for device in devices:
            device_item = rumps.MenuItem(
                device["name"],
                callback=lambda sender, idx=device["index"], name=device["name"]: self.select_device(idx, name),
            )
            device_item.state = (device["index"] == self.transcriber.input_device_index)
            device_menu.append(device_item)
            self.device_menu_items[device["index"]] = device_item

        return device_menu

    def refresh_devices(self, _=None):
        """Refresh the device list after hardware changes"""
        log("Refreshing audio devices...")
        devices = self.transcriber.refresh_devices()

        # Clear and rebuild the Input Device submenu
        device_submenu = self.menu["Input Device"]
        device_submenu.clear()

        self.device_menu_items.clear()

        # Add Default option
        default_item = rumps.MenuItem(
            "Default (System)", callback=lambda _: self.select_device(None)
        )
        default_item.state = self.transcriber.input_device_index is None
        device_submenu.add(default_item)
        self.device_menu_items[None] = default_item

        # Add discovered devices
        for device in devices:
            device_item = rumps.MenuItem(
                device["name"],
                callback=lambda sender, idx=device["index"], name=device["name"]: self.select_device(idx, name),
            )
            device_item.state = (device["index"] == self.transcriber.input_device_index)
            device_submenu.add(device_item)
            self.device_menu_items[device["index"]] = device_item

        log(f"Found {len(devices)} input devices")
        safe_notification(
            title="Devices Refreshed",
            subtitle=f"Found {len(devices)} input devices",
            message="",
        )

    def select_model(self, model_key):
        """Handle model selection"""
        self.transcriber.set_model(model_key)
        self.update_model_menu()

    def update_model_menu(self):
        """Update checkmarks for model menu"""
        current = self.transcriber.current_model_key
        for key, item in self.model_menu_items.items():
            item.state = (key == current)

    def select_device(self, device_index, device_name=None):
        """Handle device selection"""
        log(f"Device selection: index={device_index}, name={device_name}")

        # Update checkmarks
        for idx, item in self.device_menu_items.items():
            item.state = idx == device_index

        # Set the device
        self.transcriber.set_input_device(device_index, device_name)

    def load_model(self):
        """Load the Parakeet model"""
        self.transcriber.load_model()
        self.menu["Status: Loading..."].title = "Status: Ready"

    def start_keyboard_listener(self):
        """Start listening for Alt+R hotkey"""

        def event_handler(proxy, event_type, event, refcon):
            try:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                flags = Quartz.CGEventGetFlags(event)

                # Check modifiers
                alt_pressed = (flags & Quartz.kCGEventFlagMaskAlternate) != 0
                shift_pressed = (flags & Quartz.kCGEventFlagMaskShift) != 0
                cmd_pressed = (flags & Quartz.kCGEventFlagMaskCommand) != 0

                # Key code 15 is 'r', only respond to Alt+R (no shift, no cmd)
                if event_type == kCGEventKeyDown and keycode == 15:
                    if alt_pressed and not shift_pressed and not cmd_pressed:
                        current_time = time.time()
                        if current_time - self.last_hotkey_time > self.hotkey_debounce:
                            self.last_hotkey_time = current_time
                            log("Alt+R pressed")
                            threading.Thread(
                                target=self.transcriber.toggle_recording,
                                daemon=True,
                            ).start()
                            return None

            except Exception as e:
                log(f"Event handler error: {e}")
                import traceback
                log(traceback.format_exc())

            return event

        # Create event tap
        try:
            self.event_tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,
                CGEventMaskBit(kCGEventKeyDown),
                event_handler,
                None,
            )

            if self.event_tap is None:
                log("Failed to create event tap - accessibility permissions needed!")
                return

            run_loop_source = Quartz.CFMachPortCreateRunLoopSource(
                None, self.event_tap, 0
            )
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetCurrent(),
                run_loop_source,
                Quartz.kCFRunLoopCommonModes,
            )

            Quartz.CGEventTapEnable(self.event_tap, True)
            log("Hotkey listener started (Alt+R)")

        except Exception as e:
            log(f"Failed to setup event tap: {e}")
            import traceback
            log(traceback.format_exc())

    def quit_app(self, _):
        """Clean up and quit"""
        self.transcriber.cleanup()
        rumps.quit_application()

    def restart_app(self, _):
        """Restart the application"""
        log("Restarting application...")
        self.transcriber.cleanup()

        # Restart the process
        python = sys.executable
        os.execl(python, python, *sys.argv)


def main():
    VoiceTranscriptionApp().run()


if __name__ == "__main__":
    main()
