#!/usr/bin/env python3
"""
Voice Transcription Menu Bar App
Alt+R: Quick transcription to clipboard

Apple Silicon: Powered by NVIDIA Parakeet via MLX (Metal GPU-accelerated)
Intel Mac:     Powered by Whisper via faster-whisper (CTranslate2/int8 CPU-optimized)
"""

import pyaudio
import wave
import pyperclip
import threading
import rumps
import sys
import platform
from pathlib import Path
from datetime import datetime
from Cocoa import NSSound
from Quartz import CGEventMaskBit, kCGEventKeyDown
import Quartz
import numpy as np
import tempfile
import torch
import time

# Detect CPU architecture: arm64 = Apple Silicon, x86_64 = Intel
ARCH = platform.machine()
IS_APPLE_SILICON = (ARCH == "arm64")

if IS_APPLE_SILICON:
    from parakeet_mlx import from_pretrained
else:
    from faster_whisper import WhisperModel

# Setup logging to file
import os
import json

CONFIG_FILE = os.path.expanduser("~/.parakeet_config.json")
LOG_FILE = os.path.expanduser("~/Library/Logs/VoiceTranscription_debug.log")

# Model Constants — architecture-specific
if IS_APPLE_SILICON:
    # MLX Metal GPU-accelerated Parakeet (Apple Silicon only)
    MODELS = {
        "fast": {
            "name": "Fast (0.6b)",
            "path": "mlx-community/parakeet-tdt-0.6b-v3",
        },
        "accurate": {
            "name": "Accurate (1.1b)",
            "path": "mlx-community/parakeet-tdt-1.1b",
        },
    }
else:
    # CTranslate2/int8 Whisper models — optimized for Intel CPU via AVX/AVX2
    MODELS = {
        "fast": {
            "name": "Fast (Whisper Medium)",
            "size": "medium",
        },
        "accurate": {
            "name": "Accurate (Whisper Large-v3)",
            "size": "large-v3",
        },
    }

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
RECORDING_ARCHIVE_PATH = Path.home() / "MEDIA" / "recordings"
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
        """Load the transcription model with warmup for fast first inference.

        Apple Silicon: loads parakeet-mlx (MLX/Metal GPU-accelerated)
        Intel Mac:     loads faster-whisper (CTranslate2/int8 CPU-optimized)
        """
        if self.model_loaded:
            return

        self.app.title = "⏳"
        model_info = MODELS.get(self.current_model_key, MODELS["fast"])

        try:
            if IS_APPLE_SILICON:
                log(f"Loading Parakeet model: {model_info['name']} ({model_info['path']})...")
                self.model = from_pretrained(model_info['path'])
                log("Warming up model (JIT compilation)...")
                self._warmup_model()
            else:
                model_size = model_info['size']
                log(f"Loading Whisper {model_info['name']} on Intel CPU (CTranslate2/int8)...")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                log(f"Whisper {model_info['name']} loaded (Intel CPU optimized)")

            self.model_loaded = True
            self.app.title = "🎙️"
            log(f"Model ({model_info['name']}) loaded and ready!")
            safe_notification(
                title="Model Ready",
                subtitle=f"Loaded {model_info['name']}",
                message="Press Alt+R to transcribe",
            )
        except Exception as e:
            self.app.title = "❌"
            log(f"Error loading model: {e}")
            import traceback
            log(traceback.format_exc())
            safe_notification(
                title="Error Loading Model",
                subtitle=f"Failed to load {model_info['name']}",
                message=str(e),
            )

        # Load VAD model in parallel
        threading.Thread(target=self.load_vad_model, daemon=True).start()

    def _warmup_model(self):
        """Run a warmup transcription to trigger JIT compilation (Apple Silicon only).

        Enables local attention mode (reduces compute) and runs a silent audio pass
        so MLX/Metal compiles the kernels before the first real recording.
        """
        if not IS_APPLE_SILICON:
            return

        try:
            self.model.encoder.set_attention_model("rel_pos_local_attn", (256, 256))
            log("Set local attention mode (256, 256)")

            warmup_samples = np.zeros(8000, dtype=np.float32)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                warmup_path = f.name
                with wave.open(warmup_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes((warmup_samples * 32767).astype(np.int16).tobytes())

            _ = self.model.transcribe(warmup_path)
            Path(warmup_path).unlink(missing_ok=True)
            log("Model warmup complete")
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
        """Start recording audio for quick transcription to clipboard"""
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
                safe_notification(
                    title="Recording Error",
                    subtitle="Failed to start recording",
                    message=str(e2),
                )
                return

        play_sound("Tink")

        def record():
            while self.is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    log(f"Recording error: {e}")
                    break

        self.record_thread = threading.Thread(target=record, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and transcribe to clipboard"""
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

        if not self.frames:
            log("No audio data recorded")
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
            log(f"Error saving audio: {e}")
            import traceback
            log(traceback.format_exc())
            self.app.title = "❌"
            play_sound("Funk")
            return

        # Transcribe in background
        threading.Thread(
            target=self._transcribe_and_copy, args=(audio_path,), daemon=True
        ).start()

    def _transcribe_and_copy(self, audio_path):
        """Transcribe audio and copy to clipboard.

        Apple Silicon: uses parakeet-mlx (.transcribe returns a result with .text)
        Intel Mac:     uses faster-whisper (.transcribe yields Segment objects)
        """
        try:
            log("Starting transcription...")
            start_time = time.time()

            if IS_APPLE_SILICON:
                result = self.model.transcribe(
                    str(audio_path),
                    chunk_duration=30.0,
                    overlap_duration=2.0,
                )
                transcribed_text = result.text.strip()
            else:
                segments, _ = self.model.transcribe(str(audio_path), beam_size=5)
                transcribed_text = " ".join(seg.text for seg in segments).strip()

            elapsed = time.time() - start_time
            log(f"Transcription ({elapsed:.2f}s): '{transcribed_text}'")

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
        except Exception as e:
            log(f"Transcription error: {e}")
            import traceback
            log(traceback.format_exc())
            play_sound("Funk")
        finally:
            self.app.title = "🎙️"

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
            rumps.MenuItem(
                ("Alt+R" if IS_APPLE_SILICON else "Ctrl+R") + ": Transcribe to clipboard",
                callback=None
            ),
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
        """Start listening for the transcription hotkey.

        Apple Silicon: Alt+R (Option+R)
        Intel Mac:     Ctrl+R — Option+R on Intel produces ® and conflicts
        """
        if IS_APPLE_SILICON:
            HOTKEY_LABEL = "Alt+R"
        else:
            HOTKEY_LABEL = "Ctrl+R"

        def event_handler(proxy, event_type, event, refcon):
            try:
                keycode = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
                flags = Quartz.CGEventGetFlags(event)

                # Check modifiers
                alt_pressed = (flags & Quartz.kCGEventFlagMaskAlternate) != 0
                ctrl_pressed = (flags & Quartz.kCGEventFlagMaskControl) != 0
                shift_pressed = (flags & Quartz.kCGEventFlagMaskShift) != 0
                cmd_pressed = (flags & Quartz.kCGEventFlagMaskCommand) != 0

                # Key code 15 is 'r'
                if event_type == kCGEventKeyDown and keycode == 15:
                    if IS_APPLE_SILICON:
                        # Alt+R (no shift, no cmd)
                        triggered = alt_pressed and not shift_pressed and not cmd_pressed
                    else:
                        # Ctrl+R (no alt, no shift, no cmd) — avoids ® conflict on Intel
                        triggered = ctrl_pressed and not alt_pressed and not shift_pressed and not cmd_pressed

                    if triggered:
                        current_time = time.time()
                        if current_time - self.last_hotkey_time > self.hotkey_debounce:
                            self.last_hotkey_time = current_time
                            log(f"{HOTKEY_LABEL} pressed")
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
            log(f"Hotkey listener started ({HOTKEY_LABEL})")

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
