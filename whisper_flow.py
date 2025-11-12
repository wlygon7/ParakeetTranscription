#!/usr/bin/env python3
"""
Voice Transcription Tool - Press hotkey to record and transcribe speech
Inspired by Whisper Flow, powered by NVIDIA Parakeet on Apple Silicon via MLX
"""

import pyaudio
import wave
import tempfile
import pyperclip
import threading
from pathlib import Path
from pynput import keyboard
from parakeet_mlx import from_pretrained

# Configuration
HOTKEY = keyboard.Key.f12  # Change to your preferred hotkey (F13, F14, etc.)
SAMPLE_RATE = 16000  # Parakeet expects 16kHz
CHANNELS = 1  # Mono
CHUNK = 1024
FORMAT = pyaudio.paInt16


class VoiceTranscriber:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.model = None

        print("Loading Parakeet model...")
        self.model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
        print("Model loaded! Ready to transcribe.")
        print(f"Press {HOTKEY} to start/stop recording")

    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return

        self.is_recording = True
        self.frames = []

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        print("\n🎤 Recording... (press hotkey again to stop)")

        # Record in a separate thread
        def record():
            while self.is_recording:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break

        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()

    def stop_recording(self):
        """Stop recording and transcribe"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_thread.join()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        print("⏹️  Recording stopped. Transcribing...")

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name

            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b"".join(self.frames))

        # Transcribe
        try:
            result = self.model.transcribe(temp_path)
            transcribed_text = result.text.strip()

            if transcribed_text:
                # Copy to clipboard (will paste at cursor when you Cmd+V)
                pyperclip.copy(transcribed_text)
                print(f"✅ Transcribed: {transcribed_text}")
                print("📋 Copied to clipboard - paste with Cmd+V")
            else:
                print("⚠️  No speech detected")
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

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


def main():
    transcriber = VoiceTranscriber()

    def on_press(key):
        if key == HOTKEY:
            transcriber.toggle_recording()

    # Start keyboard listener
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            transcriber.cleanup()


if __name__ == "__main__":
    main()
