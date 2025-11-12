"""
Setup script to build macOS app bundle
"""
from setuptools import setup

APP = ['whisper_flow_app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,
    'plist': {
        'CFBundleName': 'VoiceTranscription',
        'CFBundleDisplayName': 'Voice Transcription',
        'CFBundleGetInfoString': 'Voice Transcription Tool',
        'CFBundleIdentifier': 'com.yourname.voicetranscription',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSMicrophoneUsageDescription': 'This app needs microphone access to record your voice for transcription.',
        'LSUIElement': False,  # Set to True to hide from dock
    },
    'packages': [
        'rumps',
        'pyaudio',
        'pynput',
        'pyperclip',
        'parakeet_mlx',
        'mlx',
        'numpy',
        'wave',
        'tempfile',
        'threading',
    ],
    'includes': [
        'parakeet_mlx',
    ],
}

setup(
    name='VoiceTranscription',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
