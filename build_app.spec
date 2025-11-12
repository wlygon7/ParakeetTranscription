# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['whisper_flow_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'parakeet_mlx',
        'mlx',
        'mlx.core',
        'mlx.nn',
        'mlx.utils',
        'rumps',
        'pyaudio',
        'pynput',
        'pynput.keyboard',
        'pynput.keyboard._darwin',
        'pyperclip',
        'wave',
        'tempfile',
        'numpy',
        'huggingface_hub',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VoiceTranscription',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoiceTranscription',
)

app = BUNDLE(
    coll,
    name='VoiceTranscription.app',
    icon=None,
    bundle_identifier='com.yourname.voicetranscription',
    info_plist={
        'NSMicrophoneUsageDescription': 'This app needs microphone access to record your voice for transcription.',
        'LSUIElement': False,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'NSHighResolutionCapable': True,
    },
)
