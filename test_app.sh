#!/bin/bash
# Quick test script to check if Voice Transcription app is working

echo "=== Voice Transcription App Status ==="
echo ""

# Check if process is running
PID=$(launchctl list | grep voicetranscription | awk '{print $1}')
if [ -n "$PID" ] && [ "$PID" != "-" ]; then
    echo "✅ Service is running (PID: $PID)"
else
    echo "❌ Service is NOT running"
    exit 1
fi

# Check process details
echo ""
echo "Process details:"
ps aux | grep whisper_flow_app | grep -v grep | head -1

echo ""
echo "=== Recent Logs ==="
echo ""
echo "--- Application Log (last 10 lines) ---"
tail -10 /Users/gassandrid/Library/Logs/VoiceTranscription.log 2>/dev/null || echo "(No logs yet)"

echo ""
echo "--- Errors (excluding accessibility warnings) ---"
grep -v "This process is not trusted" /Users/gassandrid/Library/Logs/VoiceTranscription.error.log 2>/dev/null | grep -v "^$" | tail -5 || echo "(No errors)"

echo ""
echo "=== Instructions ==="
echo "1. Look for the microphone icon (🎙️) in your menu bar"
echo "2. Grant Accessibility permission: System Preferences → Security & Privacy → Accessibility"
echo "3. Press Alt+R to start recording"
echo "4. Press Alt+R again to stop and transcribe"
echo "5. Text will be copied to clipboard - paste with Cmd+V"
echo ""
echo "To view live logs:"
echo "  tail -f ~/Library/Logs/VoiceTranscription.log"
