"""Audible alarm with per-Brainrot cooldown. Windows-only (winsound),
read-only side effect (sound output) — never touches input devices or any
other process.

Cooldown is keyed by the detected Brainrot (its normalized name+tier), not
global: it stops the alarm from repeating while the *same* Prestige
Brainrot is still on screen, without suppressing a genuinely new Prestige
Brainrot that happens to appear again soon after."""
import threading
import time

try:
    import winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False


class AlarmManager:
    def __init__(self, settings):
        a = settings["alarm"]
        self.enabled = a["enabled"]
        self.cooldown_seconds = a["cooldown_seconds"]
        self.frequency_hz = a["frequency_hz"]
        self.duration_ms = a["duration_ms"]
        self.beep_count = a["beep_count"]
        self.sound_file = a.get("sound_file")
        self._last_trigger_time = {}  # key -> timestamp
        self._lock = threading.Lock()
        self._play_lock = threading.Lock()

    def maybe_trigger(self, key, now=None):
        if not self.enabled:
            return False
        now = now if now is not None else time.time()
        with self._lock:
            last = self._last_trigger_time.get(key)
            if last is not None and now - last < self.cooldown_seconds:
                return False
            self._last_trigger_time[key] = now
            # prune old entries so the dict doesn't grow unbounded over a long session
            stale_cutoff = now - self.cooldown_seconds * 4
            self._last_trigger_time = {
                k: t for k, t in self._last_trigger_time.items() if t >= stale_cutoff
            }
        threading.Thread(target=self._play_safe, daemon=True).start()
        return True

    def _play_safe(self):
        # Serialize audio calls so overlapping alerts don't crash or clobber the sound driver
        with self._play_lock:
            self._play()

    def _play(self):
        if self.sound_file and _WINSOUND_AVAILABLE:
            try:
                winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception:
                pass  # Fall back to beeps if file cannot be played

        if not _WINSOUND_AVAILABLE:
            print("\a" * self.beep_count, end="", flush=True)
            return
        for _ in range(self.beep_count):
            try:
                winsound.Beep(self.frequency_hz, self.duration_ms)
            except Exception:
                break
