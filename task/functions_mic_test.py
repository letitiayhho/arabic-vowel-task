import csv
import numpy as np
from psychopy import visual, core, event, gui, sound, prefs
prefs.hardware['audioLib'] = ['ptb']
import soundfile as sf

# =========================
# Helpers
# =========================

def display_text(win, kb, text):
    print(text)
    textstim = visual.TextStim(win, text)
    event.clearEvents(eventType = None)
    textstim.draw()
    win.flip()
    kb.waitKeys(keyList = ['space'])
    win.flip()

def rms(sig):
    if len(sig) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(sig), dtype=np.float64)))

def detect_active_segments(x, sr, frame_ms=10, hangover_ms=50, z=0.5, abs_floor=0.01):
    """Return list of (start_idx, end_idx) samples judged 'active' via simple energy VAD."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop = frame_len  # non-overlapping frames for simplicity
    n = len(x)
    if n < frame_len:
        return []
    # frame RMS
    n_frames = (len(x) - frame_len) // hop + 1
    frames = np.array([x[i*hop : i*hop+frame_len] for i in range(n_frames)])
    frms = np.sqrt((frames * frames).mean(axis=1))
    thr = max(frms.mean() + z * (frms.std() + 1e-9), abs_floor)

    active = frms >= thr
    # expand active frames into sample indices with hangover
    segs = []
    on = None
    hang_frames = max(1, int(hangover_ms / frame_ms))
    hang = 0
    for i, a in enumerate(active):
        if a:
            if on is None:
                on = i
            hang = hang_frames
        else:
            if on is not None:
                if hang > 0:
                    hang -= 1
                    # stay active
                else:
                    # close segment
                    start = on * hop
                    end   = min((i) * hop + frame_len, n)
                    segs.append((start, end))
                    on = None
    if on is not None:
        start = on * hop
        end   = n
        segs.append((start, end))
    return segs

def active_stats(x, sr, frame_ms=10, hangover_ms=50, z=0.5, abs_floor=0.01):
    segs = detect_active_segments(x, sr, frame_ms=10, hangover_ms=50, z=0.5, abs_floor=0.01)
    if not segs:
        return 0.0, 0.0
    total_active = sum((e - s) for s, e in segs)
    act_sig = np.concatenate([x[s:e] for s, e in segs])
    return total_active / sr, rms(act_sig)

def save_wav(path, x, sr):
    sf.write(path, x, sr, subtype="PCM_16")
