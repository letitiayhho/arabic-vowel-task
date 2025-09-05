import os, csv, time, math, queue
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import random
from psychopy import visual, core, event, gui, sound, prefs
prefs.hardware['audioLib'] = ['ptb']
from functions_english import *

# =========================
# Ask for participant ID
# =========================
PID = input("Enter participant ID (e.g. 10X): ")

# =========================
# Config
# =========================
SAVE_DIR        = f"data/subj_{PID}/english"
SAMPLE_RATE     = 44100
CHANNELS        = 1

# Recording thresholds
MIN_DUR = 0.15   # 150 ms
MAX_REC = 2.00
MIN_ACTIVE_RMS  = 0.015

# Energy detection parameters
FRAME_MS        = 10       # frame step for activity detection
HANGOVER_MS     = 50       # keep activity “on” this long after falling below threshold
ACTIVITY_ZSCORE = 0.5      # relative threshold: frames above (mean + z * std)
ABS_FLOOR       = 0.01     # absolute floor on frame RMS to avoid too-low thresholds

# Task flow
MAX_RETRIES_PER_ITEM = 3

# =========================
# UI setup
# =========================
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)

words = ['hat', 'hit', 'heck', 'hat', 'hoot', 'hut']*5
random.shuffle(words)

# PsychoPy window & sounds
win = visual.Window(fullscr=False, size=(1200, 800), units="pix", color=[-0.5, -0.5, -0.5])
red_dot = visual.Circle(win=win, radius=10, fillColor=[1, -1, -1], lineColor=None)

# Log file
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(SAVE_DIR, f"{PID}_english_vowels_{stamp}.csv")
with open(log_path, "w", newline="", encoding="utf-8") as lf:
    writer = csv.writer(lf)
    writer.writerow([
        "pid",
        "timestamp",
        "language",
        "trial_index",
        "word",
        "rec_path",
        "active_duration_s",
        "active_rms",
        "passed",
        "retries_used",
        "sr",
        "channels",
        "min_dur_s",
        "min_rms"
    ])

# Instructions
display_text(win, "Welcome to the English pronunciation task\n\nPress SPACE to begin.")
display_text(win, "In this task you will be asked to pronounce a series of English words. English words will appear on the screen one at a time. Press space to begin your recording and say each word out loud. Try your best to speak clearly and sustain the vowel of the word, the recording will end automatically. Words with long vowels will require a longer recording than short vowels. The program will re-prompt if the response is too short or too quiet. \n \n This task will last approximately 15 minutes.")

# =========================
# Trial loop
# =========================
global_clock = core.Clock()
trial_idx = 0

for word in words:
    trial_idx += 1
    retries = 0
    passed = False
    rec_path = ""
    measured_active_dur = 0.0
    measured_active_rms = 0.0

    while retries < MAX_RETRIES_PER_ITEM and not passed:
        # Prompt
        display_text(win, f"{word}\n\n")

        # Display dot
        red_dot.draw()
        win.flip()

        # Record (fixed window)
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS
        duration = MAX_REC
        rec = sd.rec(int(duration * SAMPLE_RATE), dtype='float32')
        sd.wait()

        # End recording
        win.flip()

        # Quick trim silence at both ends (soft-trim)
        x = rec.flatten()
        # Simple endpointing by energy threshold (same parameters as VAD)
        segs = detect_active_segments(x, SAMPLE_RATE, FRAME_MS, HANGOVER_MS, ACTIVITY_ZSCORE, ABS_FLOOR)
        if segs:
            start = max(0, segs[0][0] - int(0.02 * SAMPLE_RATE))  # 20ms pre-roll
            end   = min(len(x), segs[-1][1] + int(0.02 * SAMPLE_RATE))  # 20ms post-roll
            x_trim = x[start:end]
        else:
            x_trim = x

        act_dur_s, act_rms = active_stats(x_trim, SAMPLE_RATE)
        measured_active_dur = act_dur_s
        measured_active_rms = act_rms
        passed = (act_dur_s >= MIN_DUR) and (act_rms >= MIN_ACTIVE_RMS)

        # Save WAV
        wav_name = f"{PID}_{trial_idx:03d}_{word}_try{retries}.wav"
        rec_path = os.path.join(SAVE_DIR, wav_name)
        save_wav(rec_path, x_trim, SAMPLE_RATE)

        # Feedback
        if passed:
            core.wait(1)
            continue
        else:
            if act_dur_s < MIN_DUR: 
                reason = 'short'
            if act_rms < MIN_ACTIVE_RMS: 
                reason = 'quiet'
            fb = f"Your recording was too {reason}, please try again."
            retries += 1
            print(f"retries: {retries}")
            display_text(win, fb + "\n\n Press SPACE to retry.")


    # Log
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow([
            PID,
            datetime.now().isoformat(timespec="seconds"),
            "english",
            trial_idx,
            word,
            rec_path,
            f"{measured_active_dur:.4f}",
            f"{measured_active_rms:.4f}",
            int(passed),
            retries,
            SAMPLE_RATE,
            CHANNELS,
            f"{MIN_DUR:.3f}",
            f"{MIN_ACTIVE_RMS:.3f}"
        ])

# Wrap up
txt.text = "All done. Thank you!"
sub.text = f"Data saved to: {os.path.abspath(SAVE_DIR)}\nLog file: {os.path.abspath(log_path)}\nPress any key to exit."
txt.draw(); sub.draw(); win.flip()
event.waitKeys()
win.close()
core.quit()
