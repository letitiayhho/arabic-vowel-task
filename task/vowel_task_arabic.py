import os, csv, time, math, queue
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import pandas as pd
from psychopy import visual, core, event, gui, sound, prefs, monitors
from functions_arabic import *
from motu import *
from psychopy.hardware import keyboard

# =========================
# Ask for participant ID
# =========================
PID = input("Enter participant ID (e.g. 10X): ")

# =========================
# Config
# =========================
WORDS_CSV_PATH  = "arabic_words.csv"
SAVE_DIR        = f"../data/subj_{PID}/arabic"
SAMPLE_RATE     = 48000
CHANNELS        = 1
MOTU_INDEX 	= 3

# Duration thresholds for ACTIVE speech (not total recording length)
MIN_DUR_SHORT_S = 0.12   # 120 ms
MIN_DUR_LONG_S  = 0.22   # 200 ms

# Recording window caps (enough time to speak; not used for pass/fail)
MAX_REC_SHORT_S = 1.20
MAX_REC_LONG_S  = 2.20

# Volume threshold (RMS of active segment). Start modest; adjust after a pilot.
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

items = pd.read_csv(WORDS_CSV_PATH)
items = items.sample(frac=1).reset_index(drop=True)

# PsychoPy window & sounds
kb = keyboard.Keyboard() 
win = visual.Window(
    fullscr = True, 
    size = [1920, 1200],
    screen = -1, 
    pos = (0, 0),
    units = "pix", 
    allowGUI = False,
    winType = 'glfw',
    color=[-0.5, -0.5, -0.5])
red_dot = visual.Circle(win=win, radius=10, fillColor=[1, -1, -1], lineColor=None)

# Microphone settings
sd.default.device = (MOTU_INDEX, None)   # input only
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1
sd.default.dtype = 'float32'
sd.check_input_settings(device= MOTU_INDEX, channels=1, samplerate=SAMPLE_RATE)

# Log file
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(SAVE_DIR, f"{PID}_arabic_vowels_{stamp}.csv")
with open(log_path, "w", newline="", encoding="utf-8") as lf:
    writer = csv.writer(lf)
    writer.writerow([
        "pid",
        "timestamp",
        "language",
        "trial_index",
        "word",
	"vowel",
        "vlen",
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
display_text(win, kb, "Welcome to the Arabic pronunciation task\n\nPress SPACE to begin.")
display_text(win, kb, "In this task you will be asked to pronounce a series of Arabic words. English transliterations of Arabic of words will appear on the screen one at a time. Press space to begin your recording and say each word out loud when the red dot appears on the screen. Try your best to speak clearly and sustain the vowel of the word, the recording will end automatically. Words with long vowels will require a longer recording than short vowels. The program will re-prompt if the response is too short or too quiet. \n \n This task will last approximately 10 minutes.")

# =========================
# Trial loop
# =========================
global_clock = core.Clock()

for index, item in items.iterrows():
    trial_idx = index + 1
    word = item["word"]
    vowel = item["vowel"]
    vlen = item["vlen"]
    min_dur = MIN_DUR_LONG_S if vlen == "long" else MIN_DUR_SHORT_S
    max_rec = MAX_REC_LONG_S if vlen == "long" else MAX_REC_SHORT_S

    retries = 0
    passed = False
    rec_path = ""
    measured_active_dur = 0.0
    measured_active_rms = 0.0

    while retries < MAX_RETRIES_PER_ITEM and not passed:
        # Prompt
        display_text(win, kb, f"{word}\n\n")

        # Display dot
        red_dot.draw()
        win.flip()

        # Record (fixed window)
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS
        duration = max_rec
        rec = sd.rec(int(duration * SAMPLE_RATE), dtype='float32', blocking = True)
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
        passed = (act_dur_s >= min_dur) and (act_rms >= MIN_ACTIVE_RMS)

        # Save WAV
        wav_name = f"{PID}_arabic_{trial_idx:03d}_{word}_{vowel}_{vlen}_try{retries}.wav"
        rec_path = os.path.join(SAVE_DIR, wav_name)
        save_wav(rec_path, x_trim, SAMPLE_RATE)

        # Feedback
        if passed:
            core.wait(1)
            continue
        else:
            if act_dur_s < min_dur: 
                reason = 'short'
            if act_rms < MIN_ACTIVE_RMS: 
                reason = 'quiet'
            fb = f"Your recording was too {reason}, please try again."
            retries += 1
            print(f"retries: {retries}")
            display_text(win, kb, fb + "\n\n Press SPACE to retry.")


    # Log
    with open(log_path, "a", newline="", encoding="utf-8") as lf:
        writer = csv.writer(lf)
        writer.writerow([
            PID,
            datetime.now().isoformat(timespec="seconds"),
            "arabic",
            trial_idx,
            word,
            vowel,
            vlen,
            rec_path,
            f"{measured_active_dur:.4f}",
            f"{measured_active_rms:.4f}",
            int(passed),
            retries,
            SAMPLE_RATE,
            CHANNELS,
            f"{min_dur:.3f}",
            f"{MIN_ACTIVE_RMS:.3f}"
        ])

# Wrap up
display_text(win, kb, "All done. Thank you! Press space one more time to exit the program. Your experimenter will be with you shortly.")
print(f"Data saved to: {os.path.abspath(SAVE_DIR)}\nLog file: {os.path.abspath(log_path)}\nPress any key to exit.")
win.close()
core.quit()
