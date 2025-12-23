#!/usr/bin/env python3

import sys, math, glob, os, numpy as np, pandas as pd
import parselmouth as pm
from parselmouth.praat import call
from functions import *

# ------------------------
# Config
# ------------------------
INPUT_ROOT = "../exemplars"               # where subj_*/ live
GLOB_PAT   = "subj_*/*/*.wav"        # adjust depth as needed
OUTPUT_CSV = "exemplar-formants-2.csv"

# Praat settings
MAX_FORMANT_HZ = 5500       # ~5000 male-only, 5500–6000 female/mixed
NFORMANTS      = 5.0
WINLEN_S       = 0.025
PREEMPH_HZ     = 50

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":

    pattern = os.path.join(INPUT_ROOT, GLOB_PAT)
    files = sorted(glob.glob(pattern, recursive=True))
    params = estimate_pitch_range(files)
    floor = params['floor']
    ceiling = params['ceiling']

    rows = []
    for path in files:
        meta = parse_filename(path)
        snd = pm.Sound(path)

        formants = snd.to_formant_burg(None, NFORMANTS, MAX_FORMANT_HZ, WINLEN_S, PREEMPH_HZ)
        pitch = snd.to_pitch(pitch_floor = floor, pitch_ceiling = ceiling)
        dur = snd.get_total_duration()
        tc = dur/2

        F0 = pitch.get_value_at_time(tc)
        F1 = formants.get_value_at_time(1, tc)
        F2 = formants.get_value_at_time(2, tc)
        F3 = formants.get_value_at_time(3, tc)

        row = {
            **meta,
            "file": os.path.basename(path),
            "path": os.path.abspath(path),
            "duration_s": dur,
            "t_center_s": tc,
            "F0_Hz": float("nan") if math.isnan(F0) else float(F0),
            "F1_Hz": float("nan") if math.isnan(F1) else float(F1),
            "F2_Hz": float("nan") if math.isnan(F2) else float(F2),
            "F3_Hz": float("nan") if math.isnan(F3) else float(F3),
        }
        rows.append(row)


    df = pd.DataFrame(rows, columns=[
        "subject","language", "trial","word","attempt","file","path",
        "duration_s","t_center_s","F0_Hz","F1_Hz","F2_Hz", "F3_Hz"
    ])

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Wrote {len(df)} rows to {OUTPUT_CSV}")

