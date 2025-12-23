import os
import numpy as np
import parselmouth as pm
from parselmouth.praat import call

def parse_filename(path):
    """
    Expect: .../subj_<ID>/<language>/<ID>_<language>_<trial>_<word>_<length>_<attempt>.wav
    """
    base = os.path.basename(path)                 # "101_23_bik_1.wav"
    stem, _ = os.path.splitext(base)              # "101_23_bik_1"
    parts = stem.split("_")

    if len(parts) != 6: 
        raise ValueError(f"Unable to parse filename {base}")

    trial = int(parts[2])
    attempt = parts[5].split('try')[1]

    return dict(
        subject=parts[0],
        language=parts[1],
        trial=trial,
        word=parts[3],
        length=parts[4],
        attempt=attempt,
    )



def estimate_pitch_range(
    paths,
    time_step=0.01,              # 10 ms frames
    init_floor=50.0, init_ceiling=600.0,
    voicing_threshold=None,      # None = library default
    low_clip=45.0, high_clip=800.0,
    pct_low=5.0, pct_high=95.0,
    margin_low=10.0, margin_high=20.0,
    hard_floor=50.0, hard_ceiling=600.0
):
    """
    Estimate a speaker-specific F0 floor/ceiling from multiple files.
    Returns dict with 'floor', 'ceiling' and diagnostics.
    """
    print('Estimating pitch range')

    all_f0 = []

    for p in paths:
        snd = pm.Sound(p)
        # Build a broad-range pitch track
        if voicing_threshold is None:
            pitch = snd.to_pitch(time_step=time_step, pitch_floor=init_floor, pitch_ceiling=init_ceiling)
        else:
            # Use autocorrelation variant if you want explicit thresholds
            pitch = snd.to_pitch_ac(None, init_floor, 15, False, 0.03, voicing_threshold, 0.01, 0.35, 0.14, init_ceiling)
        f0 = pitch.selected_array['frequency']  # unvoiced -> 0.0
        f0 = f0[np.isfinite(f0) & (f0 > 0)]
        if f0.size:
            all_f0.append(f0)

    if not all_f0:
        return {"floor": hard_floor, "ceiling": hard_ceiling, "n_voiced": 0, "note": "no voiced frames found"}

    f0 = np.concatenate(all_f0)

    # Clip absurd values (pre-clean)
    f0 = f0[(f0 >= low_clip) & (f0 <= high_clip)]
    if f0.size == 0:
        return {"floor": hard_floor, "ceiling": hard_ceiling, "n_voiced": 0, "note": "all frames clipped as outliers"}

    # Rough de-octave around median so halves/doubles fold toward the center
    med = np.median(f0)
    if med > 0:
        f0_fold = f0.copy()
        # fold down big doubles
        f0_fold = np.where(f0_fold/med > 1.9, f0_fold/2.0, f0_fold)
        # fold up obvious halves
        f0_fold = np.where(f0_fold/med < 0.55, f0_fold*2.0, f0_fold)
    else:
        f0_fold = f0

    # Robust outlier filter in log-Hz
    lf0 = np.log(f0_fold)
    med_l = np.median(lf0)
    mad = np.median(np.abs(lf0 - med_l)) + 1e-9
    z = np.abs(lf0 - med_l) / (1.4826 * mad)
    keep = z < 3.5
    f0_clean = f0_fold[keep]

    if f0_clean.size < 30:  # not much data; fall back to less strict filter
        f0_clean = f0_fold

    # Percentile-based bounds + margins, clamped to hard bounds
    lo = np.percentile(f0_clean, pct_low)
    hi = np.percentile(f0_clean, pct_high)
    floor = float(max(hard_floor, lo - margin_low))
    ceiling = float(min(hard_ceiling, hi + margin_high))
    # ensure sensible ordering and minimal span
    if ceiling <= floor + 40:
        ceiling = floor + 80

    return {
        "floor": round(floor, 2),
        "ceiling": round(ceiling, 2),
        "n_voiced": int(f0_clean.size),
        "median_f0": round(float(np.median(f0_clean)), 2),
        "p5": round(float(lo), 2),
        "p95": round(float(hi), 2)
    }
