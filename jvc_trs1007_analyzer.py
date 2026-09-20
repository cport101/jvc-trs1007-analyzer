#!/usr/bin/env python3
"""
-------------------
TRS1007_ANALYZER_V2.PY
-------------------

V2 JVC/Victor TRS-1007 frequency-response, distortion and crosstalk analyzer.

Inspired by Scott Wurcer's swept-frequency analysis concept and the
MIT-licensed FidelisAnalog SJPlot project [John P. Jones III of FidelisAnalog]:

    https://github.com/FidelisAnalog/SJPlot

TRS-1007 Side A organization handled here
------------------------------------------
Band 1, tracks A1-A3 : 20 Hz -> 20 kHz sweep, LEFT channel driven
Band 2, tracks A4-A6 : 20 Hz -> 20 kHz sweep, RIGHT channel driven
Band 3, tracks A7-A9 : 20 Hz -> 20 kHz stereo L+/R+ sweep (optional)

Frequency Response/Crosstalk/Distortion	Tracks 
Band 1	
A1		Pilot Signal 1,000Hz (L+R)	0:10
A2		Sweep 20Hz-20kHz (L)	0:50
A3		Non.Modulation	0:30
Band 2	
A4		Pilot Signal 1,000Hz (L+R)	0:10
A5		Sweep 20Hz-20kHz (R)	0:50
A6		Non.Modulation	0:30
Band 3	
A7		Pilot Signal 1,000Hz (L+R)	0:10
A8		Sweep 20Hz-20kHz (L+R)	0:50
A9		Non.Modulation	0:30
Band 4	
A10		Pilot Signal 1,000Hz (L-R)	0:10
A11		Sweep 20Hz-20kHz (L-R)	0:50
Reference Signal	
Band 5	
A12		Spot 1,000Hz (L)	0:10
Band 6	
A13		Spot 1,000Hz (R)	0:10
Band 7	
A14		Spot 1,000Hz (L+R)	0:10
Band 8	
A15		Spot 1,000Hz (L-R)	0:10
Frequency Response/Crosstalk/Distortion	
Band 9	
A16		Pilot Signal 1,000Hz (L+R)	0:10
A17		Sweep 20Hz-20kHz (L)	0:50
A18		Non.Modulation	0:30
Band 10	
A19		Pilot Signal 1,000Hz (L+R)	0:10
A20		Sweep 20Hz-20kHz (R)	0:50
A21		Non.Modulation	0:30
Band 11	
A22		Pilot Signal 1,000Hz (L+R)	0:10
A23		Sweep 20Hz-20kHz (L+R)	0:50
A24		Non.Modulation	0:30
Band 12	
A25		Pilot Signal 1,000Hz (L-R)	0:10
A26		Sweep 20Hz-20kHz (L-R)	0:50
Each sweep is preceded by a 1 kHz pilot and followed by a blank track.

Measurements
------------
Band 1:
    Left frequency response
    R/L crosstalk
    Left HD2 and HD3

Band 2:
    Right frequency response
    L/R crosstalk
    Right HD2 and HD3

Band 3:
    Stereo L+ and R+ frequency responses
    (no crosstalk measurement because both channels are intentionally driven)

The script accepts either:
    * an explicitly selected band (--band left/right/stereo), or
    * --band auto, which finds all pilot/sweep sections in a continuous
      recording and combines LEFT, RIGHT, and optional STEREO measurements.

Frequency response is normalized at 1 kHz.

TRS-1007 high-frequency correction
----------------------------------
The record omits the standard RIAA 75-us high-frequency recording
pre-emphasis. Captures made through a standard RIAA playback stage are
therefore corrected by the inverse 75-us playback pole before plotting.
This correction is enabled by default; use --no-trs1007-compensation
only for a capture chain that has already compensated this characteristic.

Dependencies:
    python3 -m pip install numpy scipy matplotlib

Example:
    python3 trs1007_analyzer.py trs1007.wav \
        --band left \
        --cartridge "Audio-Technica AT33PTG/II" \
        --info "47k / 100pF / SME V / Puffin / RME ADC"

SPDX-License-Identifier: MIT
"""
# ================================================================================ 
# Copyright 2026  Scott Wurcer & John P. Jones III of FidelisAnalog
# ================================================================================ 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the “Software”), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert, windows

# =====================================================================
# Constants
# =====================================================================

FMIN = 20.0
FMAX = 20_000.0
FREF = 1_000.0

WINDOW_SECONDS = 0.100
HOP_FRACTION = 0.25
POINTS_PER_OCTAVE = 24

# V2 sweep tracking / analysis
RIDGE_WINDOW_SECONDS = 0.080
RIDGE_HOP_SECONDS = 0.040
RIDGE_MIN_POINTS = 40

# Pilot detector
PILOT_LOW = 900.0
PILOT_HIGH = 1100.0
PILOT_MIN_SECONDS = 0.50
MULTI_PILOT_MIN_SECONDS = 4.0

# Nominal short gap between pilot and sweep.
SWEEP_START_DELAY = 0.035

EPS = np.finfo(float).tiny

# TRS-1007 recording characteristic:
# The high-frequency (75 us / 2122 Hz) RIAA recording time constant is
# intentionally omitted. When the record is reproduced through a normal
# RIAA phono stage, its 75-us playback de-emphasis therefore remains visible
# in the captured sweep. Correct it in the measured spectral amplitudes.
RIAA_TREBLE_TC = 75e-6
# IMPORTANT -- DEFAULT
TRS1007_COMPENSATE = True

# =====================================================================
# Utility functions
# =====================================================================


def db(x):
    """Convert linear amplitude ratio to dB."""
    return 20.0 * np.log10(np.maximum(np.asarray(x), EPS))


def safe_ratio(a, b):
    """Return a/b while protecting against divide-by-zero."""
    return np.asarray(a) / np.maximum(np.asarray(b), EPS)


def truncate(text, length=50):
    """Limit plot metadata to requested maximum length."""
    text = str(text).strip()
    return text[:length]


# =====================================================================
# WAV input
# =====================================================================


def read_wav(filename):
    """
    Read stereo WAV and convert samples to float64.

    Returns
    -------
    fs : int
        Sampling frequency.

    audio : ndarray, shape (samples, 2)
        Floating-point stereo audio.
    """

    fs, x = wavfile.read(filename)

    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(
            "TRS-1007 channel/crosstalk analysis requires a stereo WAV.")

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), info.max)
        x = x.astype(np.float64) / scale
    else:
        x = x.astype(np.float64)

    x = x[:, :2]

    # Remove ADC DC offsets independently.
    x -= np.mean(x, axis=0)

    return fs, x


# =====================================================================
# Multi-sweep pilot detection / segmentation
# =====================================================================


def find_all_pilots(audio, fs):
    """Return all substantial 1-kHz pilot regions in chronological order.

    The detector uses the summed L/R 1-kHz envelopes, so it works for
    left-only, right-only, and L+/R+ pilots. Nearby detections are merged.
    """
    env_l = pilot_envelope(audio[:, 0], fs)
    env_r = pilot_envelope(audio[:, 1], fs)
    env = env_l + env_r

    if not np.any(np.isfinite(env)) or np.nanmax(env) <= 0:
        return []

    # A global threshold works well for captures containing Band 1/2/3;
    # use a lower threshold than find_pilot() so a quieter opposite band
    # is not lost because one pilot happens to be louder.
    threshold = 0.12 * np.nanmax(env)
    active = env > threshold
    edges = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    minimum = int(MULTI_PILOT_MIN_SECONDS * fs)
    candidates = [(int(a), int(b)) for a, b in zip(starts, ends)
                  if b - a >= minimum]

    # Merge regions separated by less than 150 ms.  This prevents small
    # envelope notches inside a pilot from creating two pilots.
    merged = []
    merge_gap = int(0.150 * fs)
    for a, b in candidates:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def classify_sweep(audio, fs, pilot_end, segment_end):
    """Classify the test from channel dominance during the sweep.

    TRS-1007 pilots may be recorded in both channels even when the following
    sweep drives only Left or Right.  Classification therefore uses energetic
    0.5-second frames after the pilot, not the pilot's L/R ratio.
    """
    start = pilot_end + int(SWEEP_START_DELAY * fs)
    stop = min(segment_end, len(audio))
    if stop - start < int(1.0 * fs):
        return "stereo", 0.0

    frame = max(1, int(0.5 * fs))
    levels = []
    for pos in range(start, stop - frame + 1, frame):
        block = audio[pos:pos + frame]
        lrms = rms(block[:, 0])
        rrms = rms(block[:, 1])
        total = np.sqrt(lrms * lrms + rrms * rrms)
        levels.append((total, db((lrms + EPS) / (rrms + EPS))))

    if not levels:
        return "stereo", 0.0

    totals = np.asarray([v[0] for v in levels])
    ratios = np.asarray([v[1] for v in levels])
    peak = np.nanmax(totals)

    # Ignore blank-groove/noise frames and use the median so clicks do not
    # determine the classification.
    keep = totals >= max(peak * 0.20, EPS)
    difference = float(np.nanmedian(ratios[keep])) if np.any(keep) else 0.0

    if difference >= 6.0:
        return "left", difference
    if difference <= -6.0:
        return "right", difference
    return "stereo", difference


def analyze_audio_segment(audio, fs, band, pilot_difference, filename,
                          pilot_end, segment_end):
    """Analyze exactly one pilot-delimited TRS-1007 sweep segment."""
    sweep_start = pilot_end + int(SWEEP_START_DELAY * fs)
    sweep_end = max(sweep_start, segment_end)
    sweep = audio[sweep_start:sweep_end]

    if len(sweep) < int(0.5 * fs):
        raise RuntimeError(f"{filename}: sweep segment is too short")

    if band == "left":
        tracking_channel = 0
    elif band == "right":
        tracking_channel = 1
    else:
        tracking_channel = 0 if pilot_difference >= 0 else 1

    raw = analyze_sweep(sweep, fs, tracking_channel)
    keys = ("L", "R", "L_HD2", "R_HD2", "L_HD3", "R_HD3")
    binned = log_bin(raw["frequency"], *(raw[k] for k in keys))
    result = {"frequency": binned[0]}
    for key, values in zip(keys, binned[1:]):
        result[key] = values

    result.update({
        "band": band,
        "filename": Path(filename).name,
        "sample_rate": fs,
        "pilot_difference": pilot_difference,
    })
    if len(result["frequency"]) < 10:
        raise RuntimeError(
            f"{filename}: insufficient sweep data for {band} band")
    return result


def analyze_combined_recording(filename):
    """Find and analyze Band 1/2/3 sweeps contained sequentially in one WAV.

    Each 1-kHz pilot starts a new test.  A test is bounded by the next
    detected pilot (or EOF for the final test), preventing one sweep from
    leaking into the next sweep in a continuous capture.

    Returns a dict with optional keys: left, right, stereo.
    """
    path = Path(filename)
    fs, audio = read_wav(path)
    pilots = find_all_pilots(audio, fs)
    if not pilots:
        raise RuntimeError(f"{path.name}: no valid 1-kHz pilots found")

    print(f"{path.name}: found {len(pilots)} pilot(s)")
    found = {}

    for i, (pstart, pend) in enumerate(pilots):
        # End before the next genuine, long pilot.  The short interval during
        # which a logarithmic sweep passes through 1 kHz is deliberately not
        # considered a pilot.
        segment_end = pilots[i + 1][0] if i + 1 < len(pilots) else len(audio)
        band, difference = classify_sweep(audio, fs, pend, segment_end)
        pilot_band, pilot_difference = detect_band(audio, pstart, pend)
        print(f"  pilot {i+1}: {pstart/fs:.2f}-{pend/fs:.2f}s, "
              f"pilot={pilot_band} ({pilot_difference:+.1f} dB), "
              f"sweep={band} ({difference:+.1f} dB)")
        try:
            result = analyze_audio_segment(audio, fs, band, difference,
                                           path.name, pend, segment_end)
        except RuntimeError as exc:
            print(f"  WARNING: {exc}")
            continue

        # Keep the first valid occurrence of each band. TRS-1007 tracks 1-3,
        # 4-6, and 7-9 may contain repeated equivalent test tracks.
        if band not in found:
            found[band] = result

    return {
        "left": found.get("left"),
        "right": found.get("right"),
        "stereo": found.get("stereo"),
    }


# =====================================================================
# Analyze Recording
# =====================================================================


def analyze_recording(filename, requested_band="auto"):
    """
    Completely analyze one TRS-1007 recording.

    Parameters
    ----------
    filename:
        Stereo WAV containing one TRS-1007 sweep.

    requested_band:
        'left', 'right', 'stereo', or 'auto'.

    Returns
    -------
    Dictionary containing the band identity and binned spectral
    measurements.
    """

    path = Path(filename)

    fs, audio = read_wav(path)

    pilot_start, pilot_end = find_pilot(audio, fs)

    detected_band, pilot_difference = detect_band(
        audio,
        pilot_start,
        pilot_end,
    )

    band = (detected_band if requested_band == "auto" else requested_band)

    if requested_band != "auto" and band != detected_band:
        print(f"WARNING: {path.name}: requested '{band}' "
              f"but pilot resembles '{detected_band}'.")

    sweep = extract_sweep(
        audio,
        fs,
        pilot_end,
    )

    # The deliberately driven channel is the best frequency
    # reference for the monophonic bands.
    if band == "left":
        tracking_channel = 0

    elif band == "right":
        tracking_channel = 1

    else:
        # Stereo L+/R+: use whichever pilot is marginally stronger.
        tracking_channel = (0 if pilot_difference >= 0 else 1)

    raw = analyze_sweep(
        sweep,
        fs,
        tracking_channel,
    )

    keys = (
        "L",
        "R",
        "L_HD2",
        "R_HD2",
        "L_HD3",
        "R_HD3",
    )

    binned = log_bin(
        raw["frequency"],
        *(raw[key] for key in keys),
    )

    result = {"frequency": binned[0]}

    for key, values in zip(keys, binned[1:]):
        result[key] = values

    result["band"] = band
    result["filename"] = path.name
    result["sample_rate"] = fs
    result["pilot_difference"] = pilot_difference

    if len(result["frequency"]) < 10:
        raise RuntimeError(f"{path.name}: insufficient sweep data.")

    print(f"{path.name}: "
          f"{band}, "
          f"{len(result['frequency'])} points, "
          f"pilot L/R {pilot_difference:+.1f} dB")

    return result


# =====================================================================
# Pilot detection
# =====================================================================


def pilot_envelope(channel, fs):
    """Return smoothed envelope of energy around 1 kHz."""

    sos = butter(
        4,
        [PILOT_LOW, PILOT_HIGH],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    y = sosfiltfilt(sos, channel)

    envelope = np.abs(hilbert(y))

    # 50 ms smoothing
    n = max(1, int(0.050 * fs))

    envelope = np.convolve(
        envelope,
        np.ones(n) / n,
        mode="same",
    )

    return envelope


def find_pilot(audio, fs):
    """
    Locate first substantial 1-kHz pilot.

    Detection is performed from the sum of the L/R 1-kHz envelopes so
    that left-only, right-only and stereo pilots can all be recognized.

    Returns
    -------
    start, end : sample indices
    """

    env_l = pilot_envelope(audio[:, 0], fs)
    env_r = pilot_envelope(audio[:, 1], fs)

    env = env_l + env_r

    threshold = 0.25 * np.max(env)

    active = env > threshold

    edges = np.diff(
        active.astype(np.int8),
        prepend=0,
        append=0,
    )

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    minimum = int(PILOT_MIN_SECONDS * fs)

    for start, end in zip(starts, ends):

        if end - start >= minimum:
            return start, end

    raise RuntimeError("Unable to locate a valid 1-kHz TRS-1007 pilot.")


# =====================================================================
# Band identification
# =====================================================================


def rms(x):
    """RMS amplitude."""
    return np.sqrt(np.mean(np.square(x)))


def detect_band(audio, pilot_start, pilot_end):
    """
    Estimate TRS-1007 modulation from relative pilot amplitudes.

    LEFT:
        L substantially greater than R.

    RIGHT:
        R substantially greater than L.

    STEREO:
        L and R approximately equal.

    This is deliberately conservative.  Cartridge crosstalk means the
    supposedly undriven channel is never actually silent.
    """

    section = audio[pilot_start:pilot_end]

    left = rms(section[:, 0])
    right = rms(section[:, 1])

    difference = db(left / max(right, EPS))

    if difference >= 6.0:
        return "left", difference

    if difference <= -6.0:
        return "right", difference

    return "stereo", difference


# =====================================================================
# Sweep extraction
# =====================================================================


def extract_sweep(audio, fs, pilot_end):
    """
    Return audio beginning immediately after pilot/gap.

    The subsequent spectral analysis accepts only monotonically rising
    components between 20 Hz and 20 kHz, naturally rejecting most of
    the blank track following the sweep.
    """

    start = pilot_end + int(SWEEP_START_DELAY * fs)

    if start >= len(audio):
        raise RuntimeError("No audio remains after pilot.")

    return audio[start:]


# =====================================================================
# FFT measurements
# =====================================================================


def spectral_peak(spectrum, frequency, df, radius=2):
    """
    Return maximum FFT magnitude near requested frequency.

    A small multi-bin search reduces scalloping error from the swept
    tone falling between FFT bins.
    """

    k = int(round(frequency / df))

    lo = max(1, k - radius)
    hi = min(len(spectrum), k + radius + 1)

    if lo >= hi:
        return np.nan

    return np.max(spectrum[lo:hi])


def _parabolic_bin(mag, k):
    """Quadratic sub-bin estimate around FFT-bin k."""
    if k <= 0 or k >= len(mag) - 1:
        return float(k)
    y0, y1, y2 = np.log(np.maximum(mag[k-1:k+2], EPS))
    d = y0 - 2.0*y1 + y2
    if abs(d) <= EPS:
        return float(k)
    return float(k) + float(np.clip(0.5*(y0-y2)/d, -0.5, 0.5))


def spectral_peak_interp(spectrum, frequency, df, radius=2):
    """Interpolated spectral magnitude near a requested frequency."""
    k = int(round(frequency / df))
    lo = max(1, k-radius)
    hi = min(len(spectrum)-1, k+radius+1)
    if lo >= hi:
        return np.nan
    kp = lo + int(np.argmax(spectrum[lo:hi]))
    if kp <= 0 or kp >= len(spectrum)-1:
        return float(spectrum[kp])
    y0, y1, y2 = np.log(np.maximum(spectrum[kp-1:kp+2], EPS))
    d = y0 - 2.0*y1 + y2
    if abs(d) <= EPS:
        return float(spectrum[kp])
    delta = float(np.clip(0.5*(y0-y2)/d, -0.5, 0.5))
    return float(np.exp(y1 - 0.25*(y0-y2)*delta))


def fit_log_sweep(audio, fs, tracking_channel):
    """Fit ln(f)=a*t+b to a robust first-pass spectral ridge."""
    nfft = max(1024, int(round(RIDGE_WINDOW_SECONDS*fs)))
    nfft += nfft % 2
    hop = max(1, int(round(RIDGE_HOP_SECONDS*fs)))
    win = windows.blackmanharris(nfft, sym=False)
    df = fs/nfft
    k0=max(1,int(FMIN/df)); k1=min(nfft//2+1,int(FMAX/df)+1)
    ts=[]; fs0=[]
    for start in range(0, len(audio)-nfft+1, hop):
        mag=np.abs(np.fft.rfft(audio[start:start+nfft, tracking_channel]*win))
        if k1<=k0: continue
        k=k0+int(np.argmax(mag[k0:k1]))
        noise=np.median(mag[k0:k1])+EPS
        if mag[k]/noise < 8.0: continue
        f=_parabolic_bin(mag,k)*df
        if FMIN <= f <= FMAX:
            ts.append((start+nfft/2)/fs); fs0.append(f)
    t=np.asarray(ts); f=np.asarray(fs0)
    if len(f)<RIDGE_MIN_POINTS: raise RuntimeError('insufficient sweep-ridge points')
    y=np.log(f); mask=np.ones(len(f),dtype=bool)
    for _ in range(6):
        a,b=np.polyfit(t[mask],y[mask],1)
        r=(y-(a*t+b))/np.log(2.0)
        med=np.median(r[mask]); mad=np.median(np.abs(r[mask]-med))+1e-6
        limit=max(0.08,min(0.5,4*1.4826*mad))
        nm=np.abs(r-med)<=limit
        if np.array_equal(nm,mask): break
        mask=nm
        if mask.sum()<RIDGE_MIN_POINTS: raise RuntimeError('robust sweep fit lost too many points')
    a,b=np.polyfit(t[mask],y[mask],1)
    if a<=0: raise RuntimeError('sweep ridge is not rising')
    t20=(np.log(FMIN)-b)/a; t20k=(np.log(FMAX)-b)/a; dur=t20k-t20
    if not 35 <= dur <= 70: raise RuntimeError(f'implausible fitted sweep duration {dur:.1f}s')
    rms_oct=float(np.sqrt(np.mean(((y[mask]-(a*t[mask]+b))/np.log(2.0))**2)))
    return dict(a=float(a),b=float(b),t20=float(t20),t20k=float(t20k),duration=float(dur),points=int(mask.sum()),rms_oct=rms_oct)


def adaptive_window_seconds(f):
    """Longer LF windows, shorter HF windows for a logarithmic chirp."""
    return float(np.clip(8.0/max(float(f),FMIN), 0.025, 0.400))


def analyze_sweep(audio, fs, tracking_channel):
    """V2: fit the logarithmic sweep, then measure fixed log-frequency points."""
    try:
        fit=fit_log_sweep(audio,fs,tracking_channel)
    except RuntimeError as exc:
        print(f'  WARNING: V2 sweep fit failed ({exc}); using legacy tracker.')
        return analyze_sweep_legacy(audio,fs,tracking_channel)
    ratio=2.0**(1.0/POINTS_PER_OCTAVE)
    freqs=[]; f=FMIN
    while f <= FMAX*(1+1e-12): freqs.append(f); f*=ratio
    out={k:[] for k in ('frequency','L','R','L_HD2','R_HD2','L_HD3','R_HD3')}
    for f0 in freqs:
        t=(np.log(f0)-fit['b'])/fit['a']
        if t<0 or t>=len(audio)/fs: continue
        nfft=max(512,int(round(adaptive_window_seconds(f0)*fs))); nfft+=nfft%2
        c=int(round(t*fs)); start=c-nfft//2; stop=start+nfft
        if start<0 or stop>len(audio): continue
        win=windows.blackmanharris(nfft,sym=False)
        scale = max(np.sum(win), EPS)
        L=np.abs(np.fft.rfft(audio[start:stop,0]*win))/scale; R=np.abs(np.fft.rfft(audio[start:stop,1]*win))/scale; df=fs/nfft
        out['frequency'].append(f0); out['L'].append(spectral_peak_interp(L,f0,df)); out['R'].append(spectral_peak_interp(R,f0,df))
        for h,label in ((2,'HD2'),(3,'HD3')):
            if h*f0 < fs/2:
                out['L_'+label].append(spectral_peak_interp(L,h*f0,df)); out['R_'+label].append(spectral_peak_interp(R,h*f0,df))
            else:
                out['L_'+label].append(np.nan); out['R_'+label].append(np.nan)
    print(f"  V2 sweep fit: {fit['duration']:.2f}s, {fit['points']} ridge points, RMS {fit['rms_oct']:.4f} oct")
    return {k:np.asarray(v) for k,v in out.items()}


def analyze_sweep_legacy(audio, fs, tracking_channel):
    """
    Sequential FFT analysis of one TRS-1007 sweep.

    Returns amplitudes for BOTH channels at:
        fundamental
        second harmonic
        third harmonic

    tracking_channel selects only which channel determines the
    instantaneous sweep frequency. It does NOT determine which
    channel's distortion is measured.
    """

    nfft = int(round(WINDOW_SECONDS * fs))
    nfft += nfft % 2

    hop = max(1, int(nfft * HOP_FRACTION))
    window = windows.blackmanharris(nfft, sym=False)
    df = fs / nfft

    result = {
        "frequency": [],
        "L": [],
        "R": [],
        "L_HD2": [],
        "R_HD2": [],
        "L_HD3": [],
        "R_HD3": [],
    }

    previous_f = 0.0

    for start in range(0, len(audio) - nfft, hop):

        frame = audio[start:start + nfft]

        L = np.abs(np.fft.rfft(frame[:, 0] * window))
        R = np.abs(np.fft.rfft(frame[:, 1] * window))

        tracking = L if tracking_channel == 0 else R

        k0 = max(1, int(FMIN / df))
        k1 = min(len(tracking), int(FMAX / df) + 1)

        if k1 <= k0:
            continue

        k = k0 + np.argmax(tracking[k0:k1])
        f = k * df

        if not FMIN <= f <= FMAX:
            continue

        # TRS-1007 sweep moves monotonically upward.
        if previous_f and f < previous_f * 0.95:
            continue

        previous_f = max(previous_f, f)

        result["frequency"].append(f)

        result["L"].append(spectral_peak(L, f, df))

        result["R"].append(spectral_peak(R, f, df))

        if 2.0 * f < fs / 2:
            result["L_HD2"].append(spectral_peak(L, 2.0 * f, df))
            result["R_HD2"].append(spectral_peak(R, 2.0 * f, df))
        else:
            result["L_HD2"].append(np.nan)
            result["R_HD2"].append(np.nan)

        if 3.0 * f < fs / 2:
            result["L_HD3"].append(spectral_peak(L, 3.0 * f, df))
            result["R_HD3"].append(spectral_peak(R, 3.0 * f, df))
        else:
            result["L_HD3"].append(np.nan)
            result["R_HD3"].append(np.nan)

    return {key: np.asarray(value) for key, value in result.items()}


# =====================================================================
# Logarithmic frequency binning
# =====================================================================


def log_bin(frequency, *values):
    """
    Average measurements into equal logarithmic-frequency intervals.
    """

    ratio = 2.0**(1.0 / POINTS_PER_OCTAVE)

    edges = [FMIN]

    while edges[-1] < FMAX:
        edges.append(edges[-1] * ratio)

    edges = np.asarray(edges)

    fout = []
    outputs = [[] for _ in values]

    for lo, hi in zip(edges[:-1], edges[1:]):

        mask = (frequency >= lo) & (frequency < hi)

        if not np.any(mask):
            continue

        fout.append(np.sqrt(lo * hi))

        for output, value in zip(outputs, values):

            section = value[mask]

            if np.any(np.isfinite(section)):
                output.append(np.nanmean(section))
            else:
                output.append(np.nan)

    return (
        np.asarray(fout),
        *[np.asarray(x) for x in outputs],
    )


# =====================================================================
# TRS-1007 high-frequency compensation
# =====================================================================


def trs1007_treble_correction(frequency):
    """
    Return the linear inverse of the standard RIAA 75-us playback pole.

    TRS-1007 omits the corresponding high-frequency recording pre-emphasis.
    A capture made through a conventional RIAA phono stage therefore contains
    the 75-us treble de-emphasis.  Multiplying measured spectral amplitudes by

        sqrt(1 + (2*pi*f*75us)^2)

    removes that playback roll-off.  The arbitrary absolute gain cancels when
    the frequency response is normalized at 1 kHz.

    This function is deliberately applied to spectral measurements rather than
    filtering the WAV.  That keeps pilot/sweep detection unchanged and also
    lets HD2 and HD3 receive the correct compensation at 2*f and 3*f.
    """
    f = np.asarray(frequency, dtype=np.float64)
    return np.sqrt(1.0 + (2.0 * np.pi * f * RIAA_TREBLE_TC)**2)


def apply_trs1007_compensation(data):
    """
    Correct one analyzed TRS-1007 data set in-place.

    Fundamental L/R amplitudes are corrected at f.
    HD2 amplitudes are corrected at 2f.
    HD3 amplitudes are corrected at 3f.

    Crosstalk ratios are unaffected because wanted and leakage signals occur
    at the same fundamental frequency and receive the same correction.
    """
    if data is None or data.get("_trs1007_compensated", False):
        return data

    f = np.asarray(data["frequency"], dtype=np.float64)
    c1 = trs1007_treble_correction(f)
    c2 = trs1007_treble_correction(2.0 * f)
    c3 = trs1007_treble_correction(3.0 * f)

    for key in ("L", "R"):
        data[key] = np.asarray(data[key]) * c1
    for key in ("L_HD2", "R_HD2"):
        data[key] = np.asarray(data[key]) * c2
    for key in ("L_HD3", "R_HD3"):
        data[key] = np.asarray(data[key]) * c3

    data["_trs1007_compensated"] = True
    return data


# =====================================================================
# Normalization
# =====================================================================


def normalize_1khz(frequency, amplitude):
    """
    Convert response to dB and set interpolated 1-kHz level to 0 dB.
    """

    level = db(amplitude)

    reference = np.interp(
        np.log10(FREF),
        np.log10(frequency),
        level,
    )

    return level - reference


# =====================================================================
# Plotting
# =====================================================================
def make_combined_plot(
    left_data=None,
    right_data=None,
    stereo_data=None,
    cartridge="Unknown Cartridge",
    test_info="",
):
    """
    Produce combined SJPlot-style cartridge measurement.

    Solid:
        LEFT  frequency response = aqua blue
        RIGHT frequency response = reddish pink

    Dashed:
        HD2 / HD3 distortion products

    Dotted:
        Crosstalk

    Band 1 supplies:
        Left response
        Left HD2/HD3
        R/L crosstalk

    Band 2 supplies:
        Right response
        Right HD2/HD3
        L/R crosstalk

    Band 3 optionally supplies:
        Stereo L+/R+ response
        Independent L and R harmonic information.
    """

    LEFT = "#25C6DA"
    RIGHT = "#EC6688"

    fig, ax = plt.subplots(figsize=(14, 8.5))

    # ============================================================
    # BAND 1 -- LEFT driven
    # ============================================================

    if left_data is not None:

        f = left_data["frequency"]
        L = left_data["L"]
        R = left_data["R"]

        response = normalize_1khz(f, L)

        h2 = db(safe_ratio(left_data["L_HD2"], L))

        h3 = db(safe_ratio(left_data["L_HD3"], L))

        xtalk = db(safe_ratio(R, L))

        # Main LEFT response
        ax.semilogx(
            f,
            response,
            color=LEFT,
            linewidth=2.5,
            label="Left Response",
            zorder=10,
        )

        # LEFT distortion products
        ax.semilogx(
            f,
            h2,
            color=LEFT,
            linestyle="--",
            linewidth=1.25,
            alpha=0.85,
            label="Left HD2",
        )

        ax.semilogx(
            f,
            h3,
            color=LEFT,
            linestyle=(0, (6, 3)),
            linewidth=1.0,
            alpha=0.65,
            label="Left HD3",
        )

        # Signal leaking into RIGHT
        ax.semilogx(
            f,
            xtalk,
            color=RIGHT,
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            label="R/L Crosstalk",
        )

    # ============================================================
    # BAND 2 -- RIGHT driven
    # ============================================================

    if right_data is not None:

        f = right_data["frequency"]
        L = right_data["L"]
        R = right_data["R"]

        response = normalize_1khz(f, R)

        h2 = db(safe_ratio(right_data["R_HD2"], R))

        h3 = db(safe_ratio(right_data["R_HD3"], R))

        xtalk = db(safe_ratio(L, R))

        # Main RIGHT response
        ax.semilogx(
            f,
            response,
            color=RIGHT,
            linewidth=2.5,
            label="Right Response",
            zorder=10,
        )

        # RIGHT distortion products
        ax.semilogx(
            f,
            h2,
            color=RIGHT,
            linestyle="--",
            linewidth=1.25,
            alpha=0.85,
            label="Right HD2",
        )

        ax.semilogx(
            f,
            h3,
            color=RIGHT,
            linestyle=(0, (6, 3)),
            linewidth=1.0,
            alpha=0.65,
            label="Right HD3",
        )

        # Signal leaking into LEFT
        ax.semilogx(
            f,
            xtalk,
            color=LEFT,
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            label="L/R Crosstalk",
        )

    # ============================================================
    # BAND 3 -- stereo L+/R+
    # ============================================================

    if stereo_data is not None:

        f = stereo_data["frequency"]

        L = stereo_data["L"]
        R = stereo_data["R"]

        # Band 3 is useful primarily as a stereo-response check.
        #
        # Make these traces thinner so the dedicated Band 1/2
        # measurements remain visually dominant.

        ax.semilogx(
            f,
            normalize_1khz(f, L),
            color=LEFT,
            linewidth=1.15,
            alpha=0.50,
            label="Left Response (Stereo)",
        )

        ax.semilogx(
            f,
            normalize_1khz(f, R),
            color=RIGHT,
            linewidth=1.15,
            alpha=0.50,
            label="Right Response (Stereo)",
        )

    # ============================================================
    # Formatting
    # ============================================================

    ax.axhline(
        0,
        linewidth=0.8,
        color="black",
        alpha=0.65,
    )

    # Horizontal dB reference grid
    for level in range(-60, 11, 10):

        ax.axhline(
            level,
            linewidth=0.45,
            color="gray",
            alpha=0.20,
            zorder=0,
        )

    ax.set_xlim(
        FMIN,
        FMAX,
    )

    ax.set_ylim(
        -70,
        10,
    )

    ax.set_xlabel(
        "Frequency (Hz)",
        fontsize=11,
    )

    ax.set_ylabel(
        "Amplitude / Relative Level (dB)",
        fontsize=11,
    )

    # Audio-friendly logarithmic labels
    ticks = [
        20,
        30,
        50,
        70,
        100,
        200,
        300,
        500,
        700,
        1000,
        2000,
        3000,
        5000,
        7000,
        10000,
        20000,
    ]

    labels = [
        "20",
        "30",
        "50",
        "70",
        "100",
        "200",
        "300",
        "500",
        "700",
        "1k",
        "2k",
        "3k",
        "5k",
        "7k",
        "10k",
        "20k",
    ]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    ax.grid(
        True,
        which="major",
        linewidth=0.65,
        alpha=0.30,
    )

    ax.grid(
        True,
        which="minor",
        linewidth=0.30,
        alpha=0.15,
    )

    # ============================================================
    # Titles / metadata
    # ============================================================

    fig.suptitle(
        cartridge[:50],
        fontsize=18,
        fontweight="bold",
        y=0.965,
    )

    ax.set_title(
        "JVC TRS-1007 — 20 Hz–20 kHz Cartridge Response (75 µs HF compensated)",
        fontsize=11,
        pad=10,
    )

    if test_info:

        fig.text(
            0.5,
            0.027,
            test_info[:50],
            ha="center",
            fontsize=9,
        )

    # Describe which measurements were actually supplied.
    sources = []

    if left_data is not None:
        sources.append("Band 1 / Left")

    if right_data is not None:
        sources.append("Band 2 / Right")

    if stereo_data is not None:
        sources.append("Band 3 / L+R+")

    fig.text(
        0.01,
        0.012,
        "JVC TRS-1007 • " + " • ".join(sources),
        fontsize=7,
        alpha=0.70,
    )

    ax.legend(
        loc="lower left",
        fontsize=8.5,
        ncol=2,
        framealpha=0.92,
    )

    fig.tight_layout(rect=(0.035, 0.05, 0.985, 0.925))

    return fig



def print_measurement_diagnostics(data, label):
    """Print sample-rate-dependent harmonic validity limits."""
    if data is None: return
    fs=int(data["sample_rate"]); f=np.asarray(data["frequency"])
    if not len(f): return
    print(f"{label}: {f[0]:.1f}-{f[-1]:.1f} Hz, Fs={fs:,} Hz, "
          f"HD2 usable <= {min(FMAX,fs/4):,.0f} Hz, HD3 usable <= {min(FMAX,fs/6):,.0f} Hz")


# =====================================================================
# Main
# =====================================================================


def main():

    parser = argparse.ArgumentParser(
        description=("JVC TRS-1007 cartridge frequency-response, "
                     "distortion and crosstalk analyzer."))

    # ------------------------------------------------------------
    # Legacy / simple single-WAV operation
    # ------------------------------------------------------------

    parser.add_argument(
        "wav",
        nargs="?",
        help="Single TRS-1007 WAV recording",
    )

    parser.add_argument(
        "--band",
        choices=("auto", "left", "right", "stereo"),
        default="auto",
        help="Band identity when using a single WAV",
    )

    # ------------------------------------------------------------
    # New combined measurement operation
    # ------------------------------------------------------------

    parser.add_argument(
        "--left",
        metavar="WAV",
        help="Band 1 WAV: LEFT-channel 20 Hz-20 kHz sweep",
    )

    parser.add_argument(
        "--right",
        metavar="WAV",
        help="Band 2 WAV: RIGHT-channel 20 Hz-20 kHz sweep",
    )

    parser.add_argument(
        "--stereo",
        metavar="WAV",
        help="Band 3 WAV: stereo L+/R+ 20 Hz-20 kHz sweep",
    )

    # ------------------------------------------------------------
    # Test information
    # ------------------------------------------------------------

    parser.add_argument(
        "--cartridge",
        required=True,
        help="Cartridge name shown on graph (max 50 chars)",
    )

    parser.add_argument(
        "--info",
        default="",
        help="Test environment description (max 50 chars)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output PNG filename",
    )

    parser.add_argument(
        "--no-trs1007-compensation",
        action="store_true",
        help=("Disable TRS-1007 inverse 75-us high-frequency compensation "
              "(normally enabled)."),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Validate input mode
    # ------------------------------------------------------------

    multi_mode = any((
        args.left,
        args.right,
        args.stereo,
    ))

    if args.wav and multi_mode:
        parser.error("Use either the positional WAV or "
                     "--left/--right/--stereo, not both.")

    if not args.wav and not multi_mode:
        parser.error("Supply a WAV file or at least one of "
                     "--left, --right or --stereo.")

    cartridge = truncate(
        args.cartridge,
        50,
    )

    test_info = truncate(
        args.info,
        50,
    )

    left_data = None
    right_data = None
    stereo_data = None

    print()
    print("JVC TRS-1007 Analyzer V2")
    print("========================")
    print(f"Cartridge : {cartridge}")

    if test_info:
        print(f"Test      : {test_info}")

    print()

    # ============================================================
    # SINGLE-WAV MODE
    # ============================================================

    if args.wav:

        if args.band == "auto":
            # A continuous TRS-1007 capture may contain Band 1 followed by
            # Band 2 and optionally Band 3. Discover every pilot/sweep and
            # build one combined cartridge plot.
            combined = analyze_combined_recording(args.wav)
            left_data = combined["left"]
            right_data = combined["right"]
            stereo_data = combined["stereo"]

            if not any((left_data, right_data, stereo_data)):
                raise RuntimeError("No usable TRS-1007 sweeps were detected.")
        else:
            # Explicit single-band operation remains backward compatible.
            measurement = analyze_recording(args.wav, args.band)
            if measurement["band"] == "left":
                left_data = measurement
            elif measurement["band"] == "right":
                right_data = measurement
            else:
                stereo_data = measurement

    # ============================================================
    # MULTI-WAV MODE
    # ============================================================

    else:

        if args.left:

            print("Analyzing Band 1 / LEFT...")

            left_data = analyze_recording(
                args.left,
                "left",
            )

        if args.right:

            print("Analyzing Band 2 / RIGHT...")

            right_data = analyze_recording(
                args.right,
                "right",
            )

        if args.stereo:

            print("Analyzing Band 3 / STEREO L+/R+...")

            stereo_data = analyze_recording(
                args.stereo,
                "stereo",
            )

    # ============================================================
    # TRS-1007 recording-characteristic compensation
    # ============================================================

    print_measurement_diagnostics(left_data, "Band 1 / LEFT")
    print_measurement_diagnostics(right_data, "Band 2 / RIGHT")
    print_measurement_diagnostics(stereo_data, "Band 3 / STEREO")

    compensate = TRS1007_COMPENSATE and not args.no_trs1007_compensation

    if compensate:
        left_data = apply_trs1007_compensation(left_data)
        right_data = apply_trs1007_compensation(right_data)
        stereo_data = apply_trs1007_compensation(stereo_data)
        print("TRS-1007 HF compensation: ON (inverse RIAA 75-us pole)")
    else:
        print("TRS-1007 HF compensation: OFF")

    # ============================================================
    # Generate combined graph
    # ============================================================

    fig = make_combined_plot(
        left_data=left_data,
        right_data=right_data,
        stereo_data=stereo_data,
        cartridge=cartridge,
        test_info=test_info,
    )

    # ------------------------------------------------------------
    # Output filename
    # ------------------------------------------------------------

    if args.output:

        output = Path(args.output)

    else:

        safe_cart = "".join(c if c.isalnum() or c in "-_" else "_"
                            for c in cartridge)

        output = Path(f"{safe_cart}_TRS1007.png")

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(f"Plot written: {output}")
    print()

    plt.show()


if __name__ == "__main__":
    main()
