# JVC/Victor TRS-1007 Cartridge Analyzer V2

A Python analyzer for cartridge frequency response, harmonic distortion, and crosstalk from the JVC/Victor TRS-1007 frequency-response test record.

V2 is intended for laboratory-style analysis of digitized TRS-1007 sweeps. It detects the 1 kHz pilot sections in a continuous stereo capture, identifies the following left- or right-driven sweep, fits the logarithmic sweep trajectory, measures the fundamental plus HD2/HD3, applies the TRS-1007 high-frequency compensation when appropriate, and produces a combined SJPlot-style graph.

![Example output](examples/example_output_v2.png)

## What V2 measures

For a left-driven sweep the analyzer plots:

- Left-channel frequency response
- Right/Left crosstalk
- Left HD2 (second-harmonic distortion)
- Left HD3 (third-harmonic distortion)

For a right-driven sweep it plots:

- Right-channel frequency response
- Left/Right crosstalk
- Right HD2
- Right HD3

An optional stereo L+/R+ sweep can also be displayed as a stereo-response check.

Frequency response is normalized to 0 dB at 1 kHz. Crosstalk and HD2/HD3 are expressed in dB relative to the simultaneously measured fundamental.

## V2 analysis engine

The original analyzer tracked each FFT frame by choosing the strongest component between 20 Hz and 20 kHz. V2 retains that implementation as a fallback, but normally uses a two-pass method:

1. **Sweep-ridge pass** — short overlapping Blackman-Harris FFTs locate candidate sweep frequencies.
2. **Robust logarithmic fit** — outliers are rejected and the analyzer fits `ln(f) = a*t + b`.
3. **Measurement pass** — the fitted sweep law determines the expected excitation frequency at each logarithmic measurement point.
4. **Adaptive FFT windows** — longer windows are used at low frequencies and shorter windows at high frequencies.
5. **Sub-bin interpolation** — local quadratic interpolation reduces FFT-bin quantization error.
6. **HD2/HD3 extraction** — distortion products are measured at 2f and 3f when those frequencies are below Nyquist.

This prevents a click, harmonic, resonance, or unrelated spectral component from redefining the instantaneous fundamental simply because it is the largest FFT peak in a frame.

The analyzer reports the fitted sweep duration, number of accepted ridge points, and RMS logarithmic fit error. If a plausible fit cannot be established, it automatically falls back to the legacy monotonic peak tracker and prints a warning.

## TRS-1007 high-frequency compensation

The analyzer includes a TRS-1007-specific high-frequency correction enabled by default. The implemented model treats the disc's high-frequency section as omitting the standard RIAA 75 microsecond recording pre-emphasis. A capture made through a conventional RIAA playback chain therefore retains the corresponding playback roll-off.

The correction applied to measured spectral amplitude is:

```text
C(f) = sqrt(1 + (2*pi*f*75e-6)^2)
```

Fundamentals are corrected at `f`, HD2 at `2f`, and HD3 at `3f`. Crosstalk ratios do not require a separate correction because wanted and leakage signals are measured at the same fundamental frequency.

Disable this correction when the capture chain has already compensated the TRS-1007 characteristic:

```bash
--no-trs1007-compensation
```

The correction model is an explicit implementation assumption and should be considered when comparing results with other TRS-1007 software or published measurements.

## TRS-1007 Side A organization

The script is designed around the documented sequence used by the program:

| Band | Tracks | Function |
|---|---|---|
| 1 | A1-A3 | 1 kHz L+R pilot, 20 Hz-20 kHz LEFT sweep, blank |
| 2 | A4-A6 | 1 kHz L+R pilot, 20 Hz-20 kHz RIGHT sweep, blank |
| 3 | A7-A9 | 1 kHz L+R pilot, 20 Hz-20 kHz L+R sweep, blank |
| 4 | A10-A11 | 1 kHz L-R pilot and L-R sweep |
| 5-8 | A12-A15 | 1 kHz reference/spot signals |
| 9-12 | A16-A26 | Additional response/crosstalk/distortion bands |

V2 currently concentrates on the left, right, and optional L+R response sweeps used for the combined cartridge plot.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Dependencies are NumPy, SciPy, and Matplotlib.

## Basic usage

For a continuous WAV containing the left sweep followed by the right sweep:

```bash
python3 jvc_trs1007_analyzer.py recording.wav \
  --band auto \
  --cartridge "Audio Technica AT-150MLX" \
  --info "VTF 1.5g / 47k 250pF" \
  --output AT150MLX_TRS1007.png
```

`--band auto` is the preferred mode for a continuous capture. Long 1 kHz pilot regions are detected and used to delimit the following sweeps. Classification is based on channel dominance during the sweep rather than the stereo pilot itself.

Separate WAV files can also be supplied:

```bash
python3 jvc_trs1007_analyzer.py \
  --left left_sweep.wav \
  --right right_sweep.wav \
  --cartridge "Cartridge Name" \
  --output result.png
```

Add `--stereo stereo_sweep.wav` when a separate L+/R+ sweep is available.

## Sample-rate limits for HD2 and HD3

A harmonic can only be measured while the harmonic frequency is below Nyquist. Therefore:

```text
HD2 maximum fundamental frequency = Fs / 4
HD3 maximum fundamental frequency = Fs / 6
```

For a 44.1 kHz capture this means approximately:

- HD2: usable through 11.025 kHz fundamental frequency
- HD3: usable through 7.350 kHz fundamental frequency

For 96 kHz capture:

- HD2: usable through the full 20 kHz sweep
- HD3: usable through 16 kHz

For 192 kHz capture both HD2 and HD3 can be measured throughout a 20 kHz fundamental sweep.

The graph terminates unavailable harmonic data rather than fabricating values above Nyquist.

## Validation capture

V2 was exercised against a 44.1 kHz, 16-bit stereo continuous TRS-1007 capture. The validation run found two genuine long pilot sections and classified the following sweeps as LEFT and RIGHT. The fitted logarithmic sweeps were approximately 49.9 seconds each, consistent with the nominal 50-second sweep duration.

The validation WAV itself is deliberately not included in this repository. `.gitignore` excludes common uncompressed/lossless audio capture formats so large test-record recordings are not accidentally committed.

## Plot conventions

- Aqua solid: left frequency response
- Pink/red solid: right frequency response
- Dashed: HD2 / HD3
- Dotted: crosstalk
- Frequency axis: logarithmic, 20 Hz-20 kHz
- Response reference: 0 dB at 1 kHz

## Important measurement considerations

Results depend on the complete playback and digitization chain: cartridge, stylus condition, alignment, VTF, tonearm, loading, phono preamplifier, cabling/capacitance, ADC, sample rate, record condition, and the accuracy of the test record itself.

HD2/HD3 values derived from a swept test record are not interchangeable with every other distortion measurement method. At low levels the result can be limited by groove noise, crosstalk, ADC noise, FFT leakage, and the residual distortion of the record/playback chain.

For comparative work, keep the mechanical setup, loading, gain structure, ADC, sample rate, and analysis settings constant.

## Attribution and project history

This project was inspired by Scott Wurcer's swept-frequency cartridge analysis concepts and the MIT-licensed FidelisAnalog SJPlot work by John P. Jones III. The source retains the attribution and MIT license text inherited from the project basis.

V2 adds TRS-1007-oriented continuous-recording segmentation, sweep classification, logarithmic sweep-law fitting, adaptive analysis windows, sub-bin spectral interpolation, HD2/HD3 terminology and diagnostics, and TRS-1007 high-frequency compensation handling.

This repository is intended as an engineering/measurement tool, not as an official JVC/Victor product.

## License

MIT. See [LICENSE](LICENSE).

## Suggested GitHub repository

`https://github.com/cport101/jvc-trs1007-analyzer`

After creating an empty repository under the `cport101` account, from this directory:

```bash
git init
git branch -M main
git add .
git commit -m "Initial release: JVC TRS-1007 Analyzer V2"
git remote add origin https://github.com/cport101/jvc-trs1007-analyzer.git
git push -u origin main
```

If GitHub HTTPS authentication is used from the command line, use GitHub's currently supported authentication method rather than an account password.
