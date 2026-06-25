"""
Sliding Window Oscillation Detection, Characterization, and CDEF Analysis
https://www.techrxiv.org/doi/full/10.36227/techrxiv.176523437.74012330
author - Shuchismita Biswas (shuchismita.biswas@pnnl.gov)

Outputs (csv files written to output_dir)
-----------------------------------------
detection_summary.csv  — one row per sliding window per channel; all windows
power_terms.csv        — one row per (channel, window, oscillation freq);
                         only windows where complete sideband pairs found
timing_log.csv         — one row per detected window; computation time only
                         when oscillation is flagged
"""

import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# User inputs
CONFIG = {
    # --- data ---
    "data_path": r"C:\Users\rame388\OneDrive - PNNL\projects_backup\OEDISI_II\Oscillation_Detection\algorithm\FO_A11_1pt5Hz_pt03.csv",  # path to gzip-compressed CSV
    "output_dir": r"C:\Users\rame388\OneDrive - PNNL\projects_backup\OEDISI_II\Oscillation_Detection\algorithm",
    # --- signal properties ---
    "f0_nom": 50,  # nominal fundamental frequency (Hz)
    "fs": 500 * 50,  # POW reporting rate (sps/Hz)
    # --- sliding window ---
    "window_sec": 2,  # analysis window length (seconds)
    "overlap_sec": 1,  # overlap between consecutive windows (seconds)
    # --- detection tuning ---
    "peak_thresh": 0.003,  # detection threshold as fraction of fundamental amplitude
    "freq_max": 150,  # ignore peaks above this frequency (Hz)
    "sideband_tol": 0.1,  # tolerance for equidistance check (Hz)
    "freq_match_tol": 0.05,  # tolerance for matching V and I peak frequencies (Hz)
    # --- channel mapping: (v_col_idx, i_col_idx, label) ---
    # this is just to match the format of the simulink output files
    # will need to be replaced for real-time readiness - right now its written to process multiple
    # signals, not true distributed implementation
    "channels": [
        (0, 0, "Bus1"),
        (3, 3, "A11"),
        (6, 6, "A12"),
        (9, 9, "A13"),
        (12, 12, "Bus2"),
        (15, 15, "Bus3"),
        (18, 18, "Bus4"),
        (21, 21, "Bus5"),
    ],
}

# ───  Data Loading (simulink format specific)────────────────────────────────────────────────────


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load gzip-compressed CSV and split into voltage and current DataFrames.
    Discards the first 10 seconds (startup transient in EMT simulation).
    """
    data = pd.read_csv(data_path, compression="gzip")
    data = data[data.Time > 10].reset_index(drop=True)
    data_v = data[[c for c in data.columns if "v_abc" in c]]
    data_i = data[[c for c in data.columns if "i_abc" in c]]
    return data_v, data_i


# ──── Window Index Generation ────────────────────────────────────────


def make_window_indices(
    n_samples: int,
    fs: float,
    window_sec: float,
    overlap_sec: float,
) -> list[tuple[int, int]]:
    """
    Return (start, end) sample index pairs for every sliding window.

    Parameters
    ----------
    n_samples   : total number of samples in the signal
    fs          : sampling frequency (Hz)
    window_sec  : window length in seconds
    overlap_sec : overlap between consecutive windows in seconds

    Returns
    -------
    List of (start, end) tuples — end is exclusive
    """
    win_samples = int(window_sec * fs)
    step_samples = int((window_sec - overlap_sec) * fs)
    if step_samples <= 0:
        raise ValueError("overlap length must be lower than window length")
    starts = range(0, n_samples - win_samples + 1, step_samples)
    return [(s, s + win_samples) for s in starts]


# ─── Fundamental Frequency Estimation ───────────────────────────────────────────


def refine_frequency(fft_data: np.ndarray, k: int, fs: float, N: int) -> float:
    """
    Quadratic spectral interpolation around bin k to refine the frequency estimate

    Parameters
    ----------
    fft_data : complex FFT of the full signal
    k        : bin index of the spectral peak
    fs       : sampling frequency (Hz)
    N        : signal length (samples)

    Returns
    -------
    Refined frequency estimate (Hz)
    """
    Xk = fft_data[k]
    Xkm1 = fft_data[k - 1]
    Xkp1 = fft_data[k + 1]
    delta = -np.real((Xkp1 - Xkm1) / (2 * Xk - Xkm1 - Xkp1))
    return (k + delta) * fs / N


# ──── FFT + Fundamental Removal ──────────────────────────────────────


def compute_fft_and_remove_fundamental(
    window: np.ndarray,
    fs: float,
    f0_nom: float,
    f_est_override: float = None,  # basically use voltage signal frequency only (should this be current as current signal could be better suited for detection?)
) -> dict:
    """
    Core spectral processing for one signal window.

    Steps
    -----
    1. FFT of the raw window
    2. Find fundamental bin (search +/-1 Hz around f0_nom) could perhaps be tightened
    3. Refine fundamental estimate via quadratic interpolation
    4. Adjust window to integer number of fundamental cycles
    5. Remove fundamental via dot-product projection (paper Eq. 11)
    6. FFT of the residual — this FFT is reused later for CSD computation

    Returns
    -------
    dict with keys:
        fft_residual    complex FFT of fundamental-removed signal (length Nc)
        fft_freqs       frequency axis for the adjusted window (length Nc)
        f_est           refined fundamental frequency (Hz)
        fund_amplitude  peak amplitude of fundamental (used for threshold)
        Nc              length of the adjusted window (samples)
    """
    window = np.asarray(window, dtype=float)
    N = len(window)

    # 1 — Initial FFT (used only to locate and refine the fundamental)
    fft_raw = np.fft.fft(window)
    fft_mag = np.abs(fft_raw)
    freqs = np.fft.fftfreq(N, 1.0 / fs)

    # 2 — Locate fundamental bin: search +/-1 Hz band around nominal
    if f_est_override is not None:
        f_est = f_est_override
    else:
        half = N // 2
        k_min = int(np.argmin(np.abs(freqs[:half] - (f0_nom - 1))))
        k_max = int(np.argmin(np.abs(freqs[:half] - (f0_nom + 1))))
        k_max_inc = k_max + 1
        k = int(np.argmax(fft_mag[k_min:k_max_inc])) + k_min
        # 3 — Refine with quadratic interpolation
        f_est = refine_frequency(fft_raw, k, fs, N)

    # 4 — Adjust window to nearest integer number of fundamental cycles
    n_cycles = round(N / fs * f_est)  # cycles in original window
    Nc = round(n_cycles / f_est * fs)  # samples for exactly n_cycles cycles

    if Nc > N:
        sig_adj = np.zeros(Nc)
        sig_adj[:N] = window
    else:
        sig_adj = window[:Nc]

    # 5 — Remove fundamental via dot-product projection (Eq. 11)
    t = np.arange(Nc) / fs
    exp_ref = np.exp(-1j * 2 * np.pi * f_est * t)
    X_proj = np.dot(sig_adj, exp_ref)  # scalar projection
    fund_amp = np.abs(X_proj) * 2.0 / Nc  # peak amplitude
    x_fund = (2.0 / Nc) * X_proj * np.exp(1j * 2 * np.pi * f_est * t)
    sig_residual = sig_adj - np.real(x_fund)

    # 6 — FFT of residual — REUSED in csd_phase_from_ffts
    fft_residual = np.fft.fft(sig_residual)
    fft_freqs = np.fft.fftfreq(Nc, 1.0 / fs)

    return {
        "fft_residual": fft_residual,
        "fft_freqs": fft_freqs,
        "f_est": f_est,
        "fund_amplitude": fund_amp,
        "Nc": Nc,
    }


# ─── Peak Detection ──────────────────────────────────────────────────


def detect_peaks(
    fft_residual: np.ndarray,
    fft_freqs: np.ndarray,
    f_est: float,
    fund_amplitude: float,
    peak_thresh: float = 0.003,
    freq_max: float = 150.0,
) -> dict[float, float]:
    """
    Find oscillation peaks in the fundamental-removed FFT.

    Parameters
    ----------
    fft_residual    complex FFT of residual signal
    fft_freqs       frequency axis
    f_est           estimated fundamental (Hz) — used to exclude its neighbourhood
    fund_amplitude  peak amplitude of fundamental, used to set detection threshold (scope for improvement)
    peak_thresh     threshold as a fraction of the fundamental amplitude
    freq_max        ignore peaks above this frequency (Hz)

    Returns
    -------
    dict  {frequency_Hz (rounded to 2 decimal places): amplitude}
    """
    Nc = len(fft_residual)
    fft_mag = np.abs(fft_residual)

    # Threshold in FFT-magnitude units (undo the 2/Nc normalisation)
    threshold = peak_thresh * fund_amplitude * Nc / 2.0
    peaks, props = find_peaks(fft_mag[: Nc // 2], height=threshold, distance=3)

    result = {}
    for idx, amp in zip(peaks, props["peak_heights"]):
        f = fft_freqs[idx]
        if 0 < f < freq_max and abs(f - f_est) > 0.5:
            result[round(f, 2)] = amp * 2.0 / Nc  # physical amplitude

    return result


# ───── Detection Robustness Check ─────────────────────────────────────


def find_common_frequencies(
    peaks_v: dict, peaks_i: dict, tol: float = 0.05
) -> list[float]:
    """Return frequencies present in BOTH voltage and current peak dicts."""
    return [f for f in peaks_v if any(abs(f - fi) < tol for fi in peaks_i)]


def find_sideband_pairs(
    freqs: list[float], f0: float, tol: float = 0.1
) -> list[tuple[float, float]]:
    """
    Among a list of frequencies, find pairs (f_minus, f_plus) that are
    approximately equidistant around f0, i.e. f0 - f_minus ~ f_plus - f0.

    Returns list of (f_minus, f_plus) tuples sorted ascending.
    """
    return [
        (f1, f2)
        for f1, f2 in combinations(sorted(freqs), 2)
        if abs((f0 - f1) - (f2 - f0)) <= tol
    ]


def check_detection(
    peaks_v: dict,
    peaks_i: dict,
    f_est: float,
    freq_match_tol: float = 0.05,
    sideband_tol: float = 0.1,
) -> dict:
    """
    A detection is valid if the same frequency appears in both V and I peaks, OR
    both sub- and supersynchronous sideband peaks exist

    Returns
    -------
    dict with keys:
        valid           bool
        common_freqs    list of frequencies satisfying (A)
        sideband_pairs  list of (f_minus, f_plus) tuples satisfying (B)
    """
    common = find_common_frequencies(peaks_v, peaks_i, tol=freq_match_tol)
    if not common:
        return {"valid": False, "common_freqs": [], "sideband_pairs": []}

    pairs = find_sideband_pairs(common, f_est, tol=sideband_tol)

    return {
        "valid": True,  # condition (A) met by non-empty common
        "common_freqs": common,
        "sideband_pairs": pairs,  # empty if only (A) is met, not (B)
    }


# ──── CSD Phase (reusing already-computed FFTs) ──────────────────────


def csd_phase_from_ffts(fft_v: np.ndarray, fft_i: np.ndarray) -> np.ndarray:
    """
    Compute the voltage-current phase difference at every frequency bin.
    S_vi[k] = (1/Nc) * V[k] * conj(I[k])
    phase[k] = angle(S_vi[k])
    reusing FFT should theoretically save time
    """
    return np.angle(fft_v * np.conj(fft_i))


# ─------- Power Component Computation ────────────────────────────────────


def compute_power_terms(
    peaks_v: dict,
    peaks_i: dict,
    fft_v: np.ndarray,
    fft_i: np.ndarray,
    fft_freqs: np.ndarray,
    sideband_pairs: list[tuple[float, float]],
) -> tuple[dict, dict, dict, dict]:
    """
    Compute P+, P-, Q+, Q- for every detected oscillation.
    Phase is obtained from already-computed FFTs via csd_phase_from_ffts —

    Returns
    -------
    Pp, Pm, Qp, Qm : dicts keyed by oscillation frequency (Hz)
    """
    phase_vi = csd_phase_from_ffts(fft_v, fft_i)

    Pp, Pm, Qp, Qm = {}, {}, {}, {}

    for f_minus, f_plus in sideband_pairs:
        f_osc = round((f_plus - f_minus) / 2.0, 2)

        # Nearest bin indices for phase look-up
        idx_m = int(np.argmin(np.abs(fft_freqs - f_minus)))
        idx_p = int(np.argmin(np.abs(fft_freqs - f_plus)))

        # Amplitudes from peak dicts (nearest match)
        v_minus = peaks_v[min(peaks_v, key=lambda f: abs(f - f_minus))]
        v_plus = peaks_v[min(peaks_v, key=lambda f: abs(f - f_plus))]
        i_minus = peaks_i[min(peaks_i, key=lambda f: abs(f - f_minus))]
        i_plus = peaks_i[min(peaks_i, key=lambda f: abs(f - f_plus))]

        theta_m = phase_vi[idx_m]
        theta_p = phase_vi[idx_p]

        Pm[f_osc] = v_minus * i_minus * np.cos(theta_m)
        Pp[f_osc] = v_plus * i_plus * np.cos(theta_p)
        Qm[f_osc] = v_minus * i_minus * np.sin(theta_m)
        Qp[f_osc] = v_plus * i_plus * np.sin(theta_p)

    return Pp, Pm, Qp, Qm


# ───── Single Window Pipeline ─────────────────────────────────────────


def process_window(
    v_win: np.ndarray,
    i_win: np.ndarray,
    fs: float,
    f0_nom: float,
    cfg: dict,
) -> dict:
    """
    Full analysis pipeline for one window of voltage and current samples.

    1. Compute FFT + remove fundamental    (once each for V and I)
    2. Detect spectral peaks in residuals
    3. Robustness check
    4. If valid: compute P+/-, Q+/- reusing Step 1 FFTs (no new FFTs)

    Returns a result dict.
        valid               bool — oscillation detected by at least criterion (A)
        complete_sidebands  bool — sideband pairs found; power terms computed
    """
    # Step 1 — compute once; results forwarded to Steps 2, 7, 8
    spec_v = compute_fft_and_remove_fundamental(v_win, fs, f0_nom)
    spec_i = compute_fft_and_remove_fundamental(
        i_win, fs, f0_nom, f_est_override=spec_v["f_est"]
    )

    f_est = spec_v["f_est"]
    fft_freqs = spec_v["fft_freqs"]

    # Step 2 — peak detection (no new FFTs)
    peaks_v = detect_peaks(
        spec_v["fft_residual"],
        spec_v["fft_freqs"],
        f_est,
        spec_v["fund_amplitude"],
        cfg["peak_thresh"],
        cfg["freq_max"],
    )
    peaks_i = detect_peaks(
        spec_i["fft_residual"],
        spec_i["fft_freqs"],
        f_est,
        spec_i["fund_amplitude"],
        cfg["peak_thresh"],
        cfg["freq_max"],
    )

    # Step 3 — robustness check
    detection = check_detection(
        peaks_v,
        peaks_i,
        f_est,
        cfg["freq_match_tol"],
        cfg["sideband_tol"],
    )

    if not detection["valid"]:
        return {"valid": False, "f_est": f_est}

    # Step 4 — power terms only when complete sideband pairs exist
    sideband_pairs = detection["sideband_pairs"]
    complete = len(sideband_pairs) > 0

    Pp, Pm, Qp, Qm = {}, {}, {}, {}
    if complete:
        Pp, Pm, Qp, Qm = compute_power_terms(
            peaks_v,
            peaks_i,
            spec_v["fft_residual"],
            spec_i["fft_residual"],
            fft_freqs,
            sideband_pairs,
        )

    return {
        "valid": True,
        "complete_sidebands": complete,
        "f_est": f_est,
        "common_freqs": detection["common_freqs"],
        "sideband_pairs": sideband_pairs,
        "peaks_v": peaks_v,
        "peaks_i": peaks_i,
        "Pp": Pp,
        "Pm": Pm,
        "Qp": Qp,
        "Qm": Qm,
    }


# ──── Computation Timer (just for me for now to ensure ready for GPA)─────────────────────────────────────


def timed_process_window(
    v_win: np.ndarray,
    i_win: np.ndarray,
    fs: float,
    f0_nom: float,
    cfg: dict,
) -> tuple[dict, float | None]:
    """
    Wrapper around process_window that measures wall-clock computation time.

    Uses time.perf_counter() for sub-millisecond resolution.

    Elapsed time is returned ONLY when an oscillation is detected — so the
    timing log naturally contains only detection events, giving a clear picture
    of the computational cost of the full pipeline (detection + power terms)
    for windows where it actually runs to completion.

    Returns
    -------
    result      : the process_window result dict
    elapsed_ms  : computation time in milliseconds, or None if no detection
    """
    t0 = time.perf_counter()
    result = process_window(v_win, i_win, fs, f0_nom, cfg)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1e3 if result["valid"] else None
    return result, elapsed_ms


# ─── Sliding Window Loop ───────────────────────────────────────────


def run_channel(
    v_sig: np.ndarray,
    i_sig: np.ndarray,
    fs: float,
    f0_nom: float,
    cfg: dict,
    label: str = "",
) -> list[dict]:
    """
    Run the timed sliding window analysis on one voltage-current channel pair.
    Each window result dict is augmented with:
        window_idx       int
        time_start_s     float   window start time relative to data start
        compute_time_ms  float | None   present only when oscillation detected

    Returns a list of per-window dicts (length == number of windows).
    """
    n_samp = min(len(v_sig), len(i_sig))
    windows = make_window_indices(n_samp, fs, cfg["window_sec"], cfg["overlap_sec"])
    results = []

    print(
        f"  {label}: {len(windows)} windows  "
        f"({cfg['window_sec']} s window, {cfg['overlap_sec']} s overlap)"
    )

    for w_idx, (start, end) in enumerate(windows):
        result, elapsed_ms = timed_process_window(
            v_sig[start:end], i_sig[start:end], fs, f0_nom, cfg
        )
        result.update(
            {
                "window_idx": w_idx,
                "start_sample": start,
                "end_sample": end,
                "time_start_s": start / fs,
                "compute_time_ms": elapsed_ms,
            }
        )

        if result["valid"]:
            n_common = len(result.get("common_freqs", []))
            n_pairs = len(result.get("sideband_pairs", []))
            print(
                f"    w{w_idx:04d} t={start/fs:6.1f}s | f0={result['f_est']:.4f} Hz | "
                f"DETECTED  ({n_common} freqs, {n_pairs} pairs)  [{elapsed_ms:.2f} ms]"
            )
        else:
            print(
                f"    w{w_idx:04d} t={start/fs:6.1f}s | f0={result['f_est']:.4f} Hz | "
                f"no detection"
            )

        results.append(result)

    return results


def save_results_to_csv(all_results: dict, output_dir: str) -> None:
    """
    Write results to three CSV files.

    detection_summary.csv
    Columns: channel, window_idx, time_start_s, f_est_hz,
                 detected, complete_sidebands, n_common_freqs,
                 n_sideband_pairs, common_freqs_hz, compute_time_ms

        common_freqs_hz is a semicolon-separated string, e.g. "44.1;55.9".
        compute_time_ms is blank for windows with no detection.

    power_terms.csv
        One row per (channel, window, oscillation_frequency).
        Only windows with complete sideband pairs.
        Columns: channel, window_idx, time_start_s, f_osc_hz, Pp, Pm, Qp, Qm

    timing_log.csv
        One row per detected window only (compute_time_ms is not None).
        Columns: channel, window_idx, time_start_s, f_est_hz,
                 n_common_freqs, n_sideband_pairs, compute_time_ms
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    power_rows = []
    timing_rows = []

    for channel, windows in all_results.items():
        for r in windows:
            detected = r["valid"]
            complete = r.get("complete_sidebands", False)
            t_start = r["time_start_s"]
            w_idx = r["window_idx"]
            f_est = r["f_est"]
            ct_ms = r.get("compute_time_ms")  # None when no detection

            # ── detection_summary row (every window) ──
            summary_rows.append(
                {
                    "channel": channel,
                    "window_idx": w_idx,
                    "time_start_s": round(t_start, 4),
                    "f_est_hz": round(f_est, 5),
                    "detected": detected,
                    "complete_sidebands": complete,
                    "n_common_freqs": len(r.get("common_freqs", [])),
                    "n_sideband_pairs": len(r.get("sideband_pairs", [])),
                    # semicolon-separated — parseable with str.split(";")
                    "common_freqs_hz": ";".join(
                        str(f) for f in r.get("common_freqs", [])
                    ),
                    "I_peaks": ";".join(str(i) for i in r.get("peaks_i", [])),
                    "V_peaks": ";".join(str(v) for v in r.get("peaks_v", [])),
                    "compute_time_ms": round(ct_ms, 4) if ct_ms is not None else "",
                }
            )

            # ── power_terms rows (one per oscillation frequency) ──
            if complete:
                for f_osc in r["Pp"]:
                    power_rows.append(
                        {
                            "channel": channel,
                            "window_idx": w_idx,
                            "time_start_s": round(t_start, 4),
                            "f_osc_hz": f_osc,
                            "Pp": r["Pp"].get(f_osc),
                            "Pm": r["Pm"].get(f_osc),
                            "Qp": r["Qp"].get(f_osc),
                            "Qm": r["Qm"].get(f_osc),
                        }
                    )

            # ── timing_log rows (detected windows only) ──
            if detected and ct_ms is not None:
                timing_rows.append(
                    {
                        "channel": channel,
                        "window_idx": w_idx,
                        "time_start_s": round(t_start, 4),
                        "f_est_hz": round(f_est, 5),
                        "n_common_freqs": len(r.get("common_freqs", [])),
                        "n_sideband_pairs": len(r.get("sideband_pairs", [])),
                        "compute_time_ms": round(ct_ms, 4),
                    }
                )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "detection_summary.csv"), index=False
    )
    pd.DataFrame(power_rows).to_csv(
        os.path.join(output_dir, "power_terms.csv"), index=False
    )
    pd.DataFrame(timing_rows).to_csv(
        os.path.join(output_dir, "timing_log.csv"), index=False
    )

    print(f"\n  detection_summary.csv  — {len(summary_rows)} rows")
    print(f"  power_terms.csv        — {len(power_rows)} rows")
    print(f"  timing_log.csv         — {len(timing_rows)} rows")
    print(f"  saved to: {output_dir}")


# ──── MAIN FUNCTION ──────────────────────────────────────────────


def main(cfg: dict) -> dict:
    """
    Load data, run sliding window analysis on all channels, save CSVs, plot.

    Returns
    -------
    all_results : dict  {channel_label: [per-window result dicts]}
    """
    fs = cfg["fs"]
    f0_nom = cfg["f0_nom"]

    print(f"\n{'='*60}")
    print("  Oscillation Analysis - Sliding Window")
    print(f"  f0_nom={f0_nom} Hz | fs={fs} Hz")
    print(f"  window={cfg['window_sec']} s | overlap={cfg['overlap_sec']} s")
    print(f"{'='*60}\n")

    print(f"Loading: {cfg['data_path']}")
    data_v, data_i = load_data(cfg["data_path"])
    print(f"  Samples: {len(data_v):,}  ({len(data_v)/fs:.1f} s)\n")

    all_results = {}
    for col_v, col_i, label in cfg["channels"]:
        v_sig = data_v.iloc[:, col_v].values
        i_sig = data_i.iloc[:, col_i].values
        all_results[label] = run_channel(v_sig, i_sig, fs, f0_nom, cfg, label)
        print()

    print("Saving CSVs ...")
    save_results_to_csv(all_results, cfg["output_dir"])

    print("\nGenerating figures ...")
    from pnnl_emt_swod.plotting import plot_results

    plot_results(cfg["output_dir"], window_sec=cfg["window_sec"])

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION DATA ADAPTER  ── dataset-specific, safe to delete
# ══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE
# -------
# Translates the Simulink-logged CSV format (columns named _v_abc__N_phase,
# _i_abc__N_phase, etc.) into the plain numpy arrays that run_channel() needs.
# Everything below this banner is specific to this one simulation dataset.
#
# TO REMOVE
# ---------
# Delete from this banner to the matching END banner below, then call
# main(CONFIG) directly with a generic load_data() that suits your new dataset.
#
# COLUMN LAYOUT (one group per channel, repeated 8 times)
# --------------------------------------------------------
#   _i_abc__<N>_1/2/3   3-phase current  (phases a, b, c)
#   _i_dq__<N+1>_1/2    d-q current      (not used here)
#   _pq__<N+2>_1/2      P and Q          (not used here)
#   _v_abc__<N+4>_1/2/3 3-phase voltage  (phases a, b, c)
#   _v_dq__<N+5>_1/2    d-q voltage      (not used here)
#   _w__<N+6>           angular speed    (not used here)
#
# CHANNEL ORDER IN FILE: A1, A11, A12, A13, A2, A3, A6, A8
# ──────────────────────────────────────────────────────────────────────────────

# Mapping: channel label → (Simulink v_abc index, Simulink i_abc index)
# These are the numeric tokens that appear in the column names, e.g.
# _v_abc__5_1 has token 5, _i_abc__1_1 has token 1.
SIMULINK_CHANNEL_MAP = {
    "A1": {"v_token": 5, "i_token": 1},
    "A11": {"v_token": 12, "i_token": 8},
    "A12": {"v_token": 19, "i_token": 15},
    "A13": {"v_token": 26, "i_token": 22},
    "A2": {"v_token": 33, "i_token": 29},
    "A3": {"v_token": 40, "i_token": 36},
    "A6": {"v_token": 47, "i_token": 43},
    "A8": {"v_token": 54, "i_token": 50},
}

# Which phase to use for single-phase analysis.
# 1 = phase A, 2 = phase B, 3 = phase C.
PHASE = 1


def _col(prefix: str, token: int, phase: int) -> str:
    """Build a Simulink column name, e.g. _v_abc__5_1."""
    return f"_{prefix}__{token}_{phase}"


def load_simulation_data(
    data_path: str,
    channel_map: dict = SIMULINK_CHANNEL_MAP,
    phase: int = PHASE,
    t_start_discard: float = 10.0,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load the Simulink gzip CSV and extract per-channel voltage and current
    arrays for the requested phase.

    Parameters
    ----------
    data_path        : path to the gzip-compressed CSV
    channel_map      : dict mapping channel label to Simulink token numbers
    phase            : which phase to extract (1=A, 2=B, 3=C)
    t_start_discard  : discard all rows with Time <= this value (startup transient)

    Returns
    -------
    dict  {channel_label: {"v": np.ndarray, "i": np.ndarray}}
    """
    print(f"Loading simulation data: {data_path}")
    df = pd.read_csv(data_path, compression="gzip")
    df = df[df["Time"] > t_start_discard].reset_index(drop=True)
    print(f"  {len(df):,} samples after discarding t <= {t_start_discard} s")

    channels = {}
    for label, tokens in channel_map.items():
        v_col = _col("v_abc", tokens["v_token"], phase)
        i_col = _col("i_abc", tokens["i_token"], phase)

        if v_col not in df.columns:
            raise KeyError(
                f"Column '{v_col}' not found for channel {label}. "
                f"Check SIMULINK_CHANNEL_MAP and the CSV header."
            )
        if i_col not in df.columns:
            raise KeyError(
                f"Column '{i_col}' not found for channel {label}. "
                f"Check SIMULINK_CHANNEL_MAP and the CSV header."
            )

        channels[label] = {
            "v": df[v_col].values,
            "i": df[i_col].values,
        }
        print(f"  {label:>4s}  v←{v_col}  i←{i_col}")

    return channels


def run_simulation(
    data_path: str,
    output_dir: str,
    f0_nom: float = 50.0,
    fs: float = 500 * 50,
    window_sec: float = 2.0,
    overlap_sec: float = 1.0,
    channel_map: dict = SIMULINK_CHANNEL_MAP,
    phase: int = PHASE,
) -> dict:
    """
    End-to-end pipeline for the Simulink simulation dataset.

    Calls load_simulation_data(), then run_channel() for every channel,
    then save_results_to_csv() and plot_results().

    Parameters
    ----------
    data_path   : path to gzip CSV
    output_dir  : where to write CSVs and figures
    f0_nom      : nominal fundamental frequency (Hz)
    fs          : sampling frequency (Hz)
    window_sec  : sliding window length (seconds)
    overlap_sec : overlap between windows (seconds)
    channel_map : Simulink token map (default = SIMULINK_CHANNEL_MAP)
    phase       : which abc phase to analyse (1, 2, or 3)

    Returns
    -------
    all_results : dict  {channel_label: [per-window result dicts]}
    """
    cfg = {
        # detection settings — adjust as needed
        "peak_thresh": 0.003,
        "freq_max": 150,
        "sideband_tol": 0.1,
        "freq_match_tol": 0.05,
        # window settings (passed through to make_window_indices)
        "window_sec": window_sec,
        "overlap_sec": overlap_sec,
    }

    print(f"\n{'='*60}")
    print("  Simulation Pipeline")
    print(f"  f0_nom={f0_nom} Hz | fs={fs} Hz | phase={phase}")
    print(f"  window={window_sec} s | overlap={overlap_sec} s")
    print(f"{'='*60}\n")

    channels = load_simulation_data(data_path, channel_map, phase)
    print()

    all_results = {}
    for label, sigs in channels.items():
        all_results[label] = run_channel(sigs["v"], sigs["i"], fs, f0_nom, cfg, label)
        print()

    print("Saving CSVs ...")
    save_results_to_csv(all_results, output_dir)

    print("\nGenerating figures ...")
    from pnnl_emt_swod.plotting import plot_results

    plot_results(output_dir, window_sec=window_sec)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# END SIMULATION DATA ADAPTER
# ══════════════════════════════════════════════════════════════════════════════


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Option A: generic pipeline (edit CONFIG at top of file) ───────────────
    # results = main(CONFIG)

    # ── Option B: simulation dataset pipeline ─────────────────────────────────
    results = run_simulation(
        data_path=r"C:\Users\rame388\OneDrive - PNNL\projects_backup\OEDISI_II\Oscillation_Detection\algorithm\FO_A11_1pt5Hz_pt03.csv",  # path to gzip-compressed CSV
        output_dir=r"C:\Users\rame388\OneDrive - PNNL\projects_backup\OEDISI_II\Oscillation_Detection\algorithm",
        f0_nom=50.0,
        fs=500 * 50,
        window_sec=2.0,
        overlap_sec=1.0,
        phase=1,  # 1=phase A, 2=phase B, 3=phase C
    )
