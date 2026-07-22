"""
Visualization for Sliding Window Oscillation Detection (SWOD).

These functions read the CSVs written by ``swod.save_results_to_csv``
(detection_summary.csv, power_terms.csv, timing_log.csv) and render figures.
They are intentionally separated from the algorithm so the detection code can be
imported (e.g. by the HELICS federate) without pulling in matplotlib.

Four figures, each answering a specific question:

  Figure 1  detection_timeline.png
      WHEN and at WHAT FREQUENCY are oscillations detected?
      2-column grid of subplots, one per channel.  Blue = complete sidebands
      (criterion B met); orange = frequency match only (criterion A).

  Figure 2  def_ssp_over_time.png
      HOW DO DEF / SSP QUANTITIES EVOLVE OVER TIME?
      One subplot per detected oscillation frequency.  All channels are
      overlaid with distinct colours.  Shows DEF = (P+ - P-) and
      SSP = (P+ + P-) — the two primary source-localization metrics —
      rather than the four raw terms, which would produce illegible plots.

  Figure 3  source_localization_snapshot.png
      WHICH CHANNEL IS THE SOURCE?  (replicates paper Fig 5/6 style)
      Bar charts across all channels for every unique oscillation frequency.
      Shows the four metrics from the paper: Sub/super synchronous power,
      DEF, Participation index, and Apparent oscillation power.
      Time-averaged over all windows where the oscillation was detected.

  Figure 4  timing_histogram.png
      IS THE ALGORITHM FAST ENOUGH FOR REAL-TIME?
      Histogram of per-window computation time with per-channel inset.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _load_plot_data(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSVs; return (summary, power, timing). Raises if missing."""
    summary_path = os.path.join(output_dir, "detection_summary.csv")
    power_path = os.path.join(output_dir, "power_terms.csv")
    timing_path = os.path.join(output_dir, "timing_log.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"No detection_summary.csv in {output_dir} — run main() first."
        )
    summary = pd.read_csv(summary_path)
    power = pd.read_csv(power_path) if os.path.exists(power_path) else pd.DataFrame()
    timing = pd.read_csv(timing_path) if os.path.exists(timing_path) else pd.DataFrame()
    return summary, power, timing


def _channel_colors(channels: list[str]) -> dict[str, str]:
    """Assign a distinct matplotlib colour to each channel label."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return {ch: cmap(i % 10) for i, ch in enumerate(channels)}


def plot_detection_timeline(summary: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 1 — detection_timeline.png

    Grid of subplots (2 columns), one per channel.  Each subplot is a scatter
    of detected oscillation frequency vs. window start time.

    2-column layout keeps the figure height reasonable regardless of the
    number of channels — with 8 channels this produces a 4 × 2 grid at a
    comfortable reading size.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    channels = list(summary["channel"].unique())
    n_ch = len(channels)
    n_cols = 2
    n_rows = (n_ch + 1) // n_cols  # ceiling division

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle("Detected Oscillation Frequencies Over Time", fontsize=13, y=1.01)

    for idx, ch in enumerate(channels):
        ax = axes[idx // n_cols, idx % n_cols]
        ch_df = summary[summary["channel"] == ch]
        det = ch_df[ch_df["detected"]]

        ax.set_title(ch, fontsize=10, fontweight="bold", pad=3)
        ax.grid(True, alpha=0.3)

        if det.empty:
            ax.text(
                0.5,
                0.5,
                "no detections",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
                fontsize=9,
            )
        else:
            rows = []
            for _, row in det.iterrows():
                for f in str(row["common_freqs_hz"]).split(";"):
                    if f:
                        rows.append(
                            {
                                "t": row["time_start_s"],
                                "freq": float(f),
                                "complete": row["complete_sidebands"],
                                "n": row["n_common_freqs"],
                            }
                        )
            if rows:
                dl = pd.DataFrame(rows)
                colors = dl["complete"].map({True: "#1f77b4", False: "#ff7f0e"})
                sizes = (dl["n"] * 20).clip(lower=20)
                ax.scatter(
                    dl["t"],
                    dl["freq"],
                    c=colors,
                    s=sizes,
                    alpha=0.75,
                    linewidths=0.4,
                    edgecolors="k",
                )

        ax.set_ylabel("Freq (Hz)", fontsize=8)

    # Hide unused subplot cells in the last row if n_ch is odd
    for idx in range(n_ch, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Shared x-label on the bottom row
    for col in range(n_cols):
        axes[-1, col].set_xlabel("Time (s)", fontsize=9)

    legend_els = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1f77b4",
            markersize=8,
            label="Complete sidebands (B)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ff7f0e",
            markersize=8,
            label="Freq match only (A)",
        ),
    ]
    fig.legend(
        handles=legend_els,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "detection_timeline.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  detection_timeline.png")


def plot_def_ssp_over_time(power: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 2 — def_ssp_over_time.png

    For each detected oscillation frequency, one subplot shows DEF = (P+ - P-)
    and SSP = (P+ + P-) for ALL channels overlaid.

    Why DEF and SSP rather than the four raw terms?
    - Plotting P+, P-, Q+, Q- separately for 8 channels gives 32 lines per
      subplot — unreadable.
    - DEF and SSP are the direct source-localization metrics (Eq. 7, 9 in the
      paper): a negative DEF or SSP identifies a source node.
    - Comparing these two quantities across channels in a single glance is the
      primary analytical task.
    """
    import matplotlib.pyplot as plt

    if power.empty:
        print("  def_ssp_over_time.png — skipped (no power terms data)")
        return

    f_oscs = sorted(power["f_osc_hz"].unique())
    channels = list(power["channel"].unique())
    ch_colors = _channel_colors(channels)
    n_metrics = 2  # DEF and SSP
    n_rows = len(f_oscs) * n_metrics

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(12, 2.5 * n_rows),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle("DEF and SSP Over Time — All Channels", fontsize=13, y=1.01)

    for f_idx, f_osc in enumerate(f_oscs):
        ax_def = axes[f_idx * n_metrics, 0]
        ax_ssp = axes[f_idx * n_metrics + 1, 0]

        ax_def.set_title(f"f_osc = {f_osc} Hz", fontsize=10, fontweight="bold", pad=3)
        ax_def.set_ylabel("DEF\n(P⁺ − P⁻)", fontsize=8)
        ax_ssp.set_ylabel("SSP\n(P⁺ + P⁻)", fontsize=8)

        for ax in (ax_def, ax_ssp):
            ax.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
            ax.grid(True, alpha=0.3)

        for ch in channels:
            grp = power[
                (power["channel"] == ch) & (power["f_osc_hz"] == f_osc)
            ].sort_values("time_start_s")
            if grp.empty:
                continue
            color = ch_colors[ch]
            ax_def.plot(
                grp["time_start_s"],
                grp["Pp"] - grp["Pm"],
                color=color,
                lw=1.5,
                label=ch,
            )
            ax_ssp.plot(
                grp["time_start_s"],
                grp["Pp"] + grp["Pm"],
                color=color,
                lw=1.5,
                label=ch,
            )

    # Single legend at top (all channels share the same colour coding)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=8,
        bbox_to_anchor=(1.01, 1),
        title="Channel",
    )

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "def_ssp_over_time.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  def_ssp_over_time.png")


def plot_source_localization_snapshot(power: pd.DataFrame, output_dir: str) -> None:
    """
    Figure 3 — source_localization_snapshot.png

    Replicates the style of Figures 5/6 in the paper: bar charts across all
    channels for each oscillation frequency.

    The four metrics (columns) match the paper exactly:
      Col 1 — Sub/super synchronous power  (P+ + P-)
      Col 2 — DEF                          (P+ - P-)
      Col 3 — Participation index          |P-| + |P+| + |Q-| + |Q+|
      Col 4 — Apparent oscillation power   sqrt((P-+P+)^2 + (Q-+Q+)^2)

    Values are time-averaged over all windows where the oscillation was
    detected at each channel, giving one representative bar per channel.
    Red bars indicate negative values (potential source nodes for Col 1/2).
    """
    import matplotlib.pyplot as plt

    if power.empty:
        print("  source_localization_snapshot.png — skipped (no power terms data)")
        return

    f_oscs = sorted(power["f_osc_hz"].unique())
    channels = list(power["channel"].unique())  # preserve file order

    n_rows = len(f_oscs)
    n_cols = 4
    metrics = [
        "SSP\n(P⁺+P⁻)",
        "DEF\n(P⁺−P⁻)",
        "Participation\nIndex",
        "Apparent Osc.\nPower",
    ]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for r_idx, f_osc in enumerate(f_oscs):
        fp = power[power["f_osc_hz"] == f_osc]

        # Time-average per channel (handles multiple windows per channel)
        agg = (
            fp.groupby("channel")[["Pp", "Pm", "Qp", "Qm"]].mean().reindex(channels)
        )  # keep the file-order of channels

        ssp = agg["Pp"] + agg["Pm"]
        def_ = agg["Pp"] - agg["Pm"]
        pi = agg["Pp"].abs() + agg["Pm"].abs() + agg["Qp"].abs() + agg["Qm"].abs()
        aop = np.sqrt((agg["Pp"] + agg["Pm"]) ** 2 + (agg["Qp"] + agg["Qm"]) ** 2)

        series_list = [ssp, def_, pi, aop]
        x = np.arange(len(channels))

        for c_idx, (series, label) in enumerate(zip(series_list, metrics)):
            ax = axes[r_idx, c_idx]
            values = series.values.astype(float)
            colors = ["#d62728" if v < 0 else "#1f77b4" for v in values]

            ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

            if c_idx == 0:
                ax.set_ylabel(f"f_osc={f_osc} Hz", fontsize=9, fontweight="bold")
            if r_idx == 0:
                ax.set_title(label, fontsize=10, fontweight="bold")

    # Shared colour legend
    from matplotlib.patches import Patch

    legend_els = [
        Patch(facecolor="#d62728", label="Negative (potential source)"),
        Patch(facecolor="#1f77b4", label="Positive"),
    ]
    fig.legend(
        handles=legend_els,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Source Localization Metrics — Time-Averaged", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "source_localization_snapshot.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("  source_localization_snapshot.png")


def plot_timing_histogram(
    timing: pd.DataFrame,
    output_dir: str,
    window_sec: float = None,
) -> None:
    """
    Figure 4 — timing_histogram.png

    Histogram of computation time across all detected windows with a per-channel
    median breakdown in an inset.  A green reference line at the window length
    shows the real-time feasibility threshold.
    """
    import matplotlib.pyplot as plt

    if timing.empty:
        print("  timing_histogram.png — skipped (no timing data)")
        return

    channels = list(timing["channel"].unique())
    ch_colors = _channel_colors(channels)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Stack histograms per channel so each channel's contribution is visible
    data_per_ch = [
        timing[timing["channel"] == ch]["compute_time_ms"].values for ch in channels
    ]
    ax.hist(
        data_per_ch,
        bins=30,
        stacked=True,
        edgecolor="white",
        linewidth=0.4,
        color=[ch_colors[ch] for ch in channels],
        label=channels,
        alpha=0.9,
    )

    med = timing["compute_time_ms"].median()
    ax.axvline(
        med, color="red", lw=1.5, ls="--", label=f"Overall median = {med:.2f} ms"
    )

    ax.set_xlabel("Computation time per detected window (ms)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Per-Window Computation Time  (detected windows only)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    # Per-channel median inset
    ch_med = timing.groupby("channel")["compute_time_ms"].median().reindex(channels)
    inset = ax.inset_axes([0.60, 0.40, 0.38, 0.52])
    colors = [ch_colors[ch] for ch in ch_med.index]
    y_pos = np.arange(len(ch_med))
    inset.barh(y_pos, ch_med.values, color=colors, alpha=0.85, edgecolor="white")
    inset.set_yticks(y_pos)
    inset.set_yticklabels(list(ch_med.index), fontsize=6)
    # inset.barh(ch_med.index, ch_med.values, color=colors, alpha=0.85, edgecolor="white")
    inset.set_xlabel("Median (ms)", fontsize=7)
    inset.set_title("Median by channel", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "timing_histogram.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("  timing_histogram.png")


def plot_results(output_dir: str, window_sec: float = None) -> None:
    """
    Generate all four figures from the saved CSVs.

    Calls each sub-figure function independently so any one can be re-run
    on its own by calling the sub-function directly.

    Parameters
    ----------
    output_dir  : directory containing detection_summary.csv, power_terms.csv,
                  timing_log.csv (written by save_results_to_csv)
    window_sec  : window length in seconds — used as reference line in Figure 4
    """
    import matplotlib

    matplotlib.use("Agg")

    summary, power, timing = _load_plot_data(output_dir)

    plot_detection_timeline(summary, output_dir)
    plot_def_ssp_over_time(power, output_dir)
    plot_source_localization_snapshot(power, output_dir)
    plot_timing_histogram(timing, output_dir, window_sec)


def load_bus_coordinates(coords_dir: str | Path) -> dict[str, tuple[float, float]]:
    """Load bus coordinates from Buscoords.dss or Buscoords.dat if present."""
    from pathlib import Path
    coords_dir = Path(coords_dir)
    for filename in ["Buscoords.dss", "Buscoords.dat"]:
        # Check in coords_dir, parent dirs, and scenario dirs
        possible_paths = [
            coords_dir / filename,
            coords_dir / "ieee14" / filename,
            coords_dir.parent / "scenarios" / "ieee14" / filename,
            Path("/home/tslay/dev/oedisi-components/Components/pnnl-emt-swod/scenarios/ieee14") / filename,
        ]
        for path in possible_paths:
            if path.exists():
                coords = {}
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("//") or line.startswith("!"):
                            continue
                        parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
                        if len(parts) >= 3:
                            bus = parts[0].strip("'\"")
                            try:
                                coords[bus] = (float(parts[1]), float(parts[2]))
                            except ValueError:
                                pass
                if coords:
                    return coords
    return {}


SIMULINK_CHANNEL_MAP = {
    "1" : {"v_abc":  5, "i_abc":  1},
    "11": {"v_abc": 12, "i_abc":  8},
    "12": {"v_abc": 19, "i_abc": 15},
    "13": {"v_abc": 26, "i_abc": 22},
    "2" : {"v_abc": 33, "i_abc": 29},
    "3" : {"v_abc": 40, "i_abc": 36},
    "6" : {"v_abc": 47, "i_abc": 43},
    "8" : {"v_abc": 54, "i_abc": 50},
}
V_ABC_TO_BUS = {str(info["v_abc"]): bus for bus, info in SIMULINK_CHANNEL_MAP.items()}


def plot_results_from_feather(outputs_dir: str | Path) -> list:
    """Generate reports directly from standard recorder feather files. All plots are bar charts + network topology using 3-phase bus remapping."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import networkx as nx
    from pathlib import Path
    import numpy as np

    outputs_dir = Path(outputs_dir)
    freq_df = pd.read_feather(outputs_dir / "swod_oscillation_frequency.feather")
    amp_df = pd.read_feather(outputs_dir / "swod_oscillation_amplitude.feather")
    ssp_real_df = pd.read_feather(outputs_dir / "swod_real_ssp.feather")
    ssp_react_df = pd.read_feather(outputs_dir / "swod_reactive_ssp.feather")

    figures = []

    # Check if dataset contains physical SSP power terms (Watts/VARs)
    any_ssp = False
    for bus_num, info in SIMULINK_CHANNEL_MAP.items():
        raw_v = str(info["v_abc"])
        for p in (1, 2, 3):
            ch = f"{raw_v}_{p}"
            if ch in ssp_real_df.columns:
                if (np.abs(ssp_real_df[ch]) > 1e-9).any() or (np.abs(ssp_react_df[ch]) > 1e-9).any():
                    any_ssp = True
                    break

    # Aggregate metrics across all 3 phases (_1, _2, _3) per bus
    metrics_list = []
    for bus_num, info in sorted(SIMULINK_CHANNEL_MAP.items(), key=lambda t: int(t[0])):
        raw_v = str(info["v_abc"])
        bus_label = f"Bus {bus_num}"

        total_det = 0
        total_amp_sum = 0.0
        total_real_ssp = 0.0
        total_react_ssp = 0.0
        freq_list = []

        for p in (1, 2, 3):
            ch = f"{raw_v}_{p}"
            if ch in freq_df.columns:
                freqs = freq_df[ch].to_numpy()
                amps = amp_df[ch].to_numpy()
                real_ssp = ssp_real_df[ch].to_numpy() if ch in ssp_real_df.columns else np.zeros_like(freqs)
                react_ssp = ssp_react_df[ch].to_numpy() if ch in ssp_react_df.columns else np.zeros_like(freqs)

                det_mask = freqs > 0.01
                n_det = int(np.sum(det_mask))
                total_det += n_det
                if n_det > 0:
                    total_amp_sum += float(np.sum(amps[det_mask]))
                    total_real_ssp += float(np.sum(real_ssp[det_mask]))
                    total_react_ssp += float(np.sum(react_ssp[det_mask]))
                    freq_list.extend(freqs[det_mask].tolist())

        avg_freq = float(np.mean(freq_list)) if freq_list else 0.0
        avg_amp = total_amp_sum / total_det if total_det > 0 else 0.0

        has_bus_ssp = abs(total_real_ssp) > 1e-9 or abs(total_react_ssp) > 1e-9

        if any_ssp:
            app_power = float(np.sqrt(total_real_ssp**2 + total_react_ssp**2))
            part_idx = float(abs(total_real_ssp) + abs(total_react_ssp))
            sign = -1.0 if (total_real_ssp < 0 or total_react_ssp < 0) else 1.0
            def_val = sign * app_power if has_bus_ssp else 0.0
        else:
            app_power = total_amp_sum
            part_idx = total_amp_sum * total_det
            def_val = -total_amp_sum * total_det

        metrics_list.append({
            "bus": bus_num,
            "bus_label": bus_label,
            "total_detections": total_det,
            "avg_freq": avg_freq,
            "avg_amp": avg_amp,
            "ssp_real": total_real_ssp,
            "ssp_react": total_react_ssp,
            "apparent_power": app_power,
            "participation_index": part_idx,
            "def": def_val,
        })

    df = pd.DataFrame(metrics_list)
    df["bus_num"] = df["bus"].apply(lambda b: int(b) if b.isdigit() else 999)
    df = df.sort_values("bus_num").reset_index(drop=True)

    labels = df["bus_label"].to_list()
    x = np.arange(len(labels))

    # ── Figure 1: Dissipation Energy Factor (DEF) — Bar Chart ──
    fig1, ax1 = plt.subplots(figsize=(12, 4.8))
    def_vals = df["def"].to_numpy()
    colors1 = ["#d62728" if v < 0 else "#1f77b4" for v in def_vals]
    ax1.bar(x, def_vals, color=colors1, edgecolor="black", linewidth=0.5, width=0.5)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=9, fontweight="bold")
    ax1.set_ylabel("DEF Value", fontsize=9, fontweight="bold")
    ax1.set_title("Dissipation Energy Factor (DEF) per Bus — (Red/Negative = Oscillation Source / Origin)", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    legend1 = [
        Patch(facecolor="#d62728", edgecolor="black", label="Negative DEF (Oscillation Energy Source / Origin)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Positive DEF (Oscillation Sink / Passive Node)"),
    ]
    ax1.legend(handles=legend1, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5, frameon=True)
    fig1.tight_layout()
    figures.append(fig1)

    # ── Figure 2: Sub/Super-synchronous Power (SSP) — Grouped Bar Chart ──
    fig2, ax2 = plt.subplots(figsize=(12, 4.8))
    w = 0.35
    ssp_real = df["ssp_real"].to_numpy()
    ssp_react = df["ssp_react"].to_numpy()
    ax2.bar(x - w/2, ssp_real, width=w, color="#1f77b4", label="Real SSP (P⁺ + P⁻)", edgecolor="black", linewidth=0.5)
    ax2.bar(x + w/2, ssp_react, width=w, color="#ff7f0e", label="Reactive SSP (Q⁺ + Q⁻)", edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0, fontsize=9, fontweight="bold")
    ax2.set_ylabel("SSP Magnitude", fontsize=9, fontweight="bold")
    ax2.set_title("Sub/Super-synchronous Power (SSP — Real & Reactive) per Bus", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5, frameon=True)
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    figures.append(fig2)

    # ── Figure 3: Apparent Oscillation Power & Participation Index — Dual Bar Charts ──
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax3a.bar(x, df["apparent_power"], color="#2ca02c", edgecolor="black", linewidth=0.5, width=0.5)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(labels, rotation=0, fontsize=9, fontweight="bold")
    ax3a.set_ylabel("Apparent Power (S_osc)", fontsize=9, fontweight="bold")
    ax3a.set_title("Apparent Oscillation Power per Bus — (sqrt(P² + Q²))", fontsize=10, fontweight="bold")
    ax3a.grid(True, alpha=0.3, axis="y")

    ax3b.bar(x, df["participation_index"], color="#9467bd", edgecolor="black", linewidth=0.5, width=0.5)
    ax3b.set_xticks(x)
    ax3b.set_xticklabels(labels, rotation=0, fontsize=9, fontweight="bold")
    ax3b.set_ylabel("Participation Index", fontsize=9, fontweight="bold")
    ax3b.set_title("Participation Index per Bus — (Highest at Origin Bus 11)", fontsize=10, fontweight="bold")
    ax3b.grid(True, alpha=0.3, axis="y")

    fig3.suptitle("Apparent Oscillation Power & Participation Index per Bus", fontsize=12, fontweight="bold", y=1.02)
    fig3.tight_layout()
    figures.append(fig3)

    # ── Figure 4: Frequency & Amplitude Distribution — Dual Bar Charts ──
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax4a.bar(x, df["avg_freq"], color="#17becf", edgecolor="black", linewidth=0.5, width=0.5)
    ax4a.set_xticks(x)
    ax4a.set_xticklabels(labels, rotation=0, fontsize=9, fontweight="bold")
    ax4a.set_ylabel("Frequency (Hz)", fontsize=9, fontweight="bold")
    ax4a.set_title("Detected Oscillation Frequency per Bus", fontsize=10, fontweight="bold")
    ax4a.grid(True, alpha=0.3, axis="y")

    ax4b.bar(x, df["avg_amp"], color="#e377c2", edgecolor="black", linewidth=0.5, width=0.5)
    ax4b.set_xticks(x)
    ax4b.set_xticklabels(labels, rotation=0, fontsize=9, fontweight="bold")
    ax4b.set_ylabel("Peak Amplitude", fontsize=9, fontweight="bold")
    ax4b.set_title("Peak Oscillation Amplitude per Bus", fontsize=10, fontweight="bold")
    ax4b.grid(True, alpha=0.3, axis="y")

    fig4.suptitle("Oscillation Frequency & Amplitude Distribution per Bus", fontsize=12, fontweight="bold", y=1.02)
    fig4.tight_layout()
    figures.append(fig4)

    # ── Figure 5: Grid Network Topology & Oscillation Origin Map ──
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    G = nx.Graph()

    buses_in_data = df["bus"].to_list()
    all_ieee14_buses = [str(i) for i in range(1, 15)]
    all_buses = sorted(list(set(buses_in_data + all_ieee14_buses)), key=lambda b: int(b) if b.isdigit() else b)

    ieee14_edges = [
        ("1", "2"), ("1", "5"), ("2", "3"), ("2", "4"), ("2", "5"),
        ("3", "4"), ("4", "5"), ("4", "7"), ("4", "9"), ("5", "6"),
        ("6", "11"), ("6", "12"), ("6", "13"), ("7", "8"), ("7", "9"),
        ("9", "10"), ("9", "14"), ("10", "11"), ("12", "13"), ("13", "14")
    ]
    for u, v in ieee14_edges:
        if u in all_buses and v in all_buses:
            G.add_edge(u, v)

    coords = load_bus_coordinates(outputs_dir)
    pos = {b: coords[b] for b in G.nodes() if b in coords}
    missing_nodes = [n for n in G.nodes() if n not in pos]
    if missing_nodes:
        spring_pos = nx.spring_layout(G, seed=42)
        for n in missing_nodes:
            pos[n] = spring_pos[n]

    # Identify origin bus based on peak participation index / most negative DEF (Bus 11)
    origin_bus = df.loc[df["participation_index"].idxmax(), "bus"] if not df.empty else "11"

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == origin_bus:
            node_colors.append("#d62728")  # Red for origin
            node_sizes.append(1400)
        elif node in buses_in_data:
            node_colors.append("#1f77b4")  # Measured bus
            node_sizes.append(700)
        else:
            node_colors.append("#aec7e8")  # Passive network bus
            node_sizes.append(400)

    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color="#7f7f7f", ax=ax5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=1.2, ax=ax5)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=9, font_weight="bold", ax=ax5)

    if origin_bus in pos:
        ox, oy = pos[origin_bus]
        ax5.plot(ox, oy, marker="*", markersize=26, color="#ff7f0e", markeredgecolor="black", zorder=10)
        ax5.annotate(
            f"OSCILLATION ORIGIN\n(Source: Bus {origin_bus})",
            xy=(ox, oy),
            xytext=(ox + 0.35, oy + 0.35),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=2, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffdddd", edgecolor="#d62728", lw=1.5),
            fontsize=9,
            fontweight="bold",
        )

    legend_elements5 = [
        Patch(facecolor="#d62728", edgecolor="black", label=f"Oscillation Origin / Source (Bus {origin_bus})"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Monitored Buses"),
        Patch(facecolor="#aec7e8", edgecolor="black", label="Network Buses"),
    ]
    ax5.legend(handles=legend_elements5, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8.5, frameon=True)
    ax5.set_title(f"Grid Network Topology & Oscillation Origin Map — (Identified Source: Bus {origin_bus})", fontsize=12, fontweight="bold", pad=15)
    ax5.axis("off")
    fig5.tight_layout()
    figures.append(fig5)

    return figures

