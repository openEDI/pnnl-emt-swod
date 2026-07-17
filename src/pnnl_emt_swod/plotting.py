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


def plot_results_from_feather(outputs_dir: str | Path) -> list:
    """Generate reports directly from standard recorder feather files."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from pathlib import Path
    
    outputs_dir = Path(outputs_dir)
    freq_df = pd.read_feather(outputs_dir / "swod_oscillation_frequency.feather")
    amp_df = pd.read_feather(outputs_dir / "swod_oscillation_amplitude.feather")
    ssp_real_df = pd.read_feather(outputs_dir / "swod_real_ssp.feather")
    ssp_react_df = pd.read_feather(outputs_dir / "swod_reactive_ssp.feather")
    
    channels = [c for c in freq_df.columns if c != "time"]
    figures = []
    
    # ── Figure 1: Detection Timeline ──
    n_ch = len(channels)
    n_cols = 2
    n_rows = (n_ch + 1) // n_cols
    fig1, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig1.suptitle("Detected Oscillation Frequencies Over Time", fontsize=13, y=1.01)
    
    for idx, ch in enumerate(channels):
        ax = axes[idx // n_cols, idx % n_cols]
        times = freq_df["time"].to_numpy()
        freqs = freq_df[ch].to_numpy()
        real_ssp = ssp_real_df[ch].to_numpy()
        
        det_mask = freqs > 0.01
        
        ax.set_title(ch, fontsize=10, fontweight="bold", pad=3)
        ax.grid(True, alpha=0.3)
        
        if not np.any(det_mask):
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
            t_det = times[det_mask]
            f_det = freqs[det_mask]
            complete = np.abs(real_ssp[det_mask]) > 1e-9
            
            colors = ["#1f77b4" if c else "#ff7f0e" for c in complete]
            ax.scatter(
                t_det,
                f_det,
                c=colors,
                s=40,
                alpha=0.75,
                linewidths=0.4,
                edgecolors="k",
            )
        ax.set_ylabel("Freq (Hz)", fontsize=8)
        
    for idx in range(n_ch, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)
        
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
    fig1.legend(
        handles=legend_els,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig1.tight_layout()
    figures.append(fig1)
    
    # ── Figure 2: Real and Reactive SSP over time ──
    fig2, (ax_real, ax_react) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig2.suptitle("DEF and SSP (Real/Reactive Power) Over Time — All Channels", fontsize=13, y=1.01)
    
    ax_real.set_ylabel("Real SSP\n(Pp + Pm)", fontsize=8)
    ax_react.set_ylabel("Reactive SSP\n(Qp + Qm)", fontsize=8)
    
    for ax in (ax_real, ax_react):
        ax.axhline(0, color="k", lw=0.8, ls="--", zorder=1)
        ax.grid(True, alpha=0.3)
        
    cmap = plt.get_cmap("tab10")
    for idx, ch in enumerate(channels):
        times = freq_df["time"].to_numpy()
        det_mask = freq_df[ch].to_numpy() > 0.01
        if not np.any(det_mask):
            continue
        
        color = cmap(idx % 10)
        ax_real.plot(times[det_mask], ssp_real_df[ch].to_numpy()[det_mask], color=color, lw=1.5, label=ch)
        ax_react.plot(times[det_mask], ssp_react_df[ch].to_numpy()[det_mask], color=color, lw=1.5, label=ch)
        
    ax_real.legend(loc="upper right", fontsize=8)
    ax_react.set_xlabel("Time (s)", fontsize=10)
    fig2.tight_layout()
    figures.append(fig2)
    
    # ── Figure 3: Source Localization Snapshot ──
    avg_ssp_real = []
    avg_ssp_react = []
    avg_apparent = []
    
    for ch in channels:
        freqs = freq_df[ch].to_numpy()
        det_mask = freqs > 0.01
        if np.any(det_mask):
            real_vals = ssp_real_df[ch].to_numpy()[det_mask]
            react_vals = ssp_react_df[ch].to_numpy()[det_mask]
            avg_ssp_real.append(float(np.mean(real_vals)))
            avg_ssp_react.append(float(np.mean(react_vals)))
            avg_apparent.append(float(np.mean(np.sqrt(real_vals**2 + react_vals**2))))
        else:
            avg_ssp_real.append(0.0)
            avg_ssp_react.append(0.0)
            avg_apparent.append(0.0)
            
    fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(channels))
    
    metrics = ["SSP (Real)", "SSP (Reactive)", "Apparent Osc. Power"]
    data_lists = [avg_ssp_real, avg_ssp_react, avg_apparent]
    
    for idx, (vals, label) in enumerate(zip(data_lists, metrics)):
        ax = axes[idx]
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(label, fontsize=10, fontweight="bold")
        
    legend_els = [
        Patch(facecolor="#d62728", label="Negative (potential source)"),
        Patch(facecolor="#1f77b4", label="Positive"),
    ]
    fig3.legend(
        handles=legend_els,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig3.suptitle("Source Localization Metrics — Time-Averaged", fontsize=13, y=1.01)
    fig3.tight_layout()
    figures.append(fig3)
    
    return figures

