import sys
from pathlib import Path

import numpy as np

# Insert package source directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pnnl_emt_swod import swod  # type: ignore # noqa: E402


def test_process_window_synthetic_signal() -> None:
    """Test process_window on a synthetic fundamental plus sideband oscillation."""
    fs = 2500.0  # sampling frequency
    f0_nom = 50.0  # nominal frequency
    t_sec = 2.0  # 2 second window
    n_samples = int(fs * t_sec)
    t = np.linspace(0, t_sec, n_samples, endpoint=False)

    # 50 Hz fundamental sine wave with amplitude 1.0
    v_fundamental = np.sin(2 * np.pi * f0_nom * t)
    # Sideband at 50 + 15 = 65 Hz with amplitude 0.05
    v_sideband = 0.05 * np.sin(2 * np.pi * 65.0 * t)
    v_win = v_fundamental + v_sideband

    # Current signal matching voltage for power calculation
    i_fundamental = np.sin(2 * np.pi * f0_nom * t)
    i_sideband = 0.05 * np.sin(2 * np.pi * 65.0 * t)
    i_win = i_fundamental + i_sideband

    cfg = {
        "peak_thresh": 0.003,
        "freq_max": 150.0,
        "sideband_tol": 0.1,
        "freq_match_tol": 0.05,
    }

    result = swod.process_window(v_win, i_win, fs, f0_nom, cfg)

    assert "valid" in result
    # We should have found a valid fundamental estimate around 50 Hz
    assert "f_est" in result
    assert abs(result["f_est"] - 50.0) < 0.5
