"""
HELICS federate for Sliding Window Oscillation Detection (SWOD).

This federate subscribes to streamed voltage- and current-magnitude measurements,
buffers them per channel into fixed-length analysis windows, runs the SWOD
algorithm (``swod.process_window``) on each completed window, and publishes the
detection results as the outputs declared in ``component_definition.json``.
"""

import json
import logging
from pathlib import Path

import helics as h
import numpy as np
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    CurrentsMagnitude,
    MeasurementArray,
    PowersImaginary,
    PowersReal,
    VoltagesMagnitude,
)
from pydantic import BaseModel

from pnnl_emt_swod import swod

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class ComponentParameters(BaseModel):
    name: str
    window_length: int  # Tuned: 50000 (2 seconds analysis window length at 25 kHz)
    overlap_length: int = (
        25000  # Tuned: 25000 (1 second overlap between consecutive windows)
    )
    fs: float = 500 * 50  # Tuned: 25000.0 (POW reporting rate)
    f0_nom: float = 50.0  # Tuned: 50.0 (nominal fundamental frequency)
    peak_thresh: float = (
        0.003  # Tuned: 0.003 (detection threshold as fraction of fundamental)
    )
    freq_max: float = 150.0  # Tuned: 150.0 (ignore spectral peaks above this frequency)
    sideband_tol: float = 0.1  # Tuned: 0.1 (tolerance for equidistance check)
    freq_match_tol: float = (
        0.05  # Tuned: 0.05 (tolerance for matching V and I peak frequencies)
    )
    deltat: float = 1.0  # Tuned: 1.0 (HELICS simulation time step in seconds)


class Subscriptions:
    """Holder for the federate's HELICS input handles."""

    voltages_abs: VoltagesMagnitude
    currents_abs: CurrentsMagnitude


class Federate:
    """SWOD HELICS value federate."""

    def __init__(self, broker_config: BrokerConfig) -> None:
        self.sub = Subscriptions()
        self.fed = None
        self.info = None

        # Per-channel sample buffers, keyed by channel label.
        self.channels: list[tuple[str, str]] = []  # (voltage_id, current_id) pairs
        self.labels: list[str] = []  # channel labels (== voltage_id)
        self.v_buf: dict[str, list[float]] = {}
        self.i_buf: dict[str, list[float]] = {}
        self.samples_processed = 0

        try:
            self.load_static_inputs()
            self.load_input_mapping()
            self.initialize(broker_config)
            self.register_subscription()
            self.register_publication()
        except Exception:
            self.destroy()
            raise

        # cfg consumed by swod.process_window (only these four keys are read there)
        self.cfg = {
            "peak_thresh": self.static.peak_thresh,
            "freq_max": self.static.freq_max,
            "sideband_tol": self.static.sideband_tol,
            "freq_match_tol": self.static.freq_match_tol,
        }

    # ── Configuration loading ────────────────────────────────────────────

    def load_static_inputs(self) -> None:
        path = Path("static_inputs.json")
        with open(path, encoding="UTF-8") as file:
            config = json.load(file)
        self.static = ComponentParameters(**config)
        logger.info(f"Loaded static inputs for federate '{self.static.name}'")

    def load_input_mapping(self) -> None:
        path = Path("input_mapping.json")
        with open(path, encoding="UTF-8") as file:
            self.inputs = json.load(file)

    # ── HELICS setup ─────────────────────────────────────────────────────

    def initialize(self, broker_config: BrokerConfig) -> None:
        self.info = h.helicsCreateFederateInfo()
        self.info.core_name = self.static.name
        self.info.core_type = h.HELICS_CORE_TYPE_ZMQ
        self.info.core_init = "--federates=1"

        h.helicsFederateInfoSetBroker(self.info, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(self.info, broker_config.broker_port)
        h.helicsFederateInfoSetTimeProperty(
            self.info, h.helics_property_time_delta, self.static.deltat
        )

        self.fed = h.helicsCreateValueFederate(self.static.name, self.info)
        logger.info("Value federate created")

    def register_subscription(self) -> None:
        self.sub.voltages_abs = self.fed.register_subscription(
            self.inputs["voltages_abs"], ""
        )
        self.sub.voltages_abs.option["CONNECTION_OPTIONAL"] = True
        self.sub.voltages_abs.set_default(
            VoltagesMagnitude(ids=[], values=[], time=0.0).model_dump_json()
        )

        self.sub.currents_abs = self.fed.register_subscription(
            self.inputs["currents_abs"], ""
        )
        self.sub.currents_abs.option["CONNECTION_OPTIONAL"] = True
        self.sub.currents_abs.set_default(
            CurrentsMagnitude(ids=[], values=[], time=0.0).model_dump_json()
        )

    def register_publication(self) -> None:
        def _pub(name: str) -> h.HelicsPublication:
            return self.fed.register_publication(name, h.HELICS_DATA_TYPE_STRING, "")

        self.pub_oscillation_frequency = _pub("oscillation_frequency")
        self.pub_oscillation_amplitude = _pub("oscillation_amplitude")
        self.pub_harmonic_frequency = _pub("harmonic_frequency")
        self.pub_harmonic_amplitude = _pub("harmonic_amplitude")
        self.pub_real_ssp = _pub("real_ssp")
        self.pub_reactive_ssp = _pub("reactive_ssp")
        self.pub_watts_rms = _pub("watts_rms")
        self.pub_vars_rms = _pub("vars_rms")

    # ── Buffering ────────────────────────────────────────────────────────

    def _ensure_channels(
        self, voltages: VoltagesMagnitude, currents: CurrentsMagnitude
    ) -> None:
        """Establish the fixed channel list on the first paired message."""
        if self.channels:
            return
        n = min(len(voltages.ids), len(currents.ids))
        if n == 0:
            return
        for v_id, i_id in zip(voltages.ids[:n], currents.ids[:n]):
            label = str(v_id)
            self.channels.append((v_id, i_id))
            self.labels.append(label)
            self.v_buf[label] = []
            self.i_buf[label] = []
        logger.info(f"Established {len(self.labels)} channels: {self.labels}")

    def _append_samples(
        self, voltages: VoltagesMagnitude, currents: CurrentsMagnitude
    ) -> None:
        """Append the latest per-channel sample to each buffer."""
        v_map = dict(zip(voltages.ids, voltages.values))
        i_map = dict(zip(currents.ids, currents.values))
        for (v_id, i_id), label in zip(self.channels, self.labels):
            if v_id in v_map and i_id in i_map:
                self.v_buf[label].append(float(v_map[v_id]))
                self.i_buf[label].append(float(i_map[i_id]))

    # ── Result mapping ───────────────────────────────────────────────────

    @staticmethod
    def _dominant_freq(result: dict) -> float | None:
        """Return the common frequency with the largest voltage-peak amplitude."""
        common = result.get("common_freqs", [])
        if not common:
            return None
        peaks_v = result.get("peaks_v", {})
        # match each common freq to nearest voltage peak, pick the strongest
        best_f, best_amp = None, -1.0
        for f in common:
            if peaks_v:
                nearest = min(peaks_v, key=lambda pf: abs(pf - f))
                amp = peaks_v[nearest]
            else:
                amp = 0.0
            if amp > best_amp:
                best_f, best_amp = f, amp
        return best_f

    def _publish_results(self, results: dict[str, dict], t: float) -> None:
        """Map per-channel SWOD results onto the output publications and publish."""
        osc_freq, osc_amp, real_ssp, reactive_ssp = [], [], [], []

        for label in self.labels:
            result = results.get(label)
            f_dom = self._dominant_freq(result) if result else None

            if f_dom is None:
                osc_freq.append(0.0)
                osc_amp.append(0.0)
                real_ssp.append(0.0)
                reactive_ssp.append(0.0)
                continue

            peaks_v = result.get("peaks_v", {})
            nearest = min(peaks_v, key=lambda pf: abs(pf - f_dom)) if peaks_v else None
            osc_freq.append(float(f_dom))
            osc_amp.append(float(peaks_v.get(nearest, 0.0)) if nearest else 0.0)

            # SSP at the dominant oscillation frequency (Pp+Pm / Qp+Qm), if power
            # terms were computed (only when complete sideband pairs were found).
            Pp, Pm = result.get("Pp", {}), result.get("Pm", {})
            Qp, Qm = result.get("Qp", {}), result.get("Qm", {})
            f_osc = min(Pp, key=lambda pf: abs(pf - f_dom)) if Pp else None
            if f_osc is not None:
                real_ssp.append(float(Pp.get(f_osc, 0.0) + Pm.get(f_osc, 0.0)))
                reactive_ssp.append(float(Qp.get(f_osc, 0.0) + Qm.get(f_osc, 0.0)))
            else:
                real_ssp.append(0.0)
                reactive_ssp.append(0.0)

        ids = list(self.labels)
        self.pub_oscillation_frequency.publish(
            MeasurementArray(
                ids=ids, values=osc_freq, time=t, units="Hz"
            ).model_dump_json()
        )
        self.pub_oscillation_amplitude.publish(
            MeasurementArray(
                ids=ids, values=osc_amp, time=t, units=""
            ).model_dump_json()
        )
        self.pub_real_ssp.publish(
            MeasurementArray(
                ids=ids, values=real_ssp, time=t, units="W"
            ).model_dump_json()
        )
        self.pub_reactive_ssp.publish(
            MeasurementArray(
                ids=ids, values=reactive_ssp, time=t, units="VAR"
            ).model_dump_json()
        )

        self._publish_stubs(t)

    def _publish_stubs(self, t: float) -> None:
        """Publish empty (typed) arrays for outputs not yet implemented."""
        self.pub_harmonic_frequency.publish(
            MeasurementArray(ids=[], values=[], time=t, units="Hz").model_dump_json()
        )
        self.pub_harmonic_amplitude.publish(
            MeasurementArray(ids=[], values=[], time=t, units="").model_dump_json()
        )
        self.pub_watts_rms.publish(
            PowersReal(ids=[], values=[], equipment_ids=[], time=t).model_dump_json()
        )
        self.pub_vars_rms.publish(
            PowersImaginary(
                ids=[], values=[], equipment_ids=[], time=t
            ).model_dump_json()
        )

    # ── Run loop ─────────────────────────────────────────────────────────

    def step(self, t: float) -> None:
        """Read the latest measurements, buffer them, and process ready windows.
        """

        voltages = VoltagesMagnitude.model_validate(self.sub.voltages_abs.json)
        currents = CurrentsMagnitude.model_validate(self.sub.currents_abs.json)

        self._ensure_channels(voltages, currents)
        self._append_samples(voltages, currents)

        wl = self.static.window_length
        step = max(1, wl - self.static.overlap_length)

        if not self.labels:
            return

        first_label = self.labels[0]
        num_ready_windows = 0
        while len(self.v_buf[first_label]) >= wl + num_ready_windows * step:
            num_ready_windows += 1

        for w_idx in range(num_ready_windows):
            results_for_window: dict[str, dict] = {}
            for label in self.labels:
                v_win = np.asarray(
                    self.v_buf[label][w_idx * step : w_idx * step + wl],
                    dtype=float,
                )
                i_win = np.asarray(
                    self.i_buf[label][w_idx * step : w_idx * step + wl],
                    dtype=float,
                )
                results_for_window[label] = swod.process_window(
                    v_win, i_win, self.static.fs, self.static.f0_nom, self.cfg
                )

            self._publish_results(results_for_window, t)

            # Comparative logging matching example.py/swod.py output format
            # We log the first label or all labels; since there are 24 labels, logging the first one is enough or we can log a summary
            # Let's log A11 (which is bus4.1, bus4.2, bus4.3 - wait, let's log any detected channel to keep it concise)
            for label, r in results_for_window.items():
                start = self.samples_processed + w_idx * step
                n_common = len(r.get("common_freqs", []))
                n_pairs = len(r.get("sideband_pairs", []))
                if r["valid"]:
                    logger.info(
                        f"    {label}: t={start / self.static.fs:6.1f}s | "
                        f"f0={r['f_est']:.4f} Hz | DETECTED  ({n_common} freqs, {n_pairs} pairs)"
                    )
                else:
                    logger.info(
                        f"    {label}: t={start / self.static.fs:6.1f}s | "
                        f"f0={r['f_est']:.4f} Hz | no detection"
                    )

        if num_ready_windows > 0:
            for label in self.labels:
                self.v_buf[label] = self.v_buf[label][num_ready_windows * step :]
                self.i_buf[label] = self.i_buf[label][num_ready_windows * step :]
            self.samples_processed += num_ready_windows * step

    def run(self) -> None:
        self.fed.enter_initializing_mode()
        self.fed.enter_executing_mode()
        logger.info("Entering execution mode")

        try:
            granted = h.helicsFederateRequestTime(self.fed, h.HELICS_TIME_MAXTIME)
            while granted < h.HELICS_TIME_MAXTIME:
                if (
                    self.sub.voltages_abs.is_updated()
                    and self.sub.currents_abs.is_updated()
                ):
                    self.step(granted)
                granted = h.helicsFederateRequestTime(self.fed, h.HELICS_TIME_MAXTIME)
        finally:
            self.destroy()

    def destroy(self) -> None:
        """Clean up and disconnect the federate."""
        if self.fed is not None:
            h.helicsFederateDisconnect(self.fed)
            logger.info("Federate disconnected")
            h.helicsFederateFree(self.fed)
        if self.info is not None:
            h.helicsFederateInfoFree(self.info)
        h.helicsCloseLibrary()


def run_simulator(broker_config: BrokerConfig) -> None:
    """Entry point for running the SWOD federate."""
    schema = json.dumps(ComponentParameters.model_json_schema(), indent=2)
    with open("schema.json", "w") as f:
        f.write(schema)

    sfed = Federate(broker_config)
    try:
        sfed.run()
    except Exception:
        logger.exception("SWOD simulation failed")


def main() -> None:
    run_simulator(BrokerConfig(broker_ip="127.0.0.1", broker_port=23404))


if __name__ == "__main__":
    main()
