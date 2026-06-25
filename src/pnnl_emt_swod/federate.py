"""
HELICS federate for Sliding Window Oscillation Detection (SWOD).

This federate subscribes to streamed voltage- and current-magnitude measurements,
buffers them per channel into fixed-length analysis windows, runs the SWOD
algorithm (``swod.process_window``) on each completed window, and publishes the
detection results as the outputs declared in ``component_definition.json``.

Subscriptions (dynamic_inputs)
------------------------------
voltages_abs : VoltagesMagnitude   per-step voltage magnitude per node
currents_abs : CurrentsMagnitude   per-step current magnitude per branch

Publications (dynamic_outputs)
------------------------------
oscillation_frequency : MeasurementArray   dominant detected oscillation freq / channel
oscillation_amplitude : MeasurementArray   amplitude of that oscillation / channel
harmonic_frequency    : MeasurementArray   STUB (empty) — to be implemented
harmonic_amplitude    : MeasurementArray   STUB (empty) — to be implemented
real_ssp              : MeasurementArray   sub/super-synchronous real power (P+ + P-)
reactive_ssp          : MeasurementArray   sub/super-synchronous reactive power (Q+ + Q-)
watts_rms             : PowersReal         STUB (empty) — to be implemented
vars_rms              : PowersImaginary    STUB (empty) — to be implemented

Windowing
---------
Each granted HELICS time step delivers one scalar sample per channel.  Samples
are accumulated in a per-channel ring buffer; once a channel reaches
``window_length`` samples a window is sliced, analysed, and the buffer advanced
by ``window_length - overlap_length`` samples.

NOTE: voltage and current channels are paired by position (the i-th voltage id is
paired with the i-th current id).  This is the assumption to validate against the
upstream feeder/player — VoltagesMagnitude ids (nodes) and CurrentsMagnitude ids
(branches) generally differ, so the upstream ordering must be confirmed.
"""

import json
import logging
from pathlib import Path

import helics as h
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
    window_length: int  # samples per analysis window
    overlap_length: int = 0  # samples of overlap between consecutive windows
    fs: float = 500 * 50  # sampling frequency (Hz)
    f0_nom: float = 50.0  # nominal fundamental frequency (Hz)
    peak_thresh: float = 0.003  # detection threshold (fraction of fundamental)
    freq_max: float = 150.0  # ignore peaks above this frequency (Hz)
    sideband_tol: float = 0.1  # equidistance tolerance for sideband pairs (Hz)
    freq_match_tol: float = 0.05  # V/I peak frequency match tolerance (Hz)
    deltat: float = 0.01  # HELICS time step (s)


class Subscriptions:
    """Holder for the federate's HELICS input handles."""

    voltages_abs: VoltagesMagnitude
    currents_abs: CurrentsMagnitude


class Federate:
    """SWOD HELICS value federate."""

    def __init__(self, broker_config: BrokerConfig) -> None:
        self.sub = Subscriptions()

        # Per-channel sample buffers, keyed by channel label.
        self.channels: list[tuple[str, str]] = []  # (voltage_id, current_id) pairs
        self.labels: list[str] = []  # channel labels (== voltage_id)
        self.v_buf: dict[str, list[float]] = {}
        self.i_buf: dict[str, list[float]] = {}

        self.load_static_inputs()
        self.load_input_mapping()
        self.initialize(broker_config)
        self.register_subscription()
        self.register_publication()

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
        self.sub.currents_abs = self.fed.register_subscription(
            self.inputs["currents_abs"], ""
        )

    def register_publication(self) -> None:
        def _pub(name: str):
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

    def _ready_windows(self) -> dict[str, dict]:
        """
        For each channel that has accumulated a full window, slice it, advance
        the buffer, and return {label: process_window result}.
        """
        wl = self.static.window_length
        step = max(1, wl - self.static.overlap_length)
        results: dict[str, dict] = {}

        for label in self.labels:
            if len(self.v_buf[label]) >= wl and len(self.i_buf[label]) >= wl:
                import numpy as np

                v_win = np.asarray(self.v_buf[label][:wl], dtype=float)
                i_win = np.asarray(self.i_buf[label][:wl], dtype=float)
                results[label] = swod.process_window(
                    v_win, i_win, self.static.fs, self.static.f0_nom, self.cfg
                )
                # advance buffers
                self.v_buf[label] = self.v_buf[label][step:]
                self.i_buf[label] = self.i_buf[label][step:]

        return results

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
        """Read the latest measurements, buffer them, and process ready windows."""
        if not (
            self.sub.voltages_abs.is_updated() and self.sub.currents_abs.is_updated()
        ):
            return

        voltages = VoltagesMagnitude.model_validate(self.sub.voltages_abs.json)
        currents = CurrentsMagnitude.model_validate(self.sub.currents_abs.json)

        self._ensure_channels(voltages, currents)
        self._append_samples(voltages, currents)

        results = self._ready_windows()
        if results:
            self._publish_results(results, t)

    def run(self) -> None:
        self.fed.enter_initializing_mode()
        self.fed.enter_executing_mode()
        logger.info("Entering execution mode")

        try:
            granted = 0.0
            while granted < h.HELICS_TIME_MAXTIME - self.static.deltat:
                granted = h.helicsFederateRequestTime(
                    self.fed, granted + self.static.deltat
                )
                self.step(granted)
        finally:
            self.destroy()

    def destroy(self) -> None:
        """Clean up and disconnect the federate."""
        h.helicsFederateDisconnect(self.fed)
        logger.info("Federate disconnected")
        h.helicsFederateFree(self.fed)
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
