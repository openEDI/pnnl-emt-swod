import sys
from pathlib import Path

# Insert package source directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oedisi.types.data_types import (  # type: ignore # noqa: E402
    CurrentsMagnitude,
    MeasurementArray,
    VoltagesMagnitude,
)


def test_voltages_magnitude_adapter() -> None:
    """Test that VoltagesMagnitude correctly validates expected input formats."""
    data = {
        "ids": ["bus1", "bus2"],
        "values": [1.002, 0.998],
        "time": 12.5,
        "units": "V",
    }
    model = VoltagesMagnitude.model_validate(data)
    assert model.ids == ["bus1", "bus2"]
    assert model.values == [1.002, 0.998]
    assert model.time.timestamp() == 12.5


def test_currents_magnitude_adapter() -> None:
    """Test that CurrentsMagnitude correctly validates expected input formats."""
    data = {
        "ids": ["line1", "line2"],
        "values": [0.015, 0.022],
        "time": 12.5,
        "units": "A",
    }
    model = CurrentsMagnitude.model_validate(data)
    assert model.ids == ["line1", "line2"]
    assert model.values == [0.015, 0.022]
    assert model.time.timestamp() == 12.5


def test_measurement_array_adapter() -> None:
    """Test that MeasurementArray serializes output data correctly."""
    data = {
        "ids": ["bus1", "bus2"],
        "values": [50.01, 49.99],
        "time": 12.5,
        "units": "Hz",
    }
    model = MeasurementArray.model_validate(data)
    serialized = model.model_dump_json()
    assert "units" in serialized
    assert "Hz" in serialized
