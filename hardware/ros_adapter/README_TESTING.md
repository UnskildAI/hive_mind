# ROS Adapter Testing & Validation Suite

This suite provides comprehensive testing for the ROS Hardware Adapter, ensuring reliability, determinism, and safety.

## Prerequisites

- ROS 2 (Jazzy) installed and sourced.
- Python dependencies: `rclpy`, `requests`, `numpy`, `cv_bridge`, `pydantic`.
- Common schemas in `common/schemas`.

## Running Automated Tests

Run the full unit and integration test suite using `pytest`:

```bash
# From the hardware/ros_adapter directory
PYTHONPATH=../../ pytest tests/
```

### Test Coverage
- **Unit Tests** (`test_converters.py`): Verifies message conversion logic and schema compliance.
- **Integration Tests** (`test_adapter_node.py`): Verifies the full loop from ROS callback to HTTP request to command publication.
- **Robustness Tests** (`test_robustness.py`): Checks safety mechanisms (NaN handling, pipeline timeouts, failure modes).

## offline Verification

### Mock Pipeline
Run the mock pipeline server to simulate inference:

```bash
python3 mock_pipeline.py
# In another terminal:
curl -X POST "http://localhost:8000/config/latency/50" # Set 50ms latency
```

### Rosbag Replay
Validate the adapter using recorded sensor data:

```bash
python3 tests/rosbag_test.py /path/to/your/test.bag
```
This script replays sensor messages into the adapter callbacks and simulates the control loop, printing generated commands for verification.

### Latency Measurement
Use `tests/latency_utils.py` to instrument code for performance tracking.
