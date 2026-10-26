# Swarmchestrate-parking-forecast

Parking forecast: from the sensor messages of the last 24 hours, it predicts the next 2 hours of every parking sensor
(events and occupancy per 15 minutes). Includes data preparation, training and a web service.

### Setup
- Clone the repo
```
git clone https://github.com/<user>/Swarmchestrate-parking-forecast.git
cd Swarmchestrate-parking-forecast
```

- Install the environment with `uv`
```
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the venv
uv sync
```

### Data and checkpoint
- Unzip `data_checkpoints.zip` in the repo folder: it creates `data/` (raw and prepared data) and `checkpoints/` (trained model)
```
unzip data_checkpoints.zip
```
Prepared data in `data/processed/chania/`:
- `sensors.parquet`: cleaned messages `id, type, latitude, longitude, ts, oc, failure_after`
  (`failure_after` = true if the sensor is silent for more than 3 days after this message = sensor failure)
- `sensors_test.parquet`: the same messages, only the test period
- `state_test.parquet`: every sensor at the test start: `id, type, latitude, longitude, oc, failed`
  (`oc` = state, the same as `state` in the web service request; `failed` = sensor failed at the test start)
- `failures.parquet`: failure periods `id, start, end`
- `train.parquet`, `val.parquet`, `test.parquet`: 15-min bins `time, id, type, latitude, longitude, events, occupancy, mask`
  (`mask` = 0 during a failure)

#### Test period
| city | test start | test end | initial state | messages |
|---|---|---|---|---|
| chania | 2026-08-10 00:00 +03:00 | 2026-10-09 09:00 +03:00 | `data/processed/chania/state_test.parquet` | `data/processed/chania/sensors_test.parquet` |

#### Data preparation and training (optional)
Only needed to rebuild the data or the model; `data_checkpoints.zip` already contains both.

- Raw data in `data/raw/<city>/`: messages `id, ts, oc` (.csv) and sensor metadata (.ods)
- Prepare the data (clean messages, failures, 15-min bins and train / val / test splits; dates = midnight, local time):
```
uv run ai_component/prepare_data.py --city chania --start 2025-12-16 --val-start 2026-07-12 --test-start 2026-08-10
```
- Train the model (`itransformer_wide`: one iTransformer over all sensors, 24 h in, 2 h out):
```
uv run ai_component/train.py --city chania
```
The model and the test metrics are saved in `checkpoints/chania_itransformer_wide/`.

### Web service
- Start the service:
```
uv run forecaster_service.py --model_path ./checkpoints/chania_itransformer_wide --port 8000
  # --use_gpu
```

- Request: `time` = call time, `state` = every sensor and its state (`oc`) at `time` - 24 h,
  `messages` = all the messages of the 24 h before `time` (`ts` in unix seconds or ISO; `type`, `latitude`, `longitude` may be null)
```
{"time": "2026-08-15T10:00:00+03:00",
 "state": [{"id": "ISLPRK1-60506-22524", "type": "accessible parking space",
            "latitude": 35.507325, "longitude": 24.019731, "oc": false}, ...],
 "messages": [{"id": "ISLPRK1-60519-47185", "type": "accessible parking space",
               "latitude": 35.5104, "longitude": 24.024903, "ts": 1786690803, "oc": true}, ...]}
```

- Example (call at 2026-08-15 10:00, files in `data/examples/`):
  - `state.json`: every sensor and its state at 2026-08-14 10:00 (`time` - 24 h)
  - `messages.json`: the messages from 2026-08-14 10:00 to 2026-08-15 10:00
```
printf '{"time": "2026-08-15T10:00:00+03:00", "state": %s, "messages": %s}' \
  "$(cat data/examples/state.json)" "$(cat data/examples/messages.json)" \
  | curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @-
```

- Answer: the next 8 bins (2 h) of every sensor, sorted by time and id
```
{"time": "2026-08-15T10:00:00+03:00",
 "forecast": [{"time": "2026-08-15T10:00:00+03:00", "id": "ISLPRK1-60506-22524", "type": "accessible parking space",
               "latitude": 35.507325, "longitude": 24.019731, "events": 0.07, "occupancy": 0.04}, ...]}
```

### Assumptions
1. no data of faulty or failed sensors;
2. `state` contains all the sensors (otherwise the service answers with an error).

Duplicates and messages repeating the same state are fine.

### Acknowledgement
The model builds upon [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
