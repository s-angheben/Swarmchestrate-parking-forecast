# Swarmchestrate-parking-forecast

Parking forecast: from the sensor messages of the last 24 hours, it predicts the next 2 hours of every parking sensor
(events and occupancy per 15 minutes). Includes data preparation, training and a web service.

### Setup
- Clone the repo
```
git clone https://github.com/s-angheben/Swarmchestrate-parking-forecast.git
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
  (`oc` = state, the same as `state.sensors` in the web service request; `failed` = sensor failed at the test start)
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
- Prepare the data (clean messages, failures, 15-min bins and train / val / test splits; each date starts at 00:00 Greek time, Europe/Athens):
```
uv run ai_component/prepare_data.py --city chania --start 2025-12-16 --val-start 2026-07-12 --test-start 2026-08-10
```
- Train the model (`itransformer_wide`: one iTransformer over all sensors, 24 h in, 2 h out):
```
uv run ai_component/train.py --city chania
```
The model and the test metrics are saved in `checkpoints/chania_itransformer_wide/`.

### Web service
Forecasts the next 2 hours of every sensor from the last 24 hours.

- Start the service:
```
uv run forecaster_service.py --model_path ./checkpoints/chania_itransformer_wide --port 8000
  # --use_gpu
```

#### API (`POST /predict`)
All times are Greek time (Europe/Athens): the model learned the daily pattern of the city on the Greek clock.
Every time sent must include its timezone, so the moment is unambiguous (`2026-08-14T10:00:00+03:00` = `2026-08-14T07:00:00Z`).

**Input: the last 24 h**
- `state`: the starting point of the 24 h
  - `time`: when the state applies, any moment, with timezone (e.g. `2026-08-14T10:00:00+03:00`)
  - `sensors`: every sensor and its state at `time`: `id, type, latitude, longitude, oc`
    (`oc` = true if occupied; `type`, `latitude`, `longitude` may be null)
- `messages`: the asynchronous messages of the 24 h after `state.time`, sent by the sensors at any moment
  (not on a time grid): `id, type, latitude, longitude, ts, oc`
  (`ts` = when the message was sent, unix seconds or ISO with offset; `oc` = state of the sensor from that moment;
  `type`, `latitude`, `longitude` may be null)

**Output: the next 2 h**, from `state.time` + 24 h, for every sensor in 8 slots of 15 min, sorted by time and id: `time, id, type, latitude, longitude, events, occupancy`
- `time`: start of the slot
- `events`: expected number of state changes (free ↔ occupied) of the sensor in the slot; a decimal >= 0, it can be above 1
- `occupancy`: expected fraction of the slot the spot is occupied, from 0 (free the whole slot) to 1 (occupied the whole slot)

#### Example
Forecast from 2026-08-15 10:00, files in `data/examples/`:
- `state.json`: the `state`: time 2026-08-14 10:00 and every sensor with its state at that time
- `messages.json`: the messages from 2026-08-14 10:00 to 2026-08-15 10:00
```
printf '{"state": %s, "messages": %s}' \
  "$(cat data/examples/state.json)" "$(cat data/examples/messages.json)" \
  | curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @-
```

Answer:
```
{"time": "2026-08-15T10:00:00+03:00",
 "forecast": [{"time": "2026-08-15T10:00:00+03:00", "id": "ISLPRK1-60506-22524", "type": "accessible parking space",
               "latitude": 35.507325, "longitude": 24.019731, "events": 0.07, "occupancy": 0.04}, ...]}
```

### Assumptions
- Every sensor works: a broken sensor cannot be detected, it looks like a spot that keeps its state.
- `state.sensors` lists every sensor of the city (204 for chania); if one is missing, the service answers with an error.

### Acknowledgement
The model builds upon [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
