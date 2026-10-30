# Swarmchestrate-parking-forecast

Parking forecast: from the sensor messages of the last 24 hours, it predicts the next 2 hours of every parking sensor
(events and occupancy per 15 minutes). Includes data preparation, training of three models and a web service.

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

### Data and checkpoints
- Unzip `data_checkpoints.zip` in the repo folder: it creates `data/` (raw and prepared data, examples) and `checkpoints/` (trained models)
```
unzip data_checkpoints.zip
```

### Files
Three CSV formats; every time in a CSV is in unix seconds:
- **messages**: `id, ts, oc` (`oc` = true if occupied from that moment; true/false in any case, or 1/0)
- **metadata**: `id, type, latitude, longitude` (values may be empty)
- **state**: `id, oc` (state of every sensor at a given time)

| folder | files |
|---|---|
| `data/raw/<city>/` | `messages.csv`, `metadata.csv` |
| `data/processed/<city>/` | `train.parquet`, `val.parquet`, `test.parquet`: 15-min bins `time, id, type, latitude, longitude, events, occupancy, mask`, for training |
| | for the simulator: `messages_test.csv` (messages of the test period), `state_test.csv` (every sensor at the test start: `id, type, latitude, longitude, oc, failed`), `metadata.csv`, `failures.csv` (`id, start, end`) |
| `checkpoints/<city>_<model>[_type]/` | `config.json` (incl. `with_type`, `types`, `tz`), `scaler.json`, `sensors.csv` (the model's sensors + metadata), `model.pt`, `metrics.csv`, `log.csv` (+ `clusters.csv` or `centroids.json`) |

Failures: a sensor silent for more than 3 days is failed (`mask` = 0 in the bins, `failures.csv`; `failed` in `state_test.csv`
= silent for more than 3 days after its last message before the test, known only afterwards). The models are trained on all
data, with failures as the API sees them (no events, last state carried forward); `mask` is used only for the test metrics.

Clock: dates and 15-min bins are on the clock of `--tz` of `prepare_data.py` (default `Europe/Athens`), saved in each checkpoint
(`tz` in `config.json`): the models learn the daily pattern on that clock. A synthetic trace must start at 00:00 of that clock,
so that its daily pattern lines up.

### Pipeline
Four steps per city: raw data → prepare → train → serve. The split dates are required: days at 00:00 of the clock (`--tz`,
default Europe/Athens); prepare stops if they are not in order or a split has no messages.

#### Chania (real data, 204 sensors)
```
# 1. raw data: data/raw/chania/messages.csv and metadata.csv (in data_checkpoints.zip)

# 2. prepare: clean messages, failures, 15-min bins, train / val / test split
uv run ai_component/prepare_data.py --city chania --start 2025-12-16 --val-start 2026-07-12 --test-start 2026-08-10

# 3. train the models (one checkpoint folder each: checkpoints/chania_<model>/)
uv run ai_component/train.py --city chania --model wide
uv run ai_component/train.py --city chania --model sensor_cluster --k 10
uv run ai_component/train.py --city chania --model behavior_cluster --k 10
#    the same three with the sensor type (checkpoints/chania_<model>_type/)
uv run ai_component/train.py --city chania --model wide --with_type
uv run ai_component/train.py --city chania --model sensor_cluster --k 10 --with_type
uv run ai_component/train.py --city chania --model behavior_cluster --k 10 --with_type

# 4. serve them
uv run forecaster_service.py --model_path ./checkpoints --city chania --port 8000
```

#### Synthetic profile from the simulator (1,000 sensors, 60 days), e.g. BALANCED
```
# 1. raw data: the simulator's export, copied with the fixed names
mkdir -p data/raw/balanced
cp BALANCED-messages-1000.csv data/raw/balanced/messages.csv
cp BALANCED-metadata-1000.csv data/raw/balanced/metadata.csv

# 2. prepare: 30 days train, 15 val, 15 test
uv run ai_component/prepare_data.py --city balanced --start 2026-08-11 --val-start 2026-09-10 --test-start 2026-09-25

# 3. train (wide: batch 4, the largest that fits an 8 GB GPU with 1,000 sensors; 3 epochs ≈ 26 min, 10 ≈ 90 min)
uv run ai_component/train.py --city balanced --model wide --batch_size 4 --epochs 3
uv run ai_component/train.py --city balanced --model sensor_cluster --k 10
uv run ai_component/train.py --city balanced --model behavior_cluster --k 10

# 4. serve them
uv run forecaster_service.py --model_path ./checkpoints --city balanced --port 8000
```
No `--with_type` here: in the synthetic data `type` is the zone that generated the events (see Models).
Another profile: the same commands with its name (e.g. `COMMERCIAL_ONLY` → `data/raw/commercial_only`, `--city commercial_only`).
Each profile is its own dataset (its sensors are not the same as in the other profiles).

After step 2 the simulator replays `data/processed/<city>/messages_test.csv` from `state_test.csv`; at each step it writes
the state and the messages of the last 24 h as CSV and calls the service (see API).

### Models
All: 24 h in, 2 h out, a call at every 15-min bin, iTransformer.
- `wide`: one model over all sensors (events and occupancy of every sensor in and out)
- `sensor_cluster`: sensors grouped once by k-means on their average day over train (`--k 10`); the model forecasts the
  cluster means and each sensor gets its cluster's forecast
- `behavior_cluster`: k-means centroids from the train sensor-days (`--k 10`); at every call each sensor joins the nearest
  centroid from its last 24 h; the model forecasts the cluster means and each sensor gets its cluster's forecast

Clustering features: mean events (log) and occupancy at each hour of the day.

`--with_type`: also use the sensor type (`type` of `metadata.csv`; missing = `unknown`). Off by default; saved as a separate
checkpoint `<city>_<model>_type`.

| model | without (default) | `--with_type` |
|---|---|---|
| `wide` | column embedding = sensor id + location + target | + type |
| `sensor_cluster` | k-means on the average day (48 values) | + type one-hot: clusters follow the types first, then the behaviour |
| `behavior_cluster` | each sensor assigned from its last 24 h (48 values) | + its type one-hot (fixed per sensor): fewer cluster changes |

In k-means the type one-hot is scaled to weigh as much as the events (24 values); the type of each sensor comes from the
checkpoint. Synthetic data: `type` is the zone (COMMERCIAL, LOADING, RESIDENTIAL) that generated the events, so a model with
type is given the answer; do not use `--with_type` there for a fair comparison.

`metrics.csv`: MAE and RMSE per sensor and for
the city average, on `all` data and on `working` sensors only.

### Web service
Forecasts the next 2 hours of every sensor from the last 24 hours, with any model of the city.
```
uv run forecaster_service.py --model_path ./checkpoints --city chania --port 8000
  # --use_gpu
```

#### API (`POST /predict`)
The request carries file paths (as seen by the service); data in and out is CSV.

**`time` is the start of the 24 h of input, not the forecast moment.** To forecast from `now`, send `time` = `now` − 24 h,
the state of every sensor at that moment and the messages between `time` and `now`; the forecast covers `now` → `now` + 2 h.
```
time = now - 24 h                             now                  now + 2 h
|<--------- messages.csv: [time, now) --------->|<----- forecast.csv ----->|
^ state.csv: every sensor at time                 8 slots of 15 min
```
- `time` should be on a 15-min mark (:00, :15, :30, :45), the grid the models were trained on. Off the grid it still works:
  the slots start at the same minutes (e.g. 10:07, 10:22, …), slightly shifted from the training grid.
- Messages outside [`time`, `now`) are ignored. If `time` is the forecast moment by mistake, every message falls outside the
  window: the answer is a valid but wrong forecast for one day later, with no error.
- `time` may have any offset or be unix seconds; the service converts it to the model's clock (`tz` of the checkpoint).

```
{"model": "wide",                                  # optional: wide (default) | sensor_cluster | behavior_cluster, or with _type
 "time": "2026-08-14T10:00:00+03:00",              # state time = start of the 24 h: ISO with timezone, or unix seconds
 "state_path": ".../state.csv",                    # state: every sensor at `time`
 "messages_path": ".../messages.csv",              # messages: the 24 h after `time`
 "metadata_path": ".../metadata.csv",              # optional; not used by the current models (stored in the checkpoint)
 "output_path": ".../forecast.csv"}
```
Output `forecast.csv`: the next 2 h from `time` + 24 h, every sensor in 8 slots of 15 min, sorted by time and id:
`time, id, events, occupancy, cluster`
- `time`: start of the slot (unix seconds)
- `events`: expected number of state changes (free ↔ occupied) of the sensor in the slot; a decimal >= 0, it can be above 1
- `occupancy`: expected fraction of the slot the spot is occupied, from 0 (free the whole slot) to 1 (occupied the whole slot)
- `cluster`: the sensor's cluster (cluster models: all sensors of a cluster get the same forecast); empty for `wide`

Answer: `{"output_path": ..., "rows": ...}`. Errors (422, with the file and the value): unknown model, missing or empty
file, missing column, `oc` not true/false or 1/0, `ts` not unix seconds (e.g. ISO text or milliseconds), a sensor of the
model missing in the state, duplicate sensors in the state, `time` without timezone.

#### Example
Forecast from 2026-08-15 10:00, files in `data/examples/` (state at 2026-08-14 10:00, messages until 2026-08-15 10:00):
```
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{
  \"model\": \"behavior_cluster\", \"time\": \"2026-08-14T10:00:00+03:00\",
  \"state_path\": \"$PWD/data/examples/state.csv\", \"messages_path\": \"$PWD/data/examples/messages.csv\",
  \"metadata_path\": \"$PWD/data/examples/metadata.csv\", \"output_path\": \"$PWD/data/examples/forecast.csv\"}"
```

### Acknowledgement
The model builds upon [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
