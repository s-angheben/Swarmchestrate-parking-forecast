"""Raw data of one city (data/raw/<city>/) -> data/processed/<city>/.

    uv run ai_component/prepare_data.py --city chania --start 2025-12-16 --val-start 2026-07-12 --test-start 2026-08-10

Dates and bins on the clock of --tz (default Europe/Athens): the models learn the daily pattern on that clock.

Input (data/raw/<city>/):
- messages.csv: id, ts (unix seconds), oc (true / false)
- metadata.csv: id, type, latitude, longitude (values may be empty)

Output (data/processed/<city>/):
- train.parquet, val.parquet, test.parquet: 15-min bins (time, id, type, latitude, longitude, events, occupancy, mask),
  for training (mask = 0: no data, used only for evaluation)
- for the simulator (CSV, times in unix seconds):
  - messages_test.csv: the cleaned messages of the test period (id, ts, oc)
  - state_test.csv: every sensor at the test start (id, type, latitude, longitude, oc, failed)
  - metadata.csv: id, type, latitude, longitude
  - failures.csv: failure periods (id, start, end), one row per period
failed = silent for more than 3 days after its last message before the test (known only afterwards).
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from preprocess import BIN, SILENCE, load_csv, clean, data_end, tables

p = argparse.ArgumentParser()
p.add_argument("--city", default="chania")
p.add_argument("--start", required=True)                # first bin, e.g. 2025-12-16 (00:00 on the --tz clock)
p.add_argument("--val-start", required=True)            # first bin of validation
p.add_argument("--test-start", required=True)           # first bin of test
p.add_argument("--tz", default="Europe/Athens")         # clock of the dates and bins
args = p.parse_args()
start, val_start, test_start = (pd.Timestamp(d, tz=args.tz) for d in (args.start, args.val_start, args.test_start))

RAW = Path("data/raw") / args.city
OUT = Path("data/processed") / args.city
OUT.mkdir(parents=True, exist_ok=True)
unix = lambda t: (t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")

# messages: drop duplicates (ACK retries), cut at the end of the last full 15-min bin of the export
raw = load_csv(RAW / "messages.csv")
end = data_end(clean(raw, raw.ts.max() + BIN)).tz_convert(args.tz)
m = clean(raw, end)
sensors = pd.Index(sorted(m.id.unique()), name="id")
meta = pd.read_csv(RAW / "metadata.csv").drop_duplicates("id", keep="last").set_index("id")[["type", "latitude", "longitude"]]
meta = meta.reindex(sensors)                                          # sensors without metadata: empty
if not start < val_start < test_start < end:
    sys.exit(f"dates must be start < val-start < test-start < end of the data ({end}); first message {m.ts.min().tz_convert(args.tz)}")
for name, a, b in [("train", start, val_start), ("val", val_start, test_start), ("test", test_start, end)]:
    if not ((m.ts >= a) & (m.ts < b)).any():
        sys.exit(f"no messages in {name} ({a} -> {b}): data from {m.ts.min().tz_convert(args.tz)} to {end}")

# failures: no message for more than SILENCE (3 days, data provider rule), until the next message or the end of the data
# (no data before the first message of a sensor: not listed here, but masked in the bins)
nxt = m.groupby("id").ts.shift(-1).fillna(end.tz_convert("UTC"))     # next message of the sensor, or the end of the data
failure_after = nxt - m.ts > SILENCE                                  # True = no data from this message until the next one
pd.DataFrame({"id": m.id, "start": unix(m.ts), "end": unix(nxt)})[failure_after] \
    .sort_values(["start", "id"]).to_csv(OUT / "failures.csv", index=False)

# simulator: test messages (1 -> 1 repeats kept: real messages), state at the test start, metadata
test = m[m.ts >= test_start].sort_values(["ts", "id"])
test.assign(ts=unix(test.ts))[["id", "ts", "oc"]].to_csv(OUT / "messages_test.csv", index=False)
before = m.ts < test_start
last = m[before].drop_duplicates("id", keep="last").set_index("id")  # last message before the test (m sorted by id, ts)
failed = failure_after[before].groupby(m.id[before]).last()
meta.assign(oc=last.oc.reindex(sensors, fill_value=False), failed=failed.reindex(sensors, fill_value=True)) \
    .reset_index().to_csv(OUT / "state_test.csv", index=False)
meta.reset_index().to_csv(OUT / "metadata.csv", index=False)

# bins: events, occupancy, mask per sensor and 15-min bin (local time), long format, split by date
events, occupancy, mask = tables(m, start, end, sensors)
bins = pd.DataFrame({"events": events.stack(), "occupancy": occupancy.stack(), "mask": mask.stack()})
bins = bins.rename_axis(["time", "id"]).reset_index().join(meta, on="id")
bins = bins[["time", "id", "type", "latitude", "longitude", "events", "occupancy", "mask"]]
for name, a, b in [("train", start, val_start), ("val", val_start, test_start), ("test", test_start, end)]:
    bins[(bins.time >= a) & (bins.time < b)].to_parquet(OUT / f"{name}.parquet", index=False)

print(f"{args.city}: {len(sensors)} sensors, {len(m):,} messages, {failure_after.sum()} failures, "
      f"bins {start} -> {end}, val from {val_start}, test from {test_start} -> {OUT}")
