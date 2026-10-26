"""Raw export of one city (data/raw/<city>/) -> data/processed/<city>/.

    uv run ai_component/prepare_data.py --city chania --start 2025-12-16 --val-start 2026-07-12 --test-start 2026-08-10

Output:
- sensors.parquet: cleaned messages + metadata (id, type, latitude, longitude, ts, oc, failure_after)
- sensors_test.parquet: the same, only the test period (what the simulator replicates)
- state_test.parquet: every sensor at the start of the test (id, type, latitude, longitude, oc, failed)
- failures.parquet: failure periods (id, start, end), one row per period
- train.parquet, val.parquet, test.parquet: 15-min bins (time, id, type, latitude, longitude, events, occupancy, mask)
"""
import argparse
from pathlib import Path
import pandas as pd
from preprocess import TZ, BIN, SILENCE, load_csv, clean, data_end, tables

p = argparse.ArgumentParser()
p.add_argument("--city", default="chania")
p.add_argument("--start", default="2025-12-16")         # first bin (local time)
p.add_argument("--val-start", default="2026-07-12")     # first bin of validation
p.add_argument("--test-start", default="2026-08-10")    # first bin of test
args = p.parse_args()
start, val_start, test_start = (pd.Timestamp(d, tz=TZ) for d in (args.start, args.val_start, args.test_start))

RAW = Path("data/raw") / args.city
OUT = Path("data/processed") / args.city
OUT.mkdir(parents=True, exist_ok=True)

# messages: drop duplicates (ACK retries), cut at the end of the last full 15-min bin of the export
raw = load_csv(next(RAW.glob("*.csv")))
end = data_end(clean(raw, raw.ts.max() + BIN))
m = clean(raw, end)
sensors = pd.Index(sorted(m.id.unique()), name="id")

# metadata: all .ods files of the city; for an id in several files the newer file wins
# (files sorted by name: a dated file, e.g. ids_YYYYMMDD.ods, comes after ids.ods)
meta = pd.concat([pd.concat(pd.read_excel(f, engine="odf", sheet_name=None).values()).drop_duplicates("id")
                  for f in sorted(RAW.glob("*.ods"))])
meta = meta.drop_duplicates("id", keep="last").set_index("id")[["type", "latitude", "longitude"]].reindex(sensors)

# failures: no message for more than SILENCE (3 days, data provider rule), until the next message or the end of the data
# (no data before the first message of a sensor: not listed here, but masked in the bins)
nxt = m.groupby("id").ts.shift(-1).fillna(end.tz_convert("UTC"))     # next message of the sensor, or the end of the data
m["failure_after"] = nxt - m.ts > SILENCE                            # True = no data from this message until the next one
pd.DataFrame({"id": m.id, "start": m.ts, "end": nxt})[m.failure_after] \
    .sort_values(["start", "id"]).to_parquet(OUT / "failures.parquet", index=False)

# sensors: cleaned messages + metadata + failure_after (1 -> 1 repeats kept: real messages, they count for the silence rule)
s = m.join(meta, on="id")[["id", "type", "latitude", "longitude", "ts", "oc", "failure_after"]].sort_values(["ts", "id"])
s.to_parquet(OUT / "sensors.parquet", index=False)
s[s.ts >= test_start].to_parquet(OUT / "sensors_test.parquet", index=False)  # test period, until the end of the data

# state at the start of the test: last message before it; failed = no data at that moment (or no message yet)
last = s[s.ts < test_start].drop_duplicates("id", keep="last").set_index("id")
meta.assign(oc=last.oc.reindex(sensors, fill_value=False), failed=last.failure_after.reindex(sensors, fill_value=True)) \
    .reset_index().to_parquet(OUT / "state_test.parquet", index=False)

# bins: events, occupancy, mask per sensor and 15-min bin (local time), long format, split by date
events, occupancy, mask = tables(m, start, end, sensors)
bins = pd.DataFrame({"events": events.stack(), "occupancy": occupancy.stack(), "mask": mask.stack()})
bins = bins.rename_axis(["time", "id"]).reset_index().join(meta, on="id")
bins = bins[["time", "id", "type", "latitude", "longitude", "events", "occupancy", "mask"]]
for name, a, b in [("train", start, val_start), ("val", val_start, test_start), ("test", test_start, end)]:
    bins[(bins.time >= a) & (bins.time < b)].to_parquet(OUT / f"{name}.parquet", index=False)

print(f"{args.city}: {len(sensors)} sensors, {len(m):,} messages, {m.failure_after.sum()} failures, "
      f"bins {start} -> {end}, val from {val_start}, test from {test_start} -> {OUT}")
