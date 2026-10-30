"""Parking forecasting service: loads the models of one city once and answers POST /predict (files in, file out).

    uv run forecaster_service.py --model_path ./checkpoints --city chania [--port 8000] [--use_gpu]

Loads every <model_path>/<city>_<model>/ found (model: wide, sensor_cluster, behavior_cluster, each also with _type =
trained with --with_type).

Request:  {"model": "wide", "time": "2026-08-14T10:00:00+03:00",
           "state_path": ".../state.csv", "messages_path": ".../messages.csv", "metadata_path": ".../metadata.csv",
           "output_path": ".../forecast.csv"}
          model: optional, wide by default (one of the loaded models); time: state time = start of the 24 h, ISO with offset or unix seconds
          state.csv: id, oc (every sensor at `time`); messages.csv: id, ts (unix seconds), oc (the 24 h after `time`);
          metadata.csv: optional, id, type, latitude, longitude (values may be empty; not used by the current models)
Output:   output_path (CSV): time, id, events, occupancy, cluster: the next 8 slots of 15 min (2 h) of every sensor from
          time + 24 h, sorted by time and id (time: start of the slot, unix seconds; cluster: the sensor's cluster,
          empty for wide)
Answer:   {"output_path": ..., "rows": ...}
"""
import argparse
import sys
import threading
from pathlib import Path
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "ai_component"))
from forecast import load, forecast

MODELS = [m + t for m in ["wide", "sensor_cluster", "behavior_cluster"] for t in ["", "_type"]]   # _type: with --with_type


class PredictionRequest(BaseModel):
    model: str = "wide"
    time: AwareDatetime         # state time, with offset
    state_path: str
    messages_path: str
    metadata_path: str | None = None    # optional, not used by the current models
    output_path: str


def read(path, columns):
    """CSV with the given columns, or 422. oc: true/false (any case) or 1/0; ts: unix seconds (integers)."""
    if not Path(path).is_file():
        raise HTTPException(422, f"file not found: {path}")
    try:
        d = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise HTTPException(422, f"{path}: empty file (a header line is needed)")
    if missing := sorted(set(columns) - set(d.columns)):
        raise HTTPException(422, f"{path}: missing columns {missing}")
    d = d[columns].copy()
    if "oc" in columns:
        oc = d.oc.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if oc.isna().any():
            raise HTTPException(422, f"{path}: oc must be true/false or 1/0, found '{d.oc[oc.isna()].iloc[0]}'")
        d["oc"] = oc.astype(bool)
    if "ts" in columns:
        ts = pd.to_numeric(d.ts, errors="coerce")
        bad = ts.isna() | (ts % 1 != 0) | (ts < 1e9) | (ts >= 1e10)       # unix seconds: years 2001 - 2286
        if bad.any():
            raise HTTPException(422, f"{path}: ts must be unix seconds (integers), found '{d.ts[bad].iloc[0]}'")
        d["ts"] = pd.to_datetime(ts.astype("int64"), unit="s", utc=True)
    return d


def create_app(model_path, city, device):
    models = {m: load(p, device) for m in MODELS if (p := Path(model_path) / f"{city}_{m}").is_dir()}
    if not models:
        sys.exit(f"no checkpoint {city}_<model> in {model_path}")
    print("models:", list(models), flush=True)
    app = FastAPI(title="Parking forecasting service")
    lock = threading.Lock()

    @app.post("/predict")
    def predict(r: PredictionRequest):
        if r.model not in models:
            raise HTTPException(422, f"model {r.model} not loaded: {list(models)}")
        f = models[r.model]
        state = read(r.state_path, ["id", "oc"])
        m = read(r.messages_path, ["id", "ts", "oc"])
        metadata = read(r.metadata_path, ["id", "type", "latitude", "longitude"]) if r.metadata_path else None
        if state.id.duplicated().any():
            raise HTTPException(422, f"duplicate sensors in state: {state.id[state.id.duplicated()].unique().tolist()}")
        if len(missing := f.sensors.index.difference(state.id)):
            raise HTTPException(422, f"state missing for {len(missing)} sensors: {list(missing)}")
        with lock:                                                      # one request at a time
            out = forecast(f, m, state, r.time + pd.Timedelta("24h"), metadata)
        out["time"] = (out.time - pd.Timestamp(0, tz="UTC")) // pd.Timedelta("1s")   # unix seconds
        Path(r.output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(r.output_path, index=False)
        return {"output_path": r.output_path, "rows": len(out)}

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parking forecasting service")
    p.add_argument("--model_path", required=True, help="checkpoints folder, e.g. ./checkpoints")
    p.add_argument("--city", required=True, help="e.g. chania: loads <model_path>/<city>_<model>/")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--use_gpu", action="store_true")
    args = p.parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    uvicorn.run(create_app(args.model_path, args.city, device), host=args.host, port=args.port, access_log=False)
