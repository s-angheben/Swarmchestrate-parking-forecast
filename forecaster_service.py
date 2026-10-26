"""Parking forecasting service: loads the itransformer_wide model once and answers POST /predict.

    uv run forecaster_service.py --model_path checkpoints/chania_itransformer_wide [--port 8000] [--use_gpu]

Request:  {"time": "2026-08-15T10:00:00+03:00",
           "state": [{"id": ..., "type": ..., "latitude": ..., "longitude": ..., "oc": false}, ...],
           "messages": [{"id": ..., "type": ..., "latitude": ..., "longitude": ..., "ts": 1786777200, "oc": true}, ...]}
          state: every sensor and its state at `time` - 24 h; messages: the messages of the 24 h before `time`
          (ts: unix seconds or ISO; type / latitude / longitude may be null)
Answer:   {"time": ...,
           "forecast": [{"time": ..., "id": ..., "type": ..., "latitude": ..., "longitude": ..., "events": 0.21, "occupancy": 0.43}, ...]}
          next 8 bins (2 h) of every sensor, sorted by time and id
The itransformer_wide model uses only id, ts, oc: type and location of each sensor are stored in the checkpoint.
"""
import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "ai_component"))
from forecast import load, forecast


class SensorState(BaseModel):
    id: str
    type: str | None
    latitude: float | None
    longitude: float | None
    oc: bool


class Message(SensorState):
    ts: datetime


class PredictionRequest(BaseModel):
    time: datetime              # call time, with offset
    state: list[SensorState]    # every sensor at time - 24 h
    messages: list[Message]


def create_app(model_path, device):
    f = load(model_path, device)
    app = FastAPI(title="Parking forecasting service")
    lock = threading.Lock()

    @app.post("/predict")
    def predict(r: PredictionRequest):
        m = pd.DataFrame([x.model_dump() for x in r.messages], columns=list(Message.model_fields))
        m["ts"] = pd.to_datetime(m.ts, utc=True)
        s = pd.DataFrame([x.model_dump() for x in r.state], columns=list(SensorState.model_fields))
        missing = f.sensors.index.difference(s.id)
        if len(missing):
            raise HTTPException(422, f"state missing for {len(missing)} sensors: {list(missing)}")
        with lock:                                                      # one request at a time
            out = forecast(f, m, s, r.time)
        out["time"] = out.time.map(pd.Timestamp.isoformat)
        out = out.astype(object).where(out.notna(), None)              # unknown type / location: null (NaN is not JSON)
        return {"time": out.time.iloc[0], "forecast": out.to_dict("records")}

    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parking forecasting service")
    p.add_argument("--model_path", required=True, help="checkpoint folder, e.g. checkpoints/chania_itransformer_wide")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--use_gpu", action="store_true")
    args = p.parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    uvicorn.run(create_app(args.model_path, device), host=args.host, port=args.port, access_log=False)
