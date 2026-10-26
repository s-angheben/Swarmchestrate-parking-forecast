"""Parking forecasting service: loads the itransformer_wide model once and answers POST /predict.

    uv run forecaster_service.py --model_path checkpoints/chania_itransformer_wide [--port 8000] [--use_gpu]

Request:  {"state": {"time": "2026-08-14T10:00:00+03:00",
                     "sensors": [{"id": ..., "type": ..., "latitude": ..., "longitude": ..., "oc": false}, ...]},
           "messages": [{"id": ..., "type": ..., "latitude": ..., "longitude": ..., "ts": 1786690803, "oc": true}, ...]}
          state: every sensor and its state at state.time; messages: the messages of the 24 h after state.time
          (ts: unix seconds or ISO with offset; type / latitude / longitude may be null)
Answer:   {"forecast": [{"time": ..., "id": ..., "type": ..., "latitude": ..., "longitude": ..., "events": 0.21, "occupancy": 0.43}, ...]}
          next 8 slots of 15 min (2 h) of every sensor from state.time + 24 h, sorted by time and id
The itransformer_wide model uses only id, oc of `state` and id, ts, oc of `messages`: type and location of each sensor
are stored in the checkpoint.
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


class Sensor(BaseModel):
    id: str
    type: str | None
    latitude: float | None
    longitude: float | None
    oc: bool


class Message(Sensor):
    ts: AwareDatetime           # unix seconds, or ISO with offset


class State(BaseModel):
    time: AwareDatetime         # with offset
    sensors: list[Sensor]       # every sensor and its state at `time`


class PredictionRequest(BaseModel):
    state: State
    messages: list[Message]     # the 24 h after state.time


def create_app(model_path, device):
    f = load(model_path, device)
    app = FastAPI(title="Parking forecasting service")
    lock = threading.Lock()

    @app.post("/predict")
    def predict(r: PredictionRequest):
        m = pd.DataFrame([x.model_dump() for x in r.messages], columns=list(Message.model_fields))
        m["ts"] = pd.to_datetime(m.ts, utc=True)
        s = pd.DataFrame([x.model_dump() for x in r.state.sensors], columns=list(Sensor.model_fields))
        missing = f.sensors.index.difference(s.id)
        if len(missing):
            raise HTTPException(422, f"state missing for {len(missing)} sensors: {list(missing)}")
        with lock:                                                      # one request at a time
            out = forecast(f, m, s, r.state.time + pd.Timedelta("24h"))
        out["time"] = out.time.map(pd.Timestamp.isoformat)
        out = out.astype(object).where(out.notna(), None)              # unknown type / location: null (NaN is not JSON)
        return {"forecast": out.to_dict("records")}

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
