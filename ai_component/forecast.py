"""Load itransformer_wide from a checkpoint folder and forecast the next 2 h of every sensor from the state 24 h ago
and the messages of the last 24 h."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from model import iTransformerWide
from utils.timefeatures import time_features                          # Time-Series-Library (path set by model.py)
from preprocess import TZ, BIN, window_tables


def load(model_path, device="cpu"):
    """config.json, scaler.json, sensors.csv (the model's column order + metadata) and model.pt of a checkpoint."""
    p = Path(model_path)
    cfg = SimpleNamespace(**json.load(open(p / "config.json")))
    scaler = json.load(open(p / "scaler.json"))
    sensors = pd.read_csv(p / "sensors.csv", index_col="id")
    S = len(sensors)
    model = iTransformerWide(cfg, np.zeros(S, int), np.zeros((S, 3)))   # type / location: loaded from model.pt
    model.load_state_dict(torch.load(p / "model.pt", map_location=device))
    return SimpleNamespace(model=model.to(device).eval(), cfg=cfg, sensors=sensors, device=device,
                           mu=np.repeat(np.float32(scaler["mean"]), S), sd=np.repeat(np.float32(scaler["std"]), S))


def forecast(f, messages, state, time):
    """messages: id, ts (UTC), oc of the 24 h before `time`; state: id, oc of every sensor at `time` - 24 h.
    Returns time, id, type, latitude, longitude, events, occupancy for the next 8 bins (2 h) of every sensor, sorted by time and id."""
    t = pd.Timestamp(time).tz_convert(TZ)                               # start of the first forecast slot
    sensors, S = f.sensors.index, len(f.sensors)
    events, occupancy, _ = window_tables(messages, state, t, sensors)
    x = (np.concatenate([events.values, occupancy.values], axis=1).astype(np.float32) - f.mu) / f.sd   # (96, 2S), scaled
    xm = time_features(events.index, freq="15min").T.astype(np.float32)                                # (96, 5), local time
    with torch.no_grad():
        y = f.model(torch.tensor(x[None], device=f.device), torch.tensor(xm[None], device=f.device))[0].cpu().numpy()
    y = y * f.sd + f.mu                                                  # (8, 2S), real units
    out = pd.DataFrame({"time": np.repeat(pd.date_range(t, periods=f.cfg.pred_len, freq=BIN), S),
                        "id": np.tile(sensors, f.cfg.pred_len),
                        "events": y[:, :S].clip(0).ravel(), "occupancy": y[:, S:].clip(0, 1).ravel()})
    return out.join(f.sensors, on="id")[["time", "id", "type", "latitude", "longitude", "events", "occupancy"]]
