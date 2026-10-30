"""Load a model from a checkpoint folder (wide, sensor_cluster or behavior_cluster, with or without type) and forecast
the next 2 h of every sensor from the state 24 h ago and the messages of the last 24 h."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from model import iTransformerWide
from utils.timefeatures import time_features                          # Time-Series-Library (path set by model.py)
from preprocess import BIN, window_tables
import clusters


def load(model_path, device="cpu"):
    """config.json, scaler.json, sensors.csv (the model's sensors, in order), model.pt and the cluster file of a checkpoint."""
    p = Path(model_path)
    cfg = SimpleNamespace(**json.load(open(p / "config.json")))
    scaler = json.load(open(p / "scaler.json"))
    sensors = pd.read_csv(p / "sensors.csv", index_col="id")
    K = len(sensors) if cfg.model == "wide" else cfg.k                 # model columns per target
    model = iTransformerWide(cfg, np.zeros(K, int), np.zeros((K, 3)))  # type / location: loaded from model.pt
    model.load_state_dict(torch.load(p / "model.pt", map_location=device))
    f = SimpleNamespace(model=model.to(device).eval(), cfg=cfg, sensors=sensors, device=device, K=K,
                        mu=np.repeat(np.float32(scaler["mean"]), K), sd=np.repeat(np.float32(scaler["std"]), K))
    if cfg.model == "sensor_cluster":
        f.labels = pd.read_csv(p / "clusters.csv", index_col="id").cluster.loc[sensors.index].values
    if cfg.model == "behavior_cluster":
        c = json.load(open(p / "centroids.json"))
        f.centroids, f.scale = np.array(c["centroids"]), np.array(c["scale"])
    if cfg.with_type:                                                   # type one-hot of each sensor, columns = cfg.types
        f.onehot = np.eye(len(cfg.types))[sensors.type.fillna("unknown").map({t: i for i, t in enumerate(cfg.types)}).values]
    return f


def forecast(f, messages, state, time, metadata=None):
    """messages: id, ts (UTC), oc of the 24 h before `time`; state: id, oc of every sensor at `time` - 24 h;
    metadata: id, type, latitude, longitude of the request, if given (not used by the current models).
    Returns time, id, events, occupancy, cluster for the next 8 bins (2 h) of every sensor, sorted by time and id
    (cluster: the sensor's cluster, empty for wide)."""
    t = pd.Timestamp(time).tz_convert(f.cfg.tz)                         # start of the first forecast slot, model's clock
    ids, S, K = f.sensors.index, len(f.sensors), f.K
    events, occupancy, _ = window_tables(messages, state, t, ids)
    E, O = events.values, occupancy.values                              # (96, S)
    lab = np.arange(S)
    if f.cfg.model == "sensor_cluster":
        lab = f.labels
    if f.cfg.model == "behavior_cluster":
        F = clusters.day_features(E, O, events.index.hour.values)
        lab = clusters.assign(np.hstack([F, f.onehot]) if f.cfg.with_type else F, f.centroids, f.scale)
    if f.cfg.model != "wide":
        E, O = clusters.pool(E, lab, K)[0], clusters.pool(O, lab, K)[0]   # (96, K) cluster means
    x = (np.concatenate([E, O], axis=1).astype(np.float32) - f.mu) / f.sd                    # (96, 2K), scaled
    xm = time_features(events.index, freq="15min").T.astype(np.float32)                      # (96, 5), local time
    with torch.no_grad():
        y = f.model(torch.tensor(x[None], device=f.device), torch.tensor(xm[None], device=f.device))[0].cpu().numpy()
    y = y * f.sd + f.mu                                                  # (8, 2K), real units
    n = f.cfg.pred_len
    return pd.DataFrame({"time": np.repeat(pd.date_range(t, periods=n, freq=BIN), S), "id": np.tile(ids, n),
                         "events": y[:, lab].clip(0).ravel(), "occupancy": y[:, K + lab].clip(0, 1).ravel(),
                         "cluster": None if f.cfg.model == "wide" else np.tile(lab, n)})
