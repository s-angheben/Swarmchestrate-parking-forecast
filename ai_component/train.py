"""Train itransformer_wide on data/processed/<city>/, evaluate it on test, save it in checkpoints/<city>_itransformer_wide/.

    uv run ai_component/train.py --city chania

Settings: 24 h in, 2 h out, a call at every 15-min bin, masked MSE.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import iTransformerWide
from utils.timefeatures import time_features                          # Time-Series-Library (path set by model.py)

SEQ_LEN, PRED_LEN = 96, 8                                              # 24 h in, 2 h out

p = argparse.ArgumentParser()
p.add_argument("--city", default="chania")
args = p.parse_args()
DATA = Path("data/processed") / args.city
OUT = Path("checkpoints") / f"{args.city}_itransformer_wide"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# data: bins x sensors (sensors sorted by id), the three splits together so val/test inputs can use the previous 24 h
d = pd.concat([pd.read_parquet(DATA / f"{s}.parquet", columns=["time", "id", "events", "occupancy", "mask"]).assign(split=s)
               for s in ["train", "val", "test"]])
events, occupancy, mask = (d.pivot(index="time", columns="id", values=n) for n in ["events", "occupancy", "mask"])
split = d.drop_duplicates("time").set_index("time").split.reindex(events.index).values
sensors = events.columns
S = len(sensors)

# static information of each sensor: type index and standardised location (+ missing flag)
meta = pd.read_parquet(DATA / "test.parquet", columns=["id", "type", "latitude", "longitude"]).drop_duplicates("id").set_index("id").loc[sensors]
type_name = meta.type.fillna("unknown")                                 # sensors without metadata: own type
types = sorted(type_name.unique())
sensor_type = type_name.map({t: i for i, t in enumerate(types)}).values
lat, lon = meta.latitude.values, meta.longitude.values
missing = np.isnan(lat)
location = np.stack([np.where(missing, 0.0, (lat - np.nanmean(lat)) / np.nanstd(lat)),
                     np.where(missing, 0.0, (lon - np.nanmean(lon)) / np.nanstd(lon)), missing], axis=1).astype(np.float32)

# scaling: one mean and std per target, over all sensors and train bins with mask = 1
V = np.concatenate([events.values, occupancy.values], axis=1).astype(np.float32)   # (bins, 2S): events, then occupancy
Mk = np.concatenate([mask.values, mask.values], axis=1).astype(np.float32)
tr = split == "train"
mu, sd = np.zeros(2, np.float32), np.zeros(2, np.float32)
for k, cols in enumerate([slice(0, S), slice(S, 2 * S)]):
    v = V[tr, cols][Mk[tr, cols] == 1]
    mu[k], sd[k] = v.mean(), v.std()
Z = ((V - np.repeat(mu, S)) / np.repeat(sd, S)).astype(np.float32)
TF = time_features(events.index, freq="15min").T.astype(np.float32)   # (bins, 5) calendar features, local time


class Windows(Dataset):
    """One sample per call inside a split: input Z[c-96:c], calendar, output Z[c:c+8] and its mask."""
    def __init__(self, part):
        idx = np.where(split == part)[0]
        self.calls = np.arange(max(idx[0], SEQ_LEN), idx[-1] - PRED_LEN + 2)
    def __len__(self):
        return len(self.calls)
    def __getitem__(self, i):
        c = self.calls[i]
        return Z[c - SEQ_LEN:c], TF[c - SEQ_LEN:c], Z[c:c + PRED_LEN], Mk[c:c + PRED_LEN]


data = {s: Windows(s) for s in ["train", "val", "test"]}
cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=SEQ_LEN, pred_len=PRED_LEN,
                      e_layers=4, d_model=512, d_ff=512, n_heads=8, dropout=0.1,
                      factor=1, activation="gelu", embed="timeF", freq="t", output_attention=False,
                      enc_in=2 * S, num_class=1, n_types=len(types), types=types,
                      batch_size=32, learning_rate=1e-4, lradj="type1", train_epochs=10, patience=3, loss="masked MSE", seed=0)
json.dump(vars(cfg), open(OUT / "config.json", "w"), indent=2)
json.dump({"mean": mu.tolist(), "std": sd.tolist(), "targets": ["events", "occupancy"]}, open(OUT / "scaler.json", "w"), indent=2)
meta.to_csv(OUT / "sensors.csv")                                       # the model's column order + metadata (empty if unknown)
print(f"{S} sensors, types {types}, windows { {s: len(w) for s, w in data.items()} }, device {device}")

# training: Adam, learning rate halved every epoch, best model on validation kept, early stopping
torch.manual_seed(cfg.seed)
model = iTransformerWide(cfg, sensor_type, location).to(device)
opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
train_loader = DataLoader(data["train"], batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(data["val"], batch_size=64)


def run_epoch(loader, train):
    model.train(train)
    total, n = 0.0, 0.0
    with torch.set_grad_enabled(train):
        for x, xm, y, m in loader:
            x, xm, y, m = x.to(device), xm.to(device), y.to(device), m.to(device)
            loss = (((model(x, xm) - y) ** 2) * m).sum() / m.sum()            # masked MSE
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * m.sum().item(); n += m.sum().item()
    return total / n


log, best, wait = [], np.inf, 0
for epoch in range(1, cfg.train_epochs + 1):
    lr = cfg.learning_rate * 0.5 ** (epoch - 1)
    for g in opt.param_groups:
        g["lr"] = lr
    t0 = time.time()
    train_loss, val_loss = run_epoch(train_loader, True), run_epoch(val_loader, False)
    log.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, "seconds": round(time.time() - t0, 1)})
    pd.DataFrame(log).to_csv(OUT / "log.csv", index=False)
    print(f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.1e} | {log[-1]['seconds']} s", flush=True)
    if val_loss < best:
        best, wait = val_loss, 0
        torch.save(model.state_dict(), OUT / "model.pt")
    else:
        wait += 1
        if wait >= cfg.patience:
            print("early stopping"); break

# test: best model, one forward pass per call gives all sensors; scaled back and clipped (events >= 0, occupancy 0-1)
model.load_state_dict(torch.load(OUT / "model.pt", map_location=device))
model.eval()
with torch.no_grad():
    P = np.concatenate([model(x.to(device), xm.to(device)).cpu().numpy() for x, xm, _, _ in DataLoader(data["test"], batch_size=64)])
P = P * np.repeat(sd, S) + np.repeat(mu, S)
calls = data["test"].calls
Y = np.stack([V[c:c + PRED_LEN] for c in calls])                       # true values, real units
M = np.stack([Mk[c:c + PRED_LEN, :S] for c in calls])                  # output mask (same for both targets)
pred = {"events": P[..., :S].clip(0), "occupancy": P[..., S:].clip(0, 1)}
true = {"events": Y[..., :S], "occupancy": Y[..., S:]}


def masked_metrics(p, y, m):
    """p, y, m: (calls, steps, sensors). MAE and RMSE overall and per step, only where m == 1."""
    err, n = (p - y) * m, m.sum(axis=(0, 2))
    cols = ["overall"] + [f"+{15 * (h + 1)} min" for h in range(p.shape[1])]
    return pd.DataFrame([[np.abs(err).sum() / m.sum()] + list(np.abs(err).sum(axis=(0, 2)) / n),
                         [np.sqrt((err ** 2).sum() / m.sum())] + list(np.sqrt((err ** 2).sum(axis=(0, 2)) / n))],
                        index=["MAE", "RMSE"], columns=cols)


def city_avg(a):                                                       # average over working sensors: (calls, 8, 1)
    return ((a * M).sum(axis=2) / M.sum(axis=2))[..., None]


metrics = pd.concat({**{("per sensor", t): masked_metrics(pred[t], true[t], M) for t in pred},
                     **{("city", t): masked_metrics(city_avg(pred[t]), city_avg(true[t]), np.ones((len(calls), PRED_LEN, 1))) for t in pred}},
                    names=["level", "target", "metric"])
metrics.to_csv(OUT / "metrics.csv")
print(f"test: {len(calls)} calls")
print(metrics["overall"].unstack(["target", "metric"]).round(4).to_string())
print("saved in", OUT)
